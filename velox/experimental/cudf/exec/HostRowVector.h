/*
 * HostRowVector (2026-08-25): native row output at a CPU exit.
 *
 * The joined rows exactly as they left the GPU (D2H'd row bytes + the
 * compacted string heap + the optional null sidecar), NOT transposed into
 * Velox columns. Every arm of the layout experiments then delivers its
 * result to host memory once, in its own layout: cudf columns (CudfToVelox),
 * rows (this), or rows + host-gathered deferred payload (this, with the
 * gathered columns attached). A RowVector with zero children so sinks and
 * operator stats see size(); Velox columns are produced only on demand
 * (--include_results) through materialize().
 */
#pragma once

#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/MaterializableVector.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox {

/// Extract Velox columns from GPU-layout host rows (eager results only:
/// every output column is in the rows).
inline RowVectorPtr extractHostRows(
    memory::MemoryPool* pool,
    const RowTypePtr& type,
    vector_size_t n,
    const uint8_t* rows,
    int32_t rowWidth,
    const std::vector<FieldDesc>& fields,
    const uint8_t* chars,
    const uint8_t* nulls,
    int32_t nullStride,
    const std::vector<int32_t>* nullBits = nullptr) {
  const int numCols = static_cast<int>(type->size());
  std::vector<VectorPtr> children(numCols);
  for (int i = 0; i < numCols; i++) {
    const auto& fd = fields[i];
    if (fd.offset < 0) {
      continue; // not in the rows (deferred column): the caller supplies it
    }
    auto vec = BaseVector::create(type->childAt(i), n, pool);
    if (fd.kind == kFieldString) {
      auto* fv = vec->template asFlatVector<StringView>();
      for (vector_size_t r = 0; r < n; r++) {
        const uint8_t* slot = rows + static_cast<int64_t>(r) * rowWidth + fd.offset;
        uint32_t len;
        std::memcpy(&len, slot, 4);
        const char* src;
        if (len <= 12) {
          src = reinterpret_cast<const char*>(slot + 4);
        } else {
          uint64_t off;
          std::memcpy(&off, slot + 8, 8);
          src = reinterpret_cast<const char*>(chars + off);
        }
        fv->set(r, StringView(src, len));
      }
    } else {
      auto* dst = static_cast<uint8_t*>(const_cast<void*>(vec->valuesAsVoid()));
      VELOX_CHECK_NOT_NULL(dst);
      const int32_t w = fd.byte_width;
      for (vector_size_t r = 0; r < n; r++) {
        std::memcpy(
            dst + static_cast<int64_t>(r) * w,
            rows + static_cast<int64_t>(r) * rowWidth + fd.offset,
            w);
      }
    }
    if (nullStride > 0 && nulls != nullptr) {
      // Null bit index: the output column by default (join layout); the
      // sort's sidecar is indexed by FIELD, passed via nullBits.
      const int32_t bit = nullBits != nullptr ? (*nullBits)[i] : i;
      const uint8_t byteMask = static_cast<uint8_t>(1u << (bit & 7));
      const int32_t byteIdx = bit >> 3;
      for (vector_size_t r = 0; r < n; r++) {
        if (nulls[static_cast<int64_t>(r) * nullStride + byteIdx] & byteMask) {
          vec->setNull(r, true);
        }
      }
    }
    children[i] = std::move(vec);
  }
  return std::make_shared<RowVector>(pool, type, nullptr, n, std::move(children));
}

/// Keeps a shared host byte region (the D2H'd string heap) alive for a
/// BufferView over it.
struct SharedBytesReleaser {
  std::shared_ptr<const std::vector<uint8_t>> bytes;
  void addRef() const {}
  void release() const {}
};

/// Host extraction of GPU-layout rows into Velox columns (2026-09-28; tiled
/// 2026-09-29), used by every CPU exit of the row operators.
///
/// Rows are processed in cache-sized TILES and every column is extracted
/// per tile, so each row's cache lines come from DRAM once instead of once
/// per column (the column-at-a-time version re-streamed a 332 B row per 8 B
/// field -- ~11 s per driver at the full join; the same lesson as the CPU
/// hybrid extraction). Tiles are multiples of 64 rows, so concurrent tiles
/// never share a null word.
///
/// String slots mirror Velox's StringView byte for byte, so a view is the
/// slot itself with the heap offset of an out-of-line string replaced by its
/// address in `chars`: bodies are NOT copied. Each string column co-owns the
/// heap through a BufferView sized as the body bytes it references (honest
/// accounting, as HybridContainer::addSharedStringBufferViews; a view
/// reporting the whole heap made Velox slice the output into ~500K batches).
///
/// Usage: construct for n output rows, extract() one or more row chunks
/// (in any order; `begin` = the chunk's first output row), then finish().
/// Columns with fields[i].offset < 0 are left nullptr for the caller.
class HostRowExtractor {
 public:
  HostRowExtractor(
      memory::MemoryPool* pool,
      const RowTypePtr& type,
      vector_size_t n,
      int32_t rowWidth,
      std::vector<FieldDesc> fields,
      std::shared_ptr<const std::vector<uint8_t>> chars,
      int32_t nullStride,
      const std::vector<int32_t>* nullBits)
      : rowWidth_(rowWidth),
        fields_(std::move(fields)),
        chars_(std::move(chars)),
        heap_(chars_ != nullptr
                  ? reinterpret_cast<const char*>(chars_->data())
                  : nullptr),
        nullStride_(nullStride),
        children_(type->size()),
        outNulls_(type->size(), nullptr),
        anyNull_(std::make_unique<std::atomic<bool>[]>(type->size())),
        bodyBytes_(std::make_unique<std::atomic<uint64_t>[]>(type->size())) {
    static_assert(sizeof(StringView) == 16);
    const int numCols = static_cast<int>(type->size());
    for (int i = 0; i < numCols; i++) {
      anyNull_[i] = false;
      bodyBytes_[i] = 0;
      if (fields_[i].offset < 0) {
        continue;
      }
      children_[i] = BaseVector::create(type->childAt(i), n, pool);
      if (nullStride_ > 0 && n > 0) {
        outNulls_[i] = children_[i]->mutableRawNulls(); // all not-null
      }
      cols_.push_back(i);
      nullBit_.push_back(nullBits != nullptr ? (*nullBits)[i] : i);
    }
    // ~256 KB of rows per tile (L2-resident across the column passes).
    tileRows_ = std::max<int64_t>(
        64, ((256 << 10) / std::max(1, rowWidth_)) / 64 * 64);
  }

  /// Extract output rows [begin, begin + count): their bytes start at `rows`
  /// and their null sidecar (if any) at `nulls`. `begin` must be a multiple
  /// of 64 when several extract() calls run concurrently.
  void extract(
      int64_t begin,
      int64_t count,
      const uint8_t* rows,
      const uint8_t* nulls,
      int32_t threads) {
    const int64_t numTiles = (count + tileRows_ - 1) / tileRows_;
    auto tile = [&](int64_t t) {
      const int64_t t0 = t * tileRows_;
      const int64_t t1 = std::min<int64_t>(count, t0 + tileRows_);
      for (size_t k = 0; k < cols_.size(); k++) {
        extractColumn(k, begin, t0, t1, rows, nulls);
      }
    };
    const int32_t nThreads = static_cast<int32_t>(
        std::max<int64_t>(1, std::min<int64_t>(threads, numTiles)));
    if (nThreads <= 1) {
      for (int64_t t = 0; t < numTiles; t++) {
        tile(t);
      }
      return;
    }
    std::atomic<int64_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(nThreads);
    for (int32_t w = 0; w < nThreads; w++) {
      workers.emplace_back([&]() {
        for (int64_t t = next++; t < numTiles; t = next++) {
          tile(t);
        }
      });
    }
    for (auto& w : workers) {
      w.join();
    }
  }

  std::vector<VectorPtr> finish() {
    for (int i : cols_) {
      if (outNulls_[i] != nullptr && !anyNull_[i]) {
        children_[i]->resetNulls();
      }
      const uint64_t referenced = bodyBytes_[i];
      if (referenced > 0) {
        children_[i]->template asFlatVector<StringView>()->setStringBuffers(
            {BufferView<SharedBytesReleaser>::create(
                chars_->data(),
                std::min<uint64_t>(referenced, chars_->size()),
                SharedBytesReleaser{chars_})});
      }
    }
    return std::move(children_);
  }

 private:
  // Rows [t0, t1) of the chunk (output rows begin + t0 ...) of the k-th
  // extracted column.
  void extractColumn(
      size_t k,
      int64_t begin,
      int64_t t0,
      int64_t t1,
      const uint8_t* rows,
      const uint8_t* nulls) {
    const int i = cols_[k];
    const auto& fd = fields_[i];
    auto& vec = children_[i];
    const uint8_t* src = rows + fd.offset;
    if (fd.kind == kFieldString) {
      auto* views =
          vec->template asFlatVector<StringView>()->mutableRawValues() + begin;
      uint64_t referenced = 0;
      for (int64_t r = t0; r < t1; r++) {
        const uint8_t* slot = src + r * rowWidth_;
        auto* dst = reinterpret_cast<uint8_t*>(&views[r]);
        std::memcpy(dst, slot, 16);
        uint32_t len;
        std::memcpy(&len, slot, 4);
        if (len > 12) {
          uint64_t off;
          std::memcpy(&off, slot + 8, 8);
          const char* p = heap_ + off;
          std::memcpy(dst + 8, &p, 8);
          referenced += len;
        }
      }
      if (referenced > 0) {
        bodyBytes_[i].fetch_add(referenced, std::memory_order_relaxed);
      }
    } else {
      auto* dst = static_cast<uint8_t*>(const_cast<void*>(vec->valuesAsVoid()));
      const int32_t w = fd.byte_width;
      switch (w) {
        case 8:
          for (int64_t r = t0; r < t1; r++) {
            std::memcpy(dst + (begin + r) * 8, src + r * rowWidth_, 8);
          }
          break;
        case 4:
          for (int64_t r = t0; r < t1; r++) {
            std::memcpy(dst + (begin + r) * 4, src + r * rowWidth_, 4);
          }
          break;
        case 16:
          for (int64_t r = t0; r < t1; r++) {
            std::memcpy(dst + (begin + r) * 16, src + r * rowWidth_, 16);
          }
          break;
        case 2:
          for (int64_t r = t0; r < t1; r++) {
            std::memcpy(dst + (begin + r) * 2, src + r * rowWidth_, 2);
          }
          break;
        case 1:
          for (int64_t r = t0; r < t1; r++) {
            dst[begin + r] = src[r * rowWidth_];
          }
          break;
        default:
          for (int64_t r = t0; r < t1; r++) {
            std::memcpy(dst + (begin + r) * w, src + r * rowWidth_, w);
          }
      }
    }
    if (outNulls_[i] != nullptr && nulls != nullptr) {
      const int32_t bit = nullBit_[k];
      const uint8_t byteMask = static_cast<uint8_t>(1u << (bit & 7));
      const int32_t byteIdx = bit >> 3;
      bool wrote = false;
      for (int64_t r = t0; r < t1; r++) {
        if (nulls[r * nullStride_ + byteIdx] & byteMask) {
          bits::setNull(outNulls_[i], begin + r);
          wrote = true;
        }
      }
      if (wrote) {
        anyNull_[i] = true;
      }
    }
  }

  const int32_t rowWidth_;
  const std::vector<FieldDesc> fields_;
  const std::shared_ptr<const std::vector<uint8_t>> chars_;
  const char* const heap_;
  const int32_t nullStride_;
  std::vector<VectorPtr> children_;
  std::vector<uint64_t*> outNulls_;
  std::vector<int32_t> cols_; // extracted column indices
  std::vector<int32_t> nullBit_; // sidecar bit per extracted column
  std::unique_ptr<std::atomic<bool>[]> anyNull_;
  std::unique_ptr<std::atomic<uint64_t>[]> bodyBytes_;
  int64_t tileRows_;
};

/// One-shot extraction of n rows (see HostRowExtractor).
inline std::vector<VectorPtr> extractHostRowsParallel(
    memory::MemoryPool* pool,
    const RowTypePtr& type,
    vector_size_t n,
    const uint8_t* rows,
    int32_t rowWidth,
    const std::vector<FieldDesc>& fields,
    const std::shared_ptr<const std::vector<uint8_t>>& chars,
    const uint8_t* nulls,
    int32_t nullStride,
    const std::vector<int32_t>* nullBits,
    int32_t threads) {
  HostRowExtractor ex(
      pool, type, n, rowWidth, fields, chars, nullStride, nullBits);
  ex.extract(0, n, rows, nulls, threads);
  return ex.finish();
}

class HostRowVector : public RowVector, public MaterializableVector {
 public:
  HostRowVector(
      memory::MemoryPool* pool,
      RowTypePtr type,
      vector_size_t n,
      std::vector<uint8_t> rows,
      int32_t rowWidth,
      std::vector<FieldDesc> fields,
      std::shared_ptr<const std::vector<uint8_t>> chars,
      std::vector<uint8_t> nulls,
      int32_t nullStride,
      std::vector<int32_t> nullBits = {})
      : RowVector(pool, type, nullptr, n, std::vector<VectorPtr>{}),
        rows_(std::make_shared<const std::vector<uint8_t>>(std::move(rows))),
        offset_(0),
        rowWidth_(rowWidth),
        fields_(std::make_shared<const std::vector<FieldDesc>>(
            std::move(fields))),
        chars_(std::move(chars)),
        nulls_(std::make_shared<const std::vector<uint8_t>>(std::move(nulls))),
        nullStride_(nullStride),
        nullBits_(std::make_shared<const std::vector<int32_t>>(
            std::move(nullBits))) {}

  /// A view of rows [offset, offset + n) sharing every buffer (no copy):
  /// lets the exit emit Velox-sized output batches from one D2H'd chunk.
  VectorPtr slice(vector_size_t offset, vector_size_t n) const override {
    return std::shared_ptr<HostRowVector>(new HostRowVector(
        pool(),
        asRowType(type()),
        n,
        rows_,
        offset_ + offset,
        rowWidth_,
        fields_,
        chars_,
        nulls_,
        nullStride_,
        nullBits_));
  }

  RowVectorPtr materialize() const override {
    return extractHostRows(
        pool(),
        asRowType(type()),
        size(),
        rows_->data() + static_cast<int64_t>(offset_) * rowWidth_,
        rowWidth_,
        *fields_,
        chars_ ? chars_->data() : nullptr,
        nulls_->empty() ? nullptr
                        : nulls_->data() +
                static_cast<int64_t>(offset_) * nullStride_,
        nullStride_,
        nullBits_->empty() ? nullptr : nullBits_.get());
  }

  int64_t rowBytes() const {
    return static_cast<int64_t>(size()) * rowWidth_;
  }

  /// Bytes this vector represents (rows x stride): drives Velox's output
  /// batch sizing (outputBatchRows) the same way a flat RowVector would.
  uint64_t estimateFlatSize() const override {
    return static_cast<uint64_t>(size()) * rowWidth_;
  }

 private:
  // View constructor (slice): shares all buffers.
  HostRowVector(
      memory::MemoryPool* pool,
      RowTypePtr type,
      vector_size_t n,
      std::shared_ptr<const std::vector<uint8_t>> rows,
      vector_size_t offset,
      int32_t rowWidth,
      std::shared_ptr<const std::vector<FieldDesc>> fields,
      std::shared_ptr<const std::vector<uint8_t>> chars,
      std::shared_ptr<const std::vector<uint8_t>> nulls,
      int32_t nullStride,
      std::shared_ptr<const std::vector<int32_t>> nullBits)
      : RowVector(pool, type, nullptr, n, std::vector<VectorPtr>{}),
        rows_(std::move(rows)),
        offset_(offset),
        rowWidth_(rowWidth),
        fields_(std::move(fields)),
        chars_(std::move(chars)),
        nulls_(std::move(nulls)),
        nullStride_(nullStride),
        nullBits_(std::move(nullBits)) {}

  std::shared_ptr<const std::vector<uint8_t>> rows_;
  vector_size_t offset_; // first row of this view within rows_
  int32_t rowWidth_;
  std::shared_ptr<const std::vector<FieldDesc>> fields_;
  std::shared_ptr<const std::vector<uint8_t>> chars_; // heap, shareable
  std::shared_ptr<const std::vector<uint8_t>> nulls_;
  int32_t nullStride_;
  std::shared_ptr<const std::vector<int32_t>> nullBits_; // empty = col index
};

} // namespace facebook::velox::cudf_velox
