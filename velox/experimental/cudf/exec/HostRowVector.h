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

#include <cstring>
#include <memory>
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
        rows_(std::move(rows)),
        rowWidth_(rowWidth),
        fields_(std::move(fields)),
        chars_(std::move(chars)),
        nulls_(std::move(nulls)),
        nullStride_(nullStride),
        nullBits_(std::move(nullBits)) {}

  RowVectorPtr materialize() const override {
    return extractHostRows(
        pool(),
        asRowType(type()),
        size(),
        rows_.data(),
        rowWidth_,
        fields_,
        chars_ ? chars_->data() : nullptr,
        nulls_.empty() ? nullptr : nulls_.data(),
        nullStride_,
        nullBits_.empty() ? nullptr : &nullBits_);
  }

  int64_t rowBytes() const {
    return static_cast<int64_t>(rows_.size());
  }

 private:
  std::vector<uint8_t> rows_;
  int32_t rowWidth_;
  std::vector<FieldDesc> fields_;
  std::shared_ptr<const std::vector<uint8_t>> chars_; // heap, shareable
  std::vector<uint8_t> nulls_;
  int32_t nullStride_;
  std::vector<int32_t> nullBits_; // per output column; empty = column index
};

} // namespace facebook::velox::cudf_velox
