/*
 * RetainedMerge.h
 *
 * Column-wise merge of retained host batches into ONE contiguous flat batch.
 * Fixed-width values are copied straight through a filter's dictionary
 * indices (no per-input flatten, no per-input DecodedVector); string bodies
 * are compacted into a single buffer (as HybridContainer::coalesceBatches
 * does).
 *
 * 2026-09-27: written for a probe-store coalescing experiment (removed
 * 2026-09-29). Used by the deferred sort store to merge the retained batches
 * (BoundaryHostStore::coalesceFrom), parallel over (column, input range)
 * tasks.
 */

#pragma once

#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/FlatVector.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <vector>

namespace facebook::velox::cudf_velox {

template <typename T>
inline void appendFixedWidth(
    const VectorPtr& src,
    FlatVector<T>* out,
    vector_size_t offset) {
  const auto n = src->size();
  T* dst = out->mutableRawValues();
  if (src->encoding() == VectorEncoding::Simple::FLAT) {
    std::memcpy(
        dst + offset,
        src->asUnchecked<FlatVector<T>>()->rawValues(),
        static_cast<size_t>(n) * sizeof(T));
    if (const auto* nulls = src->rawNulls()) {
      for (vector_size_t i = 0; i < n; ++i) {
        if (bits::isBitNull(nulls, i)) {
          out->setNull(offset + i, true);
        }
      }
    }
    return;
  }
  if (src->encoding() == VectorEncoding::Simple::DICTIONARY &&
      src->rawNulls() == nullptr) {
    auto base = BaseVector::loadedVectorShared(src->valueVector());
    if (base != nullptr && base->encoding() == VectorEncoding::Simple::FLAT) {
      const auto* idx = src->wrapInfo()->as<vector_size_t>();
      const T* in = base->asUnchecked<FlatVector<T>>()->rawValues();
      const auto* baseNulls = base->rawNulls();
      if (baseNulls == nullptr) {
        for (vector_size_t i = 0; i < n; ++i) {
          dst[offset + i] = in[idx[i]];
        }
      } else {
        // All-valid null buffers are common (Parquet): test per value.
        for (vector_size_t i = 0; i < n; ++i) {
          if (bits::isBitNull(baseNulls, idx[i])) {
            out->setNull(offset + i, true);
          } else {
            dst[offset + i] = in[idx[i]];
          }
        }
      }
      return;
    }
  }
  DecodedVector decoded(*src);
  const T* in = decoded.data<T>();
  for (vector_size_t i = 0; i < n; ++i) {
    if (decoded.isNullAt(i)) {
      out->setNull(offset + i, true);
    } else {
      dst[offset + i] = in[decoded.index(i)];
    }
  }
}

template <typename T>
inline void mergeFixedWidth(
    const std::vector<VectorPtr>& srcs,
    const VectorPtr& col) {
  auto* out = col->asUnchecked<FlatVector<T>>();
  vector_size_t offset = 0;
  for (const auto& src : srcs) {
    appendFixedWidth<T>(src, out, offset);
    offset += src->size();
  }
}

inline void mergeStrings(
    const std::vector<VectorPtr>& srcs,
    const VectorPtr& col,
    memory::MemoryPool* pool) {
  auto* out = col->asUnchecked<FlatVector<StringView>>();
  std::vector<std::unique_ptr<DecodedVector>> decoded;
  decoded.reserve(srcs.size());
  uint64_t bodyBytes = 0;
  for (const auto& src : srcs) {
    auto d = std::make_unique<DecodedVector>(*src);
    const auto* views = d->data<StringView>();
    for (vector_size_t i = 0; i < src->size(); ++i) {
      if (!d->isNullAt(i)) {
        const auto& v = views[d->index(i)];
        if (!v.isInline()) {
          bodyBytes += v.size();
        }
      }
    }
    decoded.push_back(std::move(d));
  }
  char* body = nullptr;
  if (bodyBytes > 0) {
    auto buffer = AlignedBuffer::allocate<char>(bodyBytes, pool);
    buffer->setSize(bodyBytes);
    body = buffer->asMutable<char>();
    out->setStringBuffers({buffer});
  }
  auto* outViews = out->mutableRawValues();
  uint64_t bodyOffset = 0;
  vector_size_t offset = 0;
  for (size_t b = 0; b < srcs.size(); ++b) {
    const auto& d = *decoded[b];
    const auto* views = d.data<StringView>();
    const auto n = srcs[b]->size();
    for (vector_size_t i = 0; i < n; ++i) {
      if (d.isNullAt(i)) {
        out->setNull(offset + i, true);
        continue;
      }
      const auto& v = views[d.index(i)];
      if (v.isInline()) {
        outViews[offset + i] = v;
      } else {
        std::memcpy(body + bodyOffset, v.data(), v.size());
        outViews[offset + i] = StringView(body + bodyOffset, v.size());
        bodyOffset += v.size();
      }
    }
    offset += n;
  }
}

/// Merge `srcs` (one column of every retained batch, in order) into `col`,
/// pre-created with the column's type and the total row count.
inline void mergeColumn(
    const TypePtr& type,
    const std::vector<VectorPtr>& srcs,
    const VectorPtr& col,
    memory::MemoryPool* pool) {
  switch (type->kind()) {
    case TypeKind::BIGINT:
      mergeFixedWidth<int64_t>(srcs, col);
      break;
    case TypeKind::INTEGER:
      mergeFixedWidth<int32_t>(srcs, col);
      break;
    case TypeKind::SMALLINT:
      mergeFixedWidth<int16_t>(srcs, col);
      break;
    case TypeKind::TINYINT:
      mergeFixedWidth<int8_t>(srcs, col);
      break;
    case TypeKind::DOUBLE:
      mergeFixedWidth<double>(srcs, col);
      break;
    case TypeKind::REAL:
      mergeFixedWidth<float>(srcs, col);
      break;
    case TypeKind::HUGEINT:
      mergeFixedWidth<int128_t>(srcs, col);
      break;
    case TypeKind::TIMESTAMP:
      mergeFixedWidth<Timestamp>(srcs, col);
      break;
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
      mergeStrings(srcs, col, pool);
      break;
    default: {
      // BOOLEAN (bit-packed) and anything else: generic copy.
      vector_size_t offset = 0;
      for (const auto& src : srcs) {
        col->copy(src.get(), offset, 0, src->size());
        offset += src->size();
      }
    }
  }
}

namespace retained_merge_detail {

/// Run fn(task) for task in [0, numTasks) over `threads` workers (atomic task
/// counter; the first exception is rethrown after the join).
template <typename F>
inline void parallelFor(int32_t numTasks, int32_t threads, F&& fn) {
  const int32_t nThreads = std::max(1, std::min(threads, numTasks));
  if (nThreads <= 1) {
    for (int32_t t = 0; t < numTasks; ++t) {
      fn(t);
    }
    return;
  }
  std::atomic<int32_t> next{0};
  std::exception_ptr error;
  std::mutex errorMutex;
  std::vector<std::thread> workers;
  workers.reserve(nThreads);
  for (int32_t w = 0; w < nThreads; ++w) {
    workers.emplace_back([&]() {
      for (int32_t t = next++; t < numTasks; t = next++) {
        try {
          fn(t);
        } catch (...) {
          std::lock_guard<std::mutex> l(errorMutex);
          if (!error) {
            error = std::current_exception();
          }
        }
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  if (error) {
    std::rethrow_exception(error);
  }
}

/// Range tasks of one column write disjoint rows, but a range boundary can
/// fall inside a 64-row null word: null bits are cleared atomically.
inline void setNullAtomic(uint64_t* nulls, vector_size_t row) {
  std::atomic_ref<uint64_t>(nulls[row >> 6])
      .fetch_and(~(1ULL << (row & 63)), std::memory_order_relaxed);
}

/// Copy column `c` of inputs [b0, b1) to dst + offset. Returns whether any
/// null was written.
template <typename T>
inline bool copyFixedRange(
    const std::vector<RowVectorPtr>& inputs,
    int32_t c,
    size_t b0,
    size_t b1,
    vector_size_t offset,
    T* dst,
    uint64_t* nulls) {
  bool anyNull = false;
  for (size_t b = b0; b < b1; ++b) {
    const auto src = BaseVector::loadedVectorShared(inputs[b]->childAt(c));
    const auto n = src->size();
    if (src->encoding() == VectorEncoding::Simple::FLAT) {
      std::memcpy(
          dst + offset,
          src->asUnchecked<FlatVector<T>>()->rawValues(),
          static_cast<size_t>(n) * sizeof(T));
      if (const auto* srcNulls = src->rawNulls()) {
        for (vector_size_t i = 0; i < n; ++i) {
          if (bits::isBitNull(srcNulls, i)) {
            setNullAtomic(nulls, offset + i);
            anyNull = true;
          }
        }
      }
      offset += n;
      continue;
    }
    if (src->encoding() == VectorEncoding::Simple::DICTIONARY &&
        src->rawNulls() == nullptr) {
      const auto base = BaseVector::loadedVectorShared(src->valueVector());
      if (base != nullptr &&
          base->encoding() == VectorEncoding::Simple::FLAT) {
        const auto* idx = src->wrapInfo()->as<vector_size_t>();
        const T* in = base->asUnchecked<FlatVector<T>>()->rawValues();
        const auto* baseNulls = base->rawNulls();
        for (vector_size_t i = 0; i < n; ++i) {
          if (baseNulls != nullptr && bits::isBitNull(baseNulls, idx[i])) {
            setNullAtomic(nulls, offset + i);
            anyNull = true;
          } else {
            dst[offset + i] = in[idx[i]];
          }
        }
        offset += n;
        continue;
      }
    }
    DecodedVector decoded(*src);
    const T* in = decoded.data<T>();
    for (vector_size_t i = 0; i < n; ++i) {
      if (decoded.isNullAt(i)) {
        setNullAtomic(nulls, offset + i);
        anyNull = true;
      } else {
        dst[offset + i] = in[decoded.index(i)];
      }
    }
    offset += n;
  }
  return anyNull;
}

/// Out-of-line string body bytes of column `c` over inputs [b0, b1).
inline uint64_t stringBodyBytes(
    const std::vector<RowVectorPtr>& inputs,
    int32_t c,
    size_t b0,
    size_t b1) {
  uint64_t bytes = 0;
  for (size_t b = b0; b < b1; ++b) {
    const auto src = BaseVector::loadedVectorShared(inputs[b]->childAt(c));
    DecodedVector d(*src);
    const auto* views = d.data<StringView>();
    for (vector_size_t i = 0; i < src->size(); ++i) {
      if (!d.isNullAt(i)) {
        const auto& v = views[d.index(i)];
        if (!v.isInline()) {
          bytes += v.size();
        }
      }
    }
  }
  return bytes;
}

/// Copy column `c` of inputs [b0, b1): views to outViews + offset, bodies
/// compacted to body + bodyOffset. Returns whether any null was written.
inline bool copyStringRange(
    const std::vector<RowVectorPtr>& inputs,
    int32_t c,
    size_t b0,
    size_t b1,
    vector_size_t offset,
    StringView* outViews,
    char* body,
    uint64_t bodyOffset,
    uint64_t* nulls) {
  bool anyNull = false;
  for (size_t b = b0; b < b1; ++b) {
    const auto src = BaseVector::loadedVectorShared(inputs[b]->childAt(c));
    DecodedVector d(*src);
    const auto* views = d.data<StringView>();
    const auto n = src->size();
    for (vector_size_t i = 0; i < n; ++i) {
      if (d.isNullAt(i)) {
        setNullAtomic(nulls, offset + i);
        anyNull = true;
        continue;
      }
      const auto& v = views[d.index(i)];
      if (v.isInline()) {
        outViews[offset + i] = v;
      } else {
        std::memcpy(body + bodyOffset, v.data(), v.size());
        outViews[offset + i] = StringView(body + bodyOffset, v.size());
        bodyOffset += v.size();
      }
    }
    offset += n;
  }
  return anyNull;
}

template <typename T>
inline bool copyFixedRangeTo(
    const std::vector<RowVectorPtr>& inputs,
    int32_t c,
    size_t b0,
    size_t b1,
    vector_size_t offset,
    const VectorPtr& col,
    uint64_t* nulls) {
  return copyFixedRange<T>(
      inputs,
      c,
      b0,
      b1,
      offset,
      col->asUnchecked<FlatVector<T>>()->mutableRawValues(),
      nulls);
}

} // namespace retained_merge_detail

/// Merge all `inputs` (same row type `type`) into one flat batch over
/// `threads` workers. With at least 2 x threads columns, one task per column
/// (enough parallelism; the single sort's 64-128 columns). With fewer
/// (2026-09-28, Q46's 16 lineitem columns, where l_comment alone took the
/// whole 4.7 s column-per-task merge), columns are split into row-balanced
/// ranges of inputs -- strings into 2 x threads ranges, fixed-width into
/// enough for ~2 x threads fixed tasks -- and each (column, range) is a task.
/// Fixed-width ranges copy straight into the preallocated values; string
/// columns run two passes (body bytes per range, then views + compacted
/// bodies at the range's body offset). BOOLEAN and other types stay one task
/// per column.
inline RowVectorPtr mergeRetainedBatches(
    const RowTypePtr& type,
    const std::vector<RowVectorPtr>& inputs,
    memory::MemoryPool* pool,
    int32_t threads) {
  namespace d = retained_merge_detail;
  const size_t numInputs = inputs.size();
  std::vector<int64_t> starts(numInputs + 1, 0);
  for (size_t b = 0; b < numInputs; ++b) {
    starts[b + 1] = starts[b] + inputs[b]->size();
  }
  const int64_t total = starts[numInputs];
  VELOX_CHECK_LE(total, std::numeric_limits<vector_size_t>::max());
  const auto numCols = static_cast<int32_t>(type->size());
  const int32_t nThreads = std::max(1, threads);

  enum class Kind { kFixed, kString, kOther };
  std::vector<Kind> kinds(numCols, Kind::kOther);
  std::vector<VectorPtr> children(numCols);
  std::vector<uint64_t*> nulls(numCols, nullptr);
  int32_t numFixed = 0;
  for (int32_t c = 0; c < numCols; ++c) {
    const auto& t = type->childAt(c);
    switch (t->kind()) {
      case TypeKind::BIGINT:
      case TypeKind::INTEGER:
      case TypeKind::SMALLINT:
      case TypeKind::TINYINT:
      case TypeKind::DOUBLE:
      case TypeKind::REAL:
      case TypeKind::HUGEINT:
      case TypeKind::TIMESTAMP:
        kinds[c] = Kind::kFixed;
        ++numFixed;
        break;
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        kinds[c] = Kind::kString;
        break;
      default:
        break;
    }
  }

  // Ranges per column (1 = the whole column).
  const bool split = nThreads > 1 && numCols < 2 * nThreads;
  // Output vectors are created inside parallel tasks: creating them all up
  // front on one thread (2026-09-28 first cut) doubled the single sort's
  // merge (64-128 columns x 20M rows).
  auto create = [&](int32_t c) {
    children[c] = BaseVector::create(
        type->childAt(c), static_cast<vector_size_t>(total), pool);
  };
  auto mergeWhole = [&](int32_t c) {
    if (children[c] == nullptr) {
      create(c);
    }
    std::vector<VectorPtr> srcs(numInputs);
    for (size_t b = 0; b < numInputs; ++b) {
      srcs[b] = BaseVector::loadedVectorShared(inputs[b]->childAt(c));
    }
    mergeColumn(type->childAt(c), srcs, children[c], pool);
  };
  if (!split) {
    // Column per task (the merge before 2026-09-28, unchanged).
    d::parallelFor(numCols, nThreads, mergeWhole);
    return std::make_shared<RowVector>(
        pool,
        type,
        nullptr,
        static_cast<vector_size_t>(total),
        std::move(children));
  }
  d::parallelFor(numCols, nThreads, [&](int32_t c) {
    create(c);
    if (kinds[c] != Kind::kOther && total > 0) {
      // All not-null; dropped again below when no null was written.
      nulls[c] = children[c]->mutableRawNulls();
    }
  });
  auto clampRanges = [&](int64_t r) {
    return static_cast<int32_t>(
        std::max<int64_t>(1, std::min<int64_t>(r, numInputs)));
  };
  const int32_t stringRanges = clampRanges(2 * nThreads);
  const int32_t fixedRanges = clampRanges(
      (2 * nThreads + std::max(1, numFixed) - 1) / std::max(1, numFixed));
  std::vector<int32_t> colRanges(numCols, 1);
  for (int32_t c = 0; c < numCols; ++c) {
    colRanges[c] = kinds[c] == Kind::kString
        ? stringRanges
        : (kinds[c] == Kind::kFixed ? fixedRanges : 1);
  }
  // Row-balanced input boundaries for each distinct range count.
  auto makeBounds = [&](int32_t numRanges) {
    std::vector<size_t> bounds(numRanges + 1, 0);
    for (int32_t r = 1; r < numRanges; ++r) {
      const int64_t target = total * r / numRanges;
      bounds[r] = static_cast<size_t>(
          std::lower_bound(starts.begin(), starts.end(), target) -
          starts.begin());
      bounds[r] = std::max(bounds[r], bounds[r - 1]);
    }
    bounds[numRanges] = numInputs;
    return bounds;
  };
  const auto stringBounds = makeBounds(stringRanges);
  const auto fixedBounds = makeBounds(fixedRanges);
  auto boundsOf = [&](int32_t c) -> const std::vector<size_t>& {
    return kinds[c] == Kind::kString ? stringBounds : fixedBounds;
  };

  auto anyNull = std::make_unique<std::atomic<bool>[]>(numCols);
  for (int32_t c = 0; c < numCols; ++c) {
    anyNull[c] = false;
  }

  auto copyFixed = [&](int32_t c, int32_t r) {
    const auto& bounds = boundsOf(c);
    const auto b0 = bounds[r];
    const auto b1 = bounds[r + 1];
    const auto offset = static_cast<vector_size_t>(starts[b0]);
    const auto& col = children[c];
    bool wroteNull = false;
    switch (type->childAt(c)->kind()) {
      case TypeKind::BIGINT:
        wroteNull = d::copyFixedRangeTo<int64_t>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::INTEGER:
        wroteNull = d::copyFixedRangeTo<int32_t>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::SMALLINT:
        wroteNull = d::copyFixedRangeTo<int16_t>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::TINYINT:
        wroteNull = d::copyFixedRangeTo<int8_t>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::DOUBLE:
        wroteNull = d::copyFixedRangeTo<double>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::REAL:
        wroteNull = d::copyFixedRangeTo<float>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::HUGEINT:
        wroteNull = d::copyFixedRangeTo<int128_t>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      case TypeKind::TIMESTAMP:
        wroteNull = d::copyFixedRangeTo<Timestamp>(inputs, c, b0, b1, offset, col, nulls[c]);
        break;
      default:
        VELOX_UNREACHABLE();
    }
    if (wroteNull) {
      anyNull[c] = true;
    }
  };
  // Pass 1: fixed-width copies, string body sizes, whole-column merges.
  // Tasks are ordered strings first (the longest), then fixed, then other.
  struct Task {
    int32_t col;
    int32_t range; // -1: whole column via mergeColumn
  };
  std::vector<Task> pass1;
  for (auto want : {Kind::kString, Kind::kFixed}) {
    for (int32_t c = 0; c < numCols; ++c) {
      if (kinds[c] == want) {
        for (int32_t r = 0; r < colRanges[c]; ++r) {
          pass1.push_back({c, r});
        }
      }
    }
  }
  for (int32_t c = 0; c < numCols; ++c) {
    if (kinds[c] == Kind::kOther) {
      pass1.push_back({c, -1});
    }
  }
  std::vector<std::vector<uint64_t>> bodyBytes(numCols);
  for (int32_t c = 0; c < numCols; ++c) {
    if (kinds[c] == Kind::kString) {
      bodyBytes[c].assign(colRanges[c], 0);
    }
  }
  d::parallelFor(static_cast<int32_t>(pass1.size()), nThreads, [&](int32_t t) {
    const auto [c, r] = pass1[t];
    if (r < 0) {
      mergeWhole(c);
    } else if (kinds[c] == Kind::kFixed) {
      copyFixed(c, r);
    } else {
      bodyBytes[c][r] = d::stringBodyBytes(
          inputs, c, stringBounds[r], stringBounds[r + 1]);
    }
  });

  // String bodies: one compacted buffer per column, ranges at prefix offsets.
  std::vector<std::vector<uint64_t>> bodyOffset(numCols);
  std::vector<char*> bodies(numCols, nullptr);
  std::vector<Task> pass2;
  for (int32_t c = 0; c < numCols; ++c) {
    if (kinds[c] != Kind::kString) {
      continue;
    }
    bodyOffset[c].assign(colRanges[c], 0);
    uint64_t sum = 0;
    for (int32_t r = 0; r < colRanges[c]; ++r) {
      bodyOffset[c][r] = sum;
      sum += bodyBytes[c][r];
    }
    if (sum > 0) {
      auto buffer = AlignedBuffer::allocate<char>(sum, pool);
      buffer->setSize(sum);
      bodies[c] = buffer->asMutable<char>();
      children[c]->asUnchecked<FlatVector<StringView>>()->setStringBuffers(
          {buffer});
    }
    for (int32_t r = 0; r < colRanges[c]; ++r) {
      pass2.push_back({c, r});
    }
  }
  d::parallelFor(static_cast<int32_t>(pass2.size()), nThreads, [&](int32_t t) {
    const auto [c, r] = pass2[t];
    const auto b0 = stringBounds[r];
    if (d::copyStringRange(
            inputs,
            c,
            b0,
            stringBounds[r + 1],
            static_cast<vector_size_t>(starts[b0]),
            children[c]->asUnchecked<FlatVector<StringView>>()->mutableRawValues(),
            bodies[c],
            bodyOffset[c][r],
            nulls[c])) {
      anyNull[c] = true;
    }
  });
  for (int32_t c = 0; c < numCols; ++c) {
    if (nulls[c] != nullptr && !anyNull[c]) {
      children[c]->resetNulls();
    }
  }
  return std::make_shared<RowVector>(
      pool,
      type,
      nullptr,
      static_cast<vector_size_t>(total),
      std::move(children));
}

} // namespace facebook::velox::cudf_velox
