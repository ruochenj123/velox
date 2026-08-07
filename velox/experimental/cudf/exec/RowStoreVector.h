/*
 * RowStoreVector.h
 *
 * A RowVector subclass that carries GPU row-layout data.
 * Used as the output of CudfFromVelox when benchmarkRowWiseGather is enabled,
 * allowing RowHashJoinBuild/Probe to skip the col→row transpose.
 *
 * The actual data lives in a GPU device buffer (rmm::device_buffer) in a
 * packed fixed-width row format described by GpuFixedRowStore.
 */

#pragma once

#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/vector/ComplexVector.h"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

/// A RowVector that carries GPU row-store data through the Velox pipeline.
/// The CPU-side RowVector children are empty/null — the real data is on GPU.
class RowStoreVector : public RowVector {
 public:
  RowStoreVector(
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      int64_t numRows,
      rmm::device_buffer rowBuffer,
      rmm::device_buffer fieldsBuffer,
      std::vector<FieldDesc> hostFields,
      int32_t rowWidth,
      rmm::cuda_stream_view stream)
      : RowStoreVector(
            pool,
            std::move(type),
            numRows,
            std::move(rowBuffer),
            std::make_shared<rmm::device_buffer>(std::move(fieldsBuffer)),
            std::move(hostFields),
            rowWidth,
            stream) {}

  /// Shared-fields variant: batches from one producer share the single
  /// uploaded FieldDesc buffer instead of each carrying a device copy.
  RowStoreVector(
      velox::memory::MemoryPool* pool,
      RowTypePtr type,
      int64_t numRows,
      rmm::device_buffer rowBuffer,
      std::shared_ptr<rmm::device_buffer> fieldsBuffer,
      std::vector<FieldDesc> hostFields,
      int32_t rowWidth,
      rmm::cuda_stream_view stream)
      : RowVector(
            pool,
            type,
            nullptr, // nulls
            numRows,
            makeNullChildren(pool, type, numRows)),
        rowBuffer_(std::move(rowBuffer)),
        fieldsBuffer_(std::move(fieldsBuffer)),
        hostFields_(std::move(hostFields)),
        rowWidth_(rowWidth),
        stream_(stream) {}

  /// Get the GPU row store handle (pointers valid until this vector is destroyed)
  GpuFixedRowStore getGpuRowStore() {
    GpuFixedRowStore store;
    store.row_buffer = static_cast<uint8_t*>(rowBuffer_.data());
    store.row_width = rowWidth_;
    store.num_rows = size();
    store.num_fields = hostFields_.size();
    store.fields = static_cast<const FieldDesc*>(fieldsBuffer_->data());
    return store;
  }

  /// Access host-side field descriptors
  const std::vector<FieldDesc>& hostFields() const { return hostFields_; }
  int32_t rowWidth() const { return rowWidth_; }
  rmm::cuda_stream_view stream() const { return stream_; }

  /// Get raw GPU row buffer pointer
  uint8_t* gpuRowData() { return static_cast<uint8_t*>(rowBuffer_.data()); }
  const uint8_t* gpuRowData() const {
    return static_cast<const uint8_t*>(rowBuffer_.data());
  }
  int64_t gpuRowBytes() const { return static_cast<int64_t>(size()) * rowWidth_; }

 private:
  /// Create null-constant children so the RowVector base is valid
  static std::vector<VectorPtr> makeNullChildren(
      velox::memory::MemoryPool* pool,
      const RowTypePtr& type,
      int64_t numRows) {
    std::vector<VectorPtr> children(type->size());
    for (int i = 0; i < type->size(); i++) {
      children[i] = BaseVector::createNullConstant(
          type->childAt(i), numRows, pool);
    }
    return children;
  }

  rmm::device_buffer rowBuffer_; // GPU row data
  // GPU FieldDesc array. Shared: all batches from one producer point at the
  // single uploaded buffer (kept alive by this shared_ptr); the legacy ctor
  // wraps a per-batch buffer, which is equivalent but not deduplicated.
  std::shared_ptr<rmm::device_buffer> fieldsBuffer_;
  std::vector<FieldDesc> hostFields_; // CPU copy of field layout
  int32_t rowWidth_;
  rmm::cuda_stream_view stream_;
};

using RowStoreVectorPtr = std::shared_ptr<RowStoreVector>;

} // namespace facebook::velox::cudf_velox
