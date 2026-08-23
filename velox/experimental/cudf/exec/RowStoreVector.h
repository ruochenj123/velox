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
    if (nullStride_ > 0) {
      // Sidecar rides in the tail of rowBuffer_ (rows region, then
      // num_rows * nullStride_ null bytes) -- one H2D covers both.
      store.null_bytes = static_cast<const uint8_t*>(rowBuffer_.data()) +
          static_cast<int64_t>(size()) * rowWidth_;
      store.null_stride = nullStride_;
    }
    return store;
  }

  // ---- Out-of-line strings (pointer slots, 2026-08-23; format in
  // GpuFixedRowStore.h)
  //
  // Out-of-line slots hold ABSOLUTE device pointers into whatever buffer
  // already holds the bytes (the pack's uploaded heap tail of rowBuffer_, a
  // cudf strings column's chars, an upstream RowStoreVector's buffer).
  // Whoever writes such slots must register the referenced owner here so it
  // outlives this vector. hasStringRefs() == true means slots may point
  // outside rowBuffer_ (or into its tail) -- consumers that copy rows
  // elsewhere must propagate the keep-alive list.
  void addStringKeepAlive(std::shared_ptr<void> owner) {
    if (owner != nullptr) {
      stringKeepAlive_.push_back(std::move(owner));
    }
    hasStringRefs_ = true;
  }
  void addStringKeepAlives(const std::vector<std::shared_ptr<void>>& owners) {
    for (const auto& o : owners) {
      stringKeepAlive_.push_back(o);
    }
    hasStringRefs_ = true;
  }
  const std::vector<std::shared_ptr<void>>& stringKeepAlive() const {
    return stringKeepAlive_;
  }
  /// True if any string field may hold an out-of-line pointer.
  bool hasStringRefs() const {
    return hasStringRefs_;
  }
  /// Mark that this store's own rowBuffer_ tail is a referenced heap (pack
  /// path): no external owner, but rows copied out of this vector still
  /// reference it, so consumers must keep THIS vector (or rowBuffer_) alive.
  void setSelfStringHeap() {
    hasStringRefs_ = true;
  }

  /// Null sidecar (bit set = NULL) present in the tail of the row buffer.
  /// 0 = null-free store. See GpuFixedRowStore::null_bytes.
  void setNullSidecar(int32_t nullStride) { nullStride_ = nullStride; }
  int32_t nullStride() const { return nullStride_; }

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

  // ---- Boundary-hybrid (keys-only) support -------------------------------
  //
  // When benchmarkBoundaryHybrid is on, CudfFromVelox packs ONLY the join-key
  // columns into the GPU row buffer (hostFields_ then describes the keys-only
  // layout, in JOIN-KEY order, not input-column order) and attaches the
  // original host batches here. Their concatenation is row-for-row identical
  // to the GPU rows: GPU row i of this vector is row (i - start[b]) of host
  // batch b, where start[] are the prefix sums of the batch sizes. The
  // consumer (RowHashJoinBuild/Probe) uses this identity to gather payload
  // host-side for join survivors only — the payload bytes never cross PCIe.
  //
  // The batches hold LOADED, FLAT children (tryPinnedPack rejects anything
  // else), so host-side extraction reads raw flat buffers.
  void setBoundaryPayload(std::vector<RowVectorPtr> hostBatches) {
    boundaryHostBatches_ = std::move(hostBatches);
    boundaryKeysOnly_ = true;
  }
  bool boundaryKeysOnly() const {
    return boundaryKeysOnly_;
  }
  const std::vector<RowVectorPtr>& boundaryHostBatches() const {
    return boundaryHostBatches_;
  }
  /// Release the retained host batches (called by the consumer once the
  /// batch's survivors have been materialized, so host memory is not held
  /// for the lifetime of the vector).
  void clearBoundaryPayload() {
    boundaryHostBatches_.clear();
  }

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

  // Boundary-hybrid: host-retained payload batches (see setBoundaryPayload).
  // Constructed only in TUs covered by the incremental rebuild
  // (CudfConversion.cpp / RowHashJoin.cpp / CudfBatchConcat.cpp), so growing
  // this class is safe for the partial-rebuild workflow.
  std::vector<RowVectorPtr> boundaryHostBatches_;
  bool boundaryKeysOnly_ = false;
  // Null sidecar stride in bytes per row (0 = none). Appended per the
  // partial-rebuild note above.
  int32_t nullStride_ = 0;
  // Owners of buffers referenced by out-of-line string slots (see
  // addStringKeepAlive). Appended per the partial-rebuild note above.
  std::vector<std::shared_ptr<void>> stringKeepAlive_;
  bool hasStringRefs_ = false;

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
