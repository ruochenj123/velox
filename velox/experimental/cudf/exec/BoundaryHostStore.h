/*
 * BoundaryHostStore.h
 *
 * Host-side payload retention + gather for the BOUNDARY-HYBRID GPU join
 * (CudfConfig::benchmarkBoundaryHybrid): only join keys cross the device
 * boundary; payload columns stay host-resident in columnar form; the GPU
 * join returns surviving row-id pairs; the host gathers payloads once for
 * the survivors.
 *
 * This is a THIN WRAPPER over exec::HybridContainer (velox/exec/
 * RowContainer.h) in SCATTERED mode — the same machinery the CPU hybrid
 * join uses for host-side batch retention:
 *   - addPayload(): retains the batch, flattens dictionary-encoded children
 *     (so a selective filter's dictionary view does not pin its whole base
 *     vector), and pre-decodes every column into DecodedVector for O(1)
 *     valueAt access;
 *   - the extractPayloadScattered* kernels: gather by (batchId, rowInBatch)
 *     encoded in exec::HybridRowId.
 * We deliberately REUSE that machinery rather than re-implementing retention
 * and gather (per the CPU/GPU hybrid-layout design: one retention mechanism
 * on both sides of the boundary).
 *
 * What the wrapper adds/bridges:
 *   - HybridContainer requires a keys RowContainer (the CPU join stores join
 *     keys + a rowId column there). In boundary-hybrid the GPU owns the keys,
 *     so we construct the container with ZERO key types over a minimal dummy
 *     RowContainer that is never written. Only addPayload + the scattered
 *     extraction paths are exercised; those never touch the keys container.
 *   - HybridContainer's extraction API takes a `rows` array whose only use in
 *     scattered mode is `row == nullptr` => emit NULL (a probe-miss marker in
 *     the CPU outer-join path). Boundary-hybrid gathers only genuine inner-
 *     join survivors, so we pass a reusable array of non-null sentinels.
 *   - Prefix sums over the added batch sizes, mapping a row index in the
 *     CONCATENATION of all batches (== the GPU-side global row id, since the
 *     GPU keys were concatenated in exactly this order) back to
 *     (batchId, rowInBatch).
 */

#pragma once

#include "velox/exec/RowContainer.h"
#include "velox/vector/ComplexVector.h"

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

class BoundaryHostStore {
 public:
  /// `rowType` is the retained batches' full row type (all columns, in input
  /// child order); extraction addresses columns by child index in this type.
  BoundaryHostStore(const RowTypePtr& rowType, memory::MemoryPool* pool)
      : rowType_(rowType),
        dummyKeys_(std::vector<TypePtr>{BIGINT()}, pool),
        container_(
            /*keyTypes=*/{},
            rowType->children(),
            &dummyKeys_) {
    container_.setScatteredModeEnabled(true);
    container_.setId(0);
    // Register self so the single-container scattered fast path applies.
    std::unordered_map<uint8_t, exec::HybridContainer*> self{{0, &container_}};
    container_.setAllContainers(self);
  }

  // Not copyable/movable: container_ holds a pointer to dummyKeys_.
  BoundaryHostStore(const BoundaryHostStore&) = delete;
  BoundaryHostStore& operator=(const BoundaryHostStore&) = delete;

  /// Retain one batch. Batches must be added in GPU-concatenation order so
  /// that global row ids line up. Returns the batchId assigned.
  uint32_t addBatch(RowVectorPtr batch) {
    const auto batchId = static_cast<uint32_t>(batchStarts_.size());
    // 24-bit batchId in HybridRowId::encodeScattered.
    VELOX_CHECK_LT(
        batchId,
        1u << exec::HybridRowId::kBatchIdBits,
        "BoundaryHostStore: too many retained batches for scattered encoding");
    batchStarts_.push_back(totalRows_);
    totalRows_ += batch->size();
    container_.addPayload(std::move(batch));
    return batchId;
  }

  /// Drop all retained batches (per-probe-batch reuse: the probe retains one
  /// GPU batch's host payload, gathers survivors, then clears).
  void clearBatches() {
    container_.clear();
    batchStarts_.clear();
    totalRows_ = 0;
  }

  int64_t totalRows() const {
    return totalRows_;
  }

  /// The retained batches' row type (all columns, input child order).
  const RowTypePtr& rowType() const {
    return rowType_;
  }

  size_t numBatches() const {
    return batchStarts_.size();
  }

  /// Map a global row id (index into the concatenation of all added batches,
  /// == the GPU-side row id) to a scattered HybridRowId. O(log #batches).
  exec::HybridRowId idForGlobalRow(int64_t globalRow) const {
    VELOX_DCHECK_GE(globalRow, 0);
    VELOX_DCHECK_LT(globalRow, totalRows_);
    // upper_bound-1: last batch whose start <= globalRow.
    const auto it = std::upper_bound(
        batchStarts_.begin(), batchStarts_.end(), globalRow);
    const auto b = static_cast<uint32_t>(it - batchStarts_.begin() - 1);
    const auto rowInBatch = static_cast<uint32_t>(globalRow - batchStarts_[b]);
    return exec::HybridRowId{
        0, exec::HybridRowId::encodeScattered(b, rowInBatch)};
  }

  /// Bulk form of idForGlobalRow. `globalRows` are GPU-side row ids (e.g. the
  /// build-side survivor ids read back from the GPU probe).
  void idsForGlobalRows(
      const int32_t* globalRows,
      int32_t numRows,
      std::vector<exec::HybridRowId>& out) const {
    out.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      out[i] = idForGlobalRow(globalRows[i]);
    }
  }

  /// Gather column `childIdx` (index in the construction rowType) of the
  /// retained batches at `ids` into `result` (pre-created with the column's
  /// type; resized by the extraction). Runs HybridContainer's scattered
  /// single-container extraction kernel.
  ///
  /// `rowsScratch` is CALLER-OWNED scratch for the `rows` array
  /// HybridContainer's API requires: in scattered mode rows[i] is consulted
  /// only as a null-row marker (probe-miss in the CPU outer-join path), and
  /// every survivor here is a real row, so it is filled with non-null
  /// sentinels. Caller-owned so that a build-side store SHARED by several
  /// probe drivers stays immutable during gather (thread-safe: extraction
  /// only reads container state).
  void gather(
      int32_t childIdx,
      std::vector<exec::HybridRowId>& ids,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch) const {
    const auto n = static_cast<int32_t>(ids.size());
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < n) {
      rowsScratch.assign(n, &kSentinel);
    }
    // extractColumn is logically const here (pure read in scattered mode);
    // HybridContainer just does not mark it so.
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(), n, /*columnIndex=*/childIdx, result, ids);
  }

 private:
  const RowTypePtr rowType_;
  // Never-written keys container: HybridContainer's ctor dereferences its
  // column layout (for the rowId-column offset used by getRowIds, which we
  // never call), so it must be a real RowContainer with >= 1 column.
  exec::RowContainer dummyKeys_;
  exec::HybridContainer container_;
  // batchStarts_[b] = global row id of the first row of batch b.
  std::vector<int64_t> batchStarts_;
  int64_t totalRows_ = 0;
};

} // namespace facebook::velox::cudf_velox
