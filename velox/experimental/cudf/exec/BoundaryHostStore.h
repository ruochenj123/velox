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
  /// `scattered` (default, the probe's per-batch store): row ids encode
  /// (batch, row) and extraction reads the batches in place. `false` (the
  /// sort's single store): batches are merged by coalesce() into one
  /// contiguous batch and row ids are global indices (HybridContainer's
  /// coalesced mode, the CPU hybrid sort's own path).
  BoundaryHostStore(
      const RowTypePtr& rowType,
      memory::MemoryPool* pool,
      bool scattered = true)
      : rowType_(rowType),
        dummyKeys_(std::vector<TypePtr>{BIGINT()}, pool),
        container_(
            /*keyTypes=*/{},
            rowType->children(),
            &dummyKeys_) {
    container_.setScatteredModeEnabled(scattered);
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
    batches_.push_back(batch);
    container_.addPayload(std::move(batch));
    return batchId;
  }

  /// The retained batches, in the order added (a blocking consumer such as
  /// the sort re-adds them to its own single store).
  const std::vector<RowVectorPtr>& batches() const {
    return batches_;
  }

  /// Coalesced mode only: merge all retained batches into one contiguous
  /// batch (HybridContainer::coalesceBatches -- column-at-a-time, sources
  /// released as copied). Safe to run on a background thread as long as
  /// nothing else touches this store meanwhile.
  void coalesce() {
    container_.coalesceBatches();
  }
  /// Coalesced mode, after coalesce(): the single merged batch.
  RowVectorPtr coalescedBatch() const {
    return container_.getCoalescedBatch();
  }
  exec::HybridContainer& container() {
    return container_;
  }

  /// int32 global ids (the GPU probe's survivor build ids).
  void gatherCoalesced(
      int32_t childIdx,
      const int32_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      idsScratch[i] = exec::HybridRowId{0, static_cast<uint64_t>(globalRows[i])};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

  /// Coalesced mode only: gather column `childIdx` at GLOBAL row ids.
  void gatherCoalesced(
      int32_t childIdx,
      const int64_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      idsScratch[i] = exec::HybridRowId{0, static_cast<uint64_t>(globalRows[i])};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

  /// Drop all retained batches (per-probe-batch reuse: the probe retains one
  /// GPU batch's host payload, gathers survivors, then clears).
  void clearBatches() {
    container_.clear();
    batches_.clear();
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
      const std::vector<exec::HybridRowId>& ids,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch) const {
    const auto n = static_cast<int32_t>(ids.size());
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < n) {
      rowsScratch.assign(n, &kSentinel);
    }
    // extractColumn is logically const here (pure read in scattered mode,
    // ids included); HybridContainer just does not mark it so. Sharing one
    // ids vector across concurrent column gathers is therefore safe.
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(),
        n,
        /*columnIndex=*/childIdx,
        result,
        const_cast<std::vector<exec::HybridRowId>&>(ids));
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
  std::vector<RowVectorPtr> batches_;
  int64_t totalRows_ = 0;
};


/// The CPU hybrid join's build-side layout, on the GPU boundary: one
/// COALESCED per-driver store per build driver, registered as containers
/// 0..N-1 of one HybridContainer group, so a survivor's build payload is
/// extracted by container id exactly as the CPU probe does (per-container
/// grouped, pre-decoded, prefetched extraction). The only GPU-specific step
/// is mapping the matcher's global build row id to (driver chunk, local row)
/// with a search over the (<=16) chunk starts.
class MultiBoundaryHostStore {
 public:
  explicit MultiBoundaryHostStore(std::vector<std::shared_ptr<BoundaryHostStore>> chunks)
      : chunks_(std::move(chunks)) {
    VELOX_CHECK(!chunks_.empty());
    VELOX_CHECK_LE(chunks_.size(), 255, "too many build drivers for uint8 container ids");
    int64_t start = 0;
    for (size_t k = 0; k < chunks_.size(); ++k) {
      chunkStarts_.push_back(start);
      start += chunks_[k]->totalRows();
      chunks_[k]->container().setId(static_cast<uint8_t>(k));
      all_[static_cast<uint8_t>(k)] = &chunks_[k]->container();
    }
    totalRows_ = start;
    for (auto& c : chunks_) {
      c->container().setAllContainers(all_);
      // As the CPU hybrid join arm (--hybrid_join_reorder_enabled=false):
      // extract in survivor order, no reorder-by-container pass.
      c->container().setReorderEnabled(false);
    }
  }
  int64_t totalRows() const {
    return totalRows_;
  }
  const RowTypePtr& rowType() const {
    return chunks_[0]->rowType();
  }
  static constexpr int kLocalBits = 48;
  static constexpr uint64_t kLocalMask = (1ULL << kLocalBits) - 1;
  /// Gather at BUILD-TIME-ENCODED ids (containerId << 48 | localRow), the
  /// CPU mechanism: no mapping on the extraction path.
  void gatherEncoded(
      int32_t childIdx,
      const uint64_t* encoded,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      const uint64_t v = encoded[i];
      idsScratch[i] = exec::HybridRowId{
          static_cast<uint8_t>(v >> kLocalBits), v & kLocalMask};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(chunks_[0]->container())
        .extractColumn(rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

  /// Gather column `childIdx` at GLOBAL build row ids (the matcher's ids).
  void gather(
      int32_t childIdx,
      const int32_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    const auto nChunks = chunkStarts_.size();
    for (int32_t i = 0; i < numRows; ++i) {
      const int64_t g = globalRows[i];
      size_t k = nChunks - 1;
      if (nChunks > 1) {
        // last chunk whose start <= g
        k = static_cast<size_t>(
            std::upper_bound(chunkStarts_.begin(), chunkStarts_.end(), g) -
            chunkStarts_.begin() - 1);
      }
      idsScratch[i] = exec::HybridRowId{
          static_cast<uint8_t>(k), static_cast<uint64_t>(g - chunkStarts_[k])};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(chunks_[0]->container())
        .extractColumn(rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

 private:
  std::vector<std::shared_ptr<BoundaryHostStore>> chunks_;
  std::vector<int64_t> chunkStarts_;
  std::unordered_map<uint8_t, exec::HybridContainer*> all_;
  int64_t totalRows_{0};
};

} // namespace facebook::velox::cudf_velox
