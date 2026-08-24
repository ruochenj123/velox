/*
 * RowHashJoin.h
 *
 * Row-wise hash join operators for Velox cudf integration.
 *
 * These operators implement the same join semantics as CudfHashJoin but
 * use a fixed-stride row store for materialization (gather) instead of
 * columnar cudf::gather(). This enables comparing row-wise vs column-wise
 * gather performance within the Velox execution framework.
 *
 * Pipeline:
 *   CPU TableScan → Filter → Project → [CudfFromVelox] →
 *     RowHashJoinBuild (GPU: col→row transpose, extract keys, build HT)
 *     RowHashJoinProbe (GPU: col→row transpose, extract keys, probe, row gather)
 *
 * Restrictions (benchmark-only):
 *   - Inner join only
 *   - No filter expressions
 *   - Fixed-width types only (no VARCHAR)
 *   - Composite (multi-column) join keys supported
 */

#pragma once

#include "velox/experimental/cudf/exec/CudfHashJoin.h" // CudfHashJoinBridge
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/experimental/cudf/exec/GpuRowHashTable.cuh"
#include "velox/experimental/cudf/exec/GpuRowOps.cuh"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"

#include "velox/experimental/cudf/exec/BoundaryHostStore.h"

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

class RowStoreVector; // RowStoreVector.h (included by the .cpp)

// ============================================================================
// Bridge: extends CudfHashJoinBridge to also carry row-store data
// ============================================================================

class RowHashJoinBridge : public exec::JoinBridge {
 public:
  struct BuildData {
    // Row store on GPU (entire build table in row layout)
    GpuFixedRowStore gpuRowStore;
    // Ownership of device memory
    rmm::device_buffer rowBuffer;
    rmm::device_buffer fieldsBuffer;
    // Hash join object built from key column(s)
    std::shared_ptr<cudf::hash_join> hashJoin;
    // Key columns on GPU (one buffer per join key, extracted from row store).
    // cudf::hash_join references the build key table for its lifetime, so these
    // buffers must outlive `hashJoin`.
    std::vector<rmm::device_buffer> keyBuffers;
    int64_t numRows;
    int32_t rowWidth;
    // Host-side field descriptors (for probe to compute output layout)
    std::vector<FieldDesc> hostFields;

    // ---- Boundary-hybrid (CudfConfig::benchmarkBoundaryHybrid) ----
    // When set, the GPU row store above holds ONLY the join-key columns
    // (keys-only layout, join-key order: field k == rightKeys[k]) and the
    // build PAYLOAD never left the host: it lives in `hostStore`, batches
    // added in the exact GPU-concatenation order, so the build row ids
    // produced by `hashJoin` (global over the concatenated build) map to
    // (batchId, rowInBatch) via hostStore->idForGlobalRow(). The probe
    // gathers build payload host-side for survivors only.
    bool boundaryHybrid{false};
    std::shared_ptr<BoundaryHostStore> hostStore;

    // ---- Row-native N:M matcher (CudfConfig::benchmarkRowTable) ----
    // Chained multimap built directly over the key rows of gpuRowStore
    // (keys-only store under boundary-hybrid; full row store otherwise —
    // either way the keys are read row-native, never extracted to columns).
    // When set, hashJoin/keyBuffers above are NOT built and the probe runs
    // the two-pass count+refill probe from GpuRowHashTable.cuh instead of
    // cudf::hash_join::inner_join.
    bool hasRowTable{false};
    RowNativeHashTable rowTable{}; // device ptrs into the buffers below
    rmm::device_buffer rowTableHeads; // [capacity] int32
    rmm::device_buffer rowTableNextFp; // [numRows] uint64
    // [numRows] uint64 stored packed keys; non-empty iff rowTable.pack is
    // enabled without exactFp (composite-key range packing, 2026-08-19).
    rmm::device_buffer rowTablePackedKeys;

    // ---- Null sidecar (2026-08-17 null support; appended for ABI) ----
    // Concatenated per-row null bytes for the build store (bit set = NULL).
    // Present (nullStride > 0) iff any input RowStoreVector carried one;
    // batches without a sidecar contribute zeroed (all-valid) bytes.
    rmm::device_buffer nullBuffer;
    int32_t nullStride{0};
    // Out-of-line string heap of the build rows; gpuRowStore.chars points
    // into it.
    rmm::device_buffer charsBuffer;
  };

  void setBuildData(std::shared_ptr<BuildData> data);
  std::shared_ptr<BuildData> dataOrFuture(ContinueFuture* future);

  void setBuildStream(rmm::cuda_stream_view stream);
  std::optional<rmm::cuda_stream_view> getBuildStream();

 private:
  std::shared_ptr<BuildData> buildData_;
  std::optional<rmm::cuda_stream_view> buildStream_;
};

// ============================================================================
// Build operator: col→row transpose + extract keys + build hash table
// ============================================================================

class RowHashJoinBuild : public CudfOperatorBase {
 public:
  RowHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

  /// Boundary-hybrid: CudfFromVelox asks the downstream join operator which
  /// columns are join keys so it can pack ONLY those across the boundary.
  const std::shared_ptr<const core::HashJoinNode>& joinNode() const {
    return joinNode_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  std::vector<RowVectorPtr> inputs_;  // RowStoreVector or CudfVector
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

// ============================================================================
// Probe operator: col→row transpose + extract keys + probe + row gather
// ============================================================================

class RowHashJoinProbe : public CudfOperatorBase {
 public:
  RowHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

  /// Boundary-hybrid: CudfFromVelox asks the downstream join operator which
  /// columns are join keys so it can pack ONLY those across the boundary.
  const std::shared_ptr<const core::HashJoinNode>& joinNode() const {
    return joinNode_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  // ---- Boundary-hybrid (CudfConfig::benchmarkBoundaryHybrid) ----

  // Grow-on-demand pinned host staging for the survivor-id readback (the D2H
  // of the two int32 id arrays). Pinned so the copy is a true async DMA
  // followed by one stream sync, not a pageable staging crawl.
  struct PinnedIdBuffer {
    int32_t* data = nullptr;
    int64_t capacity = 0; // elements
    void ensure(int64_t n);
    ~PinnedIdBuffer();
  };

  // The entire boundary-hybrid probe path: extract keys from the keys-only
  // row store, GPU probe, read back survivor (probe-local, build-global) id
  // pairs, gather payload host-side from the retained batches, emit a CPU
  // RowVector. Returns the output vector (or nullptr for zero matches).
  RowVectorPtr boundaryProbe(
      const std::shared_ptr<RowStoreVector>& rowStoreInput,
      rmm::cuda_stream_view stream);

  // Per-batch host retention of the CURRENT probe batch's payload (cleared
  // after each batch; probe survivor ids are batch-local).
  std::unique_ptr<BoundaryHostStore> boundaryProbeStore_;
  PinnedIdBuffer boundaryProbeIds_; // survivors: probe row within batch
  PinnedIdBuffer boundaryBuildIds_; // survivors: global build row id
  // Output column -> source mapping (computed once): (output idx, child idx
  // in the probe/build input row type).
  bool boundaryLayoutReady_ = false;
  std::vector<std::pair<int32_t, int32_t>> boundaryOutFromProbe_;
  std::vector<std::pair<int32_t, int32_t>> boundaryOutFromBuild_;
  // Reused scratch: scattered ids for the two gathers, plus the non-null
  // sentinel `rows` array HybridContainer's extraction API requires
  // (per-driver so the SHARED build-side store stays immutable in gather).
  std::vector<exec::HybridRowId> boundaryProbeRowIds_;
  std::vector<exec::HybridRowId> boundaryBuildRowIds_;
  std::vector<const char*> boundarySentinelScratch_;

  std::shared_ptr<const core::HashJoinNode> joinNode_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  // Build-side data from bridge
  std::shared_ptr<RowHashJoinBridge::BuildData> buildData_;
  std::optional<rmm::cuda_stream_view> buildStream_;

  // Per-probe-batch state
  RowVectorPtr input_;

  // Row layout metadata (computed once from output type)
  std::vector<FieldDesc> probeFields_;
  int32_t probeRowWidth_ = 0;
  std::vector<FieldDesc> outputFields_;
  int32_t outputRowWidth_ = 0;

  // Column index mappings
  std::vector<cudf::size_type> leftKeyIndices_;
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  std::vector<int> leftColumnOutputIndices_;
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  std::vector<int> rightColumnOutputIndices_;

  // Selective gather: field mappings (src_offset -> dst_offset) for output
  std::vector<FieldMapping> probeGatherMappings_;
  std::vector<FieldMapping> buildGatherMappings_;
  rmm::device_buffer probeGatherMappingsBuffer_;  // FieldMapping[] on GPU
  rmm::device_buffer buildGatherMappingsBuffer_;  // FieldMapping[] on GPU
  rmm::device_buffer outputFieldsBuffer_;          // FieldDesc[] on GPU for output
  bool outputLayoutComputed_ = false;

  // ---- Null sidecar plumbing (2026-08-17; appended for ABI) ----
  // Per-mapping source/destination FIELD indices (device int32 arrays,
  // parallel to the FieldMapping buffers) so the gather can copy sidecar
  // bits, which are addressed by field index rather than byte offset.
  rmm::device_buffer probeGatherSrcFieldBuffer_;
  rmm::device_buffer probeGatherDstFieldBuffer_;
  rmm::device_buffer buildGatherSrcFieldBuffer_;
  rmm::device_buffer buildGatherDstFieldBuffer_;
  // Output null sidecar for the current batch (rows = numMatches) and the
  // output stride; allocated only when an input store carries nulls.
  rmm::device_buffer outputNullBuffer_;
  // ---- Out-of-line strings (eager compaction) ----
  rmm::device_buffer probeCharsBuffer_;   // heap of a transposed CudfVector probe
  std::vector<StringGatherField> outputStringFields_; // string fields in output
  rmm::device_buffer outputStringFieldsBuffer_;       // device copy
  rmm::device_buffer outputStrOffsetsBuffer_;         // int32 slot offsets
  rmm::device_buffer outputRowBaseBuffer_;            // int64[numMatches+1]
  rmm::device_buffer outputCharsBuffer_;              // compacted heap
  int32_t outputNullStride_ = 0;

  // Pre-allocated device buffers (reused per batch)
  rmm::device_buffer probeRowBuffer_;      // probe rows on GPU
  rmm::device_buffer probeFieldsBuffer_;   // FieldDesc on GPU
  std::vector<rmm::device_buffer> probeKeyBuffers_;  // extracted probe keys (one per join key)
  rmm::device_buffer probeGatherBuffer_;   // gathered output rows

  int64_t probeRowCapacity_ = 0;
  int64_t probeKeyCapacity_ = 0;
  int64_t gatherCapacity_ = 0;
  bool fieldsUploaded_ = false;

  // ---- Row-native matcher scratch (CudfConfig::benchmarkRowTable) ----
  // Reused across batches: per-probe-row match counts, scanned offsets, cub
  // scan temp storage, and the device total. Grown on demand.
  rmm::device_buffer rowTableCounts_; // [probeRows] int32
  rmm::device_buffer rowTableOffsets_; // [probeRows] int32
  rmm::device_buffer rowTableScanTemp_; // cub temp
  rmm::device_buffer rowTableTotal_; // 1 int32
  int64_t rowTableProbeCapacity_ = 0;
  size_t rowTableScanTempBytes_ = 0;

  // Runs the MATCHER over one probe batch — the shared front half of the
  // regular and boundary-hybrid probe paths. keyDescs[k] locates join key k
  // inside probeStore's rows (full-layout offsets for the regular path,
  // keys-only offsets for boundary). Dispatches to the row-native table
  // (bd.hasRowTable) or extract-keys + cudf::hash_join, emits
  // matcherWallNanos/matcherMatches when enabled, fills the id vectors and
  // returns the match count.
  int32_t runMatcher(
      const GpuFixedRowStore& probeStore,
      const std::vector<FieldDesc>& keyDescs,
      int32_t probeRows,
      rmm::cuda_stream_view stream,
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>& leftIndices,
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>& rightIndices);

  // Runs the row-native two-pass probe (count + scan + refill) for the
  // current batch: returns numMatches and fills left/right id vectors
  // (probe-local, build-global row ids — same contract as
  // cudf::hash_join::inner_join). Syncs `stream` once to read the total.
  int32_t rowTableProbe(
      const GpuFixedRowStore& probeStore,
      const RowKeyLayout& probeKeys,
      int32_t probeRows,
      rmm::cuda_stream_view stream,
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>& leftIndices,
      std::unique_ptr<rmm::device_uvector<cudf::size_type>>& rightIndices);

  bool initialized_ = false;
  bool finished_ = false;

  // Terminal detection: whether this probe is the LAST row-mode join in the
  // chain (its downstream consumer is a columnar cudf operator, not another
  // RowHashJoinProbe). When terminal, doGetOutput emits a column-major
  // CudfVector (row->col transpose) instead of a row-major RowStoreVector.
  // Tri-state: -1 = not yet determined, 0 = chained (emit RowStoreVector),
  // 1 = terminal (emit CudfVector).
  int emitColumnar_ = -1;

  // Build a column-major CudfVector from the gathered row buffer.
  RowVectorPtr makeColumnarOutput(
      int32_t numMatches,
      rmm::cuda_stream_view stream);
};


// ============================================================================
// Bridge translator: connects plan node to row-wise operators
// ============================================================================

class RowHashJoinBridgeTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;

  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override;

  exec::OperatorSupplier toOperatorSupplier(
      const core::PlanNodePtr& node) override;
};

} // namespace facebook::velox::cudf_velox
