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
#include "velox/experimental/cudf/exec/GpuFusedProbe.cuh"
#include "velox/experimental/cudf/exec/GpuRowHashTable.cuh"
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
    // COLUMNAR build fetch: when set, the build skipped the row-store transpose
    // and the probe reads build payload straight from these columns. buildTable
    // owns the concatenated build columns for the probe's lifetime.
    bool columnarBuild{false};
    std::shared_ptr<cudf::table> buildTable;
    // Key-only row store: under columnar payload, a fingerprint build keeps the
    // key row-native here so verify can re-pack it without a full row store.
    // Decouples key layout (always row) from payload layout (columnar vs row).
    GpuFixedRowStore keyRowStore{};
    rmm::device_buffer keyRowBuffer;
    rmm::device_buffer keyFieldsBuffer;
    std::vector<FieldDesc> keyRowFields;
    int32_t keyRowWidth{0};
    bool hasKeyRowStore{false};
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

    // ---- Fused-probe support (populated only when benchmarkFusedProbe) ----
    // Device-probeable hash table (composite key -> build row index) so the
    // fused kernel can look up matches inline without a host-orchestrated
    // cudf::hash_join::inner_join. Backed by the interleaved slot buffer below
    // (AoS layout: each slot is [key words..., val] for single-load probes).
    GpuHashTable deviceMap{};
    rmm::device_buffer mapSlots;
    bool hasDeviceMap{false};

    // ---- Composite-key packing (single-word fast path) ----
    // When packEnabled, deviceMap is a single-word table over keys packed as
    // OR_j ((val_j - packMin[j]) << packShift[j]); the fused probe must pack the
    // probe key the same way. packMax bounds the probe range check; packSentinel
    // is a value no build key produces (for out-of-range probe rows).
    bool packEnabled{false};
    std::vector<int64_t> packMin;
    std::vector<int64_t> packMax;
    std::vector<int32_t> packShift;
    uint64_t packSentinel{0};

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
// Fused probe operator: replaces a maximal run of consecutive row-wise inner
// joins with a single operator that walks each probe tuple through all N build
// hash tables in one kernel (one thread per tuple, tuple resident in a local
// accumulator). See GpuFusedProbe.cuh for the kernel-level design and the
// row-native rationale (downstream join keys produced by earlier joins are
// read from the accumulator instead of re-gathered from intermediates).
// ============================================================================

class FusedRowHashJoinProbe : public CudfOperatorBase {
 public:
  // `joinNodes` are the fused HashJoinNodes in probe order (bottom join first).
  // The operator adopts the LAST node's id/outputType as its plan-node identity
  // so downstream operators see the final chain output.
  FusedRowHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::vector<std::shared_ptr<const core::HashJoinNode>> joinNodes);

  bool needsInput() const override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  // Compute the accumulator layout + all per-step mappings once and upload the
  // device-side descriptor arrays. `probeFields` is the row layout of the probe
  // (fact) input in input-column order.
  void setupLayout(
      const std::vector<FieldDesc>& probeFields,
      rmm::cuda_stream_view stream);

  RowVectorPtr makeColumnarOutput(
      int32_t numMatches,
      rmm::cuda_stream_view stream);

  // OPT 2: wrap the already-columnar kernel output in a CudfVector. No
  // rows_to_columns pass -- the kernel wrote the columns directly.
  RowVectorPtr makeColumnarOutputDirect(
      int32_t numMatches,
      rmm::cuda_stream_view stream);

  std::vector<std::shared_ptr<const core::HashJoinNode>> joinNodes_;
  int32_t numSteps_ = 0;

  // Build-side data, one per fused join (fetched from N bridges).
  std::vector<std::shared_ptr<RowHashJoinBridge::BuildData>> builds_;
  std::vector<ContinueFuture> buildFutures_;

  // Per-probe-batch input.
  RowVectorPtr input_;

  // Probe (fact) row layout.
  std::vector<FieldDesc> probeFields_;
  int32_t probeRowWidth_ = 0;
  bool initialized_ = false;
  rmm::device_buffer probeRowBuffer_;   // Path B transpose target
  rmm::device_buffer probeFieldsBuffer_;
  int64_t probeRowCapacity_ = 0;
  bool fieldsUploaded_ = false;

  // Accumulator layout.
  int32_t accWidth_ = 0;

  // Output row layout (final chain output, column order = outputType).
  std::vector<FieldDesc> outputFields_;
  int32_t outputRowWidth_ = 0;

  // Device descriptor arrays (built once by setupLayout).
  bool layoutComputed_ = false;
  rmm::device_buffer probeToAccBuf_;   // FieldMapping[]
  int32_t numProbeFields_ = 0;
  rmm::device_buffer accToOutBuf_;     // FieldMapping[]
  int32_t numOutFields_ = 0;
  std::vector<rmm::device_buffer> stepKeyFieldBufs_;   // FusedKeyField[] per step
  std::vector<rmm::device_buffer> stepBuildToAccBufs_; // FieldMapping[] per step
  std::vector<rmm::device_buffer> stepBuildColsBufs_;  // ColBuildField[] per step
  rmm::device_buffer stepsBuf_;        // FusedJoinStep[] on device
  bool hasFingerprint_ = false;        // any step uses a composite fp slot

  // ---- OPT 1 / OPT 2: columnar seed and columnar output ----
  // hostAccOffsets_/hostAccWidths_ are the accumulator offset+width of each
  // probe input column (OPT 1) and of each output column (OPT 2), in input /
  // outputType column order. The device descriptor arrays are rebuilt per batch
  // because they embed raw column POINTERS, which change with every input
  // batch and every fresh output allocation.
  std::vector<int32_t> seedAccOffsets_;  // per SEEDED probe column
  std::vector<int32_t> seedAccWidths_;
  std::vector<int32_t> seedProbeCol_;    // c0 column index of each seed entry
  std::vector<int32_t> outAccOffsets_;   // per ACC-sourced output column
  std::vector<int32_t> accOutIdx_;       // output-col index of each acc entry
  rmm::device_buffer colSeedBuf_;        // ColSeedField[]
  rmm::device_buffer colOutBuf_;         // ColOutField[] (acc-sourced out cols)

  // OPT 3 (lazy payload): output columns read from the probe at survivor time.
  bool lazyPayload_ = false;
  std::vector<int32_t> deferredProbeCol_; // c0 column index to read from
  std::vector<int32_t> deferredOutIdx_;   // output-col index to write to
  std::vector<int32_t> deferredWidth_;
  std::vector<const uint8_t*> probeColBase_; // per c0 col base ptr (this batch)
  rmm::device_buffer deferredBuf_;           // DeferredField[]
  int32_t numDeferred_ = 0;
  // Output column buffers for OPT 2, allocated at the upper bound (one probe
  // row can produce at most one output row) and reused across batches.
  std::vector<rmm::device_buffer> outColBuffers_;
  int64_t outColCapacity_ = 0;

  // Per-batch output buffers (reused across batches).
  rmm::device_buffer probeGatherBuffer_;
  int64_t gatherCapacity_ = 0;
  rmm::device_buffer outputFieldsBuffer_; // FieldDesc[] (for RowStoreVector out)
  rmm::device_buffer outCountBuf_;        // int32 device counter

  // ---- Deferred harvest (one batch of pipeline depth) ----
  //
  // The fused kernel's output cardinality is not known until it runs, so the
  // host must read `outCount` back before it can build the output vector. Doing
  // that in the SAME doGetOutput() that launched the kernel means a blocking
  // stream.synchronize() on EVERY batch, which parks the driver thread and
  // serialises CPU ingest against GPU compute. nsys measured the cost: 21.1 s
  // across 42,777 cudaMemcpyAsync calls (493 us each, versus ~19 us for a
  // genuinely async copy) -- 72% of the operator's entire CPU-side CUDA time.
  //
  // Instead we launch the kernel and RETURN. The driver then goes upstream for
  // the next batch (scan + Velox->cuDF + H2D) while the GPU works. On the NEXT
  // doGetOutput() we harvest the now-complete batch, so the synchronize costs
  // approximately nothing.
  //
  // No double-buffering is needed: every kernel, the rowsToColumns scatter, and
  // the memset all run on the SAME stream, and CUDA stream ordering guarantees
  // the previous batch's reads complete before the next batch's writes begin.
  struct InFlight {
    bool valid = false;
    int32_t probeRows = 0;
    rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  };
  InFlight inflight_;

  bool finished_ = false;
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
