/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h" // FieldDesc
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <deque>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

// A pinned host staging slot for the fused pack path. Instances have PROCESS
// lifetime: they are handed out and recycled by PinnedPackSlotPool (see
// CudfConversion.cpp) and their memory/event are never freed per query.
struct PinnedPackSlot {
  uint8_t* host = nullptr; // cudaHostAlloc'd, retained across queries
  int64_t capacity = 0;
  // Recorded after this slot's H2D; waited on before repacking the slot.
  // A created-but-never-recorded event synchronizes instantly, so no
  // separate "in use" flag is needed.
  cudaEvent_t done = nullptr;
};

class CudfFromVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kGpuBatchSizeRows =
      "velox.cudf.gpu_batch_size_rows";

  // Transfer discipline of the pinned-pack path: "auto" (default), "sync",
  // or "async". Principle: match the discipline to the consumer's barrier
  // structure. A probe consumer has an unavoidable per-batch barrier anyway
  // (the join's match-count readback), so the simple pack→DMA→synchronize
  // sequence costs nothing extra and keeps per-operator attribution clean
  // (E2E proven identical, job 13187756). Consumers without a per-batch
  // barrier (CudfBatchConcat, build-side accumulate) get the async
  // ping-pong path, where the slot events are load-bearing. "auto" resolves
  // per OPERATOR from the downstream neighbor — one query can use both.
  static constexpr const char* kPinnedPackSync =
      "velox.cudf.pinned_pack_sync";

  CudfFromVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  bool needsInput() const override {
    return !finished_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  // Fused pinned-pack conversion (the reworked benchmarkCpuColToRow path).
  // Packs the selected input batches DIRECTLY into a pinned host slot — no
  // mergeRowVectors pass — and ships pinned-source (truly async) DMA:
  //   row_wise=true  -> ROW-major pack  -> RowStoreVector (no GPU transpose)
  //   row_wise=false -> COL-major pack  -> CudfVector     (no from_arrow)
  // Returns nullptr when the path does not apply (flag off, unsupported
  // schema, non-flat/nullable children) — caller falls through to the
  // standard merge + from_arrow path.
  // 2026-08-24 reconstruction: tryPinnedPack split into named steps.
  // Per-batch pack state threaded between the steps (positionally indexed
  // [batch * numCols + channel]; see loadChildren docs).
  struct PackBatch {
    int32_t numCols{0};
    std::vector<VectorPtr> keepAlive;
    std::vector<const uint8_t*> srcs;
    std::vector<const uint64_t*> rawNullsPtrs;
    bool anyNulls{false};
    int32_t nullStride{0};
    int64_t heapOffset{0};
  };
  void resolveRowPathOnce(bool rowWiseMode);
  bool computeLayoutOnce(
      const RowTypePtr& inRowType,
      bool rowMode,
      bool boundary);
  bool loadChildren(
      const std::vector<RowVectorPtr>& selectedInputs,
      const RowTypePtr& inRowType,
      bool rowMode,
      bool boundary,
      PackBatch& pb);
  void packIntoSlot(
      const std::vector<RowVectorPtr>& selectedInputs,
      vector_size_t totalRows,
      bool rowMode,
      bool boundary,
      const PackBatch& pb,
      uint8_t* const base);
  RowVectorPtr tryPinnedPack(
      const std::vector<RowVectorPtr>& selectedInputs,
      vector_size_t totalRows);

  const std::optional<std::string> timestampTimeZone_;
  std::vector<RowVectorPtr> inputs_;
  std::size_t currentOutputSize_ = 0;
  bool finished_ = false;

  // ---- Row-wise ingest (benchmarkCpuColToRow) ----
  //
  // Packs Velox's columnar batch straight into a fixed-stride ROW buffer on the
  // host and sends it as ONE contiguous H2D copy, producing a RowStoreVector.
  //
  // WHY THIS CAN BEAT THE COLUMNAR PATH ON THE CRITICAL PATH: E2E here is pinned
  // to ingest cost, not GPU cost. Columnar ingest produces N separate device
  // column buffers -- N allocations, N H2D copies, and N NULL-MASK MEMSETS. In
  // any nsys profile of this system cudaMemsetAsync is 54-79% of ALL CPU-side
  // CUDA time, and it is exactly those per-column null masks. A row buffer has
  // no null masks and no per-column anything: one allocation, one copy. It also
  // makes the GPU-side transposeToRows unnecessary, since the data arrives in
  // row layout already.
  //
  // The original implementation of this path was, in its own words, "deliberately
  // naive": it malloc'd AND ZEROED a fresh std::vector every batch, copied field
  // by field with scalar memcpy, transferred from PAGEABLE memory (which makes
  // cudaMemcpyAsync synchronous and ~2-3x slower), stream.synchronize()'d on
  // every batch, and re-uploaded the constant field descriptors each time. All of
  // that is fixed below; the state cached here is what makes it possible.
  // Two slots, ping-ponged: while slot A's copy is in flight we pack into slot
  // B. We wait on a slot's event before reusing it -- by then, one full batch
  // later, it has long since completed, so the wait costs nothing.
  //
  // Slots are ACQUIRED from a process-wide pool (see PinnedPackSlotPool in the
  // .cpp) and returned at close with memory and events INTACT — mirroring the
  // CUDA driver's own staging-pool design. cudaHostAlloc/cudaFreeHost globally
  // serialize the driver and cost ~ms per 100MB (un)pinned, so per-query
  // alloc/free of large slots convoys every other thread's CUDA calls
  // (measured: ~1s/thread of stalls at 1M-row batches). Total pinned memory is
  // capped at the high-water mark of concurrently live conversion operators.
  PinnedPackSlot* pinnedSlots_[2] = {nullptr, nullptr};
  static void ensurePinnedSlotCapacity(PinnedPackSlot& slot, int64_t bytes);
  int pinnedSlot_ = 0;

  // Whether this operator's DOWNSTREAM consumer can actually accept a
  // RowStoreVector. Row ingest is only legal when it can: a CudfFilterProject
  // (e.g. TPC-H Q9/Q8's year(o_orderdate) projection, which correctly runs on
  // the GPU because it WIDENS the row) accepts only CudfVector and would
  // otherwise die on "cudfInput != nullptr". Determined once, on the first
  // getOutput, by inspecting the next operator in the driver -- the same trick
  // RowHashJoinProbe uses for emitColumnar_.
  //   -1 = not yet determined, 0 = emit CudfVector, 1 = emit RowStoreVector
  int emitRowStore_ = -1;

  // ---- Boundary-hybrid (keys-only pack) ----
  //
  // When CudfConfig::benchmarkBoundaryHybrid is set and the downstream
  // consumer is RowHashJoinProbe/RowHashJoinBuild, the pinned pack transfers
  // ONLY the join-key columns (in join-key order); the full input batches are
  // retained host-side and attached to the emitted RowStoreVector (see
  // RowStoreVector::setBoundaryPayload). Resolved once alongside
  // emitRowStore_ by asking the downstream join operator for its key names:
  // probe side -> leftKeys, build side -> rightKeys.
  //   -1 = unresolved, 0 = off (normal full-row pack), 1 = keys-only pack
  int boundaryMode_ = -1;
  // Join-key column names (join-key order), stashed at resolution; resolved
  // to child indices of the ACTUAL input row type on first pack (the input
  // type can differ from outputType_, e.g. a scan batch carrying a
  // filter-only column).
  std::vector<std::string> boundaryKeyNames_;
  // Input child index of each packed key, in join-key order (parallel to the
  // keys-only rowFields_ layout).
  std::vector<int32_t> boundaryPackChannels_;
  // Join-key CHANNEL indices in the input row type, resolved lazily on the
  // first packed batch from boundaryKeyNames_ (which is filled for ANY
  // adjacent row-join op, not just boundary mode). Used by the null-KEY
  // guard: null join keys are unrepresentable in the row matcher.
  std::vector<int32_t> keyChannels_;
  bool keyChannelsResolved_{false};

  // ONE stream for every RowStoreVector this operator produces.
  //
  // The columnar path takes a fresh stream from the pool on each doGetOutput,
  // which is fine there because a CudfVector carries its own stream and cuDF
  // handles the ordering. It is NOT fine for rows: CudfBatchConcat merges N
  // RowStoreVectors with device-to-device copies issued on ONE stream, and then
  // releases the sources. If the sources were produced on DIFFERENT streams,
  // those copies read buffers with no cross-stream ordering, and the sources are
  // freed on their own streams while the copies may still be in flight -- a race
  // that shows up as nondeterministically wrong query results.
  //
  // Pinning the row path to a single per-operator stream makes every
  // RowStoreVector it emits stream-ordered against every other, which is exactly
  // the invariant concatRowStore() relies on.
  std::optional<rmm::cuda_stream_view> rowStream_;

  // Row layout is a property of the input schema, so compute it once.
  std::vector<FieldDesc> rowFields_;
  int32_t rowWidth_ = 0;   // 8-byte aligned row stride (row mode)
  int64_t dataWidth_ = 0;  // raw sum of column widths, unpadded (col mode)
  bool rowLayoutReady_ = false;
  bool hasStringFields_ = false; // any kFieldString in rowFields_ (row mode)
  // Uploaded once; SHARED by every RowStoreVector this operator emits (the
  // shared_ptr keeps it alive as long as any batch lives downstream).
  std::shared_ptr<rmm::device_buffer> rowFieldsDevice_;

  // Col-major pinned pack: cudf dtypes per column (cached with the layout).
  std::vector<cudf::data_type> colDtypes_;
  // -1 = schema unsupported, pinned pack permanently disabled for this
  // operator (falls back to the standard path without re-checking).
  int pinnedPackState_ = 0;
  // Resolved transfer discipline (kPinnedPackSync): -1 = unresolved,
  // 1 = sync (per-batch stream.synchronize, no event machinery),
  // 0 = async (ping-ponged slots guarded by events).
  int pinnedPackSyncMode_ = -1;
};

class CudfToVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kPassthroughMode =
      "velox.cudf.to_velox.passthrough_mode";

  CudfToVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  bool needsInput() const override {
    return !finished_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  bool isPassthroughMode() const;
  std::optional<uint64_t> averageRowSize();
  // Convert inputs_.front() to Velox once; slice it CPU-side per batch.
  RowVectorPtr convertFrontToVelox();
  std::optional<uint64_t> averageRowSize_;
  std::deque<CudfVectorPtr> inputs_;
  // Non-CudfVector inputs that pass through without D2H conversion
  std::deque<RowVectorPtr> passthroughInputs_;
  // Converted CPU-side buffer being drained by successive doGetOutput() calls.
  RowVectorPtr veloxBuffer_;
  // Current offset into veloxBuffer_ for the next slice.
  vector_size_t veloxOffset_{0};
  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
