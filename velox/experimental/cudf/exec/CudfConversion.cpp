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

#include "velox/experimental/cudf/exec/HostRowVector.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/BenchmarkTimelineFlag.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/exec/OperatorUtils.h"

#include <functional>
#include <limits>
#include "velox/experimental/cudf/exec/DeferralPlan.h"
#include "velox/experimental/cudf/exec/DeferralStats.h"
#include "velox/experimental/cudf/exec/RowOrderBy.h"
#include "velox/experimental/cudf/exec/DedicatedStream.h"
#include "velox/experimental/cudf/exec/RowHashJoin.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/RowStoreVector.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <mutex>

#include <algorithm>
#include <chrono>

// Local CUDA error check for the row-ingest path (cudf's CUDF_CUDA_TRY is not
// pulled in here, and a silent cudaHostAlloc failure would corrupt results).
#define VELOX_CUDA_CHECK(expr)                          \
  do {                                                  \
    cudaError_t _err = (expr);                          \
    VELOX_CHECK(                                        \
        _err == cudaSuccess,                            \
        "CUDA error in row ingest: {}",                 \
        cudaGetErrorString(_err));                      \
  } while (0)

namespace facebook::velox::cudf_velox {

namespace {
// Concatenate multiple RowVectors into a single RowVector.
// Copied from AggregationFuzzer.cpp.
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& results,
    velox::memory::MemoryPool* pool) {
  VELOX_NVTX_FUNC_RANGE();
  if (results.size() == 1) {
    return results[0];
  }
  vector_size_t totalCount = 0;
  for (const auto& result : results) {
    totalCount += result->size();
  }
  auto copy =
      BaseVector::create<RowVector>(results[0]->type(), totalCount, pool);
  auto copyCount = 0;
  for (const auto& result : results) {
    copy->copy(result.get(), copyCount, 0, result->size());
    copyCount += result->size();
  }
  return copy;
}

cudf::size_type preferredGpuBatchSizeRows(
    const facebook::velox::core::QueryConfig& queryConfig) {
  constexpr cudf::size_type kDefaultGpuBatchSizeRows = 100000;
  const auto batchSize = queryConfig.get<int32_t>(
      CudfFromVelox::kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "velox.cudf.gpu_batch_size_rows must be > 0");
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "velox.cudf.gpu_batch_size_rows must be <= max(vector_size_t)");
  return batchSize;
}
} // namespace

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfFromVelox",
          nvtx3::rgb{255, 140, 0}, // Orange
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt),
      timestampTimeZone_(driverCtx->queryConfig().get<std::string>(
          facebook::velox::core::QueryConfig::kSessionTimezone)) {}

void CudfFromVelox::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    // Materialize lazy vectors
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    input->loadedVector();

    // Timeline: initialize thread_local scan state on first addInput call
    // (this runs on the driver's actual thread, unlike the constructor).
    if (inputs_.empty() && CudfConfig::getInstance().benchmarkLogTimeline &&
        operatorCtx_->driverCtx()->driverId == 0) {
      auto& scanState = threadScanTraceState();
      if (scanState.driverId == -1) {
        scanState.driverId = operatorCtx_->driverCtx()->driverId;
        scanState.pipelineId = operatorCtx_->driverCtx()->pipelineId;
        benchmarkTimelineEnabled().store(true, std::memory_order_relaxed);
      }
    }

    // Accumulate inputs
    inputs_.push_back(input);
    currentOutputSize_ += input->size();
  }
}

RowVectorPtr CudfFromVelox::doGetOutput() {
  const auto targetOutputSize =
      preferredGpuBatchSizeRows(operatorCtx_->driverCtx()->queryConfig());

  finished_ = noMoreInput_ && inputs_.empty();

  if (finished_ or
      (currentOutputSize_ < targetOutputSize and not noMoreInput_) or
      inputs_.empty()) {
    return nullptr;
  }

  // Timeline: initialize per-thread scan state and signal scan phase complete
  const bool logTimeline = CudfConfig::getInstance().benchmarkLogTimeline;
  const auto driverId = operatorCtx_->driverCtx()->driverId;
  const auto pipelineId = operatorCtx_->driverCtx()->pipelineId;
  auto timeNow = []() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d FromVelox start %lld\n",
           (long long)timeNow(), pipelineId, (long long)inputs_[0]->size());
  }

  auto tpStart = std::chrono::steady_clock::now();

  // Select inputs that don't exceed the max vector size limit
  std::vector<RowVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;
  auto const maxVectorSize = std::numeric_limits<vector_size_t>::max();

  for (const auto& input : inputs_) {
    if (totalSize + input->size() <= maxVectorSize) {
      selectedInputs.push_back(input);
      totalSize += input->size();
    } else {
      break;
    }
  }

  auto tpBeforeMerge = std::chrono::steady_clock::now();

  // ===== Fused pinned-pack path (the reworked cpu_col_to_row) =====
  // Packs the selected inputs DIRECTLY into a pinned host slot -- fusing the
  // mergeRowVectors pass and the driver's hidden pageable-staging copy into a
  // single CPU pass -- then ships truly-async pinned-source DMA. Falls
  // through to the standard merge + from_arrow path when it does not apply.
  if (auto packed = tryPinnedPack(selectedInputs, totalSize)) {
    inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
    currentOutputSize_ -= totalSize;
    auto nanos = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
          .count();
    };
    addRuntimeStat(
        "fromVeloxSelectNanos",
        RuntimeCounter(
            nanos(tpStart, tpBeforeMerge), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxTotalNanos",
        RuntimeCounter(
            nanos(tpStart, std::chrono::steady_clock::now()),
            RuntimeCounter::Unit::kNanos));
    if (logTimeline && driverId == 0) {
      printf("OPTRACE %lld d0 p%d FromVelox end %lld\n",
             (long long)timeNow(), pipelineId, (long long)totalSize);
      threadScanTraceState().scanTraceNeeded = true;
    }
    return packed;
  }

  // Combine selected RowVectors into a single RowVector
  auto input = mergeRowVectors(selectedInputs, inputs_[0]->pool());

  auto tpAfterMerge = std::chrono::steady_clock::now();

  // Remove processed inputs
  inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
  currentOutputSize_ -= totalSize;

  // Early return if no input
  if (input->size() == 0) {
    return nullptr;
  }

  // Get a stream from the global stream pool
  auto stream = cudfGlobalStreamPool().get_stream();
  const bool logH2D = CudfConfig::getInstance().benchmarkLogGatherTime;
  // ===== Standard columnar path: Arrow → cudf =====

  // Convert RowVector to cudf table.  toCudfTable synchronizes the stream
  // internally before releasing Arrow host buffers, so no additional sync
  // is needed here.
  auto tpBeforeToCudf = std::chrono::steady_clock::now();

  with_arrow::ToCudfTiming toCudfTiming;
  auto tbl = with_arrow::toCudfTable(
      input,
      input->pool(),
      stream,
      get_output_mr(),
      timestampTimeZone_,
      &toCudfTiming);

  auto tpAfterToCudf = std::chrono::steady_clock::now();

  // Always emit detailed breakdown stats
  {
    auto nanos = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    addRuntimeStat(
        "fromVeloxSelectNanos",
        RuntimeCounter(nanos(tpStart, tpBeforeMerge), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxMergeNanos",
        RuntimeCounter(nanos(tpBeforeMerge, tpAfterMerge), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxToCudfNanos",
        RuntimeCounter(nanos(tpBeforeToCudf, tpAfterToCudf), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "toCudfExportNanos",
        RuntimeCounter(toCudfTiming.exportNanos, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "toCudfFromArrowNanos",
        RuntimeCounter(
            toCudfTiming.fromArrowNanos, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "toCudfSyncNanos",
        RuntimeCounter(toCudfTiming.syncNanos, RuntimeCounter::Unit::kNanos));
    // Device-side size of the converted table (fixed-width columns), for
    // effective H2D bandwidth: unlike h2dBytes this is not gated behind
    // benchmarkLogGatherTime, which perturbs the probe with event syncs.
    int64_t toCudfBytes = 0;
    auto tblView = tbl->view();
    for (int c = 0; c < tblView.num_columns(); c++) {
      auto const& col = tblView.column(c);
      if (cudf::is_fixed_width(col.type())) {
        toCudfBytes +=
            static_cast<int64_t>(col.size()) * cudf::size_of(col.type());
      } else if (col.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(col);
        toCudfBytes += scv.chars_size(stream) +
            static_cast<int64_t>(col.size() + 1) * sizeof(int32_t);
      }
    }
    addRuntimeStat(
        "toCudfBytes",
        RuntimeCounter(toCudfBytes, RuntimeCounter::Unit::kBytes));
    addRuntimeStat(
        "fromVeloxTotalNanos",
        RuntimeCounter(nanos(tpStart, tpAfterToCudf), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxBatchesMerged",
        RuntimeCounter(static_cast<int64_t>(selectedInputs.size())));
    addRuntimeStat(
        "fromVeloxOutputRows",
        RuntimeCounter(static_cast<int64_t>(input->size())));
  }

  if (logH2D) {
    addRuntimeStat(
        "h2dRows",
        RuntimeCounter(static_cast<int64_t>(input->size())));

    // Compute actual bytes on GPU (sum of column data sizes)
    int64_t h2dBytes = 0;
    auto tblView = tbl->view();
    for (int c = 0; c < tblView.num_columns(); c++) {
      auto const& col = tblView.column(c);
      if (cudf::is_fixed_width(col.type())) {
        h2dBytes += static_cast<int64_t>(col.size()) *
                    cudf::size_of(col.type());
      } else if (col.type().id() == cudf::type_id::STRING) {
        cudf::strings_column_view scv(col);
        h2dBytes += scv.chars_size(stream) +
            static_cast<int64_t>(col.size() + 1) * sizeof(int32_t);
      }
    }
    addRuntimeStat(
        "h2dBytes",
        RuntimeCounter(h2dBytes, RuntimeCounter::Unit::kBytes));
  }

  VELOX_CHECK_NOT_NULL(tbl);

  // Return a CudfVector that owns the cudf table
  const auto size = tbl->num_rows();
  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d FromVelox end %lld\n",
           (long long)timeNow(), pipelineId, (long long)size);
    threadScanTraceState().scanTraceNeeded = true;
  }
  return std::make_shared<CudfVector>(
      input->pool(), outputType_, size, std::move(tbl), stream);
}

namespace {
// Process-lifetime pool of pinned pack slots, mirroring the CUDA driver's own
// staging-pool design (one allocation cost per process, recycled forever).
// Rationale: cudaHostAlloc/cudaFreeHost globally serialize the CUDA driver
// and cost milliseconds per 100MB of (un)pinning. With slots owned by
// per-query operators, every query paid 48 alloc/free cycles of up to 128MB
// each, and every other thread's CUDA calls convoyed behind the driver lock
// (~1s/thread of stalls at 1M-row batches — the ">=500K regression").
// Slots return here at operator close with memory and events INTACT; total
// pinned memory is capped at the high-water mark of concurrent operators.
class PinnedPackSlotPool {
 public:
  static PinnedPackSlotPool& instance() {
    static PinnedPackSlotPool pool;
    return pool;
  }
  // Best-fit by capacity: probe-side operators (100s of MB) and build-side
  // operators (~MB) share this pool, and handing a small recycled slot to a
  // big consumer forces a regrow — cudaFreeHost + cudaHostAlloc, which
  // globally serialize the driver (measured at 10M-row batches: ~13 regrows
  // of 640MB per query, re-creating the very churn the pool exists to
  // prevent). Matching on a capacity hint makes steady-state regrowth zero.
  PinnedPackSlot* acquire(int64_t capacityHint) {
    std::lock_guard<std::mutex> lock(mutex_);
    int best = -1;
    for (int i = 0; i < static_cast<int>(free_.size()); i++) {
      if (free_[i]->capacity >= capacityHint) {
        if (best < 0 || free_[i]->capacity < free_[best]->capacity) {
          best = i;
        }
      }
    }
    if (best < 0 && !free_.empty()) {
      // Nothing big enough: take the smallest slot (cheapest old buffer to
      // free on regrow) rather than growing the pool.
      best = 0;
      for (int i = 1; i < static_cast<int>(free_.size()); i++) {
        if (free_[i]->capacity < free_[best]->capacity) {
          best = i;
        }
      }
    }
    if (best >= 0) {
      auto* slot = free_[best];
      free_.erase(free_.begin() + best);
      return slot;
    }
    return new PinnedPackSlot();
  }
  void release(PinnedPackSlot* slot) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_.push_back(slot);
  }

 private:
  std::mutex mutex_;
  std::vector<PinnedPackSlot*> free_;
};
} // namespace

// Grow a pinned slot to at least `bytes`. cudaFreeHost/cudaHostAlloc
// serialize the CUDA driver process-wide; slots are pooled and retained
// across queries, so this is a first-query (warm-up) cost only.
void CudfFromVelox::ensurePinnedSlotCapacity(
    PinnedPackSlot& slot,
    int64_t bytes) {
  if (slot.capacity >= bytes) {
    return;
  }
  if (slot.host != nullptr) {
    VELOX_CUDA_CHECK(cudaFreeHost(slot.host));
  }
  slot.capacity = std::max<int64_t>(bytes, slot.capacity * 2);
  VELOX_CUDA_CHECK(cudaHostAlloc(
      reinterpret_cast<void**>(&slot.host),
      slot.capacity,
      cudaHostAllocDefault));
  // Zero once so row-mode inter-field padding is deterministic.
  std::memset(slot.host, 0, slot.capacity);
}

// One-time (per operator instance) row-path resolution: who consumes our
// output (row store vs columnar), boundary mode, transfer discipline. See
// the section docs inside. Split out of tryPinnedPack in the 2026-08-24
// reconstruction; logic unchanged.
void CudfFromVelox::resolveRowPathOnce(bool rowWiseMode) {
  // Decide ONCE, from the downstream neighbor: (a) whether it can take a
  // RowStoreVector, and (b) the transfer discipline (kPinnedPackSync docs in
  // the header). Auto rule: a probe consumer has a per-batch barrier anyway
  // (join match-count readback), so use the simple sync path; consumers
  // without one (concat, build accumulate) get the async ping-pong path.
  if (emitRowStore_ < 0) {
    emitRowStore_ = 0;
    exec::Operator* next = nullptr;
    auto* driver = operatorCtx_->driver();
    if (driver != nullptr) {
      const auto ops = driver->operators();
      for (size_t i = 0; i + 1 < ops.size(); i++) {
        if (ops[i] == this) {
          next = ops[i + 1];
          break;
        }
      }
    }
    if (rowWiseMode && next != nullptr) {
      // CudfBatchConcat handles RowStoreVector too (concatRowStore), but
      // only counts as a row consumer when the operator AFTER it is a row
      // join -- a concat feeding a columnar operator (e.g. the agg-rebatch
      // concat, --concat_agg) must receive CudfVector.
      exec::Operator* afterNext = nullptr;
      if (auto* driver = operatorCtx_->driver()) {
        const auto ops = driver->operators();
        for (size_t i = 0; i + 2 < ops.size(); i++) {
          if (ops[i] == this) {
            afterNext = ops[i + 2];
            break;
          }
        }
      }
      if (dynamic_cast<RowHashJoinProbe*>(next) != nullptr ||
          dynamic_cast<RowHashJoinBuild*>(next) != nullptr ||
          (dynamic_cast<CudfBatchConcat*>(next) != nullptr &&
           (dynamic_cast<RowHashJoinProbe*>(afterNext) != nullptr ||
            dynamic_cast<RowHashJoinBuild*>(afterNext) != nullptr))) {
        emitRowStore_ = 1;
      }
    }
    // Row-sort consumer (branch row-sort, 2026-08-24): a RowOrderBy directly
    // after this pack, or a GATHER CudfLocalPartition whose plan parent is
    // an OrderByNode that runs as RowOrderBy (rows pass through the gather
    // untouched). Either way the pack emits RowStoreVector.
    std::shared_ptr<const core::OrderByNode> sortNode;
    if (rowWiseMode && next != nullptr) {
      const auto& planRoot = operatorCtx_->task()->planFragment().planNode;
      if (auto* sort = dynamic_cast<RowOrderBy*>(next)) {
        sortNode = sort->orderByNode();
      } else if (rowSortConsumesGather(next, planRoot)) {
        sortNode = orderByBehindGather(planRoot, next->planNodeId());
      }
      if (sortNode != nullptr) {
        emitRowStore_ = 1;
      }
    }
    // ---- Boundary-hybrid resolution (keys-only pack) ----
    // Ask the downstream join operator which columns are join keys: only
    // those cross the boundary; everything else is retained host-side and
    // attached to the emitted RowStoreVector. Probe side packs leftKeys,
    // build side packs rightKeys — both in JOIN-KEY order, so probe key k
    // and build key k line up field-for-field in the keys-only row layout.
    // Key names are captured for ANY adjacent row-join op: boundary mode
    // packs exactly these; the row pack's null-KEY guard checks exactly
    // these (null join keys are unrepresentable in the row matcher).
    // Spine deferral v2 (2026-08-24): boundary (crossing-set + host
    // retention) applies ONLY to a pack feeding a join PROBE -- the spine.
    // Build-side packs are always EAGER; with pruning they carry just the
    // columns the join needs (keys + join output).
    boundaryMode_ = 0;
    if (rowWiseMode && next != nullptr) {
      std::shared_ptr<const core::HashJoinNode> adjJoin;
      if (auto* probe = dynamic_cast<RowHashJoinProbe*>(next)) {
        for (const auto& key : probe->joinNode()->leftKeys()) {
          boundaryKeyNames_.push_back(key->name());
        }
        adjJoin = probe->joinNode();
        const auto& cfg = CudfConfig::getInstance();
        if (!boundaryKeyNames_.empty() && cfg.benchmarkBoundaryHybrid) {
          // Adaptive (batch-level, 2026-08-24): start EAGER; tryPinnedPack
          // consults DeferralStats per BATCH and flips one-way to deferred
          // once the chain endpoint has observed enough reduction.
          // Non-adaptive: always defer.
          deferralEligible_ = true;
          boundaryMode_ = cfg.benchmarkDeferralAdaptive ? 0 : 1;
        }
      } else if (auto* build = dynamic_cast<RowHashJoinBuild*>(next)) {
        for (const auto& key : build->joinNode()->rightKeys()) {
          boundaryKeyNames_.push_back(key->name());
        }
        adjJoin = build->joinNode();
        // Build-side deferral (2026-09-03, --boundary_build_defer): the
        // build's payload also stays on the host -- the ingress ships keys +
        // rowRef, RowHashJoinBuild retains the host batches in its
        // BoundaryHostStore, and the probe gathers survivors' build payload
        // host-side. Static (always defer): a build materializes at the
        // probe's exit, which for a CPU exit is the deferred case (i).
        // PLAN-TIME RULE: the build is complete before any probe statistic
        // exists, so its verdict cannot be adaptive. Defer the build payload
        // iff the plan has a SINGLE hash join (its payload then materializes
        // once, at the query's CPU exit); with more joins the build payload
        // of an inner join would have to re-enter the device, so stay eager.
        const auto& bcfg = CudfConfig::getInstance();
        if (!boundaryKeyNames_.empty() && bcfg.benchmarkBoundaryHybrid &&
            bcfg.benchmarkBoundaryBuildDefer) {
          int numJoins = 0;
          int numSorts = 0;
          std::function<void(const core::PlanNodePtr&)> countJoins =
              [&](const core::PlanNodePtr& node) {
                if (std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
                  ++numJoins;
                }
                if (std::dynamic_pointer_cast<const core::OrderByNode>(node)) {
                  ++numSorts;
                }
                for (const auto& src : node->sources()) {
                  countJoins(src);
                }
              };
          countJoins(operatorCtx_->task()->planFragment().planNode);
          // Sort-after-join (2026-09-10): a row sort above the join consumes
          // the probe's rows on the device and carries only the PROBE's
          // rowRef to the CPU exit; build ids are not propagated, so the
          // build payload must be in the rows -> stay eager when the plan
          // has a sort (paper Fig. deferral (c): build R crosses eagerly).
          // Width gate on the BUILD side's own deferrable payload (2026-09-17;
          // before, the build deferred only if the probe would, to avoid the
          // mixed mode "probe payload eager + build gathered on the host",
          // measured at 10.5 s vs 5.7 s for plain Row with 8 fixed-width
          // projected columns under the old 48 B gate). With the gate at the
          // rowRef width (8 B) the probe defers whenever it carries any
          // payload, so that mixed mode can only arise with a payload-free
          // probe, where the eager crossing is keys-only and cheap. The two
          // sides therefore decide independently: a build-only payload
          // (e.g. a probe projecting keys only) is deferred as well.
          const int32_t minBytes = bcfg.benchmarkDeferMinPayloadBytes;
          const int32_t probeBytes = probeSideDeferredBytes(*adjJoin);
          const int32_t buildBytes = buildSideDeferredBytes(*adjJoin);
          addRuntimeStat(
              "fromVeloxBuildProbePayloadBytes", RuntimeCounter(probeBytes));
          addRuntimeStat(
              "fromVeloxBuildPayloadBytes", RuntimeCounter(buildBytes));
          // Endpoint rule (2026-09-19): the build's payload materializes at
          // the probe's exit, and only the CPU-exit path (makeHostOutput)
          // can gather deferred BUILD columns from the build's host store.
          // When a GPU operator consumes the join (aggregation, later join,
          // sort), the columnar output path cannot fill them, and the
          // payload would have to re-enter the device anyway -- so the
          // build stays eager. The probe side keeps its own (adaptive)
          // rule; a deferred probe with an eager build is supported.
          bool buildGpuConsumer = false;
          {
            const auto& root = operatorCtx_->task()->planFragment().planNode;
            std::function<bool(const core::PlanNodePtr&)> walk =
                [&](const core::PlanNodePtr& node) -> bool {
              for (const auto& src : node->sources()) {
                if (src->id() == adjJoin->id()) {
                  // ANY parent counts: a Project/Filter above the join also
                  // runs on the device (CudfFilterProject) and takes the
                  // join's columnar output, so deferred build columns could
                  // not be filled there either (2026-09-19: MSR-style
                  // join -> project -> sum crashed on exactly that). Only a
                  // join at the plan root exits straight to the CPU.
                  buildGpuConsumer = true;
                  return true;
                }
                if (walk(src)) {
                  return true;
                }
              }
              return false;
            };
            walk(root);
          }
          if (numJoins == 1 && numSorts == 0 && !buildGpuConsumer &&
              (minBytes <= 0 || buildBytes >= minBytes)) {
            deferralEligible_ = true;
            boundaryMode_ = 1;
            payloadGateChecked_ = true; // plan rule + probe-width gate
            addRuntimeStat("fromVeloxBuildDefer", RuntimeCounter(1));
          } else if (numJoins == 1 && numSorts == 0 && buildGpuConsumer) {
            addRuntimeStat("fromVeloxBuildGpuConsumerEager", RuntimeCounter(1));
          } else if (numJoins == 1 && numSorts == 0) {
            addRuntimeStat("fromVeloxBuildGateEager", RuntimeCounter(1));
          }
        }
      } else if (sortNode != nullptr) {
        // Row sort: the SORT KEYS are the crossing set (the sort hashes
        // nothing, but it orders on them); the sort node is the chain
        // endpoint. Pruning keeps exactly the sort's output columns.
        boundaryKeyNames_ = RowOrderBy::sortKeyNames(*sortNode);
        joinOutputNames_ = sortNode->outputType()->names();
        const auto& cfg = CudfConfig::getInstance();
        if (!boundaryKeyNames_.empty() && cfg.benchmarkBoundaryHybrid) {
          deferralEligible_ = true;
          boundaryMode_ = cfg.benchmarkDeferralAdaptive ? 0 : 1;
          // Plan-time rule for sorts (2026-08-30): a sort never reduces, so
          // its endpoint would report S = rows and the adaptive controller
          // would never defer; but at a CPU EXIT deferral is the right
          // choice regardless (the payload is gathered once, on the host,
          // by the parallel scattered gather; nothing returns to the GPU).
          // A sort is a CPU exit unless a GPU operator consumes it.
          if (cfg.benchmarkDeferralAdaptive) {
            bool gpuConsumer = false;
            const auto& root = operatorCtx_->task()->planFragment().planNode;
            std::function<bool(const core::PlanNodePtr&)> walk =
                [&](const core::PlanNodePtr& node) -> bool {
              for (const auto& src : node->sources()) {
                if (src->id() == sortNode->id()) {
                  // Parent of the sort: a GPU consumer would be another
                  // cudf-adaptable relational operator (join/agg/sort);
                  // an exchange, project or the root means a CPU exit.
                  gpuConsumer =
                      std::dynamic_pointer_cast<const core::HashJoinNode>(node) != nullptr ||
                      std::dynamic_pointer_cast<const core::AggregationNode>(node) != nullptr ||
                      std::dynamic_pointer_cast<const core::OrderByNode>(node) != nullptr;
                  return true;
                }
                if (walk(src)) {
                  return true;
                }
              }
              return false;
            };
            walk(root);
            if (!gpuConsumer) {
              boundaryMode_ = 1;
              addRuntimeStat("fromVeloxStaticDefer", RuntimeCounter(1));
            }
          }
        }
        const auto& planRoot = operatorCtx_->task()->planFragment().planNode;
        const auto chain = collectChainKeyNames(planRoot, sortNode->id());
        laterKeyNames_ = chain.later;
        endpointJoinId_ = chain.endpointJoinId;
      }
      if (adjJoin != nullptr) {
        joinOutputNames_ = adjJoin->outputType()->names();
        // Crossing set: keys of LATER joins in the chain must also cross
        // as values (a later join hashes on them). Cached per plan walk.
        const auto& planRoot = operatorCtx_->task()->planFragment().planNode;
        const auto chain = collectChainKeyNames(planRoot, adjJoin->id());
        laterKeyNames_ = chain.later;
        endpointJoinId_ = chain.endpointJoinId;
        // Plan-time rule (2026-08-28): when the chain's TERMINAL probe exits
        // to the CPU, deferral is always right -- the survivors' payload is
        // gathered per batch from the retained host batch (cache/TLB-local)
        // and nothing returns to the GPU -- so the adaptive mode starts
        // DEFERRED instead of paying one eager observation batch per driver.
        // Only chains whose endpoint feeds a GPU consumer stay batch-adaptive
        // (there the survivors' payload must cross, and S/P decides).
        if (deferralEligible_ && boundaryMode_ == 0 &&
            CudfConfig::getInstance().benchmarkDeferralAdaptive) {
          bool endpointExitsToHost = false;
          if (auto* driver = operatorCtx_->driver()) {
            const auto ops = driver->operators();
            for (size_t i = 0; i + 1 < ops.size(); i++) {
              if (ops[i]->planNodeId() == endpointJoinId_ &&
                  dynamic_cast<RowHashJoinProbe*>(ops[i]) != nullptr) {
                endpointExitsToHost =
                    dynamic_cast<CudfToVelox*>(ops[i + 1]) != nullptr;
                break;
              }
            }
          }
          if (endpointExitsToHost) {
            boundaryMode_ = 1;
            addRuntimeStat("fromVeloxStaticDefer", RuntimeCounter(1));
          }
        }
      }
    }
    addRuntimeStat(
        "fromVeloxBoundaryMode",
        RuntimeCounter(static_cast<int64_t>(boundaryMode_)));
    const auto syncCfg =
        operatorCtx_->driverCtx()->queryConfig().get<std::string>(
            kPinnedPackSync, "auto");
    if (syncCfg == "sync") {
      pinnedPackSyncMode_ = 1;
    } else if (syncCfg == "async") {
      pinnedPackSyncMode_ = 0;
    } else {
      pinnedPackSyncMode_ =
          (next != nullptr &&
           dynamic_cast<RowHashJoinProbe*>(next) != nullptr)
          ? 1
          : 0;
    }
    addRuntimeStat(
        "fromVeloxEmitRowStore",
        RuntimeCounter(static_cast<int64_t>(emitRowStore_)));
    addRuntimeStat(
        "fromVeloxPackSyncMode",
        RuntimeCounter(static_cast<int64_t>(pinnedPackSyncMode_)));
  }
}

// One-time layout computation (boundary keys-only | pruned subset | full),
// from the INPUT's actual row type. Split out of tryPinnedPack in the
// 2026-08-24 reconstruction; logic unchanged. Returns false on a schema
// that the pinned pack cannot represent (caller falls back).
namespace {
// Nominal width of one field, as the deferred-payload gate counts it: the
// same hybridPayloadNominalWidth() the CPU gates use (exact for fixed-width
// types, 32 B for VARCHAR = 16 B slot + 16 B nominal out-of-line body), so
// the CPU and GPU width gates count a string the same way. (Before
// 2026-09-16 this counted only the 16 B slot, so two probe strings measured
// 32 B and stayed eager under the 48 B default.)
int32_t deferredFieldBytes(const TypePtr& type) {
  return static_cast<int32_t>(exec::hybridPayloadNominalWidth(type));
}
} // namespace

// The probe side's deferred width for 'join', from the plan alone: the join's
// output columns that come from the probe (left) source, minus the probe
// keys. Lets the build ingress apply the same width gate the probe ingress
// applies, so the two sides make one decision.
int32_t CudfFromVelox::probeSideDeferredBytes(const core::HashJoinNode& join) {
  const auto& probeType = join.sources()[0]->outputType();
  std::vector<std::string> keys;
  for (const auto& k : join.leftKeys()) {
    keys.push_back(k->name());
  }
  int32_t bytes = 0;
  for (const auto& name : join.outputType()->names()) {
    if (std::find(keys.begin(), keys.end(), name) != keys.end()) {
      continue;
    }
    auto idx = probeType->getChildIdxIfExists(name);
    if (!idx.has_value()) {
      continue;
    }
    bytes += deferredFieldBytes(probeType->childAt(*idx));
  }
  return bytes;
}

// The build side's deferred width for 'join': the join's output columns that
// come from the build (right) source, minus the build keys.
int32_t CudfFromVelox::buildSideDeferredBytes(const core::HashJoinNode& join) {
  const auto& buildType = join.sources()[1]->outputType();
  std::vector<std::string> keys;
  for (const auto& k : join.rightKeys()) {
    keys.push_back(k->name());
  }
  int32_t bytes = 0;
  for (const auto& name : join.outputType()->names()) {
    if (std::find(keys.begin(), keys.end(), name) != keys.end()) {
      continue;
    }
    auto idx = buildType->getChildIdxIfExists(name);
    if (!idx.has_value()) {
      continue;
    }
    bytes += deferredFieldBytes(buildType->childAt(*idx));
  }
  return bytes;
}

int32_t CudfFromVelox::deferredPayloadBytes(const RowTypePtr& inRowType) const {
  // Crossing set: the adjacent join's keys plus any later chain keys present
  // here; everything else the join outputs would stay on the host.
  auto crossing = [&](const std::string& n) {
    return std::find(boundaryKeyNames_.begin(), boundaryKeyNames_.end(), n) !=
        boundaryKeyNames_.end() ||
        std::find(laterKeyNames_.begin(), laterKeyNames_.end(), n) !=
        laterKeyNames_.end();
  };
  int32_t bytes = 0;
  for (const auto& name : joinOutputNames_) {
    if (crossing(name) || !inRowType->getChildIdxIfExists(name).has_value()) {
      continue;
    }
    // Same nominal width as the build-side and CPU gates (2026-09-16).
    bytes += deferredFieldBytes(
        inRowType->childAt(inRowType->getChildIdx(name)));
  }
  return bytes;
}

bool CudfFromVelox::computeLayoutOnce(
    const RowTypePtr& inRowType,
    bool rowMode,
    bool boundary) {
  // ---- One-time layout: offsets/widths + cudf dtypes ----
  // ONE builder (2026-08-25, review round 6) over a chosen column list:
  //   boundary : the crossing set (adjacent join keys + later chain keys
  //              present here) + a hidden __rowid; batch retained on host.
  //   default  : the adjacent join's keys + its output columns present here
  //              (the scan's filter-only columns never cross); when no
  //              adjacent join is known, or nothing would be pruned, every
  //              input column.
  if (rowLayoutReady_) {
    return pinnedPackState_ >= 0;
  }
  const int numCols = static_cast<int>(inRowType->size());
  std::vector<std::string> names;
  bool subset = false;
  if (boundary) {
    names = boundaryKeyNames_;
    for (const auto& n : laterKeyNames_) {
      if (inRowType->getChildIdxIfExists(n).has_value() &&
          std::find(names.begin(), names.end(), n) == names.end()) {
        names.push_back(n);
      }
    }
    subset = true;
  } else {
    if (rowMode && !joinOutputNames_.empty()) {
      names = boundaryKeyNames_;
      for (const auto& n : joinOutputNames_) {
        if (inRowType->getChildIdxIfExists(n).has_value() &&
            std::find(names.begin(), names.end(), n) == names.end()) {
          names.push_back(n);
        }
      }
      subset = static_cast<int>(names.size()) < numCols;
    }
    if (!subset) {
      names = inRowType->names();
    }
  }

  boundaryPackChannels_.clear();
  hasStringFields_ = false;
  std::vector<FieldDesc> fields;
  std::vector<cudf::data_type> dtypes;
  fields.reserve(names.size() + 1);
  dtypes.reserve(names.size());
  int32_t offset = 0;
  for (const auto& name : names) {
    const auto ch = static_cast<int32_t>(inRowType->getChildIdx(name));
    const auto& type = inRowType->childAt(ch);
    FieldDesc fd;
    switch (type->kind()) {
      case TypeKind::TINYINT:
        fd.byte_width = 1;
        break;
      case TypeKind::SMALLINT:
        fd.byte_width = 2;
        break;
      case TypeKind::INTEGER: // includes DATE
      case TypeKind::REAL:
        fd.byte_width = 4;
        break;
      case TypeKind::BIGINT:
      case TypeKind::DOUBLE:
        fd.byte_width = 8;
        break;
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        if (boundary) {
          VELOX_FAIL(
              "boundary-hybrid: unsupported join-key type {} for column {}",
              type->toString(),
              name);
        }
        if (!rowMode) {
          pinnedPackState_ = -1;
          return false;
        }
        fd.byte_width = kRowStrSlotBytes;
        fd.kind = kFieldString;
        hasStringFields_ = true;
        break;
      default:
        if (boundary) {
          VELOX_FAIL(
              "boundary-hybrid: unsupported join-key type {} for column {}",
              type->toString(),
              name);
        }
        pinnedPackState_ = -1;
        return false;
    }
    cudf::data_type dtype;
    try {
      dtype = veloxToCudfDataType(type);
    } catch (const std::exception&) {
      pinnedPackState_ = -1;
      return false;
    }
    if (fd.kind == kFieldString) {
      offset = (offset + 7) & ~7;
    } else {
      if (cudf::size_of(dtype) != fd.byte_width) {
        pinnedPackState_ = -1; // width mismatch (e.g. decimals)
        return false;
      }
      offset = (offset + fd.byte_width - 1) & ~(fd.byte_width - 1);
    }
    fd.offset = offset;
    if (subset) {
      boundaryPackChannels_.push_back(ch);
    }
    fields.push_back(fd);
    dtypes.push_back(dtype);
    offset += fd.byte_width;
  }
  if (boundary) {
    offset = (offset + 7) & ~7;
    rowIdField_ = static_cast<int32_t>(fields.size());
    fields.push_back({offset, 8});
    offset += 8;
    crossingNames_ = names;
    crossingNames_.push_back(kRowIdName);
  } else if (subset) {
    prunedPack_ = true;
    subsetPackNames_ = std::move(names);
  }
  rowFields_ = std::move(fields);
  colDtypes_ = std::move(dtypes);
  rowWidth_ = (offset + 7) & ~7; // 8-byte aligned stride (row mode)
  dataWidth_ = offset; // raw width sum, unpadded (col mode)
  rowLayoutReady_ = true;
  return pinnedPackState_ >= 0;
}

// Load/validate the children of every selected input into positional
// (batch, channel) slots: srcs = raw value pointers, keepAlive pins the
// loaded vectors, rawNulls collects nullable FLAT payload channels (row
// mode). Returns false when a batch cannot be packed (per-batch fallback);
// boundary mode hard-fails instead where a fallback would corrupt results.
// Split out of tryPinnedPack in the 2026-08-24 reconstruction; logic
// unchanged.
bool CudfFromVelox::loadChildren(
    const std::vector<RowVectorPtr>& selectedInputs,
    const RowTypePtr& inRowType,
    bool rowMode,
    bool boundary,
    PackBatch& pb) {
  const int numCols = static_cast<int>(inRowType->size());
  auto& keepAlive = pb.keepAlive;
  auto& srcs = pb.srcs;
  auto& rawNullsPtrs = pb.rawNullsPtrs;
  bool& anyNulls = pb.anyNulls;
  // ---- Load children; require flat, null-free, materialized ----
  // Loading a lazy child materializes the scan column -- work the merge path
  // did anyway. keepAlive pins loaded vectors for the duration of the pack.
  // POSITIONALLY INDEXED [b * numCols + c]: the boundary host-retention
  // block below rebuilds each batch from these slots, so every (b, c) must
  // land at its own index (a dense push_back breaks the moment any channel
  // is skipped).

  // Null sidecar (2026-08-17): nullable FLAT children are packable in row
  // mode -- their validity rides in a per-row sidecar appended after the row
  // region. Boundary (keys-only) and col-major modes keep the old refusal.

  const bool allowNulls = rowMode && !boundary;
  // Resolve join-key channels once (names came from the adjacent join op).
  // Under boundary the whole CROSSING set behaves like keys (later joins
  // hash on them; nulls unrepresentable, flat required).
  if (!keyChannelsResolved_ && rowMode) {
    const auto& names = boundary ? crossingNames_ : boundaryKeyNames_;
    for (const auto& name : names) {
      if (name == kRowIdName) {
        continue;
      }
      auto idx = inRowType->getChildIdxIfExists(name);
      if (idx.has_value()) {
        keyChannels_.push_back(static_cast<int32_t>(idx.value()));
      }
    }
    keyChannelsResolved_ = true;
  }
  auto isKeyChannel = [&](int c) {
    return std::find(keyChannels_.begin(), keyChannels_.end(), c) !=
        keyChannels_.end();
  };
  for (size_t b = 0; b < selectedInputs.size(); b++) {
    for (int c = 0; c < numCols; c++) {
      // Boundary mode reads srcs[] only at the key channels; payload
      // channels are retained host-side as whole batches, and
      // BoundaryHostStore/HybridContainer::addPayload handles their
      // loading, flattening, and nulls natively. No constraints here.
      if (boundary && !isKeyChannel(c)) {
        // Payload channel: LOAD it (host-side extraction needs loaded
        // children) but impose no flat/null constraints — BoundaryHostStore/
        // HybridContainer::addPayload decodes and flattens natively.
        keepAlive[b * numCols + c] =
            BaseVector::loadedVectorShared(selectedInputs[b]->childAt(c));
        continue;
      }
      auto child =
          BaseVector::loadedVectorShared(selectedInputs[b]->childAt(c));
      // String channels (2026-08-21): Parquet strings often arrive
      // dictionary-encoded; the pack reads FlatVector<StringView>, so
      // flatten here (a copy of the StringViews, not of the bytes).
      // Subset-pack channels (boundary crossing / pruned) must also be
      // flat: filter pushdown yields DICTIONARY columns.
      if (child != nullptr && rowMode &&
          child->encoding() != VectorEncoding::Simple::FLAT &&
          (boundary || prunedPack_ ||
           inRowType->childAt(c)->kind() == TypeKind::VARCHAR ||
           inRowType->childAt(c)->kind() == TypeKind::VARBINARY)) {
        BaseVector::flattenVector(child);
        child = BaseVector::loadedVectorShared(child);
      }
      // Null JOIN KEYS are unrepresentable in the row matcher (it hashes
      // raw key bytes; NULL must never match). Fail loudly rather than
      // silently joining on residual bytes. Applies to both boundary
      // (keys-only) and full-row packs.
      if (child != nullptr && isKeyChannel(c) && child->mayHaveNulls() &&
          child->rawNulls() != nullptr) {
        VELOX_CHECK_EQ(
            BaseVector::countNulls(child->nulls(), child->size()),
            0,
            "row pack: join key column '{}' (channel {}) contains NULLs; "
            "null join keys are not supported by the row-wise join path",
            inRowType->nameOf(c),
            c);
      }
      // Key channel that passed the null guard (zero actual nulls): accept
      // directly even if the nulls buffer exists, so a mayHaveNulls-flagged
      // but clean key does not force a per-batch fallback.
      if (child != nullptr && isKeyChannel(c) &&
          child->encoding() == VectorEncoding::Simple::FLAT &&
          child->valuesAsVoid() != nullptr) {
        srcs[b * numCols + c] =
            static_cast<const uint8_t*>(child->valuesAsVoid());
        keepAlive[b * numCols + c] = std::move(child);
        continue;
      }
      if (child != nullptr && allowNulls && !isKeyChannel(c) &&
          child->mayHaveNulls() &&
          child->encoding() == VectorEncoding::Simple::FLAT &&
          child->valuesAsVoid() != nullptr) {
        rawNullsPtrs[b * numCols + c] = child->rawNulls();
        anyNulls = anyNulls || child->rawNulls() != nullptr;
        srcs[b * numCols + c] =
            static_cast<const uint8_t*>(child->valuesAsVoid());
        keepAlive[b * numCols + c] = std::move(child);
        continue;
      }
      if (child == nullptr ||
          child->encoding() != VectorEncoding::Simple::FLAT ||
          child->mayHaveNulls() || child->valuesAsVoid() == nullptr) {
        // Boundary mode cannot fall back per-batch: the downstream probe/
        // build requires EVERY batch in keys-only layout with host payload
        // attached; a silent full-width batch would corrupt results. (Only
        // KEY channels reach this check under boundary.)
        VELOX_CHECK(
            !boundary,
            "boundary-hybrid: join key column {} is non-flat or nullable; "
            "the keys-only pinned pack cannot handle it",
            c);
        return false; // per-batch fallback; schema itself may be fine
      }
      srcs[b * numCols + c] =
          static_cast<const uint8_t*>(child->valuesAsVoid());
      keepAlive[b * numCols + c] = std::move(child);
    }
  }

  return true;
}

// The CPU pack itself: one pass over (batch, channel) into the pinned
// slot -- row-major (with null sidecar + string heap tail) or col-major.
// Split out of tryPinnedPack in the 2026-08-24 reconstruction; logic
// unchanged.
void CudfFromVelox::packIntoSlot(
    const std::vector<RowVectorPtr>& selectedInputs,
    vector_size_t totalRows,
    bool rowMode,
    bool boundary,
    const PackBatch& pb,
    uint8_t* const base,
    size_t inputBegin,
    size_t inputEnd) {
  inputEnd = std::min(inputEnd, selectedInputs.size());
  const int numCols = pb.numCols;
  const auto& srcs = pb.srcs;
  const auto& rawNullsPtrs = pb.rawNullsPtrs;
  const bool anyNulls = pb.anyNulls;
  const int32_t nullStride = pb.nullStride;
  const int64_t heapOffset = pb.heapOffset;
  // ---- Pack: ONE CPU pass, fused with the (former) merge ----
  if (rowMode) {
    const int32_t rowWidth = rowWidth_;
    // Boundary mode packs only the key channels (rowFields_ was computed
    // over boundaryPackChannels_ / subsetPackNames_); normal mode packs
    // every channel, where
    // rowFields_.size() == numCols and packed index == channel index.
    const int numPacked = static_cast<int>(rowFields_.size());
    uint8_t* const heapBase = base + heapOffset;
    int64_t heapCursor = 0;
    int64_t rowsSoFar = 0;
    // Cache tiling (2026-08-29): the pack is column-at-a-time, so every
    // field write lands in a different row's cache line and the line is
    // only complete after all columns have visited it. With wide rows an
    // input batch (e.g. 8K x 1.3 KB = 11 MB) no longer stays cache-resident
    // across the column passes -- and 16 drivers' tiles evict each other
    // from L3 -- so each field write becomes a DRAM read-modify-write.
    // Packing in row tiles of T rows (T x rowWidth ~ L2) keeps the tile's
    // lines resident; T = 0 keeps one tile per input batch.
    const int64_t tileRows =
        CudfConfig::getInstance().benchmarkPackTileRows > 0
        ? CudfConfig::getInstance().benchmarkPackTileRows
        : std::numeric_limits<int64_t>::max();
    for (size_t b = inputBegin; b < inputEnd; b++) {
      const int64_t nb = selectedInputs[b]->size();
      for (int64_t t0 = 0; t0 < nb; t0 += tileRows) {
      const int64_t n = std::min<int64_t>(tileRows, nb - t0);
      uint8_t* const tile = base + (rowsSoFar + t0) * rowWidth;
      for (int k = 0; k < numPacked; k++) {
        // Subset packs (boundary crossing set / pruned): field k comes
        // from channel boundaryPackChannels_[k]; the hidden __rowid field
        // (boundary only) is filled below, not from a channel.
        if (boundary && k == rowIdField_) {
          const int32_t off = rowFields_[k].offset;
          uint8_t* d = tile + off;
          for (int64_t r = 0; r < n; r++, d += rowWidth) {
            const int64_t rowid = pb.rowIdBias + rowsSoFar + t0 + r;
            std::memcpy(d, &rowid, 8);
          }
          continue;
        }
        const int c = (boundary || prunedPack_) ? boundaryPackChannels_[k] : k;
        const int32_t off = rowFields_[k].offset;
        const int32_t w = rowFields_[k].byte_width;
        const uint8_t* src = srcs[b * numCols + c];
        if (rowFields_[k].kind == kFieldString) {
          // Inline StringViews (<= 12B) are byte-identical to the slot.
          // Longer ones: len + prefix stay, the pointer becomes a heap
          // offset and the bytes are appended to the heap region.
          const auto* sv = reinterpret_cast<const StringView*>(src) + t0;
          const uint64_t* nulls = rawNullsPtrs[b * numCols + c];
          uint8_t* d = tile + off;
          for (int64_t r = 0; r < n; r++, d += rowWidth) {
            if (nulls != nullptr && velox::bits::isBitNull(nulls, t0 + r)) {
              std::memset(d, 0, kRowStrSlotBytes);
              continue;
            }
            const auto& v = sv[r];
            if (v.size() <= kRowStrInlineMax) {
              std::memcpy(d, &v, kRowStrSlotBytes);
            } else {
              const uint32_t len = v.size();
              std::memcpy(d, &len, 4);
              std::memcpy(d + 4, v.data(), 4);
              const uint64_t hoff =
                  static_cast<uint64_t>(pb.heapBias + heapCursor);
              std::memcpy(d + 8, &hoff, 8);
              std::memcpy(heapBase + heapCursor, v.data(), len);
              heapCursor += len;
            }
          }
          continue;
        }
        if (w == 8) {
          const uint64_t* s = reinterpret_cast<const uint64_t*>(src) + t0;
          uint8_t* d = tile + off;
          for (int64_t r = 0; r < n; r++, d += rowWidth) {
            *reinterpret_cast<uint64_t*>(d) = s[r];
          }
        } else if (w == 4) {
          const uint32_t* s = reinterpret_cast<const uint32_t*>(src) + t0;
          uint8_t* d = tile + off;
          for (int64_t r = 0; r < n; r++, d += rowWidth) {
            *reinterpret_cast<uint32_t*>(d) = s[r];
          }
        } else {
          const uint8_t* s = src + t0 * w;
          uint8_t* d = tile + off;
          for (int64_t r = 0; r < n; r++, d += rowWidth) {
            std::memcpy(d, s + r * w, w);
          }
        }
      }
      }
      rowsSoFar += nb;
    }
    // ---- Null sidecar fill (2026-08-17): bit SET = NULL ----
    if (anyNulls) {
      uint8_t* const nullBase =
          base + static_cast<int64_t>(totalRows) * rowWidth;
      std::memset(nullBase, 0, static_cast<size_t>(totalRows) * nullStride);
      int64_t rowsBefore = 0;
      for (size_t b = inputBegin; b < inputEnd; b++) {
        const int64_t n = selectedInputs[b]->size();
        // Bit index = PACKED FIELD index (consumers map fields, not
        // channels); for identity packs field == channel.
        for (int k = 0; k < numPacked; k++) {
          const int c = prunedPack_ ? boundaryPackChannels_[k] : k;
          const uint64_t* nulls = rawNullsPtrs[b * numCols + c];
          if (nulls == nullptr) {
            continue;
          }
          const uint8_t byteMask = static_cast<uint8_t>(1u << (k & 7));
          const int32_t byteIdx = k >> 3;
          for (int64_t r = 0; r < n; r++) {
            if (velox::bits::isBitNull(nulls, r)) {
              nullBase[(rowsBefore + r) * nullStride + byteIdx] |= byteMask;
            }
          }
        }
        rowsBefore += n;
      }
    }
  } else {
    // Col-major: per column region, sequential memcpy per input chunk --
    // this IS the merge, done straight into pinned memory.
    std::vector<int64_t> colOff(numCols);
    int64_t acc = 0;
    for (int c = 0; c < numCols; c++) {
      colOff[c] = acc;
      acc += static_cast<int64_t>(rowFields_[c].byte_width) * totalRows;
    }
    int64_t rowsSoFar = 0;
    for (size_t b = 0; b < selectedInputs.size(); b++) {
      const int64_t n = selectedInputs[b]->size();
      for (int c = 0; c < numCols; c++) {
        const int32_t w = rowFields_[c].byte_width;
        std::memcpy(
            base + colOff[c] + rowsSoFar * w,
            srcs[b * numCols + c],
            static_cast<size_t>(n) * w);
      }
      rowsSoFar += n;
    }
  }

}

RowVectorPtr CudfFromVelox::tryPinnedPack(
    const std::vector<RowVectorPtr>& selectedInputs,
    vector_size_t totalRows) {
  if (!CudfConfig::getInstance().benchmarkCpuColToRow || totalRows == 0 ||
      pinnedPackState_ < 0 || selectedInputs.empty()) {
    return nullptr;
  }
  const bool rowWiseMode = CudfConfig::getInstance().benchmarkRowWiseGather;

  resolveRowPathOnce(rowWiseMode);
  // Nothing to defer => nothing to gain. Decided ONCE from the pack's
  // ACTUAL input type (outputType_ can be the join's type): if every input
  // column the chain needs is already in the crossing set (keys + later
  // keys), boundary mode would only add the host-exit path. Resolve EAGER
  // (measured 2026-08-24: keys-only probes lost 0.6-0.9x under deferral).
  if (deferralEligible_ && !nothingToDeferChecked_) {
    nothingToDeferChecked_ = true;
    if (auto inType = std::dynamic_pointer_cast<const RowType>(
            selectedInputs[0]->type())) {
      bool anyDeferred = false;
      for (const auto& n : inType->names()) {
        const bool crossing =
            std::find(
                boundaryKeyNames_.begin(), boundaryKeyNames_.end(), n) !=
                boundaryKeyNames_.end() ||
            std::find(laterKeyNames_.begin(), laterKeyNames_.end(), n) !=
                laterKeyNames_.end();
        const bool needed = joinOutputNames_.empty() ||
            std::find(joinOutputNames_.begin(), joinOutputNames_.end(), n) !=
                joinOutputNames_.end();
        if (!crossing && needed) {
          anyDeferred = true;
          break;
        }
      }
      if (!anyDeferred) {
        deferralEligible_ = false;
        boundaryMode_ = 0;
        addRuntimeStat("fromVeloxNothingToDefer", RuntimeCounter(1));
      }
    }
  }
  const bool rowMode = rowWiseMode && emitRowStore_ == 1;
  if (rowWiseMode && !rowMode) {
    // Row mode requested but downstream only takes CudfVector: leave the
    // standard path (from_arrow + GPU transpose downstream) intact.
    return nullptr;
  }
  // Batch-level adaptive deferral (2026-08-24): the eager/deferred choice is
  // re-checked per BATCH against the chain endpoint's observed survival.
  // One-way: once deferred, never back. The layout caches are reset so the
  // next computeLayoutOnce builds the crossing-set layout; the downstream
  // probe re-initializes when it sees the new field-name signature.
  if (deferralEligible_ && boundaryMode_ == 0 &&
      CudfConfig::getInstance().benchmarkDeferralAdaptive &&
      DeferralStats::instance().shouldDefer(
          endpointJoinId_,
          CudfConfig::getInstance().benchmarkDeferralThreshold)) {
    boundaryMode_ = 1;
    rowLayoutReady_ = false;
    // The crossing set behaves as keys under boundary: re-resolve, or the
    // later-join-key channels stay classified as payload and never load.
    keyChannelsResolved_ = false;
    keyChannels_.clear();
    prunedPack_ = false;
    hasStringFields_ = false;
    rowIdField_ = -1;
    subsetPackNames_.clear();
    crossingNames_.clear();
    boundaryPackChannels_.clear();
    rowFields_.clear();
    colDtypes_.clear();
    addRuntimeStat("fromVeloxDeferralSwitch", RuntimeCounter(1));
  }
  // ---- Selective hybrid execution (2026-08-30) ----
  // Deferral only pays when the columns it keeps on the host are wider than
  // the 8-byte rowRef it ships in their place: with a key-only projection the
  // "payload" is a couple of narrow columns, and gathering them for every
  // surviving tuple costs more than the crossing it saves. Plan-time test on
  // the deferred width -- the GPU counterpart of the CPU gate, not an
  // adaptive decision.
  if (boundaryMode_ == 1 && rowMode && !payloadGateChecked_) {
    payloadGateChecked_ = true;
    const auto minBytes =
        CudfConfig::getInstance().benchmarkDeferMinPayloadBytes;
    if (minBytes > 0) {
      auto t =
          std::dynamic_pointer_cast<const RowType>(selectedInputs[0]->type());
      const int32_t bytes = t != nullptr ? deferredPayloadBytes(t) : 0;
      addRuntimeStat("fromVeloxDeferredPayloadBytes", RuntimeCounter(bytes));
      if (bytes < minBytes) {
        boundaryMode_ = 0;
        deferralEligible_ = false;
        addRuntimeStat("fromVeloxPayloadGateEager", RuntimeCounter(1));
      }
    }
  }
  // Boundary-hybrid is only meaningful on the row path straight into a
  // RowHashJoinBuild/Probe (resolved above).
  const bool boundary = boundaryMode_ == 1 && rowMode;
  // Per-batch record of the (possibly adaptive) decision: sum = number of
  // batches packed in boundary mode, count = batches. fromVeloxBoundaryMode
  // is the resolution-time value only (0 for adaptive before the switch).
  addRuntimeStat(
      "fromVeloxDeferredBatches", RuntimeCounter(boundary ? 1 : 0));
  // Spine denominator for the endpoint's survival ratio: every row this
  // pack ships (eager or deferred) enters the chain.
  if (deferralEligible_ && rowMode) {
    DeferralStats::instance().recordPacked(endpointJoinId_, totalRows);
  }

  // NOTE: layout derives from the INPUT's actual row type, not outputType_
  // -- the two can disagree (e.g. the build-side scan batch carries fewer
  // children than the operator's declared output type).
  auto inRowType =
      std::dynamic_pointer_cast<const RowType>(selectedInputs[0]->type());
  if (inRowType == nullptr) {
    return nullptr;
  }
  const int numCols = static_cast<int>(inRowType->size());
  if (!computeLayoutOnce(inRowType, rowMode, boundary)) {
    return nullptr;
  }
  auto tpPackStart = std::chrono::steady_clock::now();

  PackBatch pb;
  pb.numCols = numCols;
  pb.keepAlive.resize(selectedInputs.size() * numCols);
  pb.srcs.resize(selectedInputs.size() * numCols);
  pb.rawNullsPtrs.assign(selectedInputs.size() * numCols, nullptr);
  if (!loadChildren(selectedInputs, inRowType, rowMode, boundary, pb)) {
    return nullptr;
  }
  const bool anyNulls = pb.anyNulls;
  // ---- Stream ----
  auto stream = cudfGlobalStreamPool().get_stream();
  if (rowMode) {
    // All RowStoreVectors from this operator MUST share one stream (see
    // rowStream_ docs in the header). DEDICATED, not pooled: see
    // DedicatedStream.h for the stall this avoids.
    if (!rowStream_.has_value()) {
      rowStream_ = acquireDedicatedStream();
    }
    stream = rowStream_.value();
  }

  // ---- Acquire a pinned slot (ping-ponged; event-guarded reuse) ----
  const int32_t nullStride = (numCols + 7) / 8;
  const int64_t nullRegionBytes =
      (rowMode && anyNulls) ? static_cast<int64_t>(totalRows) * nullStride : 0;
  // Out-of-line string heap (2026-08-21): rides after the sidecar so one
  // H2D carries rows + nulls + chars. Sized by a pre-pass over the
  // StringViews (lengths only; no byte traffic).
  int64_t heapBytes = 0;
  // Per-input heap bytes: chunked staging groups whole input batches, so it
  // needs each batch's heap contribution to size a group.
  std::vector<int64_t> inputHeapBytes(selectedInputs.size(), 0);
  if (rowMode && !boundary && hasStringFields_) {
    const int numPackedFields = static_cast<int>(rowFields_.size());
    for (size_t b = 0; b < selectedInputs.size(); b++) {
      const int64_t n = selectedInputs[b]->size();
      const int64_t heapBefore = heapBytes;
      for (int k = 0; k < numPackedFields; k++) {
        if (rowFields_[k].kind != kFieldString) {
          continue;
        }
        const int c = prunedPack_ ? boundaryPackChannels_[k] : k;
        const auto* sv =
            reinterpret_cast<const StringView*>(pb.srcs[b * numCols + c]);
        const uint64_t* nulls = pb.rawNullsPtrs[b * numCols + c];
        for (int64_t r = 0; r < n; r++) {
          if (nulls != nullptr && velox::bits::isBitNull(nulls, r)) {
            continue;
          }
          if (sv[r].size() > kRowStrInlineMax) {
            heapBytes += sv[r].size();
          }
        }
      }
      inputHeapBytes[b] = heapBytes - heapBefore;
    }
  }
  const int64_t heapOffset = rowMode
      ? static_cast<int64_t>(totalRows) * rowWidth_ + nullRegionBytes
      : 0;
  const int64_t totalBytes = (rowMode
      ? static_cast<int64_t>(totalRows) * rowWidth_
      : dataWidth_ * static_cast<int64_t>(totalRows)) + nullRegionBytes +
      heapBytes;
  // Chunked staging (2026-08-30): a slot sized to the whole GPU batch has to
  // be page-locked before the first batch can ship, and cudaHostAlloc takes a
  // global driver lock -- with 16 drivers and a 5M-row batch that first-touch
  // cost (seconds) dwarfs the copy it serves. Staging through a bounded block
  // caps it: the pinned buffer is allocated once at block size and reused for
  // every group, while the DEVICE batch stays exactly as before (one
  // contiguous row store, same downstream kernels) because each group copies
  // into its own offsets of the single device buffer.
  const int64_t blockBytes =
      rowMode ? CudfConfig::getInstance().benchmarkPackBlockBytes : 0;
  const bool chunked = blockBytes > 0 && totalBytes > blockBytes;
  if (pinnedSlots_[0] == nullptr) {
    // totalBytes of the first batch is the size hint: probe-side operators
    // draw big recycled slots, build-side operators draw small ones.
    const int64_t hint = chunked ? blockBytes : totalBytes;
    pinnedSlots_[0] = PinnedPackSlotPool::instance().acquire(hint);
    pinnedSlots_[1] = PinnedPackSlotPool::instance().acquire(hint);
  }
  const bool syncMode = pinnedPackSyncMode_ == 1;
  // Sync mode relies on the batch-level stream.synchronize() at ship time to
  // guarantee a slot is free before it is repacked -- which only holds when a
  // batch uses each slot once. Chunked staging repacks a slot every second
  // GROUP, so it needs the per-slot event guard even in sync mode.
  const bool eventGuard = !syncMode || chunked;
  // Ping-pong: hand out the next slot, waiting only if its previous copy is
  // still in flight (with two slots that copy is a whole group behind).
  auto nextSlot = [&]() -> PinnedPackSlot& {
    auto& s = *pinnedSlots_[pinnedSlot_];
    pinnedSlot_ ^= 1;
    if (!eventGuard) {
      // Sync discipline: the per-batch stream.synchronize() at ship time
      // guarantees the previous copy out of this slot completed before we
      // returned it downstream — no event guard needed to repack.
    } else if (s.done == nullptr) {
      VELOX_CUDA_CHECK(
          cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming));
    } else {
      // A never-recorded event also synchronizes instantly.
      VELOX_CUDA_CHECK(cudaEventSynchronize(s.done));
    }
    return s;
  };

  pb.nullStride = nullStride;
  pb.heapOffset = heapOffset;
  std::unique_ptr<rmm::device_buffer> stagedBuf;
  PinnedPackSlot* lastSlot = nullptr;
  if (chunked) {
    // The device batch is allocated up front; groups copy into their slice
    // of it. Rows, null sidecar and string heap live in three disjoint
    // device regions, so a group issues (up to) three copies.
    stagedBuf = std::make_unique<rmm::device_buffer>(totalBytes, stream);
    auto* dev = static_cast<uint8_t*>(stagedBuf->data());
    const int64_t rowRegion = static_cast<int64_t>(totalRows) * rowWidth_;
    size_t gBegin = 0;
    int64_t rowsDone = 0, heapDone = 0, groups = 0;
    while (gBegin < selectedInputs.size()) {
      // Grow the group while it fits the block; always take at least one
      // input batch (a single batch larger than the block grows the slot).
      size_t gEnd = gBegin;
      int64_t gRows = 0, gHeap = 0, gBytes = 0;
      while (gEnd < selectedInputs.size()) {
        const int64_t r = selectedInputs[gEnd]->size();
        const int64_t h = inputHeapBytes[gEnd];
        const int64_t want = (gRows + r) * (rowWidth_ + (anyNulls ? nullStride : 0)) + gHeap + h;
        if (gEnd > gBegin && want > blockBytes) {
          break;
        }
        gRows += r;
        gHeap += h;
        gBytes = want;
        ++gEnd;
      }
      auto& gs = nextSlot();
      ensurePinnedSlotCapacity(gs, gBytes);
      const int64_t gNullBytes = anyNulls ? gRows * nullStride : 0;
      pb.heapOffset = gRows * rowWidth_ + gNullBytes;
      pb.rowIdBias = rowsDone;
      pb.heapBias = heapDone;
      packIntoSlot(
          selectedInputs,
          static_cast<vector_size_t>(gRows),
          rowMode,
          boundary,
          pb,
          gs.host,
          gBegin,
          gEnd);
      VELOX_CUDA_CHECK(cudaMemcpyAsync(
          dev + rowsDone * rowWidth_,
          gs.host,
          gRows * rowWidth_,
          cudaMemcpyHostToDevice,
          stream.value()));
      if (gNullBytes > 0) {
        VELOX_CUDA_CHECK(cudaMemcpyAsync(
            dev + rowRegion + rowsDone * nullStride,
            gs.host + gRows * rowWidth_,
            gNullBytes,
            cudaMemcpyHostToDevice,
            stream.value()));
      }
      if (gHeap > 0) {
        VELOX_CUDA_CHECK(cudaMemcpyAsync(
            dev + heapOffset + heapDone,
            gs.host + pb.heapOffset,
            gHeap,
            cudaMemcpyHostToDevice,
            stream.value()));
      }
      if (eventGuard) {
        VELOX_CUDA_CHECK(cudaEventRecord(gs.done, stream.value()));
      }
      lastSlot = &gs;
      rowsDone += gRows;
      heapDone += gHeap;
      ++groups;
      gBegin = gEnd;
    }
    VELOX_CHECK_EQ(rowsDone, totalRows);
    if (!packBlockStatsLogged_) {
      packBlockStatsLogged_ = true;
      addRuntimeStat("fromVeloxPackGroups", RuntimeCounter(groups));
    }
    pb.heapOffset = heapOffset;
    pb.rowIdBias = 0;
    pb.heapBias = 0;
  } else {
    auto& s = nextSlot();
    ensurePinnedSlotCapacity(s, totalBytes);
    packIntoSlot(selectedInputs, totalRows, rowMode, boundary, pb, s.host);
    lastSlot = &s;
  }
  auto& slot = *lastSlot;
  uint8_t* const base = slot.host;
  auto tpPacked = std::chrono::steady_clock::now();

  // ---- Ship: pinned-source cudaMemcpyAsync = microsecond enqueue ----
  RowVectorPtr result;
  auto* pool = selectedInputs[0]->pool();
  if (rowMode) {
    if (rowFieldsDevice_ == nullptr) {
      rowFieldsDevice_ = std::make_shared<rmm::device_buffer>(
          rowFields_.data(), rowFields_.size() * sizeof(FieldDesc), stream);
      stream.synchronize(); // one-time, first batch only
    }
    // Chunked staging already filled (and event-guarded) the device buffer.
    rmm::device_buffer gpuRowBuf = stagedBuf != nullptr
        ? std::move(*stagedBuf)
        : rmm::device_buffer(totalBytes, stream);
    if (stagedBuf == nullptr) {
      VELOX_CUDA_CHECK(cudaMemcpyAsync(
          gpuRowBuf.data(),
          slot.host,
          totalBytes,
          cudaMemcpyHostToDevice,
          stream.value()));
      if (!syncMode) {
        VELOX_CUDA_CHECK(cudaEventRecord(slot.done, stream.value()));
      }
    }
    auto fieldsHostCopy = rowFields_;
    // All batches share the one uploaded FieldDesc buffer (no per-batch
    // device alloc + D2D copy); the shared_ptr keeps it alive downstream.
    auto rowStoreResult = std::make_shared<RowStoreVector>(
        pool,
        outputType_,
        totalRows,
        std::move(gpuRowBuf),
        rowFieldsDevice_,
        std::move(fieldsHostCopy),
        rowWidth_,
        stream);
    if (anyNulls) {
      // The single H2D above carried rows + sidecar (totalBytes includes
      // the null region); record the stride so consumers can find it.
      rowStoreResult->setNullSidecar(nullStride);
    }
    if (heapBytes > 0) {
      // The single H2D above carried rows + sidecar + chars; record where
      // the heap tail starts so consumers can address it.
      rowStoreResult->setCharsInTail(heapOffset, heapBytes);
    }
    if (boundary) {
      // ---- Spine deferral v2: retain THIS emitted batch's host rows as a
      // per-batch BoundaryHostStore; the hidden __rowid field indexes it.
      // Payload never crosses; materialization gathers survivors later.
      if (boundaryStoreType_ == nullptr) {
        boundaryStoreType_ = inRowType;
      }
      auto store = std::make_shared<BoundaryHostStore>(
          std::dynamic_pointer_cast<const RowType>(boundaryStoreType_),
          selectedInputs[0]->pool());
      for (size_t b = 0; b < selectedInputs.size(); b++) {
        std::vector<VectorPtr> children(numCols);
        for (int c = 0; c < numCols; c++) {
          children[c] = pb.keepAlive[b * numCols + c];
        }
        store->addBatch(std::make_shared<RowVector>(
            selectedInputs[b]->pool(),
            inRowType,
            nullptr,
            selectedInputs[b]->size(),
            std::move(children)));
      }
      rowStoreResult->setProvenance(std::move(store), rowIdField_);
      rowStoreResult->setFieldNames(crossingNames_);
    }
    if (prunedPack_) {
      rowStoreResult->setFieldNames(subsetPackNames_);
    }
    result = std::move(rowStoreResult);
  } else {
    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.reserve(numCols);
    int64_t off = 0;
    for (int c = 0; c < numCols; c++) {
      const int64_t bytes =
          static_cast<int64_t>(rowFields_[c].byte_width) * totalRows;
      rmm::device_buffer dbuf(bytes, stream);
      VELOX_CUDA_CHECK(cudaMemcpyAsync(
          dbuf.data(),
          base + off,
          bytes,
          cudaMemcpyHostToDevice,
          stream.value()));
      off += bytes;
      cols.push_back(std::make_unique<cudf::column>(
          colDtypes_[c],
          static_cast<cudf::size_type>(totalRows),
          std::move(dbuf),
          rmm::device_buffer{},
          0));
    }
    if (!syncMode) {
      VELOX_CUDA_CHECK(cudaEventRecord(slot.done, stream.value()));
    }
    auto tbl = std::make_unique<cudf::table>(std::move(cols));
    result = std::make_shared<CudfVector>(
        pool, outputType_, totalRows, std::move(tbl), stream);
    // No synchronize: downstream cudf operators consume this vector's stream
    // (CudfBatchConcat joins input streams), so ordering is preserved; the
    // pinned slot outlives the in-flight copy via slot.done.
  }

  // Sync discipline (kPinnedPackSync resolved to sync — see header docs):
  // wait for this batch's DMA before returning. For probe consumers this is
  // free (the join's match-count readback would wait anyway — verified E2E-
  // identical, job 13187756) and it keeps per-operator attribution clean:
  // the full boundary cost lands here, in the conversion.
  if (syncMode) {
    stream.synchronize();
  }

  auto tpDone = std::chrono::steady_clock::now();
  auto nanos = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
  };
  addRuntimeStat(
      "toCudfPackNanos",
      RuntimeCounter(
          nanos(tpPackStart, tpPacked), RuntimeCounter::Unit::kNanos));
  addRuntimeStat(
      "toCudfDmaNanos",
      RuntimeCounter(nanos(tpPacked, tpDone), RuntimeCounter::Unit::kNanos));
  addRuntimeStat(
      "toCudfBytes",
      RuntimeCounter(totalBytes, RuntimeCounter::Unit::kBytes));
  addRuntimeStat(
      "fromVeloxToCudfNanos",
      RuntimeCounter(
          nanos(tpPackStart, tpDone), RuntimeCounter::Unit::kNanos));
  addRuntimeStat(
      "fromVeloxOutputRows",
      RuntimeCounter(static_cast<int64_t>(totalRows)));
  addRuntimeStat(
      "fromVeloxBatchesMerged",
      RuntimeCounter(static_cast<int64_t>(selectedInputs.size())));
  return result;
}

void CudfFromVelox::doClose() {
  // TODO(kn): Remove default stream after redesign of CudfFromVelox
  cudf::get_default_stream(cudf::allow_default_stream).synchronize();

  // Return the pinned slots to the process-wide pool with memory and events
  // INTACT. Freeing here (the previous design) made every query pay
  // cudaFreeHost/cudaHostAlloc cycles on large slots — operations that
  // globally serialize the CUDA driver and convoy all other threads.
  // The pool bounds total pinned memory at the high-water mark of
  // concurrently live conversion operators, so nothing leaks over time.
  for (auto*& slot : pinnedSlots_) {
    if (slot != nullptr) {
      if (slot->done != nullptr) {
        // The last batch's copy may still be reading slot->host.
        cudaEventSynchronize(slot->done);
      }
      PinnedPackSlotPool::instance().release(slot);
      slot = nullptr;
    }
  }

  Operator::close();
  inputs_.clear();
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfToVelox",
          nvtx3::rgb{148, 0, 211}, // Purple
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt) {}

bool CudfToVelox::isPassthroughMode() const {
  return operatorCtx_->driverCtx()->queryConfig().get<bool>(
      kPassthroughMode, true);
}

void CudfToVelox::doAddInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    if (cudfInput) {
      inputs_.push_back(std::move(cudfInput));
    } else {
      // Non-CudfVector (e.g. skip_output dummy RowVector) — pass through
      passthroughInputs_.push_back(std::move(input));
    }
  }
}

std::optional<uint64_t> CudfToVelox::averageRowSize() {
  if (!averageRowSize_) {
    averageRowSize_ =
        inputs_.front()->estimateFlatSize() / inputs_.front()->size();
  }
  return averageRowSize_;
}

// Pop inputs_.front(), convert its GPU table to a Velox RowVector via a
// single to_arrow_host + synchronize, and return it.  The caller is
// responsible for any further slicing.
RowVectorPtr CudfToVelox::convertFrontToVelox() {
  auto cudfVector = std::move(inputs_.front());
  inputs_.pop_front();
  auto stream = cudfVector->stream();
  auto tableView = cudfVector->getTableView();
  auto output = with_arrow::toVeloxColumn(
      tableView, pool(), outputType_, "", stream, get_temp_mr());
  stream.synchronize();
  reportD2HWall();
  output->setType(outputType_);
  return output;
}

void CudfToVelox::reportD2HWall() {
  auto& nanos = with_arrow::toVeloxD2HNanos();
  if (nanos > 0) {
    addRuntimeStat(
        "toVeloxD2HNanos",
        RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
    nanos = 0;
  }
}

// Output batching strategy
// ========================
// The key constraint is minimising D->H (device-to-host) transfers.
// Each call to toVeloxColumn / to_arrow_host triggers one D->H copy per
// column, so calling it once per output batch (rather than once per row
// or once per input batch) is critical for performance.
//
// Two cases arise depending on the size of the front GPU input relative
// to targetBatchSize:
//
//  (A) Front input >= targetBatchSize  (e.g. CudfOrderBy: one large sorted
//      table).  We convert the whole input to Velox in one shot and then
//      slice it purely on the CPU using BaseVector::slice().  Subsequent
//      getOutput() calls return successive CPU slices with no additional
//      D->H work until veloxBuffer_ is exhausted.
//
//  (B) Front input < targetBatchSize  (e.g. CudfFilterProject with high
//      selectivity: many small GPU batches).  We concatenate inputs on device
//      until we accumulate targetBatchSize rows, then convert the concat
//      result to Velox in one shot.  This preserves the GPU-side merge
//      that avoids emitting many undersized Velox batches downstream.
//
// In both cases exactly one toVeloxColumn + stream.synchronize() is issued
// per output batch, regardless of how many GPU inputs were consumed.
RowVectorPtr CudfToVelox::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  // Drain passthrough inputs first (skip_output CPU RowVectors; the row
  // path's native exits: HostRowVector / deferred columnar RowVector).
  // Emit them in Velox-sized output batches (outputBatchRows), exactly as
  // the cudf path slices its converted output -- parity across arms.
  if (!passthroughInputs_.empty()) {
    auto& front = passthroughInputs_.front();
    const auto total = front->size();
    const auto target = std::max<vector_size_t>(
        1, outputBatchRows(front->estimateFlatSize() / std::max<vector_size_t>(1, total)));
    RowVectorPtr result;
    if (passthroughCursor_ == 0 && total <= target) {
      result = std::move(front);
      passthroughInputs_.pop_front();
      passthroughCursor_ = 0;
    } else {
      const auto n = std::min<vector_size_t>(target, total - passthroughCursor_);
      // HostRowVector overrides slice() with a shared view; a columnar
      // RowVector slices its children.
      result = std::static_pointer_cast<RowVector>(
          front->slice(passthroughCursor_, n));
      passthroughCursor_ += n;
      if (passthroughCursor_ >= total) {
        passthroughInputs_.pop_front();
        passthroughCursor_ = 0;
      }
    }
    finished_ = noMoreInput_ && inputs_.empty() && passthroughInputs_.empty();
    return result;
  }

  if (outputType_->size() == 0) {
    // cuDF zero-column tables do not have a row count, so we sum the sizes
    // of all CudfVectors in the inputs_, to maintain the logical count.
    // This is necessary to ensure correct behavior for e.g. `count` operators.
    vector_size_t totalSize = 0;
    while (!inputs_.empty()) {
      totalSize += inputs_.front()->size();
      inputs_.pop_front();
    }
    finished_ = noMoreInput_ && inputs_.empty();
    if (totalSize == 0) {
      return nullptr;
    }
    return BaseVector::create<RowVector>(outputType_, totalSize, pool());
  }

  // Drain veloxBuffer_ (populated on a previous call) before consuming
  // more GPU inputs.
  if (!veloxBuffer_) {
    if (inputs_.empty()) {
      finished_ = noMoreInput_ && passthroughInputs_.empty();
      return nullptr;
    }

    // Passthrough mode: emit each GPU input as a single Velox batch with no
    // re-batching.  Used when the caller knows the batch size is already
    // correct (e.g. default pipeline without explicit batch-size overrides).
    if (isPassthroughMode()) {
      auto output = convertFrontToVelox();
      finished_ = noMoreInput_ && inputs_.empty();
      if (output->size() == 0) {
        return nullptr;
      }
      return output;
    }

    const auto targetBatchSize = outputBatchRows(averageRowSize());

    if (static_cast<vector_size_t>(inputs_.front()->size()) >=
        targetBatchSize) {
      // Case A: large input.  Convert once; subsequent calls slice CPU-side.
      veloxBuffer_ = convertFrontToVelox();
      veloxOffset_ = 0;
      averageRowSize_ = std::nullopt; // recompute from next input
    } else {
      // Case B: small inputs.  GPU-concat until we reach targetBatchSize,
      // then convert the merged table in one D->H transfer.
      auto stream = inputs_.front()->stream();
      std::vector<CudfVectorPtr> toConcat;
      vector_size_t accumulated = 0;
      while (!inputs_.empty() && accumulated < targetBatchSize) {
        accumulated += static_cast<vector_size_t>(inputs_.front()->size());
        toConcat.push_back(std::move(inputs_.front()));
        inputs_.pop_front();
      }
      VELOX_CHECK_LE(
          accumulated,
          std::numeric_limits<cudf::size_type>::max(),
          "Accumulated row count exceeds cudf int32 limit");
      auto concatTable = getConcatenatedTable(
          std::move(toConcat), outputType_, stream, get_temp_mr());
      auto tableView = concatTable->view();
      veloxBuffer_ = with_arrow::toVeloxColumn(
          tableView, pool(), outputType_, "", stream, get_temp_mr());
      stream.synchronize();
      reportD2HWall();
      veloxBuffer_->setType(outputType_);
      veloxOffset_ = 0;
      averageRowSize_ = std::nullopt;
    }
  }

  // Slice veloxBuffer_ on the CPU to produce the next output batch.
  const auto totalRows = static_cast<vector_size_t>(veloxBuffer_->size());
  if (veloxOffset_ >= totalRows) {
    veloxBuffer_.reset();
    finished_ = noMoreInput_ && inputs_.empty();
    return nullptr;
  }

  const auto targetBatchSize = outputBatchRows(
      veloxBuffer_->estimateFlatSize() /
      static_cast<uint64_t>(std::max<vector_size_t>(totalRows, 1)));
  const auto take = std::min(targetBatchSize, totalRows - veloxOffset_);

  auto slice = std::dynamic_pointer_cast<RowVector>(
      veloxBuffer_->slice(veloxOffset_, take));
  VELOX_CHECK_NOT_NULL(slice);
  veloxOffset_ += take;

  if (veloxOffset_ >= totalRows) {
    veloxBuffer_.reset();
    finished_ = noMoreInput_ && inputs_.empty();
  }

  return slice;
}

void CudfToVelox::doClose() {
  Operator::close();
  inputs_.clear();
  veloxBuffer_.reset();
}

} // namespace facebook::velox::cudf_velox
