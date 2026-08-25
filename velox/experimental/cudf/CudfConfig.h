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

#include <cudf/types.hpp>

#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::cudf_velox {

struct CudfConfig {
  /// Keys used by the initialize() method.
  static constexpr const char* kCudfEnabled{"cudf.enabled"};
  static constexpr const char* kCudfDebugEnabled{"cudf.debug_enabled"};
  static constexpr const char* kCudfMemoryResource{"cudf.memory_resource"};
  static constexpr const char* kCudfMemoryPercent{"cudf.memory_percent"};
  static constexpr const char* kCudfFunctionNamePrefix{
      "cudf.function_name_prefix"};
  static constexpr const char* kCudfAstExpressionEnabled{
      "cudf.ast_expression_enabled"};
  static constexpr const char* kCudfAstExpressionPriority{
      "cudf.ast_expression_priority"};
  static constexpr const char* kCudfJitExpressionEnabled{
      "cudf.jit_expression_enabled"};
  static constexpr const char* kCudfJitExpressionPriority{
      "cudf.jit_expression_priority"};
  static constexpr const char* kCudfOutputMr{"cudf.output_mr"};
  static constexpr const char* kCudfAllowCpuFallback{"cudf.allow_cpu_fallback"};
  static constexpr const char* kCudfLogFallback{"cudf.log_fallback"};
  static constexpr const char* kCudfBatchSizeMinThreshold{
      "cudf.batch_size_min_threshold"};
  static constexpr const char* kCudfBatchSizeMaxThreshold{
      "cudf.batch_size_max_threshold"};
  static constexpr const char* kCudfConcatOptimizationEnabled{
      "cudf.concat_optimization_enabled"};
  static constexpr const char* kCudfTimestampUnit{"cudf.timestamp_unit"};
  /// Query session configs for the cuDF Operators.
  static constexpr const char* kCudfTopNBatchSize{"cudf.topk_batch_size"};

  /// Singleton CudfConfig instance.
  /// Clients must set the configs below before invoking registerCudf().
  static CudfConfig& getInstance();

  /// Initialize from a map with the above keys.
  void initialize(std::unordered_map<std::string, std::string>&&);

  /// Enable cudf by default.
  /// Clients can disable here and enable it via the QueryConfig as well.
  bool enabled{true};

  /// Enable debug printing.
  bool debugEnabled{false};

  /// Allow fallback to CPU operators if GPU operator replacement fails.
  bool allowCpuFallback{true};

  /// Memory resource for cuDF.
  /// Possible values are (cuda, pool, async, arena, managed, managed_pool).
  std::string memoryResource{"async"};

  /// The initial percent of GPU memory to allocate for pool or arena memory
  /// resources.
  int32_t memoryPercent{50};

  /// Memory resource for output vectors. When set to a value different from
  /// memoryResource, a separate MR is created for output allocations.
  /// When empty, the main memoryResource is used.
  std::string outputMemoryResource;

  /// Register all the functions with the functionNamePrefix.
  std::string functionNamePrefix;

  /// Enable AST in expression evaluation.
  bool astExpressionEnabled{true};

  /// Enable JIT in expression evaluation.
  bool jitExpressionEnabled{true};

  /// Priority of AST expression. Expression with higher priority is chosen for
  /// a given root expression.
  /// Example:
  /// Priority of expression that uses individual cuDF functions is 50.
  /// If AST priority is 100 then for a velox expression node that is supported
  /// by both, AST will be chosen as replacement for cudf execution, if AST
  /// priority is 25 then standalone cudf function is chosen.
  int astExpressionPriority{100};

  /// Priority of JIT expression.
  int jitExpressionPriority{101};

  /// Whether to log a reason for falling back to Velox CPU execution.
  bool logFallback{true};

  /// Whether to insert CudfBatchConcat operators before supported Cudf
  /// operators.
  /// This can improve performance by reducing the number of cuda kernel
  /// launches on addInput of certain operators by collecting a minimum number
  /// of rows before concatenating and passing on to the next operator.
  /// This batch size is determined by batchSizeMinThreshold and
  /// batchSizeMaxThreshold
  bool concatOptimizationEnabled{false};

  /// Minimum rows to accumulate before GPU-side concatenation in
  /// `CudfBatchConcat` (default 100k).
  int32_t batchSizeMinThreshold{100000};

  /// Maximum rows allowed in a concatenated batch (user configurable).
  /// When not set, cuDF's own `size_type::max()` is used.
  std::optional<int32_t> batchSizeMaxThreshold;
  // Query config key for the TopN batch size in the cuDF TopN operator.
  int32_t topNBatchSize{5};

  /// Timestamp unit for cuDF timestamp types.
  /// Can be configured via kCudfTimestampUnit with string values:
  /// "s" (seconds), "ms" (milliseconds), "us" (microseconds), "ns"
  /// (nanoseconds).
  cudf::type_id timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;

  /// [Benchmark] When true, CudfHashJoinProbe returns an empty CudfVector
  /// instead of the actual join output, eliminating D2H transfer overhead.
  bool benchmarkSkipOutput{false};

  /// [Benchmark] When true, CudfHashJoinProbe logs accumulated gather time
  /// (the two cudf::gather calls in unfilteredOutput) when finished.
  bool benchmarkLogGatherTime{false};

  /// Spine deferral: adaptive defer-vs-eager. false = --boundary_hybrid
  /// always defers; true = eager until a prior execution of the same join
  /// observed match rate <= benchmarkDeferralThreshold (see DeferralStats).
  bool benchmarkDeferralAdaptive{false};
  /// Native row output at a CPU exit (2026-08-25): the row-path join emits
  /// HostRowVector (GPU-layout rows D2H'd) instead of transposing to Velox
  /// columns; results are extracted only when printed.
  bool benchmarkRowOutputNative{false};
  double benchmarkDeferralThreshold{0.5};

  /// [Benchmark] When true, CudfHashJoinProbe skips cudf::gather entirely
  /// and returns a dummy 1-row output. Use to measure gather's true e2e
  /// impact by comparing e2e with vs without this flag.
  bool benchmarkSkipGather{false};

  /// [Benchmark] When true, CudfHashJoinProbe uses row-wise warp gather
  /// instead of column-wise cudf::gather. Both sides (probe+build) are
  /// transposed to fixed-stride rows, gathered via warp-collaborative copy,
  /// then the result is discarded (skip_output semantics apply).
  bool benchmarkRowWiseGather{false};

  /// [Benchmark] When true AND benchmarkRowWiseGather is true, CudfFromVelox
  /// does CPU col-to-row conversion and outputs RowStoreVector.
  /// When false, CudfFromVelox uses the standard columnar Arrow->cudf path
  /// and RowHashJoinProbe/Build do GPU-side transpose.
  bool benchmarkCpuColToRow{false};

  /// [Benchmark] When true, operators emit per-batch trace lines to stdout:
  ///   TRACE <timestamp_us> <driver_id> <op_name> <event> <rows> <bytes>
  /// Use for Gantt-chart timeline reconstruction.
  bool benchmarkLogTimeline{false};

  /// [Benchmark] When true, FilterProject nodes are kept on the CPU instead
  /// of being replaced by CudfFilterProject. This pushes elementwise
  /// projections (e.g. TPC-H revenue = extendedprice*(1-discount)) to the CPU
  /// so the fused output column crosses PCIe instead of the raw inputs,
  /// reducing H2D transfer. Requires allowCpuFallback=true.
  bool benchmarkKeepProjectOnCpu{false};

  /// [Deprecated 2026-08-20] Fused-probe path removed (see exec/attic/).
  /// Field retained ONLY for stale-object ABI layout; never read.
  bool benchmarkFusedProbe{false};

  /// [Benchmark] Make the COLUMNAR probe use cudf::distinct_hash_join instead of
  /// the general cudf::hash_join. distinct_hash_join is the N:1-aware primitive
  /// (undefined if the build has duplicate keys) -- single-pass, no count phase.
  /// This is the FAIR columnar baseline for FK->PK workloads: the general join
  /// pays a two-pass count+retrieve tax (measured ~8ms fixed) that distinct
  /// avoids for free. Only applied to inner joins with no join filter (the build
  /// keys must be unique). Set it only when the workload is genuinely FK->PK.
  bool benchmarkDistinctHashJoin{false};

  /// [Deprecated 2026-08-20] Fused-probe path removed (see exec/attic/).
  /// Field retained ONLY for stale-object ABI layout; never read.
  bool benchmarkFuseSingleJoin{false};

  /// [Deprecated 2026-08-20] Fused-probe path removed (see exec/attic/).
  /// Field retained ONLY for stale-object ABI layout; never read.
  bool fusedColumnarSeed{false};

  /// [Deprecated 2026-08-20] Fused-probe path removed (see exec/attic/).
  /// Field retained ONLY for stale-object ABI layout; never read.
  bool fusedColumnarOutput{false};


  /// [Deprecated 2026-08-20] Fused-probe path removed (see exec/attic/).
  /// Field retained ONLY for stale-object ABI layout; never read.
  bool benchmarkFusedLazyPayload{false};

  /// [Benchmark] When true, a CudfBatchConcat is inserted before
  /// CudfHashJoinProbe, accumulating probe batches to batchSizeMinThreshold
  /// rows before probing. This is the "just re-batch on the GPU" alternative to
  /// row-wise gather: it decouples the join's batch size from the pipeline's,
  /// so scan/H2D can stream CPU-friendly small batches while the join still
  /// sees a GPU-friendly large one.
  ///
  /// Only the probe side is wrapped. CudfHashJoinBuild already accumulates all
  /// of its input and concatenates once in noMoreInput(), so it is effectively
  /// at unbounded batch size already and concat there would be pure copy cost.
  ///
  /// Independent of concatOptimizationEnabled, which covers aggregation only.
  bool benchmarkConcatBeforeJoin{false};

 private:
  // [Deprecated 2026-08-20] fused-probe registry; retained for ABI layout.
  std::unordered_set<std::string> fusedProbeJoinIds_;
  mutable std::mutex fusedProbeMutex_;

 public:
  /// [Benchmark] BOUNDARY-HYBRID mode (the paper's §3 single-join design):
  /// ONLY the join-key columns cross the device boundary. CudfFromVelox packs
  /// just the key columns (in join-key order) into the pinned row slot and
  /// RETAINS each input batch's full payload host-side; RowHashJoinBuild keeps
  /// a keys-only GPU row store plus a host-side payload store mapping global
  /// build row id -> (batchId, rowInBatch); RowHashJoinProbe probes on the
  /// GPU, reads back only the surviving (probe-local, build-global) id pairs,
  /// and materializes the output RowVector on the CPU from the retained
  /// batches. Requires benchmarkRowWiseGather + benchmarkCpuColToRow (the
  /// keys-only pack rides the pinned-pack path) and applies only to the
  /// non-fused RowHashJoinBuild/Probe pair (inner join, no filter,
  /// fixed-width types, N:1 probe).
  ///
  /// NOTE deliberately declared AFTER the private members: this struct is
  /// compiled into several libraries that the fast incremental rebuild
  /// (rebuild_cudf_exec.sh) does not touch. Appending at the very end keeps
  /// every pre-existing member at its old offset, so stale objects that only
  /// touch old members remain layout-compatible. The singleton instance is
  /// constructed in ToCudf.cpp, which IS rebuilt, so it has the new size.
  bool benchmarkBoundaryHybrid{false};

  /// [Benchmark] ROW-NATIVE MATCHER: replace cudf::hash_join as the single
  /// join's match-finder with RowNativeHashTable (GpuRowHashTable.cuh) — a
  /// chained N:M GPU multimap built directly over the build side's key row
  /// store and probed with keys read straight from the probe row store. No
  /// key extraction to columnar buffers on either side (the extract_keys
  /// step does not run; keyBuffers/hashJoin are not built). Applies to BOTH
  /// the standard GPU-resident RowHashJoinProbe (pair lists feed the row
  /// gather) and the boundary-hybrid probe (pair lists feed the id D2H +
  /// host gather). Requires benchmarkRowWiseGather; incompatible with
  /// benchmarkFusedProbe (the fused path keeps its own N:1 table).
  ///
  /// NOTE appended at the very end for the same stale-object layout reason
  /// as benchmarkBoundaryHybrid above.
  bool benchmarkRowTable{false};

  /// [Benchmark] COMPOSITE-KEY RANGE PACKING for the row-native matcher
  /// (benchmarkRowTable): at table-build time, compute per-key min/max over
  /// the build side and, when the combined bit-width fits 64 bits, pack the
  /// K key columns into ONE uint64 word ((v_k - min_k) << shift_k, the
  /// Eiger/GpuFusedProbe scheme). Hash, fingerprint, and verify then operate
  /// on the single packed word; probe keys outside the build range are
  /// rejected before hashing. No effect for single-key joins or when the
  /// ranges do not fit. NOTE appended at the very end for the same
  /// stale-object layout reason as benchmarkBoundaryHybrid above.
  bool benchmarkRowTablePackKeys{true};
};

} // namespace facebook::velox::cudf_velox
