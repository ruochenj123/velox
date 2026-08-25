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

#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/benchmarks/CudfTpchBenchmark.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveConfig.h"
#include "velox/experimental/cudf/connectors/hive/CudfHiveTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfHiveConnectorTestBase.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"

#include <experimental/cudf/connectors/hive/CudfHiveConnector.h>

DECLARE_int64(max_coalesced_bytes);
DECLARE_string(max_coalesced_distance_bytes);
DECLARE_int32(parquet_prefetch_rowgroups);

using namespace facebook::velox;
using namespace facebook::velox::common::testutil;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

DEFINE_string(
    cudf_memory_resource,
    "async",
    "Memory resource for cudf operators.");

DEFINE_int32(
    cudf_memory_percent,
    50,
    "Percentage of GPU memory to allocate for cudf operators.");

// Hybrid-layout fork: scans stay on the CPU by default (the CPU does
// data-reductive work; only key-centric operators run on the GPU).
DEFINE_bool(velox_cudf_table_scan, false, "Enable cuDF table scan");

// ---- Hybrid-layout / whole-query-deferral arms (see DESIGN-whole-query-
// deferral.md). Mirrors VeloxCudfJoinBench's flag -> CudfConfig mapping so
// query-level runs and the single-join bench share one vocabulary. ----
DEFINE_bool(gpu, true,
    "Register the cuDF operator replacements. false = pure CPU Velox in the "
    "SAME binary (same scan path), the apples-to-apples baseline.");
DEFINE_bool(row_wise, false,
    "Row-wise GPU joins (RowHashJoinBuild/Probe over fixed-stride row "
    "stores) instead of the columnar cudf::hash_join operators.");
DEFINE_bool(cpu_col_to_row, false,
    "With --row_wise: CPU-side pinned pack into row stores (no GPU "
    "transpose).");
DEFINE_bool(boundary_hybrid, false,
    "Boundary-hybrid: only join keys cross the boundary; payloads stay "
    "host-resident and are gathered for survivors. Implies --row_wise "
    "--cpu_col_to_row.");
DEFINE_bool(row_table, false,
    "Row-native chained matcher (RowNativeHashTable) instead of "
    "cudf::hash_join inside the row-wise join operators.");
DEFINE_bool(
    row_output_native,
    false,
    "Row-path CPU exit emits HostRowVector (native GPU row layout, D2H only) "
    "instead of transposing to Velox columns; extracted only when printed.");
DEFINE_bool(deferral_adaptive, false,
    "With --boundary_hybrid: eager until a prior repeat observed match "
    "rate <= --deferral_threshold at the adjacent join (DeferralStats).");
DEFINE_double(deferral_threshold, 0.5, "Adaptive deferral match-rate bound.");
DEFINE_bool(row_table_pack_keys, true,
    "With --row_table: composite-key range packing when the ranges fit.");
DEFINE_bool(keep_project_on_cpu, true,
    "Keep FilterProject nodes on the CPU (elementwise projections such as "
    "revenue = price*(1-discount) are computed before the boundary).");
DEFINE_int32(concat_agg_rows, 0,
    "With --concat_agg: explicit concat target rows (0 = use "
    "cudf_gpu_batch_size_rows).");
DEFINE_bool(concat_agg, false,
    "Insert CudfBatchConcat before every CudfHashAggregation (upstream "
    "concatOptimizationEnabled): re-batch small inputs to the GPU agg.");
DEFINE_bool(concat_join, false,
    "Columnar arm: insert CudfBatchConcat before each probe (GPU re-batch "
    "to cudf_gpu_batch_size_rows).");
DEFINE_bool(log_gather_time, false,
    "Emit matcher/gather/boundary phase timings as operator stats.");
DEFINE_int32(batch_size, 0,
    "Engine output batch rows (preferred/max); 0 = Velox default (1024).");
DEFINE_string(pinned_pack_sync, "auto",
    "velox.cudf.pinned_pack_sync for the row-wise pack (auto|sync|async).");

DEFINE_bool(cudf_debug_enabled, false, "Enable debug printing");

void CudfTpchBenchmark::initialize() {
  TpchBenchmark::initialize();

  if (FLAGS_velox_cudf_table_scan) {
    connector::ConnectorRegistry::global().erase(
        facebook::velox::exec::test::kHiveConnectorId);

    // Add new values into the cudfHive configuration...
    auto cudfHiveConfigurationValues =
        std::unordered_map<std::string, std::string>();
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    cudfHiveConfigurationValues
        [cudf_velox::connector::hive::CudfHiveConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    cudfHiveConfigurationValues[cudf_velox::connector::hive::CudfHiveConfig::
                                    kAllowMismatchedCudfHiveSchemas] =
        std::to_string(true);
    auto cudfHiveProperties = std::make_shared<const config::ConfigBase>(
        std::move(cudfHiveConfigurationValues));

    // Create cudfHive connector with config...
    cudf_velox::connector::hive::CudfHiveConnectorFactory cudfHiveFactory;
    auto cudfHiveConnector = cudfHiveFactory.newConnector(
        facebook::velox::exec::test::kHiveConnectorId,
        cudfHiveProperties,
        ioExecutor_.get());
    connector::ConnectorRegistry::global().insert(
        cudfHiveConnector->connectorId(), cudfHiveConnector);
  }

  cudf_velox::CudfConfig::getInstance().memoryResource =
      FLAGS_cudf_memory_resource;
  cudf_velox::CudfConfig::getInstance().memoryPercent =
      FLAGS_cudf_memory_percent;

  cudf_velox::CudfConfig::getInstance().debugEnabled = FLAGS_cudf_debug_enabled;

  // ---- Hybrid-layout arms ----
  auto& cfg = cudf_velox::CudfConfig::getInstance();
  if (FLAGS_boundary_hybrid) {
    FLAGS_row_wise = true;
    FLAGS_cpu_col_to_row = true;
  }
  cfg.benchmarkRowWiseGather = FLAGS_row_wise;
  cfg.benchmarkCpuColToRow = FLAGS_cpu_col_to_row;
  cfg.benchmarkBoundaryHybrid = FLAGS_boundary_hybrid;
  cfg.benchmarkRowTable = FLAGS_row_table;
  cfg.benchmarkRowTablePackKeys = FLAGS_row_table_pack_keys;
  cfg.benchmarkKeepProjectOnCpu = FLAGS_keep_project_on_cpu;
  cfg.benchmarkDeferralAdaptive = FLAGS_deferral_adaptive;
  cfg.benchmarkRowOutputNative = FLAGS_row_output_native;
  cfg.benchmarkDeferralThreshold = FLAGS_deferral_threshold;
  cfg.benchmarkLogGatherTime = FLAGS_log_gather_time;
  cfg.benchmarkConcatBeforeJoin = FLAGS_concat_join;
  cfg.concatOptimizationEnabled = FLAGS_concat_agg;
  if (FLAGS_concat_agg) {
    // Concat target: explicit --concat_agg_rows, else aligned with the GPU
    // batch size (the default 100K would re-batch to SMALLER than the
    // conversion already produces).
    cfg.batchSizeMinThreshold = FLAGS_concat_agg_rows > 0
        ? FLAGS_concat_agg_rows
        : FLAGS_cudf_gpu_batch_size_rows;
  }
  if (FLAGS_concat_join) {
    cfg.batchSizeMinThreshold = FLAGS_cudf_gpu_batch_size_rows;
  }
  // Scans/filters/projects stay on the CPU; the per-operator fallback is
  // what lets the plan mix CPU and GPU operators.
  cfg.allowCpuFallback = true;

  // Enable cuDF operators
  if (FLAGS_gpu) {
    cudf_velox::registerCudf();
  }

  // Add custom configs
  queryConfigs_[facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
      std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  queryConfigs_[facebook::velox::cudf_velox::CudfFromVelox::kPinnedPackSync] =
      FLAGS_pinned_pack_sync;
  if (FLAGS_batch_size > 0) {
    const auto bs = std::to_string(FLAGS_batch_size);
    queryConfigs_["preferred_output_batch_rows"] = bs;
    queryConfigs_["max_output_batch_rows"] = bs;
    queryConfigs_["preferred_output_batch_bytes"] =
        std::to_string(1ULL << 30);
  }
}

std::shared_ptr<config::ConfigBase>
CudfTpchBenchmark::makeConnectorProperties() {
  auto cfg = TpchBenchmark::makeConnectorProperties();
  using CudfHiveCfg = cudf_velox::connector::hive::CudfHiveConfig;

  // CuDF-specific properties.
  cfg->set(
      CudfHiveCfg::kMaxChunkReadLimit,
      std::to_string(FLAGS_cudf_chunk_read_limit));
  cfg->set(
      CudfHiveCfg::kMaxPassReadLimit,
      std::to_string(FLAGS_cudf_pass_read_limit));
  cfg->set(CudfHiveCfg::kAllowMismatchedCudfHiveSchemas, "true");

  return cfg;
}

std::vector<std::shared_ptr<connector::ConnectorSplit>>
CudfTpchBenchmark::listSplits(
    const std::string& path,
    int32_t numSplitsPerFile,
    const exec::test::TpchPlan& plan) {
  // TODO (dm): Figure out a way to enforce 1 split per file in
  // CudfHiveDataSource outside of this benchmark
  if (FLAGS_velox_cudf_table_scan) {
    // TODO (dm): Instead of this, we can maybe use
    // makeHiveConnectorSplits(vector<shared_ptr<TempFilePath>>&
    // filePaths)
    std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
    auto temp = HiveConnectorTestBase::makeHiveConnectorSplits(
        path, 1, plan.dataFileFormat);
    for (auto& i : temp) {
      result.push_back(i);
    }
    return result;
  }

  return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
}

void CudfTpchBenchmark::shutdown() {
  if (FLAGS_gpu) {
    cudf_velox::unregisterCudf();
  }
  TpchBenchmark::shutdown();
}

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  tpchBenchmarkMain();
}
