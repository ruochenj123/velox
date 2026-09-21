#include "velox/experimental/cudf/benchmarks/GpuSynthQueryBuilder.h"

#include "velox/exec/tests/utils/PlanBuilder.h"

#include <fmt/format.h>
#include <gflags/gflags.h>

#include <algorithm>

DECLARE_int32(synth_payload_cols);
DECLARE_int32(synth_join_keys);
DECLARE_int32(synth_build_payload_cols);
DECLARE_int32(synth_wide_payload_cols);
DECLARE_int32(synth_wide_sort_keys);
DECLARE_bool(synth_sort_gather);

namespace facebook::velox::cudf_velox {

using exec::test::PlanBuilder;
using exec::test::TpchPlan;

TpchPlan GpuSynthQueryBuilder::getQueryPlan(int queryId) const {
  if (queryId != kGpuJoin && queryId != kGpuSort) {
    return TpchQueryBuilder::getQueryPlan(queryId);
  }
  auto plan = queryId == kGpuJoin ? getGpuJoinPlan() : getGpuSortPlan();
  if (resident_) {
    makeResident(plan);
  }
  return plan;
}

// Build: R projected to the join keys (keys only, 100M rows).
// Probe: P = keys + the first `--synth_payload_cols` payloads (200M rows,
// probe order scrambled, hit rate encoded in row_id).
// Output: probe keys + payloads (no build-side columns cross the boundary).
TpchPlan GpuSynthQueryBuilder::getGpuJoinPlan() const {
  // Key candidates are all BIGINT: the row-native matcher takes fixed-width
  // 4/8-byte keys only (the CPU workload's 3rd/4th keys are VARCHARs). The
  // 3rd/4th keys are R payloads copied into P at the same row, so they agree
  // exactly on matching rows; the two VARCHAR keys move to the payload list.
  static const std::vector<std::string> kKeys = {
      "row_id", "l_suppkey", "l_orderkey", "l_partkey"};
  static const std::vector<std::string> kPayloadOrder = {
      "l_returnflag", "l_linestatus",  "l_extendedprice", "l_shipmode",
      "l_shipinstruct", "l_quantity",  "l_discount",      "l_tax",
      "l_linenumber",   "l_shipdate",  "l_commitdate",    "l_receiptdate",
      "l_comment"};
  static const std::vector<std::string> kBuildRenames = {
      "row_id AS b_row_id",
      "l_suppkey AS b_suppkey",
      "l_orderkey AS b_orderkey",
      "l_partkey AS b_partkey"};
  static const std::vector<std::string> kBuildKeys = {
      "b_row_id", "b_suppkey", "b_orderkey", "b_partkey"};
  const int numPayloads = std::clamp(FLAGS_synth_payload_cols, 0, 13);
  const int numJoinKeys = std::clamp(FLAGS_synth_join_keys, 1, 4);
  // Build-side payload (2026-08-31): R carries the same 13 payload columns
  // as P; --synth_build_payload_cols projects the LAST N of the payload
  // order from R (renamed b_*), so probe and build payloads never share a
  // column. Projection convention: output = the join keys actually used +
  // probe payload + build payload (unused key candidates are NOT projected).
  const int numBuildPayloads =
      std::clamp(FLAGS_synth_build_payload_cols, 0, 13 - numPayloads);

  std::vector<std::string> probeColumns(
      kKeys.begin(), kKeys.begin() + numJoinKeys);
  probeColumns.insert(
      probeColumns.end(),
      kPayloadOrder.begin(),
      kPayloadOrder.begin() + numPayloads);
  std::vector<std::string> buildColumns(
      kKeys.begin(), kKeys.begin() + numJoinKeys);
  std::vector<std::string> buildRenames(
      kBuildRenames.begin(), kBuildRenames.begin() + numJoinKeys);
  std::vector<std::string> outputColumns = probeColumns;
  for (int i = 0; i < numBuildPayloads; ++i) {
    const auto& c = kPayloadOrder[kPayloadOrder.size() - 1 - i];
    buildColumns.push_back(c);
    buildRenames.push_back(fmt::format("{} AS b_{}", c, c));
    outputColumns.push_back("b_" + c);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId buildScanId;
  core::PlanNodeId probeScanId;

  auto build = PlanBuilder(planNodeIdGenerator, pool_.get())
                   .filtersAsNode(filtersAsNode_)
                   .tableScan(
                       kTableR,
                       getRowType(kTableR, buildColumns),
                       getFileColumnNames(kTableR),
                       {})
                   .captureScanNodeId(buildScanId)
                   .project(buildRenames)
                   .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kTableP,
                      getRowType(kTableP, probeColumns),
                      getFileColumnNames(kTableP),
                      {})
                  .captureScanNodeId(probeScanId)
                  .hashJoin(
                      {kKeys.begin(), kKeys.begin() + numJoinKeys},
                      {kBuildKeys.begin(), kBuildKeys.begin() + numJoinKeys},
                      build,
                      "",
                      outputColumns)
                  .planNode();

  TpchPlan context;
  context.planName = fmt::format(
      "gjoin_p{}_b{}_j{}", numPayloads, numBuildPayloads, numJoinKeys);
  context.plan = std::move(plan);
  context.dataFiles[probeScanId] = getTableFilePaths(kTableP);
  context.dataFiles[buildScanId] = getTableFilePaths(kTableR);
  context.dataFileFormat = format_;
  return context;
}

// Scan `wide` (k1..k{keys}, c0..c{payloads-1}) -> OrderBy on the keys; full
// sort, no LIMIT (the Velox row-sort blog setting).
TpchPlan GpuSynthQueryBuilder::getGpuSortPlan() const {
  const int numPayloads = std::clamp(FLAGS_synth_wide_payload_cols, 0, 256);
  const int numKeys = std::clamp(FLAGS_synth_wide_sort_keys, 1, 16);
  std::vector<std::string> sortKeys;
  for (int i = 1; i <= numKeys; ++i) {
    sortKeys.push_back(fmt::format("k{}", i));
  }
  std::vector<std::string> columns = sortKeys;
  for (int i = 0; i < numPayloads; ++i) {
    columns.push_back(fmt::format("c{}", i));
  }

  core::PlanNodeId scanId;
  PlanBuilder builder(pool_.get());
  builder.filtersAsNode(filtersAsNode_)
      .tableScan(
          kTableWide,
          getRowType(kTableWide, columns),
          getFileColumnNames(kTableWide),
          {})
      .captureScanNodeId(scanId);
  if (FLAGS_synth_sort_gather) {
    // Velox's production shape: N scan (+pack) drivers feed ONE sort driver
    // through a gather LocalPartition. Without it the whole ingestion
    // (arrow conversion / row pack) runs on the sort's single driver.
    builder.localPartition(std::vector<std::string>{});
  }
  auto plan = builder.orderBy(sortKeys, false).planNode();

  TpchPlan context;
  context.planName = fmt::format(
      "gsort_k{}_p{}{}", numKeys, numPayloads, FLAGS_synth_sort_gather ? "_gather" : "");
  context.plan = std::move(plan);
  context.dataFiles[scanId] = getTableFilePaths(kTableWide);
  context.dataFileFormat = format_;
  return context;
}

} // namespace facebook::velox::cudf_velox
