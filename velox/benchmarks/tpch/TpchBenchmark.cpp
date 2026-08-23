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

#include "velox/benchmarks/tpch/TpchBenchmark.h"
#include <unordered_set>
#include <map>
#include <functional>
#include <iostream>
#include "velox/exec/OperatorType.h"
#include "velox/exec/PlanNodeStats.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DECLARE_int32(num_drivers);
DEFINE_bool(
    resident_tables,
    false,
    "Serve every TableScan from process-resident, pre-decoded RowVector "
    "batches (loaded once per process by running the real scan; filters "
    "stay as CPU FilterNodes). Removes Parquet decode from the timed "
    "region for every arm. See ResidentTables.h.");
DEFINE_string(
    data_path,
    "",
    "Root path of TPC-H data. Data layout must follow Hive-style partitioning. "
    "Example layout for '-data_path=/data/tpch10'\n"
    "       /data/tpch10/customer\n"
    "       /data/tpch10/lineitem\n"
    "       /data/tpch10/nation\n"
    "       /data/tpch10/orders\n"
    "       /data/tpch10/part\n"
    "       /data/tpch10/partsupp\n"
    "       /data/tpch10/region\n"
    "       /data/tpch10/supplier\n"
    "If the above are directories, they contain the data files for "
    "each table. If they are files, they contain a file system path for each "
    "data file, one per line. This allows running against cloud storage or "
    "HDFS");
namespace {
static bool notEmpty(const char* /*flagName*/, const std::string& value) {
  return !value.empty();
}
} // namespace

DEFINE_validator(data_path, &notEmpty);

DEFINE_int32(
    run_query_verbose,
    -1,
    "Run a given query and print execution statistics");
DEFINE_int32(
    io_meter_column_pct,
    0,
    "Percentage of lineitem columns to "
    "include in IO meter query. The columns are sorted by name and the n% first "
    "are scanned");
DEFINE_int32(
    s_selectivity_pct,
    100,
    "Selectivity percentage for S table in Q31 (10, 30, 60, 90, or 100)");

DEFINE_int32(
    synth_payload_cols,
    12,
    "Synthetic benches (Q31 join / Q40 sort): number of payload/projection "
    "columns to carry (1..12; VARCHARs are the last two).");
DEFINE_int32(
    synth_sort_keys,
    4,
    "Q40 sort bench: number of sort keys (1..4, low-cardinality first).");

DEFINE_bool(
    synth_join_sort,
    false,
    "Q31 join bench: append the 4-key sort after the join.");

DEFINE_int32(
    synth_join_keys,
    4,
    "Q31 join bench: number of join keys (1..4; row_id first, so match "
    "cardinality is identical for any count).");
DEFINE_int32(
    synth_wide_payload_cols,
    64,
    "Number of BIGINT payload columns for the wide sort benchmark (Q41), "
    "0-256. Mirrors the 64/128/256 widths used in Velox's row-based-sort "
    "evaluation; requires the 'wide' table from gen_widesort.sh");
DEFINE_int32(
    synth_wide_sort_keys,
    2,
    "Number of sort key columns for the wide sort benchmark (Q41), 1-16");



DEFINE_bool(
    result_checksum,
    false,
    "Print an order-sensitive checksum over all columns of all output rows "
    "(correctness gate; --include_results only compares the first 5 columns)");

DEFINE_bool(
    use_tpcds,
    false,
    "Interpret --run_query_verbose as a TPC-DS query number (1, 3, 7, 10, 13, "
    "15, 18, 19, 25, 26, 29, 30, 42, 43, 46, 52, 53, 55, 63, 68, 73, 74, 79, "
    "81, 89, 93, 96) and use the TPC-DS query builder. --data_path "
    "must point to a TPC-DS dataset laid out one sub-directory per table.");

void TpchBenchmark::initQueryBuilder() {
  if (FLAGS_use_tpcds) {
    tpcdsQueryBuilder_ =
        std::make_shared<TpcdsQueryBuilder>(toFileFormat(FLAGS_data_format));
    tpcdsQueryBuilder_->initialize(FLAGS_data_path);
    return;
  }
  queryBuilder_ =
      std::make_shared<TpchQueryBuilder>(toFileFormat(FLAGS_data_format));
  queryBuilder_->setResidentTables(FLAGS_resident_tables, FLAGS_num_drivers);
  queryBuilder_->initialize(FLAGS_data_path);
}

void TpchBenchmark::initialize() {
  QueryBenchmarkBase::initialize();
  initQueryBuilder();
}

void TpchBenchmark::shutdown() {
  QueryBenchmarkBase::shutdown();
  queryBuilder_.reset();
  tpcdsQueryBuilder_.reset();
}

void TpchBenchmark::runMain(
    std::ostream& out,
    facebook::velox::RunStats& runStats) {
  if (FLAGS_run_query_verbose == -1 && FLAGS_io_meter_column_pct == 0) {
    folly::runBenchmarks();
  } else {
    auto queryPlan = FLAGS_io_meter_column_pct > 0
        ? queryBuilder_->getIoMeterPlan(FLAGS_io_meter_column_pct)
        : getQueryPlan(FLAGS_run_query_verbose);
    auto [cursor, actualResults] = run(queryPlan, queryConfigs_);
    if (!cursor) {
      LOG(ERROR) << "Query terminated with error. Exiting";
      exit(1);
    }
    auto task = cursor->task();
    ensureTaskCompletion(task.get());
    if (FLAGS_include_results) {
      printResults(actualResults, out);
      out << std::endl;
    }
    if (FLAGS_result_checksum) {
      // Order-sensitive checksum over EVERY column of EVERY output row.
      // printResults() cannot serve as a correctness gate here because
      // RowVector::toString() truncates to the first 5 children, so wide
      // results would be compared on a handful of columns only.
      // Two digests are printed. The ordered one chains every row into a
      // single FNV-1a stream and is the strict gate for order-defined
      // results (sort). Join output order is not defined across drivers, so
      // for those the unordered digest -- a commutative combination of
      // per-row hashes -- is the gate: it is invariant to row order but
      // still sensitive to any changed, dropped, or duplicated row.
      uint64_t ordered = 1469598103934665603ULL; // FNV offset basis
      uint64_t unorderedSum = 0;
      uint64_t unorderedXor = 0;
      int64_t nrows = 0;
      for (const auto& vector : actualResults) {
        auto* rowVector = vector->as<RowVector>();
        const auto ncols = rowVector->childrenSize();
        for (vector_size_t i = 0; i < vector->size(); ++i) {
          uint64_t row = 1469598103934665603ULL;
          for (size_t c = 0; c < ncols; ++c) {
            const auto& child = rowVector->childAt(c);
            const uint64_t h =
                child->isNullAt(i) ? 0xDEADBEEFULL : child->hashValueAt(i);
            row = (row ^ h) * 1099511628211ULL; // FNV-1a mix
          }
          ordered = (ordered ^ row) * 1099511628211ULL;
          unorderedSum += row;
          unorderedXor ^= row;
          ++nrows;
        }
      }
      out << "Result checksum: " << std::hex << ordered
          << " unordered: " << unorderedSum << "/" << unorderedXor << std::dec
          << " rows: " << nrows << std::endl;
    }
    const auto stats = task->taskStats();
    int64_t rawInputBytes = 0;
    for (auto& pipeline : stats.pipelineStats) {
      auto& first = pipeline.operatorStats[0];
      if (first.operatorType == OperatorType::kTableScan) {
        rawInputBytes += first.rawInputBytes;
      }
    }
    runStats.rawInputBytes = rawInputBytes;
    out << fmt::format(
               "Execution time: {}",
               facebook::velox::succinctMillis(
                   stats.executionEndTimeMs - stats.executionStartTimeMs))
        << std::endl;
    out << fmt::format(
               "Splits total: {}, finished: {}",
               stats.numTotalSplits,
               stats.numFinishedSplits)
        << std::endl;
    out << printPlanWithStats(
               *queryPlan.plan, stats, FLAGS_include_custom_stats)
        << std::endl;
    // Operators registered under synthetic plan-node ids (e.g. the cudf
    // "<id>-from-velox" / "<id>-to-velox" conversion operators) are not in
    // the plan tree and printPlanWithStats drops them. Print them here,
    // aggregated per (planNodeId, operatorType) across drivers.
    {
      std::unordered_set<std::string> planIds;
      std::function<void(const core::PlanNode&)> collect =
          [&](const core::PlanNode& n) {
            planIds.insert(n.id());
            for (const auto& src : n.sources()) {
              collect(*src);
            }
          };
      collect(*queryPlan.plan);
      std::map<std::string, exec::OperatorStats> extra;
      for (const auto& pipeline : stats.pipelineStats) {
        for (const auto& op : pipeline.operatorStats) {
          if (planIds.count(op.planNodeId)) {
            continue;
          }
          const auto key = op.planNodeId + " " + op.operatorType;
          auto it = extra.find(key);
          if (it == extra.end()) {
            extra.emplace(key, op);
          } else {
            it->second.add(op);
          }
        }
      }
      for (const auto& [key, op] : extra) {
        out << "-- [extra] " << key << ": Input: " << op.inputPositions
            << " rows (" << succinctBytes(op.inputBytes) << ", "
            << op.inputVectors << " batches), Output: " << op.outputPositions
            << " rows (" << succinctBytes(op.outputBytes)
            << "), Cpu time: "
            << succinctNanos(
                   op.addInputTiming.cpuNanos + op.getOutputTiming.cpuNanos +
                   op.finishTiming.cpuNanos + op.isBlockedTiming.cpuNanos)
            << ", Wall time: "
            << succinctNanos(
                   op.addInputTiming.wallNanos + op.getOutputTiming.wallNanos +
                   op.finishTiming.wallNanos + op.isBlockedTiming.wallNanos)
            << ", Blocked wall time: " << succinctNanos(op.blockedWallNanos)
            << ", drivers: " << op.numDrivers << std::endl;
        if (FLAGS_include_custom_stats) {
          for (const auto& [name, st] : op.runtimeStats) {
            out << "      " << name << "  sum: " << st.sum
                << ", count: " << st.count << ", min: " << st.min
                << ", max: " << st.max << std::endl;
          }
        }
      }
    }
  }
}

std::unique_ptr<TpchBenchmark> benchmark;

BENCHMARK(q1) {
  benchmark->runQuery(1);
}

BENCHMARK(q2) {
  benchmark->runQuery(2);
}

BENCHMARK(q3) {
  benchmark->runQuery(3);
}

BENCHMARK(q4) {
  benchmark->runQuery(4);
}

BENCHMARK(q5) {
  benchmark->runQuery(5);
}

BENCHMARK(q6) {
  benchmark->runQuery(6);
}

BENCHMARK(q7) {
  benchmark->runQuery(7);
}

BENCHMARK(q8) {
  benchmark->runQuery(8);
}

BENCHMARK(q9) {
  benchmark->runQuery(9);
}

BENCHMARK(q10) {
  benchmark->runQuery(10);
}

BENCHMARK(q11) {
  benchmark->runQuery(11);
}

BENCHMARK(q12) {
  benchmark->runQuery(12);
}

BENCHMARK(q13) {
  benchmark->runQuery(13);
}

BENCHMARK(q14) {
  benchmark->runQuery(14);
}

BENCHMARK(q15) {
  benchmark->runQuery(15);
}

BENCHMARK(q16) {
  benchmark->runQuery(16);
}

BENCHMARK(q17) {
  benchmark->runQuery(17);
}

BENCHMARK(q18) {
  benchmark->runQuery(18);
}

BENCHMARK(q19) {
  benchmark->runQuery(19);
}

BENCHMARK(q20) {
  benchmark->runQuery(20);
}

BENCHMARK(q21) {
  benchmark->runQuery(21);
}

BENCHMARK(q22) {
  benchmark->runQuery(22);
}

void tpchBenchmarkMain() {
  VELOX_CHECK_NOT_NULL(benchmark);
  benchmark->initialize();
  if (FLAGS_test_flags_file.empty()) {
    RunStats ignore;
    benchmark->runMain(std::cout, ignore);
  } else {
    benchmark->runAllCombinations();
  }
  benchmark->shutdown();
}
