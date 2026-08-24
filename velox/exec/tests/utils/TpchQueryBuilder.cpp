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
#include "velox/exec/tests/utils/TpchQueryBuilder.h"

#include <unordered_set>
#include "velox/connectors/hive/TableHandle.h"
#include "velox/exec/Cursor.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/exec/tests/utils/ResidentTables.h"

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/tpch/gen/TpchGen.h"

#include <folly/json.h>

#include <algorithm>
#include <iostream>
#include <atomic>
#include <fstream>
#include <thread>
#include <limits>
#include <sstream>
#include <gflags/gflags.h>

DECLARE_int32(s_selectivity_pct);
DECLARE_int32(synth_payload_cols);
DECLARE_int32(synth_sort_keys);
DECLARE_bool(synth_join_sort);
DECLARE_bool(synth_sort_gather);
DECLARE_bool(synth_join_flip);
DECLARE_int32(synth_join_keys);
DECLARE_int32(synth_wide_payload_cols);
DECLARE_int32(synth_wide_sort_keys);

namespace facebook::velox::exec::test {

namespace {

/// DWRF does not support Date type and Varchar is used.
/// Return the Date filter expression as per data format.
std::string formatDateFilter(
    const std::string& stringDate,
    const RowTypePtr& rowType,
    const std::string& lowerBound,
    const std::string& upperBound) {
  bool isDwrf = rowType->findChild(stringDate)->isVarchar();
  auto suffix = isDwrf ? "" : "::DATE";

  if (!lowerBound.empty() && !upperBound.empty()) {
    return fmt::format(
        "{} between {}{} and {}{}",
        stringDate,
        lowerBound,
        suffix,
        upperBound,
        suffix);
  } else if (!lowerBound.empty()) {
    return fmt::format("{} > {}{}", stringDate, lowerBound, suffix);
  } else if (!upperBound.empty()) {
    return fmt::format("{} < {}{}", stringDate, upperBound, suffix);
  }

  VELOX_FAIL(
      "Date range check expression must have either a lower or an upper bound");
}

std::vector<std::string> mergeColumnNames(
    const std::vector<std::string>& firstColumnVector,
    const std::vector<std::string>& secondColumnVector) {
  std::vector<std::string> mergedColumnVector = std::move(firstColumnVector);
  mergedColumnVector.insert(
      mergedColumnVector.end(),
      secondColumnVector.begin(),
      secondColumnVector.end());
  return mergedColumnVector;
};
} // namespace

void TpchQueryBuilder::readFileSchema(
    const std::string& tableName,
    const std::string& filePath,
    const std::vector<std::string>& columns) {
  dwio::common::ReaderOptions readerOptions{pool_.get()};
  readerOptions.setFileFormat(format_);
  auto uniqueReadFile =
      filesystems::getFileSystem(filePath, nullptr)->openFileForRead(filePath);
  std::shared_ptr<ReadFile> readFile;
  readFile.reset(uniqueReadFile.release());
  auto input = std::make_unique<dwio::common::BufferedInput>(
      readFile, readerOptions.memoryPool());
  std::unique_ptr<dwio::common::Reader> reader =
      dwio::common::getReaderFactory(readerOptions.fileFormat())
          ->createReader(std::move(input), readerOptions);
  const auto fileType = reader->rowType();
  const auto fileColumnNames = fileType->names();
  // A NARROWER file than the table's declared schema (e.g. the 2-key
  // 'widesort' dataset against the 16-key 'wide' declaration): the file's
  // columns must be a by-name subset of the declaration, and are then
  // mapped by name in FILE order (positional mapping would misalign).
  std::vector<std::string> columnsUsed = columns;
  if (fileColumnNames.size() < columns.size()) {
    std::unordered_set<std::string> declared(columns.begin(), columns.end());
    for (const auto& n : fileColumnNames) {
      VELOX_CHECK(
          declared.count(n) > 0,
          "file column '{}' not in the declared schema (narrower file)",
          n);
    }
    columnsUsed = fileColumnNames;
  }
  // There can be extra columns in the file towards the end.
  VELOX_CHECK_GE(fileColumnNames.size(), columnsUsed.size());
  std::unordered_map<std::string, std::string> fileColumnNamesMap(
      columnsUsed.size());
  std::transform(
      columnsUsed.begin(),
      columnsUsed.end(),
      fileColumnNames.begin(),
      std::inserter(fileColumnNamesMap, fileColumnNamesMap.begin()),
      [](std::string a, std::string b) { return std::make_pair(a, b); });
  auto columnNames = columnsUsed;
  auto types = fileType->children();
  // Trailing extra file columns (e.g. the <col>_str twins of an encoded
  // dataset) are exposed under their own names so queries can select them.
  for (size_t i = columnsUsed.size(); i < fileColumnNames.size(); i++) {
    columnNames.push_back(fileColumnNames[i]);
    fileColumnNamesMap[fileColumnNames[i]] = fileColumnNames[i];
    if (fileColumnNames[i].size() > 4 &&
        fileColumnNames[i].substr(fileColumnNames[i].size() - 4) == "_str") {
      encStrTwins_.insert(
          fileColumnNames[i].substr(0, fileColumnNames[i].size() - 4));
    }
  }
  types.resize(columnNames.size());
  tableMetadata_[tableName].type =
      std::make_shared<RowType>(std::move(columnNames), std::move(types));
  tableMetadata_[tableName].fileColumnNames = std::move(fileColumnNamesMap);
}

void TpchQueryBuilder::initialize(const std::string& dataPath) {
  // SSB dataset? (lineorder fact table present.)
  {
    std::error_code ec;
    if (fs::exists(dataPath + "/lineorder", ec)) {
      ssb_ = true;
      static const std::vector<std::pair<std::string, std::vector<std::string>>>
          kSsbTables = {
              {"lineorder",
               {"lo_orderkey",      "lo_linenumber",  "lo_custkey",
                "lo_partkey",       "lo_suppkey",     "lo_orderdate",
                "lo_orderpriority", "lo_shippriority", "lo_quantity",
                "lo_extendedprice", "lo_ordtotalprice", "lo_discount",
                "lo_revenue",       "lo_supplycost",  "lo_tax",
                "lo_commitdate",    "lo_shipmode"}},
              {"customer",
               {"c_custkey", "c_name", "c_address", "c_city", "c_nation",
                "c_region", "c_phone", "c_mktsegment"}},
              {"supplier",
               {"s_suppkey", "s_name", "s_address", "s_city", "s_nation",
                "s_region", "s_phone"}},
              {"part",
               {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1",
                "p_color", "p_type", "p_size", "p_container"}},
              {"date",
               {"d_datekey", "d_date", "d_dayofweek", "d_month", "d_year",
                "d_yearmonthnum", "d_yearmonth", "d_daynuminweek",
                "d_daynuminmonth", "d_daynuminyear", "d_monthnuminyear",
                "d_weeknuminyear", "d_sellingseason", "d_lastdayinweekfl",
                "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl"}},
          };
      for (const auto& [tableName, columns] : kSsbTables) {
        const fs::path tablePath{dataPath + "/" + tableName};
        for (auto const& dirEntry : fs::directory_iterator{tablePath}) {
          if (!dirEntry.is_regular_file() ||
              dirEntry.path().filename().c_str()[0] == '.') {
            continue;
          }
          if (tableMetadata_[tableName].dataFiles.empty()) {
            readFileSchema(tableName, dirEntry.path().string(), columns);
          }
          tableMetadata_[tableName].dataFiles.push_back(dirEntry.path());
        }
      }
      return;
    }
  }
  // Encoded dataset? (see encCodes_ in the header)
  {
    std::ifstream codes(dataPath + "/codes.json");
    if (codes.good()) {
      std::stringstream buf;
      buf << codes.rdbuf();
      auto json = folly::parseJson(buf.str());
      for (const auto& [col, dict] : json.items()) {
        for (const auto& [text, code] : dict.items()) {
          encCodes_[col.asString()][text.asString()] =
              static_cast<int32_t>(code.asInt());
        }
      }
      encoded_ = true;
      LOG(INFO) << "TpchQueryBuilder: string-encoded dataset, "
                << encCodes_.size() << " dictionaries";
    }
  }
  for (const auto& [tableName, columns] : kTables_) {
    const fs::path tablePath{dataPath + "/" + tableName};
    std::error_code error;
    bool anyFound = false;
    for (auto const& dirEntry : fs::directory_iterator{
             tablePath, std::filesystem::directory_options(), error}) {
      if (!dirEntry.is_regular_file()) {
        continue;
      }
      // Ignore hidden files.
      if (dirEntry.path().filename().c_str()[0] == '.') {
        continue;
      }
      if (tableMetadata_[tableName].dataFiles.empty()) {
        anyFound = true;
        readFileSchema(tableName, dirEntry.path().string(), columns);
      }
      tableMetadata_[tableName].dataFiles.push_back(dirEntry.path());
    }
    if (!anyFound && error) {
      std::ifstream file(tablePath);
      std::string line;
      while (std::getline(file, line)) {
        if (tableMetadata_[tableName].dataFiles.empty()) {
          readFileSchema(tableName, line, columns);
        }
        tableMetadata_[tableName].dataFiles.push_back(line);
      }
    }
  }
}

std::string TpchQueryBuilder::encLit(
    const std::string& col,
    const std::string& value) const {
  if (!encodedCol(col)) {
    return "'" + value + "'";
  }
  const auto& d = encCodes_.at(col);
  auto it = d.find(value);
  if (it == d.end()) {
    // e.g. TPC-H Q19's literal 'AIR REG' (data has 'REG AIR'): a text
    // predicate that matches nothing must keep matching nothing.
    LOG(WARNING) << "no code for " << col << "='" << value << "'";
    return "-1";
  }
  return std::to_string(it->second);
}

std::string TpchQueryBuilder::encIn(
    const std::string& col,
    const std::vector<std::string>& values) const {
  std::string out = col + " IN (";
  for (size_t i = 0; i < values.size(); i++) {
    out += (i ? ", " : "") + encLit(col, values[i]);
  }
  return out + ")";
}

std::string TpchQueryBuilder::encLikePrefix(
    const std::string& col,
    const std::string& prefix) const {
  if (!encodedCol(col)) {
    return col + " LIKE '" + prefix + "%'";
  }
  int32_t lo = std::numeric_limits<int32_t>::max(), hi = -1;
  for (const auto& [text, code] : encCodes_.at(col)) {
    if (text.compare(0, prefix.size(), prefix) == 0) {
      lo = std::min(lo, code);
      hi = std::max(hi, code);
    }
  }
  VELOX_CHECK(hi >= 0, "no code matches {} LIKE '{}%'", col, prefix);
  return "(" + col + " BETWEEN " + std::to_string(lo) + " AND " +
      std::to_string(hi) + ")";
}

std::string TpchQueryBuilder::encLikeSuffix(
    const std::string& col,
    const std::string& suffix) const {
  if (!encodedCol(col)) {
    return col + " LIKE '%" + suffix + "'";
  }
  // OR-chain rather than IN (...): remaining filters are parsed without a
  // memory pool and an IN-list would be a complex (array) literal.
  std::string out = "(";
  bool first = true;
  for (const auto& [text, code] : encCodes_.at(col)) {
    if (text.size() >= suffix.size() &&
        text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0) {
      out += (first ? "" : " OR ") + col + " = " + std::to_string(code);
      first = false;
    }
  }
  VELOX_CHECK(!first, "no code matches {} LIKE '%{}'", col, suffix);
  return out + ")";
}

const std::vector<std::string>& TpchQueryBuilder::getTableNames() {
  return kTableNames_;
}

TpchQueryBuilder::~TpchQueryBuilder() {
  // The process-wide registry must release the resident batches while the
  // pool that owns their buffers (pool_) is still alive.
  ResidentTableRegistry::instance().clear();
  residentCache_.clear();
  unregisterResidentTablesAdapter();
}

TpchPlan TpchQueryBuilder::getQueryPlan(int queryId) const {
  auto plan = queryId >= 101 && queryId <= 113 ? getSsbPlan(queryId)
                                               : buildQueryPlan(queryId);
  if (resident_) {
    makeResident(plan);
  }
  return plan;
}

namespace {
void collectScans(
    const core::PlanNodePtr& node,
    std::vector<std::shared_ptr<const core::TableScanNode>>& out) {
  if (auto scan = std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
    out.push_back(scan);
  }
  for (const auto& src : node->sources()) {
    collectScans(src, out);
  }
}
} // namespace

void TpchQueryBuilder::makeResident(TpchPlan& plan) const {
  registerResidentTablesAdapter();
  std::vector<std::shared_ptr<const core::TableScanNode>> scans;
  collectScans(plan.plan, scans);
  for (const auto& scan : scans) {
    auto hive = std::dynamic_pointer_cast<const connector::hive::HiveTableHandle>(
        scan->tableHandle());
    VELOX_CHECK_NOT_NULL(hive);
    const auto& table = hive->tableName();
    const auto& outType = scan->outputType();
    std::string key = table;
    for (const auto& n : outType->names()) {
      key += "|" + n;
    }
    auto it = residentCache_.find(key);
    std::cerr << "[resident] table " << table << " cols=" << outType->size()
              << (it == residentCache_.end() ? " loading..." : " cached")
              << std::endl;
    if (it == residentCache_.end()) {
      // Run the real (unfiltered) scan once and retain its batches.
      auto start = std::chrono::steady_clock::now();
      // The round-robin local partition after the scan forces lazy
      // vectors to be loaded inside the (parallel) scan drivers rather than
      // serially on this thread.
      core::PlanNodeId scanId;
      // Distinct id space (1M+): the ResidentTables adapter matches scans by
      // plan node id, and the preload's own scan must never collide with a
      // query scan id already registered.
      auto preloadIds = std::make_shared<core::PlanNodeIdGenerator>(1000000);
      auto scanPlan = PlanBuilder(preloadIds, pool_.get())
                          .tableScan(table, outType, getFileColumnNames(table))
                          .capturePlanNodeId(scanId)
                          .localPartitionRoundRobin()
                          .planNode();
      CursorParameters params;
      params.planNode = scanPlan;
      params.maxDrivers = residentDrivers_;
      // The preload is a plain CPU scan: keep GPU operator replacement (the
      // cudf driver adapter) out of it, or the localPartition becomes a GPU
      // round trip.
      params.queryConfigs["cudf.enabled"] = "false";
      auto cursor = TaskCursor::create(params);
      auto* task = cursor->task().get();
      for (const auto& path : getTableFilePaths(table)) {
        for (auto& split : HiveConnectorTestBase::makeHiveConnectorSplits(
                 path, residentDrivers_ * 4, format_)) {
          task->addSplit(scanId, exec::Split(std::move(split)));
        }
      }
      task->noMoreSplits(scanId);
      cursor->start();
      std::vector<RowVectorPtr> produced;
      while (cursor->moveNext()) {
        auto v = cursor->current();
        std::vector<VectorPtr> children(v->childrenSize());
        for (size_t c = 0; c < children.size(); c++) {
          children[c] = BaseVector::loadedVectorShared(v->childAt(c));
        }
        produced.push_back(std::make_shared<RowVector>(
            v->pool(), v->type(), nullptr, v->size(), std::move(children)));
      }
      waitForTaskCompletion(task);
      // Deep FLAT copy into OUR pool, in parallel (batches are independent):
      // the task's pools die with the cursor; transferOrCopyTo leaves
      // reader-owned buffer views behind and testingCopyPreserveEncodings
      // shares string buffers / drops the pool for dictionary bases.
      // BaseVector::copy allocates every buffer in the target pool.
      auto batches =
          std::make_shared<std::vector<RowVectorPtr>>(produced.size());
      {
        const size_t nThreads = std::max<int32_t>(1, residentDrivers_);
        std::vector<std::thread> workers;
        std::atomic<size_t> nextIdx{0};
        for (size_t t = 0; t < nThreads; t++) {
          workers.emplace_back([&]() {
            for (size_t i = nextIdx++; i < produced.size(); i = nextIdx++) {
              const auto& src = produced[i];
              auto row = std::static_pointer_cast<RowVector>(
                  BaseVector::create(src->type(), src->size(), pool_.get()));
              row->copy(src.get(), 0, 0, src->size());
              (*batches)[i] = std::move(row);
            }
          });
        }
        for (auto& w : workers) {
          w.join();
        }
      }
      int64_t rows = 0;
      for (const auto& b : *batches) {
        rows += b->size();
      }
      produced.clear();
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count();
      std::cerr << "[resident] loaded " << table << ": " << rows
                << " rows, " << batches->size() << " batches, " << ms
                << " ms" << std::endl;
      LOG(INFO) << "Resident table " << table << " (" << outType->size()
                << " cols): " << rows << " rows in " << batches->size()
                << " batches, " << ms << " ms";
      it = residentCache_.emplace(key, std::move(batches)).first;
    }
    ResidentTableRegistry::instance().set(scan->id(), it->second);
  }
  std::cerr << "[resident] all tables ready" << std::endl;
  // No splits: the scans are served from memory.
  plan.dataFiles.clear();
}

// ============================================================================
// SSB (Star Schema Benchmark) Q1.1..Q4.3 as ids 101..113. All queries share
// one template: lineorder probe spine, dimension builds, aggregation, then
// (for flights 2-4) an order-by. Filters sit on the dimension scans and on
// lineorder (flight 1).
// ============================================================================
TpchPlan TpchQueryBuilder::getSsbPlan(int queryId) const {
  struct Dim {
    std::string table;
    std::vector<std::string> columns; // scan columns (key first)
    std::string probeKey;
    std::string buildKey;
    std::vector<std::string> filters;
  };
  std::vector<std::string> loColumns;
  std::vector<std::string> loFilters;
  std::vector<Dim> dims;
  std::string aggExprInput; // projected column feeding the SUM
  std::vector<std::string> groupBy;
  bool orderByGroup = true;

  auto dateDim = [](std::vector<std::string> filters,
                    std::vector<std::string> extraCols =
                        {}) -> Dim {
    std::vector<std::string> cols = {"d_datekey"};
    for (auto& c : extraCols) cols.push_back(c);
    return {"date", cols, "lo_orderdate", "d_datekey", std::move(filters)};
  };

  switch (queryId) {
    case 101: // Q1.1
      loColumns = {"lo_orderdate", "lo_extendedprice", "lo_discount",
                   "lo_quantity"};
      loFilters = {"lo_discount between 1 and 3", "lo_quantity < 25"};
      dims = {dateDim({"d_year = 1993"}, {"d_year"})};
      aggExprInput = "lo_extendedprice * lo_discount";
      break;
    case 102: // Q1.2
      loColumns = {"lo_orderdate", "lo_extendedprice", "lo_discount",
                   "lo_quantity"};
      loFilters = {"lo_discount between 4 and 6",
                   "lo_quantity between 26 and 35"};
      dims = {dateDim({"d_yearmonthnum = 199401"}, {"d_yearmonthnum"})};
      aggExprInput = "lo_extendedprice * lo_discount";
      break;
    case 103: // Q1.3
      loColumns = {"lo_orderdate", "lo_extendedprice", "lo_discount",
                   "lo_quantity"};
      loFilters = {"lo_discount between 5 and 7",
                   "lo_quantity between 26 and 35"};
      dims = {dateDim(
          {"d_weeknuminyear = 6", "d_year = 1994"},
          {"d_weeknuminyear", "d_year"})};
      aggExprInput = "lo_extendedprice * lo_discount";
      break;
    case 104: // Q2.1
    case 105: // Q2.2
    case 106: // Q2.3
      loColumns = {"lo_orderdate", "lo_partkey", "lo_suppkey", "lo_revenue"};
      dims = {
          {"part", {"p_partkey", "p_brand1", "p_category"}, "lo_partkey",
           "p_partkey",
           queryId == 104
               ? std::vector<std::string>{"p_category = 'MFGR#12'"}
               : queryId == 105
               ? std::vector<std::string>{
                     "p_brand1 between 'MFGR#2221' and 'MFGR#2228'"}
               : std::vector<std::string>{"p_brand1 = 'MFGR#2239'"}},
          {"supplier", {"s_suppkey", "s_region"}, "lo_suppkey", "s_suppkey",
           {queryId == 104 ? "s_region = 'AMERICA'"
                           : queryId == 105 ? "s_region = 'ASIA'"
                                            : "s_region = 'EUROPE'"}},
          dateDim({}, {"d_year"})};
      aggExprInput = "lo_revenue";
      groupBy = {"d_year", "p_brand1"};
      break;
    case 107: // Q3.1
      loColumns = {"lo_orderdate", "lo_custkey", "lo_suppkey", "lo_revenue"};
      dims = {
          {"customer", {"c_custkey", "c_nation", "c_region"}, "lo_custkey",
           "c_custkey", {"c_region = 'ASIA'"}},
          {"supplier", {"s_suppkey", "s_nation", "s_region"}, "lo_suppkey",
           "s_suppkey", {"s_region = 'ASIA'"}},
          dateDim({"d_year between 1992 and 1997"}, {"d_year"})};
      aggExprInput = "lo_revenue";
      groupBy = {"c_nation", "s_nation", "d_year"};
      break;
    case 108: // Q3.2
      loColumns = {"lo_orderdate", "lo_custkey", "lo_suppkey", "lo_revenue"};
      dims = {
          {"customer", {"c_custkey", "c_city", "c_nation"}, "lo_custkey",
           "c_custkey", {"c_nation = 'UNITED STATES'"}},
          {"supplier", {"s_suppkey", "s_city", "s_nation"}, "lo_suppkey",
           "s_suppkey", {"s_nation = 'UNITED STATES'"}},
          dateDim({"d_year between 1992 and 1997"}, {"d_year"})};
      aggExprInput = "lo_revenue";
      groupBy = {"c_city", "s_city", "d_year"};
      break;
    case 109: // Q3.3
    case 110: // Q3.4
      loColumns = {"lo_orderdate", "lo_custkey", "lo_suppkey", "lo_revenue"};
      dims = {
          {"customer", {"c_custkey", "c_city"}, "lo_custkey", "c_custkey",
           {"c_city IN ('UNITED KI1', 'UNITED KI5')"}},
          {"supplier", {"s_suppkey", "s_city"}, "lo_suppkey", "s_suppkey",
           {"s_city IN ('UNITED KI1', 'UNITED KI5')"}},
          queryId == 109
              ? dateDim({"d_year between 1992 and 1997"}, {"d_year"})
              : dateDim({"d_yearmonth = 'Dec1997'"}, {"d_yearmonth", "d_year"})};
      aggExprInput = "lo_revenue";
      groupBy = {"c_city", "s_city", "d_year"};
      break;
    case 111: // Q4.1
      loColumns = {"lo_orderdate", "lo_custkey", "lo_suppkey", "lo_partkey",
                   "lo_revenue", "lo_supplycost"};
      dims = {
          {"customer", {"c_custkey", "c_nation", "c_region"}, "lo_custkey",
           "c_custkey", {"c_region = 'AMERICA'"}},
          {"supplier", {"s_suppkey", "s_region"}, "lo_suppkey", "s_suppkey",
           {"s_region = 'AMERICA'"}},
          {"part", {"p_partkey", "p_mfgr"}, "lo_partkey", "p_partkey",
           {"p_mfgr IN ('MFGR#1', 'MFGR#2')"}},
          dateDim({}, {"d_year"})};
      aggExprInput = "lo_revenue - lo_supplycost";
      groupBy = {"d_year", "c_nation"};
      break;
    case 112: // Q4.2
      loColumns = {"lo_orderdate", "lo_custkey", "lo_suppkey", "lo_partkey",
                   "lo_revenue", "lo_supplycost"};
      dims = {
          {"customer", {"c_custkey", "c_region"}, "lo_custkey", "c_custkey",
           {"c_region = 'AMERICA'"}},
          {"supplier", {"s_suppkey", "s_nation", "s_region"}, "lo_suppkey",
           "s_suppkey", {"s_region = 'AMERICA'"}},
          {"part", {"p_partkey", "p_mfgr", "p_category"}, "lo_partkey",
           "p_partkey", {"p_mfgr IN ('MFGR#1', 'MFGR#2')"}},
          dateDim({"d_year IN (1997, 1998)"}, {"d_year"})};
      aggExprInput = "lo_revenue - lo_supplycost";
      groupBy = {"d_year", "s_nation", "p_category"};
      break;
    case 113: // Q4.3
      loColumns = {"lo_orderdate", "lo_custkey", "lo_suppkey", "lo_partkey",
                   "lo_revenue", "lo_supplycost"};
      dims = {
          {"customer", {"c_custkey", "c_region"}, "lo_custkey", "c_custkey",
           {"c_region = 'AMERICA'"}},
          {"supplier", {"s_suppkey", "s_city", "s_nation"}, "lo_suppkey",
           "s_suppkey", {"s_nation = 'UNITED STATES'"}},
          {"part", {"p_partkey", "p_brand1", "p_category"}, "lo_partkey",
           "p_partkey", {"p_category = 'MFGR#14'"}},
          dateDim({"d_year IN (1997, 1998)"}, {"d_year"})};
      aggExprInput = "lo_revenue - lo_supplycost";
      groupBy = {"d_year", "s_city", "p_brand1"};
      break;
    default:
      VELOX_FAIL("Unknown SSB query id {}", queryId);
  }

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  TpchPlan context;
  context.dataFileFormat = format_;

  auto loType = getRowType("lineorder", loColumns);
  core::PlanNodeId loScanId;
  auto pb = PlanBuilder(planNodeIdGenerator, pool_.get())
                .filtersAsNode(filtersAsNode_)
                .tableScan(
                    "lineorder",
                    loType,
                    getFileColumnNames("lineorder"),
                    loFilters)
                .capturePlanNodeId(loScanId);
  context.dataFiles[loScanId] = getTableFilePaths("lineorder");

  // Columns carried through the join chain: agg input + group-by columns
  // as they appear; each dim join drops its probe key.
  for (const auto& d : dims) {
    auto dimType = getRowType(d.table, d.columns);
    core::PlanNodeId dimScanId;
    auto dimPlan = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           d.table,
                           dimType,
                           getFileColumnNames(d.table),
                           d.filters)
                       .capturePlanNodeId(dimScanId)
                       .planNode();
    context.dataFiles[dimScanId] = getTableFilePaths(d.table);
    // Join output = columns needed AFTER this join: later dims' probe
    // keys, the aggregate's input columns, and group-by columns seen so
    // far (from lineorder or from this/earlier dims).
    std::vector<std::string> aggCols;
    for (const auto& c : {"lo_extendedprice", "lo_discount", "lo_revenue",
                          "lo_supplycost"}) {
      if (aggExprInput.find(c) != std::string::npos) {
        aggCols.push_back(c);
      }
    }
    std::vector<std::string> neededAfter = aggCols;
    for (const auto& later : dims) {
      if (&later > &d) {
        neededAfter.push_back(later.probeKey);
      }
    }
    for (const auto& g : groupBy) {
      neededAfter.push_back(g);
    }
    std::vector<std::string> outCols;
    auto want = [&](const std::string& c) {
      return std::find(neededAfter.begin(), neededAfter.end(), c) !=
          neededAfter.end() &&
          std::find(outCols.begin(), outCols.end(), c) == outCols.end();
    };
    for (const auto& c : loColumns) {
      if (want(c)) {
        outCols.push_back(c);
      }
    }
    for (const auto& c : d.columns) {
      if (want(c)) {
        outCols.push_back(c);
      }
    }
    pb.hashJoin({d.probeKey}, {d.buildKey}, dimPlan, "", outCols);
    loColumns = outCols;
  }

  pb.project([&] {
    std::vector<std::string> exprs = groupBy;
    exprs.push_back(aggExprInput + " AS agg_input");
    return exprs;
  }());
  pb.partialAggregation(groupBy, {"sum(agg_input) AS revenue"})
      .localPartition(std::vector<std::string>{})
      .finalAggregation();
  if (!groupBy.empty() && orderByGroup) {
    std::vector<std::string> ob;
    for (const auto& g : groupBy) {
      ob.push_back(g);
    }
    pb.orderBy(ob, false);
  }
  context.plan = pb.planNode();
  context.planName = "SSB Q" + std::to_string(queryId - 100);
  if (resident_) {
    // getQueryPlan applies makeResident to the returned plan.
  }
  return context;
}

TpchPlan TpchQueryBuilder::buildQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 2:
      return getQ2Plan();
    case 3:
      return getQ3Plan();
    case 4:
      return getQ4Plan();
    case 5:
      return getQ5Plan();
    case 6:
      return getQ6Plan();
    case 7:
      return getQ7Plan();
    case 8:
      return getQ8Plan();
    case 9:
      return getQ9Plan();
    case 10:
      return getQ10Plan();
    case 11:
      return getQ11Plan();
    case 12:
      return getQ12Plan();
    case 13:
      return getQ13Plan();
    case 14:
      return getQ14Plan();
    case 15:
      return getQ15Plan();
    case 16:
      return getQ16Plan();
    case 17:
      return getQ17Plan();
    case 18:
      return getQ18Plan();
    case 19:
      return getQ19Plan();
    case 20:
      return getQ20Plan();
    case 21:
      return getQ21Plan();
    case 22:
      return getQ22Plan();
    // VLDB experiment queries
    case 23:
      return getScan1PctPlan();
    case 24:
      return getScan3PctPlan();
    case 25:
      return getScan10PctPlan();
    case 26:
      return getScan30PctPlan();
    case 27:
      return getJoinLOPlan();
    case 28:
      return getJoinLPPlan();
    case 29:
      return getJoinLSPlan();
    case 30:
      return getJoinOCPlan();
    case 32:
      return getCaseAPlan();
    // Hybrid single-operator benchmarks (synthetic tables R/S)
    case 31:
      return getQ31Plan();
    case 40:
      return getQ40Plan();
    case 41:
      return getQ41Plan();
    case 43:
      return getQ43Plan();
    default:
      VELOX_NYI("TPC-H query {} is not supported yet", queryId);
  }
}

TpchPlan TpchQueryBuilder::getQ1Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_returnflag",
      "l_linestatus",
      "l_quantity",
      "l_extendedprice",
      "l_discount",
      "l_tax",
      "l_shipdate"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  // shipdate <= '1998-09-02'
  const auto shipDate = "l_shipdate";
  auto filter = formatDateFilter(shipDate, selectedRowType, "", "'1998-09-03'");

  core::PlanNodeId lineitemPlanNodeId;

  auto plan =
      PlanBuilder(pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, selectedRowType, fileColumnNames, {filter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project(
              {"l_returnflag",
               "l_linestatus",
               "l_quantity",
               "l_extendedprice",
               "l_extendedprice * (1.0 - l_discount) AS l_sum_disc_price",
               "l_extendedprice * (1.0 - l_discount) * (1.0 + l_tax) AS l_sum_charge",
               "l_discount"})
          .partialAggregation(
              {"l_returnflag", "l_linestatus"},
              {"sum(l_quantity)",
               "sum(l_extendedprice)",
               "sum(l_sum_disc_price)",
               "sum(l_sum_charge)",
               "avg(l_quantity)",
               "avg(l_extendedprice)",
               "avg(l_discount)",
               "count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_returnflag", "l_linestatus"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ2Plan() const {
  std::vector<std::string> supplierColumnsSubQuery = {
      "s_suppkey", "s_nationkey"};
  std::vector<std::string> nationColumnsSubQuery = {
      "n_nationkey", "n_regionkey"};
  std::vector<std::string> supplierColumns = {
      "s_acctbal",
      "s_name",
      "s_address",
      "s_phone",
      "s_comment",
      "s_suppkey",
      "s_nationkey"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_mfgr", "p_size", "p_type"};
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto supplierSelectedRowTypeSubQuery =
      getRowType(kSupplier, supplierColumnsSubQuery);
  const auto& supplierFileColumnsSubQuery = getFileColumnNames(kSupplier);
  auto nationSelectedRowTypeSubQuery =
      getRowType(kNation, nationColumnsSubQuery);
  const auto& nationFileColumnsSubQuery = getFileColumnNames(kNation);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  const std::string regionNameFilter = "r_name = 'EUROPE'";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanIdSubQuery;
  core::PlanNodeId partsuppScanIdSubQuery;
  core::PlanNodeId nationScanIdSubQuery;
  core::PlanNodeId regionScanIdSubQuery;
  core::PlanNodeId partScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId nationScanId;
  core::PlanNodeId regionScanId;

  auto regionSubQuery = PlanBuilder(planNodeIdGenerator)
                            .filtersAsNode(filtersAsNode_)
                            .tableScan(
                                kRegion,
                                regionSelectedRowType,
                                regionFileColumns,
                                {regionNameFilter})
                            .captureScanNodeId(regionScanIdSubQuery)
                            .planNode();

  auto nationJoinRegionSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowTypeSubQuery, nationFileColumnsSubQuery)
          .captureScanNodeId(nationScanIdSubQuery)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              regionSubQuery,
              "",
              {"n_nationkey"})
          .planNode();

  auto supplierJoinNationJoinRegionSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kSupplier,
              supplierSelectedRowTypeSubQuery,
              supplierFileColumnsSubQuery)
          .captureScanNodeId(supplierScanIdSubQuery)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegionSubQuery,
              "",
              {"s_suppkey"})
          .planNode();

  auto part = PlanBuilder(planNodeIdGenerator)
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {},
                      encLikeSuffix("p_type", "BRASS"))
                  .captureScanNodeId(partScanId)
                  .filter("p_size = 15")
                  .planNode();

  auto region = PlanBuilder(planNodeIdGenerator)
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter})
                    .captureScanNodeId(regionScanId)
                    .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              mergeColumnNames(supplierColumns, {"s_suppkey", "n_name"}))
          .planNode();

  auto partsuppJoinPartJoinSupplierJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_suppkey", "ps_supplycost", "p_partkey", "p_mfgr"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationJoinRegion,
              "",
              mergeColumnNames(
                  supplierColumns,
                  {"ps_supplycost", "p_partkey", "p_mfgr", "n_name"}))
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanIdSubQuery)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationJoinRegionSubQuery,
              "",
              {"ps_supplycost", "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"min(ps_supplycost) AS min_supplycost"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              partsuppJoinPartJoinSupplierJoinNationJoinRegion,
              "ps_supplycost = min_supplycost",
              mergeColumnNames(
                  supplierColumns, {"p_partkey", "p_mfgr", "n_name"}))
          .orderBy({"s_acctbal DESC", "n_name", "s_name", "p_partkey"}, false)
          .project(
              {"s_acctbal",
               "s_name",
               "n_name",
               "p_partkey",
               "p_mfgr",
               "s_address",
               "s_phone",
               "s_comment"})
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[supplierScanIdSubQuery] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanIdSubQuery] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanIdSubQuery] = getTableFilePaths(kNation);
  context.dataFiles[regionScanIdSubQuery] = getTableFilePaths(kRegion);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ3Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_orderkey", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_shippriority", "o_custkey", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_mktsegment"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  const auto orderDate = "o_orderdate";
  const auto shipDate = "l_shipdate";
  auto orderDateFilter =
      formatDateFilter(orderDate, ordersSelectedRowType, "", "'1995-03-15'");
  auto shipDateFilter =
      formatDateFilter(shipDate, lineitemSelectedRowType, "'1995-03-15'", "");
  auto customerFilter = "c_mktsegment = 'BUILDING'";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;
  core::PlanNodeId customerPlanNodeId;

  auto customers = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kCustomer,
                           customerSelectedRowType,
                           customerFileColumns,
                           {customerFilter})
                       .captureScanNodeId(customerPlanNodeId)
                       .planNode();

  auto custkeyJoinNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter})
          .captureScanNodeId(ordersPlanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"o_orderdate", "o_shippriority", "o_orderkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              custkeyJoinNode,
              "",
              {"l_orderkey", "o_orderdate", "o_shippriority", "part_revenue"})
          .partialAggregation(
              {"l_orderkey", "o_orderdate", "o_shippriority"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"l_orderkey", "revenue", "o_orderdate", "o_shippriority"})
          .orderBy({"revenue DESC", "o_orderdate"}, false)
          .limit(0, 10, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ4Plan() const {
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderpriority", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_commitdate", "l_receiptdate"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  std::string orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1993-07-01'", "'1993-09-30'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter})
                    .captureScanNodeId(ordersPlanNodeId)
                    .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {},
              "l_commitdate < l_receiptdate")
          .captureScanNodeId(lineitemPlanNodeId)
          .partialAggregation({"l_orderkey"}, {}, {})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"o_orderpriority"},
              core::JoinType::kRightSemiFilter)
          .partialAggregation({"o_orderpriority"}, {"count(0) AS order_count"})
          .finalAggregation()
          .orderBy({"o_orderpriority"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ5Plan() const {
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_custkey", "o_orderkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> nationColumns = {
      "n_nationkey", "n_name", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_regionkey", "r_name"};

  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  std::string regionNameFilter = "r_name = 'ASIA'";
  const auto orderDate = "o_orderdate";
  std::string orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1994-01-01'", "'1994-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId regionScanNodeId;

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        regionSelectedRowType,
                        regionFileColumns,
                        {regionNameFilter})
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter})
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto customer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"},
              {"r_regionkey"},
              region,
              "",
              {"n_nationkey", "n_name"})
          .planNode();

  auto supplierJoinNationRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"s_suppkey", "n_name", "s_nationkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
               "l_orderkey",
               "l_suppkey"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNationRegion,
              "",
              {"n_name", "part_revenue", "s_nationkey", "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"n_name", "part_revenue", "s_nationkey", "o_custkey"})
          .hashJoin(
              {"s_nationkey", "o_custkey"},
              {"c_nationkey", "c_custkey"},
              customer,
              "",
              {"n_name", "part_revenue"})
          .partialAggregation({"n_name"}, {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"revenue DESC"}, false)
          .project({"n_name", "revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ6Plan() const {
  std::vector<std::string> selectedColumns = {
      "l_shipdate", "l_extendedprice", "l_quantity", "l_discount"};

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  const auto shipDate = "l_shipdate";
  auto shipDateFilter = formatDateFilter(
      shipDate, selectedRowType, "'1994-01-01'", "'1994-12-31'");

  core::PlanNodeId lineitemPlanNodeId;
  auto plan = PlanBuilder(pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kLineitem,
                      selectedRowType,
                      fileColumnNames,
                      {shipDateFilter,
                       "l_discount between 0.05 and 0.07",
                       "l_quantity < 24.0"})
                  .captureScanNodeId(lineitemPlanNodeId)
                  .project({"l_extendedprice * l_discount"})
                  .partialAggregation({}, {"sum(p0)"})
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .planNode();
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ7Plan() const {
  std::vector<std::string> supplierColumns = {"s_nationkey", "s_suppkey"};
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_suppkey", "l_orderkey", "l_discount", "l_extendedprice"};
  std::vector<std::string> ordersColumns = {"o_custkey", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey", "c_nationkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  const std::string nationFilter = encIn("n_name", {"FRANCE", "GERMANY"});
  auto shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1995-01-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId suppNationScanNodeId;
  core::PlanNodeId custNationScanNodeId;

  auto custNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowType, nationFileColumns, {nationFilter})
          .captureScanNodeId(custNationScanNodeId)
          .planNode();

  auto customerJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              custNation,
              "",
              {"n_name", "c_custkey"})
          .project({"n_name as cust_nation", "c_custkey"})
          .planNode();

  auto ordersJoinCustomer =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNation,
              "",
              {"cust_nation", "o_orderkey"})
          .planNode();

  auto suppNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowType, nationFileColumns, {nationFilter})
          .captureScanNodeId(suppNationScanNodeId)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              suppNation,
              "",
              {"n_name", "s_suppkey"})
          .project({"n_name as supp_nation", "s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"supp_nation",
               "l_extendedprice",
               "l_discount",
               "l_shipdate",
               "l_orderkey"})
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomer,
              "(((cust_nation = " + encLit("n_name", "FRANCE") +
                  ") AND (supp_nation = " + encLit("n_name", "GERMANY") +
                  ")) OR ((cust_nation = " + encLit("n_name", "GERMANY") +
                  ") AND (supp_nation = " + encLit("n_name", "FRANCE") + ")))",
              {"supp_nation",
               "cust_nation",
               "l_extendedprice",
               "l_discount",
               "l_shipdate"})
          .project(
              {"cust_nation",
               "supp_nation",
               "l_extendedprice * (1.0 - l_discount) as part_revenue",
               "year(l_shipdate) as l_year"})
          .partialAggregation(
              {"supp_nation", "cust_nation", "l_year"},
              {"sum(part_revenue) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supp_nation", "cust_nation", "l_year"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[suppNationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[custNationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ8Plan() const {
  std::vector<std::string> partColumns = {"p_partkey", "p_type"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_orderkey", "l_partkey", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderkey", "o_custkey"};
  std::vector<std::string> customerColumns = {"c_nationkey", "c_custkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_regionkey"};
  std::vector<std::string> nationColumnsWithName = {
      "n_name", "n_nationkey", "n_regionkey"};
  std::vector<std::string> regionColumns = {"r_name", "r_regionkey"};

  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  const auto nationSelectedRowTypeWithName =
      getRowType(kNation, nationColumnsWithName);
  const auto& nationFileColumnsWithName = getFileColumnNames(kNation);
  const auto regionSelectedRowType = getRowType(kRegion, regionColumns);
  const auto& regionFileColumns = getFileColumnNames(kRegion);

  const auto orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1995-01-01'", "'1996-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId nationScanNodeIdWithName;
  core::PlanNodeId regionScanNodeId;

  auto nationWithName =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationSelectedRowTypeWithName, nationFileColumnsWithName)
          .captureScanNodeId(nationScanNodeIdWithName)
          .planNode();

  auto region = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kRegion,
                        regionSelectedRowType,
                        regionFileColumns,
                        {"r_name = 'AMERICA'"})
                    .captureScanNodeId(regionScanNodeId)
                    .planNode();

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {"p_type = " + encLit("p_type", "ECONOMY ANODIZED STEEL")})
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto nationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .hashJoin(
              {"n_regionkey"}, {"r_regionkey"}, region, "", {"n_nationkey"})
          .planNode();

  auto customerJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nationJoinRegion,
              "",
              {"c_custkey"})
          .planNode();

  auto ordersJoinCustomerJoinNationJoinRegion =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {orderDateFilter})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customerJoinNationJoinRegion,
              "",
              {"o_orderkey", "o_orderdate"})
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationWithName,
              "",
              {"s_suppkey", "n_name"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              ordersJoinCustomerJoinNationJoinRegion,
              "",
              {"l_partkey",
               "l_suppkey",
               "o_orderdate",
               "l_extendedprice",
               "l_discount"})
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"n_name",
               "o_orderdate",
               "l_partkey",
               "l_extendedprice",
               "l_discount"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"n_name", "o_orderdate", "l_extendedprice", "l_discount"})
          .project(
              {"l_extendedprice * (1.0 - l_discount) as volume",
               "n_name",
               "o_orderdate"})
          .project(
              {"volume",
               "(CASE WHEN n_name = " + encLit("n_name", "BRAZIL") +
                   " THEN volume ELSE 0.0 END) as brazil_volume",
               "year(o_orderdate) AS o_year"})
          .partialAggregation(
              {"o_year"},
              {"sum(brazil_volume) as volume_brazil",
               "sum(volume) as volume_all"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"o_year"}, false)
          .project({"o_year", "(volume_brazil / volume_all) as mkt_share"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[nationScanNodeIdWithName] = getTableFilePaths(kNation);
  context.dataFiles[regionScanNodeId] = getTableFilePaths(kRegion);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ9Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_suppkey",
      "l_partkey",
      "l_discount",
      "l_extendedprice",
      "l_orderkey",
      "l_quantity"};
  std::vector<std::string> partColumns = {"p_name", "p_partkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderdate"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partsuppScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId nationScanNodeId;

  const std::vector<std::string> lineitemCommonColumns = {
      "l_extendedprice", "l_discount", "l_quantity"};

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {},
                      "p_name like '%green%'")
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto supplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .planNode();

  auto nation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .planNode();

  auto lineitemJoinPartJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .hashJoin({"l_partkey"}, {"p_partkey"}, part, "", lineitemColumns)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              mergeColumnNames(lineitemColumns, {"s_nationkey"}))
          .planNode();

  auto partsuppJoinLineitemJoinPartJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanNodeId)
          .hashJoin(
              {"ps_partkey", "ps_suppkey"},
              {"l_partkey", "l_suppkey"},
              lineitemJoinPartJoinSupplier,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"l_orderkey", "s_nationkey", "ps_supplycost"}))
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              partsuppJoinLineitemJoinPartJoinSupplier,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"s_nationkey", "ps_supplycost", "o_orderdate"}))
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              mergeColumnNames(
                  lineitemCommonColumns,
                  {"ps_supplycost", "o_orderdate", "n_name"}))
          .project(
              {"n_name AS nation",
               "year(o_orderdate) AS o_year",
               "l_extendedprice * (1.0 - l_discount) - ps_supplycost * l_quantity AS amount"})
          .partialAggregation(
              {"nation", "o_year"}, {"sum(amount) AS sum_profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"nation", "o_year DESC"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ10Plan() const {
  std::vector<std::string> customerColumns = {
      "c_nationkey",
      "c_custkey",
      "c_acctbal",
      "c_name",
      "c_address",
      "c_phone",
      "c_comment"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"};
  std::vector<std::string> ordersColumns = {
      "o_orderdate", "o_orderkey", "o_custkey"};

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);
  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto lineitemReturnFlagFilter =
      "l_returnflag = " + encLit("l_returnflag", "R");
  const auto orderDate = "o_orderdate";
  auto orderDateFilter = formatDateFilter(
      orderDate, ordersSelectedRowType, "'1993-10-01'", "'1993-12-31'");

  const std::vector<std::string> customerOutputColumns = {
      "c_name", "c_acctbal", "c_phone", "c_address", "c_custkey", "c_comment"};

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId nationScanNodeId;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto nation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kNation, nationSelectedRowType, nationFileColumns)
          .captureScanNodeId(nationScanNodeId)
          .planNode();

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter})
                    .captureScanNodeId(ordersScanNodeId)
                    .planNode();

  auto partialPlan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              mergeColumnNames(
                  customerOutputColumns, {"c_nationkey", "o_orderkey"}))
          .hashJoin(
              {"c_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              mergeColumnNames(customerOutputColumns, {"n_name", "o_orderkey"}))
          .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kLineitem,
                      lineitemSelectedRowType,
                      lineitemFileColumns,
                      {lineitemReturnFlagFilter})
                  .captureScanNodeId(lineitemScanNodeId)
                  .project(
                      {"l_extendedprice * (1.0 - l_discount) AS part_revenue",
                       "l_orderkey"})
                  .hashJoin(
                      {"l_orderkey"},
                      {"o_orderkey"},
                      partialPlan,
                      "",
                      mergeColumnNames(
                          customerOutputColumns, {"part_revenue", "n_name"}))
                  .partialAggregation(
                      {"c_custkey",
                       "c_name",
                       "c_acctbal",
                       "n_name",
                       "c_address",
                       "c_phone",
                       "c_comment"},
                      {"sum(part_revenue) as revenue"})
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .orderBy({"revenue DESC"}, false)
                  .project(
                      {"c_custkey",
                       "c_name",
                       "revenue",
                       "c_acctbal",
                       "n_name",
                       "c_address",
                       "c_phone",
                       "c_comment"})
                  .limit(0, 20, false)
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ11Plan() const {
  // Column definitions.
  std::vector<std::string> partsuppColumns = {
      "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"};
  std::vector<std::string> partsuppColumnsSubQuery = {
      "ps_availqty", "ps_suppkey", "ps_supplycost"};
  std::vector<std::string> supplierColumns = {"s_suppkey", "s_nationkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  // Row type definitions.
  auto partsuppRowType = getRowType(kPartsupp, partsuppColumns);
  auto partsuppRowTypeSubQuery = getRowType(kPartsupp, partsuppColumnsSubQuery);
  auto supplierRowType = getRowType(kSupplier, supplierColumns);
  auto nationRowType = getRowType(kNation, nationColumns);

  // File columns.
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  const auto& partsuppFileColumnsSubQuery = getFileColumnNames(kPartsupp);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  // Filter for nation name.
  const std::string nationNameFilter =
      "n_name = " + encLit("n_name", "GERMANY");

  // Plan node ID generator.
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  // Subquery plan nodes.
  core::PlanNodeId partsuppScanIdSubQuery;
  core::PlanNodeId supplierScanIdSubQuery;
  core::PlanNodeId nationScanIdSubQuery;

  // Main query plan nodes.
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId nationScanId;

  // Subquery plan to calculate filter cost.
  auto nationSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationRowType, nationFileColumns, {nationNameFilter})
          .captureScanNodeId(nationScanIdSubQuery)
          .planNode();

  auto supplierJoinNationSubQuery =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanIdSubQuery)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nationSubQuery,
              "",
              {"s_suppkey"})
          .planNode();

  auto subQueryPlan =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kPartsupp, partsuppRowTypeSubQuery, partsuppFileColumnsSubQuery)
          .captureScanNodeId(partsuppScanIdSubQuery)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNationSubQuery,
              "",
              {"ps_availqty", "ps_suppkey", "ps_supplycost"})
          .project({"ps_supplycost * ps_availqty AS product_cost_qty"})
          .partialAggregation({}, {"sum(product_cost_qty) AS sum_cost_qty"})
          .localPartition({})
          .finalAggregation()
          .project({"sum_cost_qty * 0.0001 AS filter_cost_qty"})
          .planNode();

  // Query plan.
  auto nation =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kNation, nationRowType, nationFileColumns, {nationNameFilter})
          .captureScanNodeId(nationScanId)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin({"s_nationkey"}, {"n_nationkey"}, nation, "", {"s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator)
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"ps_partkey", "ps_availqty", "ps_supplycost"})
          .project(
              {"ps_supplycost * ps_availqty AS product_cost_qty", "ps_partkey"})
          .partialAggregation(
              {"ps_partkey"}, {"sum(product_cost_qty) AS value"})
          .localPartition({"ps_partkey"})
          .finalAggregation()
          .nestedLoopJoin(
              subQueryPlan, {"ps_partkey", "value", "filter_cost_qty"})
          .filter("value > filter_cost_qty")
          .project({"ps_partkey", "value"})
          .orderBy({"value DESC"}, false)
          .planNode();

  // Create context and return.
  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partsuppScanIdSubQuery] = getTableFilePaths(kPartsupp);
  context.dataFiles[supplierScanIdSubQuery] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanIdSubQuery] = getTableFilePaths(kNation);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ12Plan() const {
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderpriority"};
  std::vector<std::string> lineitemColumns = {
      "l_receiptdate",
      "l_orderkey",
      "l_commitdate",
      "l_shipmode",
      "l_shipdate"};

  auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  const std::string receiptDateFilter = formatDateFilter(
      "l_receiptdate", lineitemSelectedRowType, "'1994-01-01'", "'1994-12-31'");
  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "", "'1995-01-01'");
  const std::string commitDateFilter = formatDateFilter(
      "l_commitdate", lineitemSelectedRowType, "", "'1995-01-01'");

  auto lineitem = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kLineitem,
                          lineitemSelectedRowType,
                          lineitemFileColumns,
                          {receiptDateFilter,
                           encIn("l_shipmode", {"MAIL", "SHIP"}),
                           shipDateFilter,
                           commitDateFilter},
                          "l_commitdate < l_receiptdate")
                      .captureScanNodeId(lineitemScanNodeId)
                      .filter("l_shipdate < l_commitdate")
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns, {})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              lineitem,
              "",
              {"l_shipmode", "o_orderpriority"})
          .project(
              {"l_shipmode",
               "(CASE WHEN o_orderpriority = " + encLit("o_orderpriority", "1-URGENT") +
                   " OR o_orderpriority = " + encLit("o_orderpriority", "2-HIGH") +
                   " THEN 1 ELSE 0 END) AS high_line_count_partial",
               "(CASE WHEN o_orderpriority <> " + encLit("o_orderpriority", "1-URGENT") +
                   " AND o_orderpriority <> " + encLit("o_orderpriority", "2-HIGH") +
                   " THEN 1 ELSE 0 END) AS low_line_count_partial"})
          .partialAggregation(
              {"l_shipmode"},
              {"sum(high_line_count_partial) as high_line_count",
               "sum(low_line_count_partial) as low_line_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"l_shipmode"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ13Plan() const {
  std::vector<std::string> ordersColumns = {
      "o_custkey", "o_comment", "o_orderkey"};
  std::vector<std::string> customerColumns = {"c_custkey"};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;

  auto customers =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerSelectedRowType, customerFileColumns)
          .captureScanNodeId(customerScanNodeId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {},
              "o_comment not like '%special%requests%'")
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customers,
              "",
              {"c_custkey", "o_orderkey"},
              core::JoinType::kRight)
          .partialAggregation({"c_custkey"}, {"count(o_orderkey) as c_count"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .singleAggregation({"c_count"}, {"count(0) as custdist"})
          .orderBy({"custdist DESC", "c_count DESC"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ14Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey", "l_extendedprice", "l_discount", "l_shipdate"};
  std::vector<std::string> partColumns = {"p_partkey", "p_type"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  const std::string shipDate = "l_shipdate";
  const std::string shipDateFilter = formatDateFilter(
      shipDate, lineitemSelectedRowType, "'1995-09-01'", "'1995-09-30'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partScanNodeId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kPart, partSelectedRowType, partFileColumns)
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {},
              shipDateFilter)
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_extendedprice * (1.0 - l_discount) as part_revenue",
               "l_shipdate",
               "l_partkey"})
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"part_revenue", "p_type"})
          .project(
              {"(CASE WHEN " + encLikePrefix("p_type", "PROMO") +
                   " THEN part_revenue ELSE 0.0 END) as filter_revenue",
               "part_revenue"})
          .partialAggregation(
              {},
              {"sum(part_revenue) as total_revenue",
               "sum(filter_revenue) as total_promo_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"100.00 * total_promo_revenue/total_revenue as promo_revenue"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ15Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", "s_name", "s_address", "s_phone"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1996-01-01'", "'1996-03-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeIdSubQuery;
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId supplierScanNodeId;

  auto maxRevenue =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanNodeId)
          .project(
              {"l_suppkey",
               "l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .partialAggregation(
              {"l_suppkey"}, {"sum(part_revenue) as total_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .singleAggregation({}, {"max(total_revenue) as max_revenue"})
          .planNode();

  auto supplierWithMaxRevenue =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanNodeIdSubQuery)
          .project(
              {"l_suppkey as supplier_no",
               "l_extendedprice * (1.0 - l_discount) as part_revenue"})
          .partialAggregation(
              {"supplier_no"}, {"sum(part_revenue) as total_revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .hashJoin(
              {"total_revenue"},
              {"max_revenue"},
              maxRevenue,
              "",
              {"supplier_no", "total_revenue"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_suppkey"},
              {"supplier_no"},
              supplierWithMaxRevenue,
              "",
              {"s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"})
          .orderBy({"s_suppkey"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeIdSubQuery] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ16Plan() const {
  std::vector<std::string> partColumns = {
      "p_brand", "p_type", "p_size", "p_partkey"};
  std::vector<std::string> supplierColumns = {
      "s_suppkey", strCol("s_comment")};
  std::vector<std::string> partsuppColumns = {"ps_partkey", "ps_suppkey"};

  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  const auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  const auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId partScanNodeId;
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId partsuppScanNodeId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {"p_size in (49, 14, 23, 45, 19, 3, 36, 9)"},
                      "NOT " + encLikePrefix("p_type", "MEDIUM POLISHED"))
                  .captureScanNodeId(partScanNodeId)
                  // Neq is unsupported as a tableScan subfield filter for
                  // Parquet source.
                  .filter("p_brand <> " + encLit("p_brand", "Brand#45"))
                  .planNode();

  auto supplier = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kSupplier,
                          supplierSelectedRowType,
                          supplierFileColumns,
                          {},
                          strCol("s_comment") +
                              " LIKE '%Customer%Complaints%'")
                      .captureScanNodeId(supplierScanNodeId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanNodeId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              {"ps_suppkey", "p_brand", "p_type", "p_size"},
              core::JoinType::kAnti,
              true /*nullAware*/)
          // Empty aggregate is used here to get the distinct count of
          // ps_suppkey.
          // approx_distinct could be used instead for getting the count of
          // distinct ps_suppkey but since approx_distinct is non deterministic
          // and the standard error can not be set to 0, it is not used here.
          .partialAggregation({"p_brand", "p_type", "p_size", "ps_suppkey"}, {})
          .localPartition({"p_brand", "p_type", "p_size", "ps_suppkey"})
          .finalAggregation()
          .partialAggregation(
              {"p_brand", "p_type", "p_size"},
              {"count(ps_suppkey) as supplier_cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"supplier_cnt DESC", "p_brand", "p_type", "p_size"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanNodeId] = getTableFilePaths(kPartsupp);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ17Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey", "l_extendedprice", "l_quantity"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand", "p_container"};

  auto lineitemRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId lineitemAggScanId;
  core::PlanNodeId partScanId;
  core::PlanNodeId partAggScanId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partRowType,
                      partFileColumns,
                      {"p_brand = " + encLit("p_brand", "Brand#23"),
                       "p_container = " + encLit("p_container", "MED BOX")})
                  .captureScanNodeId(partScanId)
                  .planNode();

  auto partAgg = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .filtersAsNode(filtersAsNode_)
                     .tableScan(kPart, partRowType, partFileColumns, {})
                     .captureScanNodeId(partAggScanId)
                     .planNode();

  auto lineitemJoinPart =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"l_quantity", "p_partkey", "l_extendedprice"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemAggScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partAgg,
              "",
              {"l_partkey", "l_quantity"})
          .partialAggregation({"l_partkey"}, {"avg(l_quantity) as avg_"})
          .localPartition({"l_partkey"})
          .finalAggregation()
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              lineitemJoinPart,
              "l_quantity < 0.2 * avg_",
              {"l_extendedprice"})
          .partialAggregation({}, {"sum(l_extendedprice) as partial_sum"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"(partial_sum / 7.0) as avg_yearly"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitemAggScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[partAggScanId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ18Plan() const {
  std::vector<std::string> lineitemColumns = {"l_orderkey", "l_quantity"};
  std::vector<std::string> ordersColumns = {
      "o_orderkey", "o_custkey", "o_orderdate", "o_totalprice"};
  std::vector<std::string> customerColumns = {"c_name", "c_custkey"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId lineitemScanNodeId;

  auto bigOrders =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitemSelectedRowType, lineitemFileColumns)
          .captureScanNodeId(lineitemScanNodeId)
          .partialAggregation({"l_orderkey"}, {"sum(l_quantity) AS quantity"})
          .localPartition({"l_orderkey"})
          .finalAggregation()
          .filter("quantity > 300.0")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey"},
              bigOrders,
              "",
              {"o_orderkey",
               "o_custkey",
               "o_orderdate",
               "o_totalprice",
               "l_orderkey",
               "quantity"})
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kCustomer, customerSelectedRowType, customerFileColumns)
                  .captureScanNodeId(customerScanNodeId)
                  .planNode(),
              "",
              {"c_name",
               "c_custkey",
               "o_orderkey",
               "o_orderdate",
               "o_totalprice",
               "quantity"})
          .localPartition(std::vector<std::string>{})
          .orderBy({"o_totalprice DESC", "o_orderdate"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ19Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_partkey",
      "l_shipmode",
      "l_shipinstruct",
      "l_extendedprice",
      "l_discount",
      "l_quantity"};
  std::vector<std::string> partColumns = {
      "p_partkey", "p_brand", "p_container", "p_size"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanNodeId;
  core::PlanNodeId partScanNodeId;

  const std::string shipModeFilter = encIn("l_shipmode", {"AIR", "AIR REG"});
  const std::string shipInstructFilter =
      "(l_shipinstruct = 'DELIVER IN PERSON')";
  const std::string joinFilterExpr =
      "     ((p_brand = " + encLit("p_brand", "Brand#12") + ")"
      "     AND (l_quantity between 1.0 and 11.0)"
      "     AND (" + encIn("p_container", {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}) + ")"
      "     AND (p_size BETWEEN 1 AND 5))"
      " OR  ((p_brand = " + encLit("p_brand", "Brand#23") + ")"
      "     AND (" + encIn("p_container", {"MED BAG", "MED BOX", "MED PKG", "MED PACK"}) + ")"
      "     AND (l_quantity between 10.0 and 20.0)"
      "     AND (p_size BETWEEN 1 AND 10))"
      " OR  ((p_brand = " + encLit("p_brand", "Brand#34") + ")"
      "     AND (" + encIn("p_container", {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}) + ")"
      "     AND (l_quantity between 20.0 and 30.0)"
      "     AND (p_size BETWEEN 1 AND 15))";

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kPart, partSelectedRowType, partFileColumns)
                  .captureScanNodeId(partScanNodeId)
                  .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kLineitem,
                      lineitemSelectedRowType,
                      lineitemFileColumns,
                      {shipModeFilter, shipInstructFilter})
                  .captureScanNodeId(lineitemScanNodeId)
                  .project(
                      {"l_extendedprice * (1.0 - l_discount) as part_revenue",
                       "l_shipmode",
                       "l_shipinstruct",
                       "l_partkey",
                       "l_quantity"})
                  .hashJoin(
                      {"l_partkey"},
                      {"p_partkey"},
                      part,
                      joinFilterExpr,
                      {"part_revenue"})
                  .partialAggregation({}, {"sum(part_revenue) as revenue"})
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ20Plan() const {
  std::vector<std::string> lineitemColumns = {
      "l_shipdate", "l_suppkey", "l_partkey", "l_quantity"};
  std::vector<std::string> partColumns = {"p_partkey", "p_name"};
  std::vector<std::string> supplierColumns = {
      "s_nationkey", "s_address", "s_name", "s_suppkey"};
  std::vector<std::string> partsuppColumns = {
      "ps_availqty", "ps_partkey", "ps_suppkey"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);
  auto supplierSelectedRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto partsuppSelectedRowType = getRowType(kPartsupp, partsuppColumns);
  const auto& partsuppFileColumns = getFileColumnNames(kPartsupp);
  auto nationSelectedRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  const std::string shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemSelectedRowType, "'1994-01-01'", "'1994-12-31'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId partScanId;
  core::PlanNodeId partAggScanId;
  core::PlanNodeId supplierScanId;
  core::PlanNodeId partsuppScanId;
  core::PlanNodeId nationScanId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {},
                      "p_name like 'forest%'")
                  .captureScanNodeId(partScanId)
                  .planNode();

  auto partAgg = PlanBuilder(planNodeIdGenerator, pool_.get())
                     .filtersAsNode(filtersAsNode_)
                     .tableScan(
                         kPart,
                         partSelectedRowType,
                         partFileColumns,
                         {},
                         "p_name like 'forest%'")
                     .captureScanNodeId(partAggScanId)
                     .planNode();

  auto nation = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        nationSelectedRowType,
                        nationFileColumns,
                        {"n_name = " + encLit("n_name", "CANADA")})
                    .captureScanNodeId(nationScanId)
                    .planNode();

  auto partsuppJoinPart =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPartsupp, partsuppSelectedRowType, partsuppFileColumns)
          .captureScanNodeId(partsuppScanId)
          .hashJoin(
              {"ps_partkey"},
              {"p_partkey"},
              part,
              "",
              {"ps_partkey", "ps_suppkey", "ps_availqty"},
              core::JoinType::kLeftSemiFilter)
          .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierSelectedRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_name", "s_address", "s_suppkey"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {shipDateFilter})
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              partAgg,
              "",
              {"l_partkey", "l_suppkey", "l_quantity"})
          .partialAggregation(
              {"l_partkey", "l_suppkey"}, {"sum(l_quantity) AS sum_qty"})
          .localPartition({"l_partkey", "l_suppkey"})
          .finalAggregation()
          .project({"l_partkey", "l_suppkey", "0.5 * sum_qty AS filter_qty"})
          .hashJoin(
              {"l_partkey", "l_suppkey"},
              {"ps_partkey", "ps_suppkey"},
              partsuppJoinPart,
              "ps_availqty > filter_qty",
              {"ps_suppkey"})
          .hashJoin(
              {"ps_suppkey"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"s_name", "s_address"},
              core::JoinType::kRightSemiFilter)
          .orderBy({"s_name"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[partScanId] = getTableFilePaths(kPart);
  context.dataFiles[partAggScanId] = getTableFilePaths(kPart);
  context.dataFiles[supplierScanId] = getTableFilePaths(kSupplier);
  context.dataFiles[partsuppScanId] = getTableFilePaths(kPartsupp);
  context.dataFiles[nationScanId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ21Plan() const {
  std::vector<std::string> supplierColumns = {
      "s_nationkey", "s_name", "s_suppkey"};
  std::vector<std::string> lineitemColumnsWithDates = {
      "l_suppkey", "l_commitdate", "l_orderkey", "l_receiptdate"};
  std::vector<std::string> lineitemColumns = {"l_suppkey", "l_orderkey"};
  std::vector<std::string> ordersColumns = {"o_orderkey", "o_orderstatus"};
  std::vector<std::string> nationColumns = {"n_nationkey", "n_name"};

  auto supplierRowType = getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);
  auto lineitem1RowType = getRowType(kLineitem, lineitemColumnsWithDates);
  const auto& lineitem1FileColumns = getFileColumnNames(kLineitem);
  auto lineitem2RowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitem2FileColumns = getFileColumnNames(kLineitem);
  auto lineitem3RowType = getRowType(kLineitem, lineitemColumnsWithDates);
  const auto& lineitem3FileColumns = getFileColumnNames(kLineitem);
  auto ordersRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  auto nationRowType = getRowType(kNation, nationColumns);
  const auto& nationFileColumns = getFileColumnNames(kNation);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId supplierScanNodeId;
  core::PlanNodeId lineitem1ScanNodeId;
  core::PlanNodeId lineitem2ScanNodeId;
  core::PlanNodeId lineitem3ScanNodeId;
  core::PlanNodeId ordersScanNodeId;
  core::PlanNodeId nationScanNodeId;
  const std::string receiptCommitFilter = "l_receiptdate > l_commitdate";

  auto lineitem3 =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitem3RowType,
              lineitem3FileColumns,
              {},
              receiptCommitFilter)
          .captureScanNodeId(lineitem3ScanNodeId)
          .project({"l_orderkey as l_orderkey_3", "l_suppkey as l_suppkey_3"})
          .planNode();

  auto nation = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kNation,
                        nationRowType,
                        nationFileColumns,
                        {"n_name = " + encLit("n_name", "SAUDI ARABIA")})
                    .captureScanNodeId(nationScanNodeId)
                    .planNode();

  auto supplierJoinNation =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kSupplier, supplierRowType, supplierFileColumns)
          .captureScanNodeId(supplierScanNodeId)
          .hashJoin(
              {"s_nationkey"},
              {"n_nationkey"},
              nation,
              "",
              {"s_suppkey", "s_name", "s_nationkey"})
          .planNode();

  auto lineitemJoinSupplier =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitem1RowType,
              lineitem1FileColumns,
              {},
              receiptCommitFilter)
          .captureScanNodeId(lineitem1ScanNodeId)
          .project({"l_orderkey as l_orderkey_1", "l_suppkey as l_suppkey_1"})
          .hashJoin(
              {"l_suppkey_1"},
              {"s_suppkey"},
              supplierJoinNation,
              "",
              {"l_orderkey_1", "s_nationkey", "l_suppkey_1", "s_name"})
          .planNode();

  auto ordersJoinLineitem1 =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersRowType,
              ordersFileColumns,
              {"o_orderstatus = 'F'"})
          .captureScanNodeId(ordersScanNodeId)
          .hashJoin(
              {"o_orderkey"},
              {"l_orderkey_1"},
              lineitemJoinSupplier,
              "",
              {"s_nationkey", "l_orderkey_1", "l_suppkey_1", "s_name"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, lineitem2RowType, lineitem2FileColumns)
          .captureScanNodeId(lineitem2ScanNodeId)
          .project({"l_orderkey as l_orderkey_2", "l_suppkey as l_suppkey_2"})
          .hashJoin(
              {"l_orderkey_2"},
              {"l_orderkey_1"},
              ordersJoinLineitem1,
              "l_suppkey_2 <> l_suppkey_1",
              {"l_orderkey_1", "l_suppkey_1", "s_name"},
              core::JoinType::kRightSemiFilter)
          .hashJoin(
              {"l_orderkey_1"},
              {"l_orderkey_3"},
              lineitem3,
              "l_suppkey_3 <> l_suppkey_1",
              {"s_name"},
              core::JoinType::kAnti,
              false /*nullAware*/)
          .partialAggregation({"s_name"}, {"count(1) as numwait"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"numwait DESC", "s_name"}, false)
          .limit(0, 100, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[supplierScanNodeId] = getTableFilePaths(kSupplier);
  context.dataFiles[lineitem1ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitem2ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[lineitem3ScanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[nationScanNodeId] = getTableFilePaths(kNation);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ22Plan() const {
  std::vector<std::string> ordersColumns = {"o_custkey"};
  // Encoded dataset: c_phone is a code; the country code is computed from
  // the text twin right after the scan (CPU side) as an INTEGER, so only a
  // fixed-width value crosses the boundary.
  std::vector<std::string> customerColumns = {"c_acctbal", strCol("c_phone")};
  std::vector<std::string> customerColumnsWithKey = {
      "c_custkey", "c_acctbal", strCol("c_phone")};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType = getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);
  const auto customerSelectedRowTypeWithKey =
      getRowType(kCustomer, customerColumnsWithKey);
  const auto& customerFileColumnsWithKey = getFileColumnNames(kCustomer);

  const std::string phoneFilter = "substr(" + strCol("c_phone") +
      ", 1, 2) IN ('13', '31', '23', '29', '30', '18', '17')";

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanNodeId;
  core::PlanNodeId customerScanNodeIdWithKey;
  core::PlanNodeId ordersScanNodeId;

  auto orders =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kOrders, ordersSelectedRowType, ordersFileColumns)
          .captureScanNodeId(ordersScanNodeId)
          .planNode();

  auto customerAvgAccountBalance =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              customerSelectedRowType,
              customerFileColumns,
              {"c_acctbal > 0.0"},
              phoneFilter)
          .captureScanNodeId(customerScanNodeId)
          .partialAggregation({}, {"avg(c_acctbal) as avg_acctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  auto customerScan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomer,
              customerSelectedRowTypeWithKey,
              customerFileColumnsWithKey,
              {},
              phoneFilter)
          .captureScanNodeId(customerScanNodeIdWithKey);
  if (encodedCol("c_phone")) {
    customerScan.project(
        {"c_custkey",
         "c_acctbal",
         "cast(substr(c_phone_str, 1, 2) as integer) AS c_phone"});
  }
  auto plan =
      customerScan
          .nestedLoopJoin(
              customerAvgAccountBalance,
              {"c_acctbal", "avg_acctbal", "c_custkey", "c_phone"})
          .filter("c_acctbal > avg_acctbal")
          .hashJoin(
              {"c_custkey"},
              {"o_custkey"},
              orders,
              "",
              {"c_acctbal", "c_phone"},
              core::JoinType::kAnti,
              false /*nullAware*/)
          .project(
              {encodedCol("c_phone")
                   ? "c_phone AS country_code"
                   : "substr(c_phone, 1, 2) AS country_code",
               "c_acctbal"})
          .partialAggregation(
              {"country_code"},
              {"count(0) AS numcust", "sum(c_acctbal) AS totacctbal"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"country_code"}, false)
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersScanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerScanNodeId] = getTableFilePaths(kCustomer);
  context.dataFiles[customerScanNodeIdWithKey] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getIoMeterPlan(int columnPct) const {
  VELOX_CHECK(columnPct > 0 && columnPct <= 100);
  auto columns = getFileColumnNames(kLineitem);
  std::vector<std::string> names;
  for (auto& pair : columns) {
    names.push_back(pair.first);
  }
  std::sort(names.begin(), names.end());
  names.resize(names.size() * columnPct / 100);
  if (std::find(names.begin(), names.end(), "l_partkey") == names.end()) {
    names.push_back("l_partkey");
  }

  const auto selectedRowType = getRowType(kLineitem, names);
  std::vector<std::string> aggregates;
  std::vector<std::string> projectExprs;

  for (auto i = 0; i < selectedRowType->size(); ++i) {
    if (selectedRowType->childAt(i)->kind() == TypeKind::VARCHAR) {
      projectExprs.push_back(
          fmt::format("length({}) as l{}", selectedRowType->nameOf(i), i));
      aggregates.push_back(fmt::format("max(l{})", i));
    } else {
      projectExprs.push_back(selectedRowType->nameOf(i));
      aggregates.push_back(fmt::format("max({})", selectedRowType->nameOf(i)));
    }
  }

  std::string filter = "l_partkey between 2000000 and 2500000";

  core::PlanNodeId lineitemPlanNodeId;
  std::unordered_map<std::string, std::string> aliases;
  for (auto& name : names) {
    aliases[name] = name;
  }
  auto plan = PlanBuilder(pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kLineitem, selectedRowType, aliases, {filter})
                  .captureScanNodeId(lineitemPlanNodeId)
                  .project(projectExprs)
                  .partialAggregation({}, aggregates)
                  .localPartition(std::vector<std::string>{})
                  .finalAggregation()
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

// ============================================================
// VLDB Experiment Queries Q23-Q30
// ============================================================

// Q23: Scan 1% selectivity
// SELECT l_orderkey, l_discount, l_extendedprice
// FROM lineitem WHERE l_shipdate < '1992-02-01'
TpchPlan TpchQueryBuilder::getScan1PctPlan() const {
  std::vector<std::string> selectedColumns = {
      "l_orderkey", "l_discount", "l_extendedprice", "l_shipdate"};
  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);
  auto filter =
      formatDateFilter("l_shipdate", selectedRowType, "", "'1992-02-01'");

  core::PlanNodeId lineitemPlanNodeId;
  auto plan =
      PlanBuilder(pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, selectedRowType, fileColumnNames, {filter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project({"l_orderkey", "l_discount", "l_extendedprice"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

// Q24: Scan 3% selectivity
// SELECT l_orderkey, l_discount, l_extendedprice
// FROM lineitem WHERE l_shipdate < '1992-04-15'
TpchPlan TpchQueryBuilder::getScan3PctPlan() const {
  std::vector<std::string> selectedColumns = {
      "l_orderkey", "l_discount", "l_extendedprice", "l_shipdate"};
  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);
  auto filter =
      formatDateFilter("l_shipdate", selectedRowType, "", "'1992-04-15'");

  core::PlanNodeId lineitemPlanNodeId;
  auto plan =
      PlanBuilder(pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, selectedRowType, fileColumnNames, {filter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project({"l_orderkey", "l_discount", "l_extendedprice"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

// Q25: Scan 10% selectivity
// SELECT l_orderkey, l_discount, l_extendedprice
// FROM lineitem WHERE l_shipdate < '1992-10-01'
TpchPlan TpchQueryBuilder::getScan10PctPlan() const {
  std::vector<std::string> selectedColumns = {
      "l_orderkey", "l_discount", "l_extendedprice", "l_shipdate"};
  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);
  auto filter =
      formatDateFilter("l_shipdate", selectedRowType, "", "'1992-10-01'");

  core::PlanNodeId lineitemPlanNodeId;
  auto plan =
      PlanBuilder(pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, selectedRowType, fileColumnNames, {filter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project({"l_orderkey", "l_discount", "l_extendedprice"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

// Q26: Scan 30% selectivity
// SELECT l_orderkey, l_discount, l_extendedprice
// FROM lineitem WHERE l_shipdate < '1994-03-01'
TpchPlan TpchQueryBuilder::getScan30PctPlan() const {
  std::vector<std::string> selectedColumns = {
      "l_orderkey", "l_discount", "l_extendedprice", "l_shipdate"};
  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);
  auto filter =
      formatDateFilter("l_shipdate", selectedRowType, "", "'1994-03-01'");

  core::PlanNodeId lineitemPlanNodeId;
  auto plan =
      PlanBuilder(pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kLineitem, selectedRowType, fileColumnNames, {filter})
          .captureScanNodeId(lineitemPlanNodeId)
          .project({"l_orderkey", "l_discount", "l_extendedprice"})
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}

// Q27: Join lineitem ⋈ orders
// SELECT count(*) FROM lineitem, orders WHERE l_orderkey = o_orderkey
TpchPlan TpchQueryBuilder::getJoinLOPlan() const {
  std::vector<std::string> lineitemColumns = {"l_orderkey"};
  std::vector<std::string> ordersColumns = {"o_orderkey"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {})
                    .captureScanNodeId(ordersPlanNodeId)
                    .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {})
          .captureScanNodeId(lineitemPlanNodeId)
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              {"l_orderkey"})
          .partialAggregation({}, {"count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

// Q32 (case-a microbench): SINGLE join whose output goes straight to the
// CPU -- no aggregation. The build filter is highly selective, so the probe
// spine survives at ~1%: the batch-level adaptive pack should flip to
// deferred, and the terminal probe (a CPU exit) emits a host RowVector with
// the deferred payload gathered host-side, never uploaded.
// SELECT l_orderkey, l_extendedprice, l_discount, l_returnflag, o_orderdate
// (5 output columns: the --include_results row printer elides past 5)
// FROM lineitem, orders
// WHERE l_orderkey = o_orderkey
//   AND o_orderdate >= DATE '1998-07-01' AND o_orderpriority = '1-URGENT'
TpchPlan TpchQueryBuilder::getCaseAPlan() const {
  std::vector<std::string> lineitemColumns = {
      "l_orderkey", "l_extendedprice", "l_discount", "l_returnflag"};
  std::vector<std::string> ordersColumns = {
      "o_orderkey", "o_orderdate", "o_orderpriority"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  auto orderDateFilter = formatDateFilter(
      "o_orderdate", ordersSelectedRowType, "'1998-07-01'", "");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId ordersPlanNodeId;

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kOrders,
                        ordersSelectedRowType,
                        ordersFileColumns,
                        {orderDateFilter, "o_orderpriority = '1-URGENT'"})
                    .captureScanNodeId(ordersPlanNodeId)
                    .planNode();

  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kLineitem,
                      lineitemSelectedRowType,
                      lineitemFileColumns,
                      {})
                  .captureScanNodeId(lineitemPlanNodeId)
                  .hashJoin(
                      {"l_orderkey"},
                      {"o_orderkey"},
                      orders,
                      "",
                      {"l_orderkey",
                       "l_extendedprice",
                       "l_discount",
                       "l_returnflag",
                       "o_orderdate"})
                  .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

// Q28: Join lineitem ⋈ part
// SELECT count(*) FROM lineitem, part WHERE l_partkey = p_partkey
TpchPlan TpchQueryBuilder::getJoinLPPlan() const {
  std::vector<std::string> lineitemColumns = {"l_partkey"};
  std::vector<std::string> partColumns = {"p_partkey"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto partSelectedRowType = getRowType(kPart, partColumns);
  const auto& partFileColumns = getFileColumnNames(kPart);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId partPlanNodeId;

  auto part = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(
                      kPart,
                      partSelectedRowType,
                      partFileColumns,
                      {})
                  .captureScanNodeId(partPlanNodeId)
                  .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {})
          .captureScanNodeId(lineitemPlanNodeId)
          .hashJoin(
              {"l_partkey"},
              {"p_partkey"},
              part,
              "",
              {"l_partkey"})
          .partialAggregation({}, {"count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[partPlanNodeId] = getTableFilePaths(kPart);
  context.dataFileFormat = format_;
  return context;
}

// Q29: Join lineitem ⋈ supplier
// SELECT count(*) FROM lineitem, supplier WHERE l_suppkey = s_suppkey
TpchPlan TpchQueryBuilder::getJoinLSPlan() const {
  std::vector<std::string> lineitemColumns = {"l_suppkey"};
  std::vector<std::string> supplierColumns = {"s_suppkey"};

  const auto lineitemSelectedRowType = getRowType(kLineitem, lineitemColumns);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto supplierSelectedRowType =
      getRowType(kSupplier, supplierColumns);
  const auto& supplierFileColumns = getFileColumnNames(kSupplier);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemPlanNodeId;
  core::PlanNodeId supplierPlanNodeId;

  auto supplier = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kSupplier,
                          supplierSelectedRowType,
                          supplierFileColumns,
                          {})
                      .captureScanNodeId(supplierPlanNodeId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem,
              lineitemSelectedRowType,
              lineitemFileColumns,
              {})
          .captureScanNodeId(lineitemPlanNodeId)
          .hashJoin(
              {"l_suppkey"},
              {"s_suppkey"},
              supplier,
              "",
              {"l_suppkey"})
          .partialAggregation({}, {"count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFiles[supplierPlanNodeId] = getTableFilePaths(kSupplier);
  context.dataFileFormat = format_;
  return context;
}

// Q30: Join orders ⋈ customer
// SELECT count(*) FROM orders, customer WHERE o_custkey = c_custkey
TpchPlan TpchQueryBuilder::getJoinOCPlan() const {
  std::vector<std::string> ordersColumns = {"o_custkey"};
  std::vector<std::string> customerColumns = {"c_custkey"};

  const auto ordersSelectedRowType = getRowType(kOrders, ordersColumns);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);
  const auto customerSelectedRowType =
      getRowType(kCustomer, customerColumns);
  const auto& customerFileColumns = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId ordersPlanNodeId;
  core::PlanNodeId customerPlanNodeId;

  auto customer = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kCustomer,
                          customerSelectedRowType,
                          customerFileColumns,
                          {})
                      .captureScanNodeId(customerPlanNodeId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kOrders,
              ordersSelectedRowType,
              ordersFileColumns,
              {})
          .captureScanNodeId(ordersPlanNodeId)
          .hashJoin(
              {"o_custkey"},
              {"c_custkey"},
              customer,
              "",
              {"o_custkey"})
          .partialAggregation({}, {"count(0)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[ordersPlanNodeId] = getTableFilePaths(kOrders);
  context.dataFiles[customerPlanNodeId] = getTableFilePaths(kCustomer);
  context.dataFileFormat = format_;
  return context;
}

// Q31: Single join S x R with Sort (for late-m testing with custom tables)
// Uses join_benchmark_v2 data: R (100M rows, 17 cols) and S (200M rows, 5 cols)
// Pattern: S (probe) JOIN R (build) -> Sort by (row_id, l_suppkey,
// l_returnflag, l_linestatus)
// Output: 16 columns from R (4 keys + 12 payloads, excluding l_comment)
TpchPlan TpchQueryBuilder::getQ31Plan() const {
  // Build-side columns (R): 4 join keys + up to 12 payload columns.
  // Payload order interleaves the two VARCHARs early so width sweeps
  // (--synth_payload_cols=2/4/8/12) include string payloads from N=4 on.
  static const std::vector<std::string> kJoinKeys = {
      "row_id", "l_suppkey", "l_returnflag", "l_linestatus"};
  static const std::vector<std::string> kPayloadOrder = {
      "l_orderkey",       "l_partkey",  "l_extendedprice", "l_shipmode",
      "l_shipinstruct",   "l_quantity", "l_discount",      "l_tax",
      "l_linenumber",     "l_shipdate", "l_commitdate",    "l_receiptdate",
      "l_comment"}; // 13th payload: wide VARCHAR (~27B avg)
  const int numPayloads = std::clamp(FLAGS_synth_payload_cols, 0, 13);
  std::vector<std::string> rColumns = kJoinKeys;
  rColumns.insert(
      rColumns.end(),
      kPayloadOrder.begin(),
      kPayloadOrder.begin() + numPayloads);

  // Probe-side columns (S) - only join keys (will be renamed to avoid
  // conflicts)
  std::vector<std::string> sColumns = {
      "row_id", // join key
      "l_suppkey", // join key
      "l_returnflag", // join key
      "l_linestatus" // join key
  };

  auto rSelectedRowType = getRowType(kTableR, rColumns);
  const auto& rFileColumns = getFileColumnNames(kTableR);
  auto sSelectedRowType = getRowType(kTableS, sColumns);
  const auto& sFileColumns = getFileColumnNames(kTableS);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId rPlanNodeId;
  core::PlanNodeId sPlanNodeId;

  // Build side: R (has the payloads)
  auto r = PlanBuilder(planNodeIdGenerator, pool_.get())
               .filtersAsNode(filtersAsNode_)
               .tableScan(kTableR, rSelectedRowType, rFileColumns, {})
               .captureScanNodeId(rPlanNodeId)
               .planNode();

  // HashJoin: S (probe) x R (build) on the first --synth_join_keys of the
  // 4 key columns (row_id alone is unique, so match cardinality is
  // identical for any key count; extra keys only add key-processing cost).
  // --s_selectivity_pct thins the probe stream via row_id % 10 (uniform,
  // matching rows only); --synth_join_sort appends the 4-key sort.
  const int numJoinKeys = std::clamp(FLAGS_synth_join_keys, 1, 4);
  static const std::vector<std::string> kProbeRenames = {
      "row_id AS s_row_id",
      "l_suppkey AS s_suppkey",
      "l_returnflag AS s_returnflag",
      "l_linestatus AS s_linestatus"};
  static const std::vector<std::string> kProbeKeys = {
      "s_row_id", "s_suppkey", "s_returnflag", "s_linestatus"};
  PlanBuilder sBuilder(planNodeIdGenerator, pool_.get());
  if (FLAGS_synth_join_flip) {
    // FLIPPED: probe = R (keys + payloads), thinned by the same
    // row_id % 10 predicate; build = S (keys only, renamed s_*). Output =
    // R's columns, so the payload now rides on the PROBE side.
    auto sBuild = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kTableS, sSelectedRowType, sFileColumns, {})
                      .captureScanNodeId(sPlanNodeId)
                      .project({kProbeRenames.begin(),
                                kProbeRenames.begin() + numJoinKeys})
                      .planNode();
    sBuilder.filtersAsNode(filtersAsNode_)
        .tableScan(kTableR, rSelectedRowType, rFileColumns, {})
        .captureScanNodeId(rPlanNodeId);
    if (FLAGS_s_selectivity_pct < 100) {
      sBuilder.filter(
          fmt::format("(row_id % 10) < {}", FLAGS_s_selectivity_pct / 10));
    }
    sBuilder.hashJoin(
        {kJoinKeys.begin(), kJoinKeys.begin() + numJoinKeys},
        {kProbeKeys.begin(), kProbeKeys.begin() + numJoinKeys},
        sBuild,
        "",
        rColumns);
  } else {
    sBuilder.filtersAsNode(filtersAsNode_)
        .tableScan(kTableS, sSelectedRowType, sFileColumns, {})
        .captureScanNodeId(sPlanNodeId);
    if (FLAGS_s_selectivity_pct < 100) {
      sBuilder.filter(
          fmt::format("(row_id % 10) < {}", FLAGS_s_selectivity_pct / 10));
    }
    sBuilder
        .project(
            {kProbeRenames.begin(), kProbeRenames.begin() + numJoinKeys})
        .hashJoin(
            {kProbeKeys.begin(), kProbeKeys.begin() + numJoinKeys},
            {kJoinKeys.begin(), kJoinKeys.begin() + numJoinKeys},
            r,
            "", // no filter
            rColumns); // output: 4 key cols + selected payloads
  }
  if (FLAGS_synth_join_sort) {
    if (FLAGS_synth_sort_gather) {
      sBuilder.localPartition(std::vector<std::string>{});
    }
    sBuilder.orderBy(kJoinKeys, false);
  }
  auto plan = sBuilder.planNode();

  TpchPlan context;
  context.planName = fmt::format(
      "q31{}_p{}_sel{}_j{}{}",
      FLAGS_synth_join_flip ? "flip" : "",
      numPayloads,
      FLAGS_s_selectivity_pct,
      numJoinKeys,
      FLAGS_synth_join_sort ? "_sort" : "");
  context.plan = std::move(plan);
  context.dataFiles[sPlanNodeId] = getTableFilePaths(kTableS);
  context.dataFiles[rPlanNodeId] = getTableFilePaths(kTableR);
  context.dataFileFormat = format_;
  return context;
}

// Q40: Configurable sort benchmark on lineitem
// Uses 4 sort keys (l_linenumber, l_suppkey, l_partkey, l_orderkey)
// and 16 columns (full lineitem schema)
//
// This query tests hybrid sort performance on a simple TableScan -> Sort
// pattern without join overhead.
TpchPlan TpchQueryBuilder::getQ40Plan() const {
  // Sort keys ordered by increasing cardinality; --synth_sort_keys=k
  // takes the first k. --synth_payload_cols=N controls carried payload
  // width (same ordering as Q31: VARCHARs from N=4 on).
  static const std::vector<std::string> kAllSortKeys = {
      "l_linenumber", // INTEGER - 1-7 values
      "l_suppkey", // BIGINT - ~10K unique values
      "l_partkey", // BIGINT - ~200K unique values
      "l_orderkey" // BIGINT - ~60M unique values
  };
  static const std::vector<std::string> kPayloadOrder = {
      "l_extendedprice", "l_shipmode",   "l_shipinstruct", "l_quantity",
      "l_discount",      "l_tax",        "l_shipdate",     "l_commitdate",
      "l_receiptdate",   "l_returnflag", "l_linestatus",   "l_comment"};
  const int numKeys = std::clamp(FLAGS_synth_sort_keys, 1, 4);
  const int numPayloads = std::clamp(FLAGS_synth_payload_cols, 0, 12);
  std::vector<std::string> sortKeys(
      kAllSortKeys.begin(), kAllSortKeys.begin() + numKeys);
  // Scan carries all sort-key candidates plus the selected payloads so
  // total width is k-independent within a payload setting.
  std::vector<std::string> selectedColumns = kAllSortKeys;
  selectedColumns.insert(
      selectedColumns.end(),
      kPayloadOrder.begin(),
      kPayloadOrder.begin() + numPayloads);

  const auto selectedRowType = getRowType(kLineitem, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kLineitem);

  core::PlanNodeId lineitemPlanNodeId;

  PlanBuilder q40(pool_.get());
  q40.filtersAsNode(filtersAsNode_)
      .tableScan(kLineitem, selectedRowType, fileColumnNames, {})
      .captureScanNodeId(lineitemPlanNodeId);
  if (FLAGS_synth_sort_gather) {
    q40.localPartition(std::vector<std::string>{});
  }
  auto plan = q40.orderBy(sortKeys, false).planNode();

  TpchPlan context;
  context.planName = fmt::format("q40_k{}_p{}", numKeys, numPayloads);
  context.plan = std::move(plan);
  context.dataFiles[lineitemPlanNodeId] = getTableFilePaths(kLineitem);
  context.dataFileFormat = format_;
  return context;
}


// Wide-payload sort. Mirrors the benchmark in Velox's "Why Sort is row-based
// in Velox" post (velox-lib.io/blog/why-row-based-sort, Dec 2025), which
// measured a keys-only sort gathering payloads from the ORIGINAL per-batch
// vectors at 1.9-3.9x SLOWER than row-based sort at 64/128/256 payload
// columns. Our hybrid sort coalesces payload batches into one contiguous base
// (overlapped with the key sort), removing the per-batch indirection they
// identified as the root cause; this query tests whether that fix holds at
// THEIR column counts -- the lineitem-based Q40 tops out at 12 payload cols.
// LIMIT omitted so the final sort does full work.

// Q43: TPC-H lineitem |><| orders on l_orderkey = o_orderkey, BUILD ON
// ORDERS -- the smaller side and the exact plan of the GPU offload bench
// (intro Fig 1: same query, both engines, both building on orders).
// lineitem is filtered l_shipdate < '1994-02-21' (~30% of rows at any SF).
// Output = 10 fixed-width lineitem columns + 4 orders payloads (14 cols;
// matches the bench's supported set; orders payload = exactly 24 bytes, the
// hybrid join gate threshold). A final sum/count aggregation keeps the
// client from materializing ~SF*1.8M output rows while the join's
// getOutput still performs the full extraction.
TpchPlan TpchQueryBuilder::getQ43Plan() const {
  std::vector<std::string> lineitemCols = {
      "l_orderkey",      "l_partkey",  "l_suppkey",  "l_linenumber",
      "l_quantity",      "l_extendedprice", "l_discount", "l_tax",
      "l_commitdate",    "l_receiptdate",   "l_shipdate"};
  std::vector<std::string> ordersCols = {
      "o_orderkey", "o_custkey", "o_totalprice", "o_orderdate",
      "o_shippriority"};

  auto lineitemType = getRowType(kLineitem, lineitemCols);
  auto ordersType = getRowType(kOrders, ordersCols);
  const auto& lineitemFileColumns = getFileColumnNames(kLineitem);
  const auto& ordersFileColumns = getFileColumnNames(kOrders);

  auto shipDateFilter = formatDateFilter(
      "l_shipdate", lineitemType, "", "'1994-02-21'");

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId lineitemScanId;
  core::PlanNodeId ordersScanId;

  auto orders = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(kOrders, ordersType, ordersFileColumns, {})
                    .captureScanNodeId(ordersScanId)
                    .planNode();

  std::vector<std::string> outputCols = {
      "l_orderkey",      "l_partkey",  "l_suppkey",  "l_linenumber",
      "l_quantity",      "l_extendedprice", "l_discount", "l_tax",
      "l_commitdate",    "l_receiptdate",   "o_custkey",  "o_totalprice",
      "o_orderdate",     "o_shippriority"};

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kLineitem, lineitemType, lineitemFileColumns, {shipDateFilter})
          .captureScanNodeId(lineitemScanId)
          .hashJoin(
              {"l_orderkey"},
              {"o_orderkey"},
              orders,
              "",
              outputCols)
          // Integer-cent sums: double addition is non-associative, so
          // cross-arm accumulation-order differences flip the last ulp of a
          // raw double sum and break exact checksum gating. Integer sums are
          // order-independent. Aggregation inputs must be field accesses, so
          // project the casts first.
          .project(
              {"cast(l_extendedprice * 100.0 as bigint) as l_price_cents",
               "cast(o_totalprice * 100.0 as bigint) as o_price_cents",
               "cast(l_quantity as bigint) as l_qty_int"})
          .partialAggregation(
              {},
              {"count(1)",
               "sum(l_price_cents)",
               "sum(o_price_cents)",
               "sum(l_qty_int)"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan context;
  context.plan = std::move(plan);
  context.dataFiles[lineitemScanId] = getTableFilePaths(kLineitem);
  context.dataFiles[ordersScanId] = getTableFilePaths(kOrders);
  context.dataFileFormat = format_;
  return context;
}

TpchPlan TpchQueryBuilder::getQ41Plan() const {
  const int numPayloads = std::clamp(FLAGS_synth_wide_payload_cols, 0, 256);
  const int numKeys = std::clamp(FLAGS_synth_wide_sort_keys, 1, 16);
  std::vector<std::string> sortKeys;
  sortKeys.reserve(numKeys);
  for (int i = 1; i <= numKeys; ++i) {
    sortKeys.push_back(fmt::format("k{}", i));
  }
  std::vector<std::string> selectedColumns = sortKeys;
  selectedColumns.reserve(2 + numPayloads);
  for (int i = 0; i < numPayloads; ++i) {
    selectedColumns.push_back(fmt::format("c{}", i));
  }

  const auto selectedRowType = getRowType(kTableWide, selectedColumns);
  const auto& fileColumnNames = getFileColumnNames(kTableWide);

  core::PlanNodeId widePlanNodeId;
  auto plan = PlanBuilder(pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kTableWide, selectedRowType, fileColumnNames, {})
                  .captureScanNodeId(widePlanNodeId)
                  .orderBy(sortKeys, false)
                  .planNode();

  TpchPlan context;
  context.planName = fmt::format("q41_k{}_p{}", numKeys, numPayloads);
  context.plan = std::move(plan);
  context.dataFiles[widePlanNodeId] = getTableFilePaths(kTableWide);
  context.dataFileFormat = format_;
  return context;
}

const std::vector<std::string> TpchQueryBuilder::kTableNames_ = {
    kLineitem,
    kOrders,
    kCustomer,
    kNation,
    kRegion,
    kPart,
    kSupplier,
    kPartsupp,
    kTableR,
    kTableS,
    kTableWide};

const std::unordered_map<std::string, std::vector<std::string>>
    TpchQueryBuilder::kTables_ = {
        std::make_pair(
            "lineitem",
            tpch::getTableSchema(tpch::Table::TBL_LINEITEM)->names()),
        std::make_pair(
            "orders",
            tpch::getTableSchema(tpch::Table::TBL_ORDERS)->names()),
        std::make_pair(
            "customer",
            tpch::getTableSchema(tpch::Table::TBL_CUSTOMER)->names()),
        std::make_pair(
            "nation",
            tpch::getTableSchema(tpch::Table::TBL_NATION)->names()),
        std::make_pair(
            "region",
            tpch::getTableSchema(tpch::Table::TBL_REGION)->names()),
        std::make_pair(
            "part",
            tpch::getTableSchema(tpch::Table::TBL_PART)->names()),
        std::make_pair(
            "supplier",
            tpch::getTableSchema(tpch::Table::TBL_SUPPLIER)->names()),
        std::make_pair(
            "partsupp",
            tpch::getTableSchema(tpch::Table::TBL_PARTSUPP)->names()),
        // Wide-payload sort table (gen_widesort.sh): 2 sort keys + 256
        // BIGINT payload columns, sized to the payload widths used in
        // Velox's "Why Sort is row-based" evaluation (64/128/256).
        std::make_pair(
            "wide",
            [] {
              std::vector<std::string> names;
              names.reserve(272);
              for (int i = 1; i <= 16; ++i) {
                names.push_back(fmt::format("k{}", i));
              }
              for (int i = 0; i < 256; ++i) {
                names.push_back(fmt::format("c{}", i));
              }
              return names;
            }()),
        // Custom join benchmark tables R and S
        std::make_pair(
            "R",
            std::vector<std::string>{
                "row_id",
                "l_suppkey",
                "l_returnflag",
                "l_linestatus",
                "l_orderkey",
                "l_partkey",
                "l_linenumber",
                "l_quantity",
                "l_extendedprice",
                "l_discount",
                "l_tax",
                "l_shipdate",
                "l_commitdate",
                "l_receiptdate",
                "l_shipinstruct",
                "l_shipmode",
                "l_comment"}),
        std::make_pair(
            "S",
            std::vector<std::string>{
                "row_id",
                "l_suppkey",
                "l_returnflag",
                "l_linestatus",
                "s_orderkey"})};

} // namespace facebook::velox::exec::test
