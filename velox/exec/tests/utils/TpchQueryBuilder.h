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
#include <unordered_set>
#include <unordered_map>

#include "velox/dwio/common/Options.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::test {

/// Contains the query plan and input data files keyed on source plan node ID.
/// All data files use the same file format specified in 'dataFileFormat'.
struct TpchPlan {
  core::PlanNodePtr plan;
  std::unordered_map<core::PlanNodeId, std::vector<std::string>> dataFiles;
  dwio::common::FileFormat dataFileFormat;
  std::string planName;
};

/// Contains type information, data files, and file column names for a table.
/// This information is inferred from the input data files.
/// The type names are mapped to the standard names.
/// Example: If the file has a 'returnflag' column, the corresponding type name
/// will be 'l_returnflag'. fileColumnNames store the mapping between standard
/// names and the corresponding name in the file.
struct TpchTableMetadata {
  RowTypePtr type;
  std::vector<std::string> dataFiles;
  std::unordered_map<std::string, std::string> fileColumnNames;
};

/// Builds TPC-H queries using TPC-H data files located in the specified
/// directory. Each table data must be placed in hive-style partitioning. That
/// is, the top-level directory is expected to contain a sub-directory per table
/// name and the name of the sub-directory must match the table name. Example:
/// ls -R data/
///  customer   lineitem
///
///  data/customer:
///  customer1.parquet  customer2.parquet
///
///  data/lineitem:
///  lineitem1.parquet  lineitem2.parquet  lineitem3.parquet

/// The column names can vary. Additional columns may exist towards the end.
/// The class uses standard names (example: l_returnflag) to build TPC-H plans.
/// Since the column names in the file can vary, they are mapped to the standard
/// names. Therefore, the order of the columns in the file is important and
/// should be in the same order as in the TPC-H standard.
class TpchQueryBuilder {
 public:
  explicit TpchQueryBuilder(
      dwio::common::FileFormat format,
      bool filtersAsNode = false)
      : format_(format), filtersAsNode_(filtersAsNode) {}

  /// Read each data file, initialize row types, and determine data paths for
  /// each table.
  /// @param dataPath path to the data files
  void initialize(const std::string& dataPath);

  /// Get the query plan for a given TPC-H query number.
  /// @param queryId TPC-H query number
  TpchPlan getQueryPlan(int queryId) const;

  ~TpchQueryBuilder();

  /// Resident-table mode (2026-08-22): every TableScan of a query plan is
  /// served from process-resident, pre-decoded RowVector batches (see
  /// ResidentTables.h). Scan filters are kept as FilterNodes (filtersAsNode)
  /// so the CPU-side filtering stays in the timed region; only the Parquet
  /// decode leaves it. Batches are loaded once per (table, columns) per
  /// process, by running the real scan; numDrivers sizes that preload.
  void setResidentTables(bool resident, int32_t numDrivers) {
    resident_ = resident;
    residentDrivers_ = numDrivers;
    if (resident) {
      filtersAsNode_ = true;
    }
  }

  /// Returns a plan for select max(c1), max(c2), .. from lineitem where partkey
  /// between
  /// ... such that 'columnPct' columns are aggregated with max(c) for numbers
  /// and with max(length(c)) for strings. To select the columns, we sort them
  /// by column name and take the first columnPct percent.
  TpchPlan getIoMeterPlan(int columnPct) const;

  /// Get the TPC-H table names present.
  static const std::vector<std::string>& getTableNames();

 private:
  // Initializes the schema information for 'tableName' from sample file at
  // 'filePath'.
  void readFileSchema(
      const std::string& tableName,
      const std::string& filePath,
      const std::vector<std::string>& columns);

  TpchPlan getQ1Plan() const;
  TpchPlan getQ2Plan() const;
  TpchPlan getQ3Plan() const;
  TpchPlan getQ4Plan() const;
  TpchPlan getQ5Plan() const;
  TpchPlan getQ6Plan() const;
  TpchPlan getQ7Plan() const;
  TpchPlan getQ8Plan() const;
  TpchPlan getQ9Plan() const;
  TpchPlan getQ10Plan() const;
  TpchPlan getQ11Plan() const;
  TpchPlan getQ12Plan() const;
  TpchPlan getQ13Plan() const;
  TpchPlan getQ14Plan() const;
  TpchPlan getQ15Plan() const;
  TpchPlan getQ16Plan() const;
  TpchPlan getQ17Plan() const;
  TpchPlan getQ18Plan() const;
  TpchPlan getQ19Plan() const;
  TpchPlan getQ20Plan() const;
  TpchPlan getQ21Plan() const;
  TpchPlan getQ22Plan() const;

  // VLDB experiment queries (scan selectivity + equi-join microbenchmarks)
  TpchPlan getScan1PctPlan() const;   // Q23
  TpchPlan getScan3PctPlan() const;   // Q24
  TpchPlan getScan10PctPlan() const;  // Q25
  TpchPlan getScan30PctPlan() const;  // Q26
  TpchPlan getJoinLOPlan() const;     // Q27: lineitem ⋈ orders
  TpchPlan getJoinLPPlan() const;     // Q28: lineitem ⋈ part
  TpchPlan getJoinLSPlan() const;     // Q29: lineitem ⋈ supplier
  TpchPlan getJoinOCPlan() const;     // Q30: orders ⋈ customer

  // Hybrid single-operator benchmarks (synthetic tables R/S; see Bolt).
  TpchPlan getQ31Plan() const;  // S JOIN R -> Sort (2-way, 16 output cols)
  TpchPlan getQ40Plan() const;  // Configurable sort benchmark on lineitem (4 sort keys, 16 cols)
  TpchPlan getQ41Plan() const;  // Wide-payload sort on the synthetic 'wide' table (up to 256 payload cols)
  TpchPlan getQ43Plan() const;  // lineitem |><| orders join, build on orders (intro Fig 1 workload)

  const std::vector<std::string>& getTableFilePaths(
      const std::string& tableName) const {
    return tableMetadata_.at(tableName).dataFiles;
  }

  std::shared_ptr<const RowType> getRowType(
      const std::string& tableName,
      const std::vector<std::string>& columnNames) const {
    auto columnSelector = std::make_shared<dwio::common::ColumnSelector>(
        tableMetadata_.at(tableName).type, columnNames);
    return columnSelector->buildSelectedReordered();
  }

  const std::unordered_map<std::string, std::string>& getFileColumnNames(
      const std::string& tableName) const {
    return tableMetadata_.at(tableName).fileColumnNames;
  }

  std::unordered_map<std::string, TpchTableMetadata> tableMetadata_;

  // ---- String-encoded datasets (2026-08-22 "ideal string handling" study).
  // If <dataPath>/codes.json exists, the string columns that cross joins /
  // group-bys are INTEGER codes in the files (dense rank over the sorted
  // distinct values, so code order == string order) and the original text
  // is kept as trailing <col>_str columns. The helpers below make every
  // query build correctly against both the plain and the encoded dataset.
  std::unordered_map<std::string, std::unordered_map<std::string, int32_t>>
      encCodes_;
  bool encoded_{false};
  bool encodedCol(const std::string& col) const {
    // Every encoded column has a _str twin in the files; only the small
    // dictionaries are in codes.json.
    return encoded_ && (encCodes_.count(col) > 0 || encStrTwins_.count(col) > 0);
  }
  /// Literal for `col = 'value'`: the code when encoded, else the quoted text.
  std::string encLit(const std::string& col, const std::string& value) const;
  /// `col IN (...)` predicate.
  std::string encIn(
      const std::string& col,
      const std::vector<std::string>& values) const;
  /// `col LIKE 'prefix%'` -> code range when encoded.
  std::string encLikePrefix(const std::string& col, const std::string& prefix)
      const;
  /// `col LIKE '%suffix'` -> IN-list of matching codes when encoded.
  std::string encLikeSuffix(const std::string& col, const std::string& suffix)
      const;
  /// Name of the text twin for scan-side text predicates on an encoded
  /// column (col itself when not encoded).
  std::string strCol(const std::string& col) const {
    return encodedCol(col) ? col + "_str" : col;
  }
  std::unordered_set<std::string> encStrTwins_; // columns with a _str twin
  const dwio::common::FileFormat format_;
  static const std::unordered_map<std::string, std::vector<std::string>>
      kTables_;
  static const std::vector<std::string> kTableNames_;

  static constexpr const char* kLineitem = "lineitem";
  static constexpr const char* kCustomer = "customer";
  static constexpr const char* kOrders = "orders";
  static constexpr const char* kNation = "nation";
  static constexpr const char* kRegion = "region";
  static constexpr const char* kPart = "part";
  static constexpr const char* kSupplier = "supplier";
  static constexpr const char* kPartsupp = "partsupp";
  static constexpr const char* kTableR = "R";
  static constexpr const char* kTableS = "S";
  // Wide-payload sort table: k1, k2 (sort keys) + c0..c255 (BIGINT payload).
  static constexpr const char* kTableWide = "wide";
  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
  bool filtersAsNode_;
  bool resident_{false};
  int32_t residentDrivers_{24};
  // (table|columns) -> batches, shared across queries of one process.
  mutable std::unordered_map<
      std::string,
      std::shared_ptr<const std::vector<RowVectorPtr>>>
      residentCache_;
  TpchPlan buildQueryPlan(int queryId) const;
  void makeResident(TpchPlan& plan) const;
};

} // namespace facebook::velox::exec::test
