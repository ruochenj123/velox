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

#include "velox/dwio/common/Options.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TpchQueryBuilder.h"

namespace facebook::velox::exec::test {

/// Contains type information, data files, and file column names for a TPC-DS
/// table. Same layout as TpchTableMetadata.
struct TpcdsTableMetadata {
  RowTypePtr type;
  std::vector<std::string> dataFiles;
  std::unordered_map<std::string, std::string> fileColumnNames;
};

/// Builds hand-written TPC-DS query plans (ported from Bolt's
/// bolt/exec/tests/utils/TpcdsQueryBuilder) using TPC-DS data files located in
/// the specified directory. Each table's data must be placed in hive-style
/// partitioning: the top-level directory contains one sub-directory per table
/// name. Example:
/// ls -R data/
///  store_sales   date_dim   item
///
///  data/store_sales:
///  store_sales.parquet
///
///  data/date_dim:
///  date_dim.parquet
///
/// Plans are returned as TpchPlan so the existing QueryBenchmarkBase::run()
/// harness consumes them unchanged.
///
/// NOTE (hybrid sort evaluation): unlike the TPC-DS spec, every query's final
/// LIMIT/TopN is omitted, so the trailing ORDER BY does full work.
class TpcdsQueryBuilder {
 public:
  explicit TpcdsQueryBuilder(
      dwio::common::FileFormat format,
      bool filtersAsNode = false)
      : format_(format), filtersAsNode_(filtersAsNode) {}

  /// Read each data file, initialize row types, and determine data paths for
  /// each table.
  void initialize(const std::string& dataPath);

  /// Get the query plan for a given TPC-DS query number.
  /// Supported: 1, 3, 7, 10, 13, 15, 18, 19, 25, 26, 29, 30, 40, 42, 43, 46,
  /// 52, 53, 54, 55, 63, 64, 68, 72, 73, 74, 79, 80, 81, 84, 89, 93, 96.
  TpchPlan getQueryPlan(int queryId) const;

  /// Get the TPC-DS table names present.
  static const std::vector<std::string>& getTableNames();

 private:
  void readFileSchema(
      const std::string& tableName,
      const std::string& filePath,
      const std::vector<std::string>& columns);

  // TPC-DS query plans used in the hybrid-layout paper.
  // Ported from Bolt's TpcdsQueryBuilder:
  TpchPlan getQ1Plan() const;
  TpchPlan getQ3Plan() const;
  TpchPlan getQ7Plan() const;
  TpchPlan getQ10Plan() const;
  TpchPlan getQ13Plan() const;
  TpchPlan getQ15Plan() const;
  TpchPlan getQ18Plan() const;
  TpchPlan getQ19Plan() const;
  TpchPlan getQ25Plan() const;
  TpchPlan getQ26Plan() const;
  TpchPlan getQ29Plan() const;
  TpchPlan getQ30Plan() const;
  TpchPlan getQ42Plan() const;
  TpchPlan getQ43Plan() const;
  TpchPlan getQ46Plan() const;
  TpchPlan getQ52Plan() const;
  TpchPlan getQ53Plan() const;
  TpchPlan getQ55Plan() const;
  TpchPlan getQ63Plan() const;
  TpchPlan getQ68Plan() const;
  TpchPlan getQ73Plan() const;
  TpchPlan getQ79Plan() const;
  TpchPlan getQ89Plan() const;
  TpchPlan getQ96Plan() const;
  // Hand-built from the TPC-DS qualification SQL (not in Bolt):
  TpchPlan getQ40Plan() const;
  TpchPlan getQ54Plan() const;
  TpchPlan getQ64Plan() const;
  TpchPlan getQ72Plan() const;
  TpchPlan getQ74Plan() const;
  TpchPlan getQ80Plan() const;
  TpchPlan getQ81Plan() const;
  TpchPlan getQ84Plan() const;
  TpchPlan getQ93Plan() const;

  const std::vector<std::string>& getTableFilePaths(
      const std::string& tableName) const {
    return tableMetadata_.at(tableName).dataFiles;
  }

  std::shared_ptr<const RowType> getRowType(
      const std::string& tableName,
      const std::vector<std::string>& columnNames) const {
    VELOX_CHECK_GT(tableMetadata_.size(), 0, "tableMetadata_ is empty");
    auto columnSelector = std::make_shared<dwio::common::ColumnSelector>(
        tableMetadata_.at(tableName).type, columnNames);
    return columnSelector->buildSelectedReordered();
  }

  const std::unordered_map<std::string, std::string>& getFileColumnNames(
      const std::string& tableName) const {
    return tableMetadata_.at(tableName).fileColumnNames;
  }

  std::unordered_map<std::string, TpcdsTableMetadata> tableMetadata_;
  const dwio::common::FileFormat format_;
  static const std::unordered_map<std::string, std::vector<std::string>>
      kTables_;
  static const std::vector<std::string> kTableNames_;

  // Fact tables
  static constexpr const char* kStoreSales = "store_sales";
  static constexpr const char* kStoreReturns = "store_returns";
  static constexpr const char* kCatalogSales = "catalog_sales";
  static constexpr const char* kCatalogReturns = "catalog_returns";
  static constexpr const char* kWebSales = "web_sales";
  static constexpr const char* kWebReturns = "web_returns";
  static constexpr const char* kInventory = "inventory";

  // Dimension tables
  static constexpr const char* kCustomer = "customer";
  static constexpr const char* kCustomerAddress = "customer_address";
  static constexpr const char* kCustomerDemographics = "customer_demographics";
  static constexpr const char* kDateDim = "date_dim";
  static constexpr const char* kTimeDim = "time_dim";
  static constexpr const char* kItem = "item";
  static constexpr const char* kStore = "store";
  static constexpr const char* kPromotion = "promotion";
  static constexpr const char* kHouseholdDemographics = "household_demographics";
  static constexpr const char* kWarehouse = "warehouse";
  static constexpr const char* kShipMode = "ship_mode";
  static constexpr const char* kReason = "reason";
  static constexpr const char* kIncomeBand = "income_band";
  static constexpr const char* kCallCenter = "call_center";
  static constexpr const char* kCatalogPage = "catalog_page";
  static constexpr const char* kWebPage = "web_page";
  static constexpr const char* kWebSite = "web_site";

  std::shared_ptr<memory::MemoryPool> pool_ =
      memory::memoryManager()->addLeafPool();
  const bool filtersAsNode_;
};

} // namespace facebook::velox::exec::test
