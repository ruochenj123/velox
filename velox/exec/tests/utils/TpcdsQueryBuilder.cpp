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

// Hand-built TPC-DS query plans ported from Bolt
// (bolt/exec/tests/utils/TpcdsQueryBuilder.cpp, namespace
// bytedance::bolt::exec::test). Adaptations: facebook::velox namespaces,
// VELOX_* macros, captureScanNodeId + filtersAsNode (velox PlanBuilder
// idioms), and results returned as TpchPlan so QueryBenchmarkBase::run()
// consumes them unchanged. In every query the final LIMIT is omitted
// (vs. spec) so the trailing sort does full work — hybrid sort evaluation.

#include "velox/exec/tests/utils/TpcdsQueryBuilder.h"

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/dwio/common/ReaderFactory.h"

#include <fstream>

namespace facebook::velox::exec::test {

void TpcdsQueryBuilder::readFileSchema(
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
  // There can be extra columns in the file towards the end.
  VELOX_CHECK_GE(fileColumnNames.size(), columns.size());
  std::unordered_map<std::string, std::string> fileColumnNamesMap(
      columns.size());
  std::transform(
      columns.begin(),
      columns.end(),
      fileColumnNames.begin(),
      std::inserter(fileColumnNamesMap, fileColumnNamesMap.begin()),
      [](std::string a, std::string b) { return std::make_pair(a, b); });
  auto columnNames = columns;
  auto types = fileType->children();
  types.resize(columnNames.size());
  tableMetadata_[tableName].type =
      std::make_shared<RowType>(std::move(columnNames), std::move(types));
  tableMetadata_[tableName].fileColumnNames = std::move(fileColumnNamesMap);
}

void TpcdsQueryBuilder::initialize(const std::string& dataPath) {
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

const std::vector<std::string>& TpcdsQueryBuilder::getTableNames() {
  return kTableNames_;
}

TpchPlan TpcdsQueryBuilder::getQueryPlan(int queryId) const {
  switch (queryId) {
    case 1:
      return getQ1Plan();
    case 3:
      return getQ3Plan();
    case 7:
      return getQ7Plan();
    case 10:
      return getQ10Plan();
    case 13:
      return getQ13Plan();
    case 15:
      return getQ15Plan();
    case 18:
      return getQ18Plan();
    case 19:
      return getQ19Plan();
    case 25:
      return getQ25Plan();
    case 26:
      return getQ26Plan();
    case 29:
      return getQ29Plan();
    case 30:
      return getQ30Plan();
    case 40:
      return getQ40Plan();
    case 42:
      return getQ42Plan();
    case 43:
      return getQ43Plan();
    case 46:
      return getQ46Plan();
    case 52:
      return getQ52Plan();
    case 53:
      return getQ53Plan();
    case 54:
      return getQ54Plan();
    case 55:
      return getQ55Plan();
    case 63:
      return getQ63Plan();
    case 64:
      return getQ64Plan();
    case 68:
      return getQ68Plan();
    case 72:
      return getQ72Plan();
    case 73:
      return getQ73Plan();
    case 74:
      return getQ74Plan();
    case 79:
      return getQ79Plan();
    case 80:
      return getQ80Plan();
    case 81:
      return getQ81Plan();
    case 84:
      return getQ84Plan();
    case 89:
      return getQ89Plan();
    case 93:
      return getQ93Plan();
    case 96:
      return getQ96Plan();
    default:
      VELOX_UNSUPPORTED("TPC-DS query {} is not supported", queryId);
  }
}

// Q1 (Bolt's variant): plain scan of the small 'reason' table.
// NOTE: this deviates from spec TPC-DS Q1 (which is a store_returns CTE with
// a correlated average). Bolt's "Q1" is just SELECT * FROM reason; preserved
// as-is for parity with Bolt. No join, no sort.
TpchPlan TpcdsQueryBuilder::getQ1Plan() const {
  std::vector<std::string> reasonCols = {
      "r_reason_sk", "r_reason_id", "r_reason_desc"};
  auto reasonType = getRowType(kReason, reasonCols);
  const auto& reasonFileColumnNames = getFileColumnNames(kReason);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId reasonScanId;

  auto plan = PlanBuilder(planNodeIdGenerator, pool_.get())
                  .filtersAsNode(filtersAsNode_)
                  .tableScan(kReason, reasonType, reasonFileColumnNames, {})
                  .captureScanNodeId(reasonScanId)
                  .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[reasonScanId] = getTableFilePaths(kReason);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q1";
  return result;
}

// Q3: 3-way join date_dim, store_sales, item with aggregation and sort
// SELECT d_year, i_brand_id, i_brand, sum(ss_ext_sales_price)
// FROM date_dim, store_sales, item
// WHERE d_date_sk = ss_sold_date_sk AND ss_item_sk = i_item_sk
//   AND i_manufact_id = 128 AND d_moy = 11
// GROUP BY d_year, i_brand, i_brand_id
// ORDER BY d_year, sum_agg DESC, i_brand_id
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ3Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"};
  std::vector<std::string> itemCols = {
      "i_item_sk", "i_brand_id", "i_brand", "i_manufact_id"};

  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto itemType = getRowType(kItem, itemCols);

  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId dateDimScanId;
  core::PlanNodeId storeSalesScanId;
  core::PlanNodeId itemScanId;

  // Build date_dim scan with filter d_moy = 11
  auto dateDimNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kDateDim, dateDimType, ddFileColumnNames, {"d_moy = 11"})
          .captureScanNodeId(dateDimScanId)
          .planNode();

  // Build item scan with filter i_manufact_id = 128
  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kItem,
                          itemType,
                          itemFileColumnNames,
                          {"i_manufact_id = 128"})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  // store_sales JOIN date_dim ON ss_sold_date_sk = d_date_sk
  // then JOIN item ON ss_item_sk = i_item_sk
  // then aggregate and sort
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk", "ss_ext_sales_price", "d_year"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"d_year", "i_brand_id", "i_brand", "ss_ext_sales_price"})
          .partialAggregation(
              {"d_year", "i_brand_id", "i_brand"},
              {"sum(ss_ext_sales_price) as sum_agg"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"d_year", "sum_agg DESC", "i_brand_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q3";
  return result;
}

// Q10: customer demographics aggregation (simplified)
// SELECT cd_gender, cd_marital_status, cd_education_status,
//        count(*), cd_purchase_estimate, cd_credit_rating,
//        cd_dep_count, cd_dep_employed_count, cd_dep_college_count
// FROM customer c, customer_address ca, customer_demographics cd,
//      store_sales ss, date_dim d
// WHERE c.c_current_addr_sk = ca.ca_address_sk
//   AND cd.cd_demo_sk = c.c_current_cdemo_sk
//   AND c.c_customer_sk = ss.ss_customer_sk
//   AND ss.ss_sold_date_sk = d.d_date_sk
//   AND d.d_year = 2002 AND d.d_moy BETWEEN 1 AND 4
// GROUP BY cd_gender, cd_marital_status, cd_education_status, ...
// ORDER BY cd_gender, cd_marital_status, ...
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ10Plan() const {
  std::vector<std::string> customerCols = {
      "c_customer_sk", "c_current_addr_sk", "c_current_cdemo_sk"};
  std::vector<std::string> customerAddrCols = {"ca_address_sk", "ca_county"};
  std::vector<std::string> customerDemoCols = {
      "cd_demo_sk",
      "cd_gender",
      "cd_marital_status",
      "cd_education_status",
      "cd_purchase_estimate",
      "cd_credit_rating",
      "cd_dep_count",
      "cd_dep_employed_count",
      "cd_dep_college_count"};
  std::vector<std::string> storeSalesCols = {
      "ss_customer_sk", "ss_sold_date_sk"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};

  auto customerType = getRowType(kCustomer, customerCols);
  auto customerAddrType = getRowType(kCustomerAddress, customerAddrCols);
  auto customerDemoType = getRowType(kCustomerDemographics, customerDemoCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);

  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& custAddrFileColumnNames = getFileColumnNames(kCustomerAddress);
  const auto& custDemoFileColumnNames =
      getFileColumnNames(kCustomerDemographics);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanId;
  core::PlanNodeId customerAddrScanId;
  core::PlanNodeId customerDemoScanId;
  core::PlanNodeId storeSalesScanId;
  core::PlanNodeId dateDimScanId;

  // date_dim with filter: d_year = 2002, d_moy between 1 and 4
  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2002", "d_moy between 1 and 4"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  // customer_address
  auto custAddrNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kCustomerAddress,
                              customerAddrType,
                              custAddrFileColumnNames,
                              {})
                          .captureScanNodeId(customerAddrScanId)
                          .planNode();

  // customer_demographics
  auto custDemoNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kCustomerDemographics,
                              customerDemoType,
                              custDemoFileColumnNames,
                              {})
                          .captureScanNodeId(customerDemoScanId)
                          .planNode();

  // customer
  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  // Build the join chain:
  // store_sales -> JOIN date_dim -> JOIN customer -> JOIN customer_address
  //   -> JOIN customer_demographics
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_customer_sk"})
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"c_current_addr_sk", "c_current_cdemo_sk"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              custAddrNode,
              "",
              {"c_current_cdemo_sk"})
          .hashJoin(
              {"c_current_cdemo_sk"},
              {"cd_demo_sk"},
              custDemoNode,
              "",
              {"cd_gender",
               "cd_marital_status",
               "cd_education_status",
               "cd_purchase_estimate",
               "cd_credit_rating",
               "cd_dep_count",
               "cd_dep_employed_count",
               "cd_dep_college_count"})
          .partialAggregation(
              {"cd_gender",
               "cd_marital_status",
               "cd_education_status",
               "cd_purchase_estimate",
               "cd_credit_rating",
               "cd_dep_count",
               "cd_dep_employed_count",
               "cd_dep_college_count"},
              {"count(1) as cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"cd_gender",
               "cd_marital_status",
               "cd_education_status",
               "cd_purchase_estimate",
               "cd_credit_rating"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[customerAddrScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFiles[customerDemoScanId] =
      getTableFilePaths(kCustomerDemographics);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q10";
  return result;
}

// Q25: store_sales with store_returns (low selectivity - only returned items)
// This is promising for hybrid as returns are ~2-5% of sales.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation. (Bolt's unused item scan node was dropped in this port.)
TpchPlan TpcdsQueryBuilder::getQ25Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_item_sk",
      "ss_ticket_number",
      "ss_customer_sk",
      "ss_store_sk",
      "ss_net_profit"};
  std::vector<std::string> storeReturnsCols = {
      "sr_returned_date_sk",
      "sr_item_sk",
      "sr_ticket_number",
      "sr_customer_sk",
      "sr_store_sk",
      "sr_net_loss"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> storeCols = {
      "s_store_sk", "s_store_id", "s_store_name"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto storeReturnsType = getRowType(kStoreReturns, storeReturnsCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& srFileColumnNames = getFileColumnNames(kStoreReturns);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, storeReturnsScanId, dateDim1ScanId,
      dateDim2ScanId;
  core::PlanNodeId storeScanId;

  auto dateDim1Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDimType,
                              ddFileColumnNames,
                              {"d_year = 2000"})
                          .captureScanNodeId(dateDim1ScanId)
                          .planNode();

  auto dateDim2Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDimType,
                              ddFileColumnNames,
                              {"d_year = 2001"})
                          .captureScanNodeId(dateDim2ScanId)
                          .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .planNode();

  // Build store_returns subquery first
  auto storeReturnsNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreReturns, storeReturnsType, srFileColumnNames, {})
          .captureScanNodeId(storeReturnsScanId)
          .hashJoin(
              {"sr_returned_date_sk"},
              {"d_date_sk"},
              dateDim2Node,
              "",
              {"sr_item_sk",
               "sr_ticket_number",
               "sr_customer_sk",
               "sr_store_sk",
               "sr_net_loss"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDim1Node,
              "",
              {"ss_item_sk",
               "ss_ticket_number",
               "ss_customer_sk",
               "ss_store_sk",
               "ss_net_profit"})
          // Join with returns (this is the low-selectivity join!)
          .hashJoin(
              {"ss_item_sk", "ss_ticket_number", "ss_customer_sk"},
              {"sr_item_sk", "sr_ticket_number", "sr_customer_sk"},
              storeReturnsNode,
              "",
              {"ss_store_sk", "ss_net_profit", "sr_net_loss"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_net_profit", "sr_net_loss", "s_store_id", "s_store_name"})
          .partialAggregation(
              {"s_store_id", "s_store_name"},
              {"sum(ss_net_profit) as store_profit",
               "sum(sr_net_loss) as store_loss"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"s_store_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[storeReturnsScanId] = getTableFilePaths(kStoreReturns);
  result.dataFiles[dateDim1ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[dateDim2ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q25";
  return result;
}

// Q26: catalog_sales promotional analysis (similar to Q7 but for catalog)
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ26Plan() const {
  std::vector<std::string> catalogSalesCols = {
      "cs_sold_date_sk",
      "cs_item_sk",
      "cs_bill_cdemo_sk",
      "cs_promo_sk",
      "cs_quantity",
      "cs_list_price",
      "cs_sales_price",
      "cs_coupon_amt"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_item_id"};
  std::vector<std::string> custDemoCols = {
      "cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"};
  std::vector<std::string> promoCols = {"p_promo_sk", "p_channel_email"};

  auto catalogSalesType = getRowType(kCatalogSales, catalogSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto itemType = getRowType(kItem, itemCols);
  auto custDemoType = getRowType(kCustomerDemographics, custDemoCols);
  auto promoType = getRowType(kPromotion, promoCols);

  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& cdFileColumnNames = getFileColumnNames(kCustomerDemographics);
  const auto& promoFileColumnNames = getFileColumnNames(kPromotion);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId catalogSalesScanId, dateDimScanId, itemScanId,
      custDemoScanId, promoScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2000"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto custDemoNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kCustomerDemographics,
                              custDemoType,
                              cdFileColumnNames,
                              {"cd_gender = 'M'",
                               "cd_marital_status = 'S'",
                               "cd_education_status = 'College'"})
                          .captureScanNodeId(custDemoScanId)
                          .planNode();

  auto promoNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kPromotion,
                           promoType,
                           promoFileColumnNames,
                           {"p_channel_email = 'N'"})
                       .captureScanNodeId(promoScanId)
                       .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogSales, catalogSalesType, csFileColumnNames, {})
          .captureScanNodeId(catalogSalesScanId)
          .hashJoin(
              {"cs_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"cs_item_sk",
               "cs_bill_cdemo_sk",
               "cs_promo_sk",
               "cs_quantity",
               "cs_list_price",
               "cs_sales_price",
               "cs_coupon_amt"})
          .hashJoin(
              {"cs_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"cs_bill_cdemo_sk",
               "cs_promo_sk",
               "cs_quantity",
               "cs_list_price",
               "cs_sales_price",
               "cs_coupon_amt",
               "i_item_id"})
          .hashJoin(
              {"cs_bill_cdemo_sk"},
              {"cd_demo_sk"},
              custDemoNode,
              "",
              {"cs_promo_sk",
               "cs_quantity",
               "cs_list_price",
               "cs_sales_price",
               "cs_coupon_amt",
               "i_item_id"})
          .hashJoin(
              {"cs_promo_sk"},
              {"p_promo_sk"},
              promoNode,
              "",
              {"cs_quantity",
               "cs_list_price",
               "cs_sales_price",
               "cs_coupon_amt",
               "i_item_id"})
          .partialAggregation(
              {"i_item_id"},
              {"avg(cs_quantity) as agg1",
               "avg(cs_list_price) as agg2",
               "avg(cs_coupon_amt) as agg3",
               "avg(cs_sales_price) as agg4"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"i_item_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[catalogSalesScanId] = getTableFilePaths(kCatalogSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[custDemoScanId] = getTableFilePaths(kCustomerDemographics);
  result.dataFiles[promoScanId] = getTableFilePaths(kPromotion);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q26";
  return result;
}

// Q29: store_sales with store_returns - another low selectivity opportunity
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation. (Bolt's unused catalog_sales/store column lists were dropped
// in this port.)
TpchPlan TpcdsQueryBuilder::getQ29Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_item_sk",
      "ss_ticket_number",
      "ss_customer_sk",
      "ss_quantity"};
  std::vector<std::string> storeReturnsCols = {
      "sr_returned_date_sk",
      "sr_item_sk",
      "sr_ticket_number",
      "sr_customer_sk",
      "sr_return_quantity"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_item_id", "i_item_desc"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto storeReturnsType = getRowType(kStoreReturns, storeReturnsCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto itemType = getRowType(kItem, itemCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& srFileColumnNames = getFileColumnNames(kStoreReturns);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, storeReturnsScanId, dateDim1ScanId,
      dateDim2ScanId;
  core::PlanNodeId itemScanId;

  auto dateDim1Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDimType,
                              ddFileColumnNames,
                              {"d_year = 1999", "d_moy = 9"})
                          .captureScanNodeId(dateDim1ScanId)
                          .planNode();

  auto dateDim2Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDimType,
                              ddFileColumnNames,
                              {"d_year IN (1999, 2000, 2001)"})
                          .captureScanNodeId(dateDim2ScanId)
                          .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  // store_returns with date filter
  auto storeReturnsNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreReturns, storeReturnsType, srFileColumnNames, {})
          .captureScanNodeId(storeReturnsScanId)
          .hashJoin(
              {"sr_returned_date_sk"},
              {"d_date_sk"},
              dateDim2Node,
              "",
              {"sr_item_sk",
               "sr_ticket_number",
               "sr_customer_sk",
               "sr_return_quantity"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDim1Node,
              "",
              {"ss_item_sk", "ss_ticket_number", "ss_customer_sk", "ss_quantity"})
          .hashJoin(
              {"ss_item_sk", "ss_ticket_number", "ss_customer_sk"},
              {"sr_item_sk", "sr_ticket_number", "sr_customer_sk"},
              storeReturnsNode,
              "",
              {"ss_item_sk", "ss_quantity", "sr_return_quantity"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_quantity", "sr_return_quantity", "i_item_id", "i_item_desc"})
          .partialAggregation(
              {"i_item_id", "i_item_desc"},
              {"sum(ss_quantity) as store_qty",
               "sum(sr_return_quantity) as return_qty"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"i_item_id", "i_item_desc"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[storeReturnsScanId] = getTableFilePaths(kStoreReturns);
  result.dataFiles[dateDim1ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[dateDim2ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q29";
  return result;
}

// Q30: web_returns with customer (simplified without CTE and correlated
// subquery)
// SELECT c_customer_id, c_salutation, c_first_name, c_last_name,
//        sum(wr_return_amt) as total_return
// FROM web_returns, date_dim, customer_address, customer
// WHERE wr_returned_date_sk = d_date_sk AND d_year = 2002
//   AND wr_returning_addr_sk = ca_address_sk
//   AND ca_state = 'GA'
//   AND wr_returning_customer_sk = c_customer_sk
// GROUP BY c_customer_id, c_salutation, c_first_name, c_last_name
// ORDER BY c_customer_id
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ30Plan() const {
  std::vector<std::string> webReturnsCols = {
      "wr_returned_date_sk",
      "wr_returning_customer_sk",
      "wr_returning_addr_sk",
      "wr_return_amt"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> customerAddrCols = {"ca_address_sk", "ca_state"};
  std::vector<std::string> customerCols = {
      "c_customer_sk",
      "c_customer_id",
      "c_salutation",
      "c_first_name",
      "c_last_name"};

  auto webReturnsType = getRowType(kWebReturns, webReturnsCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto customerAddrType = getRowType(kCustomerAddress, customerAddrCols);
  auto customerType = getRowType(kCustomer, customerCols);

  const auto& wrFileColumnNames = getFileColumnNames(kWebReturns);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId webReturnsScanId;
  core::PlanNodeId dateDimScanId;
  core::PlanNodeId customerAddrScanId;
  core::PlanNodeId customerScanId;

  // date_dim with filter d_year = 2002
  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2002"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  // customer_address with filter ca_state = 'GA'
  auto custAddrNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kCustomerAddress,
                              customerAddrType,
                              caFileColumnNames,
                              {"ca_state = 'GA'"})
                          .captureScanNodeId(customerAddrScanId)
                          .planNode();

  // customer
  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  // Build join chain:
  // web_returns -> JOIN date_dim -> JOIN customer_address -> JOIN customer
  //   -> aggregate -> sort
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kWebReturns, webReturnsType, wrFileColumnNames, {})
          .captureScanNodeId(webReturnsScanId)
          .hashJoin(
              {"wr_returned_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"wr_returning_customer_sk",
               "wr_returning_addr_sk",
               "wr_return_amt"})
          .hashJoin(
              {"wr_returning_addr_sk"},
              {"ca_address_sk"},
              custAddrNode,
              "",
              {"wr_returning_customer_sk", "wr_return_amt"})
          .hashJoin(
              {"wr_returning_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"c_customer_id",
               "c_salutation",
               "c_first_name",
               "c_last_name",
               "wr_return_amt"})
          .partialAggregation(
              {"c_customer_id", "c_salutation", "c_first_name", "c_last_name"},
              {"sum(wr_return_amt) as total_return"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"c_customer_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[webReturnsScanId] = getTableFilePaths(kWebReturns);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[customerAddrScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q30";
  return result;
}

// Q46: store_sales with multiple dimension joins and customer info
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ46Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_customer_sk",
      "ss_hdemo_sk",
      "ss_addr_sk",
      "ss_store_sk",
      "ss_ticket_number",
      "ss_coupon_amt",
      "ss_net_profit"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_dow"};
  std::vector<std::string> storeCols = {"s_store_sk", "s_city"};
  std::vector<std::string> hdCols = {
      "hd_demo_sk", "hd_dep_count", "hd_vehicle_count"};
  std::vector<std::string> customerCols = {
      "c_customer_sk", "c_current_addr_sk", "c_first_name", "c_last_name"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_city"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto customerType = getRowType(kCustomer, customerCols);
  auto caType = getRowType(kCustomerAddress, caCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, storeScanId, hdScanId,
      customerScanId, caScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year IN (1999, 2000, 2001)", "d_dow IN (6, 0)"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto storeNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kStore,
                           storeType,
                           storeFileColumnNames,
                           {"s_city IN ('Midway', 'Fairview')"})
                       .captureScanNodeId(storeScanId)
                       .planNode();

  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics, hdType, hdFileColumnNames, {})
                    .captureScanNodeId(hdScanId)
                    .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto caNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
          .captureScanNodeId(caScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_customer_sk",
               "ss_hdemo_sk",
               "ss_addr_sk",
               "ss_store_sk",
               "ss_ticket_number",
               "ss_coupon_amt",
               "ss_net_profit"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_customer_sk",
               "ss_hdemo_sk",
               "ss_ticket_number",
               "ss_coupon_amt",
               "ss_net_profit",
               "s_city"})
          .hashJoin(
              {"ss_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"ss_customer_sk",
               "ss_ticket_number",
               "ss_coupon_amt",
               "ss_net_profit",
               "s_city"})
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"ss_ticket_number",
               "ss_coupon_amt",
               "ss_net_profit",
               "s_city",
               "c_current_addr_sk",
               "c_first_name",
               "c_last_name"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"ss_ticket_number",
               "ss_coupon_amt",
               "ss_net_profit",
               "s_city",
               "c_first_name",
               "c_last_name",
               "ca_city"})
          .partialAggregation(
              {"c_last_name",
               "c_first_name",
               "ca_city",
               "s_city",
               "ss_ticket_number"},
              {"sum(ss_coupon_amt) as amt", "sum(ss_net_profit) as profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"c_last_name",
               "c_first_name",
               "ca_city",
               "s_city",
               "ss_ticket_number"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q46";
  return result;
}

// Q68: Customer purchases detail with multiple dimensions
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ68Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_customer_sk",
      "ss_hdemo_sk",
      "ss_addr_sk",
      "ss_store_sk",
      "ss_ticket_number",
      "ss_ext_sales_price",
      "ss_ext_list_price",
      "ss_ext_tax"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_dom"};
  std::vector<std::string> storeCols = {"s_store_sk", "s_city"};
  std::vector<std::string> hdCols = {
      "hd_demo_sk", "hd_dep_count", "hd_vehicle_count"};
  std::vector<std::string> customerCols = {
      "c_customer_sk", "c_current_addr_sk", "c_first_name", "c_last_name"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_city"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto customerType = getRowType(kCustomer, customerCols);
  auto caType = getRowType(kCustomerAddress, caCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, storeScanId, hdScanId,
      customerScanId, caScanId;

  auto dateDimNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kDateDim,
              dateDimType,
              ddFileColumnNames,
              {"d_year IN (1999, 2000, 2001)", "d_dom BETWEEN 1 AND 2"})
          .captureScanNodeId(dateDimScanId)
          .planNode();

  auto storeNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kStore,
                           storeType,
                           storeFileColumnNames,
                           {"s_city IN ('Midway', 'Fairview')"})
                       .captureScanNodeId(storeScanId)
                       .planNode();

  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics, hdType, hdFileColumnNames, {})
                    .captureScanNodeId(hdScanId)
                    .filter("hd_dep_count = 4 OR hd_vehicle_count = 3")
                    .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto caNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
          .captureScanNodeId(caScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_customer_sk",
               "ss_hdemo_sk",
               "ss_addr_sk",
               "ss_store_sk",
               "ss_ticket_number",
               "ss_ext_sales_price",
               "ss_ext_list_price",
               "ss_ext_tax"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_customer_sk",
               "ss_hdemo_sk",
               "ss_ticket_number",
               "ss_ext_sales_price",
               "ss_ext_list_price",
               "ss_ext_tax",
               "s_city"})
          .hashJoin(
              {"ss_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"ss_customer_sk",
               "ss_ticket_number",
               "ss_ext_sales_price",
               "ss_ext_list_price",
               "ss_ext_tax",
               "s_city"})
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"ss_ticket_number",
               "ss_ext_sales_price",
               "ss_ext_list_price",
               "ss_ext_tax",
               "s_city",
               "c_current_addr_sk",
               "c_first_name",
               "c_last_name"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"ss_ticket_number",
               "ss_ext_sales_price",
               "ss_ext_list_price",
               "ss_ext_tax",
               "s_city",
               "c_first_name",
               "c_last_name",
               "ca_city"})
          .partialAggregation(
              {"c_last_name",
               "c_first_name",
               "ca_city",
               "s_city",
               "ss_ticket_number"},
              {"sum(ss_ext_sales_price) as bought_city_amt",
               "sum(ss_ext_list_price) as list_amt",
               "sum(ss_ext_tax) as tax_amt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"c_last_name",
               "c_first_name",
               "ca_city",
               "s_city",
               "ss_ticket_number"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q68";
  return result;
}

// Q73: Store ticket analysis
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ73Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_customer_sk",
      "ss_hdemo_sk",
      "ss_store_sk",
      "ss_ticket_number"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_dom"};
  std::vector<std::string> storeCols = {"s_store_sk", "s_county"};
  std::vector<std::string> hdCols = {
      "hd_demo_sk", "hd_buy_potential", "hd_dep_count", "hd_vehicle_count"};
  std::vector<std::string> customerCols = {
      "c_customer_sk",
      "c_salutation",
      "c_first_name",
      "c_last_name",
      "c_preferred_cust_flag"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto customerType = getRowType(kCustomer, customerCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, storeScanId, hdScanId,
      customerScanId;

  auto dateDimNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kDateDim,
              dateDimType,
              ddFileColumnNames,
              {"d_year IN (1999, 2000, 2001)", "d_dom BETWEEN 1 AND 2"})
          .captureScanNodeId(dateDimScanId)
          .planNode();

  auto storeNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kStore,
                           storeType,
                           storeFileColumnNames,
                           {"s_county IN ('Williamson County')"})
                       .captureScanNodeId(storeScanId)
                       .planNode();

  // Note: "hd_vehicle_count > 0" is a FilterNode (not a scan subfield filter)
  // because hd_vehicle_count is INTEGER in the parquet files and velox's
  // subfield-filter conversion rejects the implicit INTEGER->BIGINT cast.
  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics,
                        hdType,
                        hdFileColumnNames,
                        {"hd_buy_potential = '>10000'"})
                    .captureScanNodeId(hdScanId)
                    .filter("hd_vehicle_count > 0")
                    .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_customer_sk", "ss_hdemo_sk", "ss_store_sk", "ss_ticket_number"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_customer_sk", "ss_hdemo_sk", "ss_ticket_number", "s_county"})
          .hashJoin(
              {"ss_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"ss_customer_sk", "ss_ticket_number", "s_county"})
          .partialAggregation(
              {"ss_customer_sk", "ss_ticket_number", "s_county"},
              {"count(1) as cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"ss_ticket_number",
               "s_county",
               "cnt",
               "c_salutation",
               "c_first_name",
               "c_last_name",
               "c_preferred_cust_flag"})
          .orderBy({"cnt DESC", "c_last_name"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q73";
  return result;
}

// Q79: store_sales with customer info - profitability by customer
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ79Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_customer_sk",
      "ss_hdemo_sk",
      "ss_store_sk",
      "ss_ticket_number",
      "ss_net_profit"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_dow"};
  std::vector<std::string> storeCols = {
      "s_store_sk", "s_number_employees", "s_city"};
  std::vector<std::string> hdCols = {
      "hd_demo_sk", "hd_dep_count", "hd_vehicle_count"};
  std::vector<std::string> customerCols = {
      "c_customer_sk", "c_first_name", "c_last_name"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto customerType = getRowType(kCustomer, customerCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, storeScanId, hdScanId,
      customerScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2000", "d_dow IN (6, 0)"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto storeNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kStore,
                           storeType,
                           storeFileColumnNames,
                           {"s_number_employees BETWEEN 200 AND 295"})
                       .captureScanNodeId(storeScanId)
                       .planNode();

  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics, hdType, hdFileColumnNames, {})
                    .captureScanNodeId(hdScanId)
                    .filter("hd_dep_count = 6 OR hd_vehicle_count > 2")
                    .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_customer_sk",
               "ss_hdemo_sk",
               "ss_store_sk",
               "ss_ticket_number",
               "ss_net_profit"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_customer_sk",
               "ss_hdemo_sk",
               "ss_ticket_number",
               "ss_net_profit",
               "s_city"})
          .hashJoin(
              {"ss_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"ss_customer_sk", "ss_ticket_number", "ss_net_profit", "s_city"})
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"ss_ticket_number",
               "ss_net_profit",
               "s_city",
               "c_first_name",
               "c_last_name"})
          .partialAggregation(
              {"c_last_name", "c_first_name", "s_city", "ss_ticket_number"},
              {"sum(ss_net_profit) as profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"c_last_name", "c_first_name", "s_city", "profit"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q79";
  return result;
}

// Q7: Promotional sales analysis
// store_sales -> date_dim -> item -> customer_demographics -> promotion
// Low selectivity: promotion and demographics filters.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ7Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_item_sk",
      "ss_cdemo_sk",
      "ss_promo_sk",
      "ss_quantity",
      "ss_list_price",
      "ss_sales_price",
      "ss_coupon_amt"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_item_id"};
  std::vector<std::string> custDemoCols = {
      "cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"};
  std::vector<std::string> promoCols = {
      "p_promo_sk", "p_channel_email", "p_channel_event"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto itemType = getRowType(kItem, itemCols);
  auto custDemoType = getRowType(kCustomerDemographics, custDemoCols);
  auto promoType = getRowType(kPromotion, promoCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& cdFileColumnNames = getFileColumnNames(kCustomerDemographics);
  const auto& promoFileColumnNames = getFileColumnNames(kPromotion);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, itemScanId, custDemoScanId,
      promoScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2000"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto custDemoNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kCustomerDemographics,
                              custDemoType,
                              cdFileColumnNames,
                              {"cd_gender = 'M'",
                               "cd_marital_status = 'S'",
                               "cd_education_status = 'College'"})
                          .captureScanNodeId(custDemoScanId)
                          .planNode();

  auto promoNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPromotion, promoType, promoFileColumnNames, {})
          .captureScanNodeId(promoScanId)
          .filter("p_channel_email = 'N' OR p_channel_event = 'N'")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk",
               "ss_cdemo_sk",
               "ss_promo_sk",
               "ss_quantity",
               "ss_list_price",
               "ss_sales_price",
               "ss_coupon_amt"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_cdemo_sk",
               "ss_promo_sk",
               "ss_quantity",
               "ss_list_price",
               "ss_sales_price",
               "ss_coupon_amt",
               "i_item_id"})
          .hashJoin(
              {"ss_cdemo_sk"},
              {"cd_demo_sk"},
              custDemoNode,
              "",
              {"ss_promo_sk",
               "ss_quantity",
               "ss_list_price",
               "ss_sales_price",
               "ss_coupon_amt",
               "i_item_id"})
          .hashJoin(
              {"ss_promo_sk"},
              {"p_promo_sk"},
              promoNode,
              "",
              {"ss_quantity",
               "ss_list_price",
               "ss_sales_price",
               "ss_coupon_amt",
               "i_item_id"})
          .partialAggregation(
              {"i_item_id"},
              {"avg(ss_quantity) as agg1",
               "avg(ss_list_price) as agg2",
               "avg(ss_coupon_amt) as agg3",
               "avg(ss_sales_price) as agg4"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"i_item_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[custDemoScanId] = getTableFilePaths(kCustomerDemographics);
  result.dataFiles[promoScanId] = getTableFilePaths(kPromotion);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q7";
  return result;
}

// Q13: store_sales with multiple dimension joins and filters.
// NOTE (Bolt deviation): global aggregate, single output row, and Bolt's plan
// has no final ORDER BY — preserved as-is (nothing to sort).
TpchPlan TpcdsQueryBuilder::getQ13Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk",
      "ss_customer_sk",
      "ss_hdemo_sk",
      "ss_addr_sk",
      "ss_store_sk",
      "ss_quantity",
      "ss_ext_sales_price",
      "ss_ext_wholesale_cost",
      "ss_net_profit"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> storeCols = {"s_store_sk", "s_store_name"};
  std::vector<std::string> hdCols = {"hd_demo_sk", "hd_dep_count"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_state"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto caType = getRowType(kCustomerAddress, caCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, storeScanId, hdScanId,
      caScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2001"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .planNode();

  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics, hdType, hdFileColumnNames, {})
                    .captureScanNodeId(hdScanId)
                    .planNode();

  auto caNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kCustomerAddress,
                        caType,
                        caFileColumnNames,
                        {"ca_state IN ('TX', 'OH')"})
                    .captureScanNodeId(caScanId)
                    .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_hdemo_sk",
               "ss_addr_sk",
               "ss_store_sk",
               "ss_quantity",
               "ss_ext_sales_price",
               "ss_ext_wholesale_cost",
               "ss_net_profit"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_hdemo_sk",
               "ss_addr_sk",
               "ss_quantity",
               "ss_ext_sales_price",
               "ss_ext_wholesale_cost",
               "ss_net_profit"})
          .hashJoin(
              {"ss_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"ss_addr_sk",
               "ss_quantity",
               "ss_ext_sales_price",
               "ss_ext_wholesale_cost",
               "ss_net_profit"})
          .hashJoin(
              {"ss_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"ss_quantity",
               "ss_ext_sales_price",
               "ss_ext_wholesale_cost",
               "ss_net_profit"})
          .partialAggregation(
              {},
              {"avg(ss_quantity) as avg_qty",
               "avg(ss_ext_sales_price) as avg_sales",
               "avg(ss_ext_wholesale_cost) as avg_cost",
               "sum(ss_net_profit) as sum_profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q13";
  return result;
}

// Q15: catalog_sales with customer and customer_address, sales by zip.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ15Plan() const {
  std::vector<std::string> catalogSalesCols = {
      "cs_sold_date_sk", "cs_bill_customer_sk", "cs_sales_price"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_qoy"};
  std::vector<std::string> customerCols = {
      "c_customer_sk", "c_current_addr_sk"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_zip"};

  auto catalogSalesType = getRowType(kCatalogSales, catalogSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto customerType = getRowType(kCustomer, customerCols);
  auto caType = getRowType(kCustomerAddress, caCols);

  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId catalogSalesScanId, dateDimScanId, customerScanId, caScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2001", "d_qoy = 2"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto caNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
          .captureScanNodeId(caScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogSales, catalogSalesType, csFileColumnNames, {})
          .captureScanNodeId(catalogSalesScanId)
          .hashJoin(
              {"cs_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"cs_bill_customer_sk", "cs_sales_price"})
          .hashJoin(
              {"cs_bill_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"cs_sales_price", "c_current_addr_sk"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"cs_sales_price", "ca_zip"})
          .partialAggregation({"ca_zip"}, {"sum(cs_sales_price) as sum_sales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"ca_zip"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[catalogSalesScanId] = getTableFilePaths(kCatalogSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q15";
  return result;
}

// Q18: catalog_sales with item carrying MANY payload columns (ideal for
// hybrid join).
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ18Plan() const {
  std::vector<std::string> itemCols = {
      "i_item_sk",
      "i_item_id",
      "i_item_desc",
      "i_current_price",
      "i_wholesale_cost",
      "i_brand",
      "i_class",
      "i_category",
      "i_manufact",
      "i_size",
      "i_color"};
  std::vector<std::string> catalogSalesCols = {
      "cs_sold_date_sk",
      "cs_item_sk",
      "cs_bill_cdemo_sk",
      "cs_quantity",
      "cs_list_price"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> customerDemoCols = {
      "cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status"};

  auto itemType = getRowType(kItem, itemCols);
  auto catalogSalesType = getRowType(kCatalogSales, catalogSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto customerDemoType = getRowType(kCustomerDemographics, customerDemoCols);

  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& cdFileColumnNames = getFileColumnNames(kCustomerDemographics);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId itemScanId;
  core::PlanNodeId catalogSalesScanId;
  core::PlanNodeId dateDimScanId;
  core::PlanNodeId customerDemoScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2001"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  // item - NO filter, many payload columns
  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto custDemoNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kCustomerDemographics,
              customerDemoType,
              cdFileColumnNames,
              {"cd_gender = 'M'", "cd_education_status = 'College'"})
          .captureScanNodeId(customerDemoScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogSales, catalogSalesType, csFileColumnNames, {})
          .captureScanNodeId(catalogSalesScanId)
          .hashJoin(
              {"cs_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"cs_item_sk", "cs_bill_cdemo_sk", "cs_quantity", "cs_list_price"})
          .hashJoin(
              {"cs_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"cs_bill_cdemo_sk",
               "cs_quantity",
               "cs_list_price",
               "i_item_id",
               "i_item_desc",
               "i_current_price",
               "i_wholesale_cost",
               "i_brand",
               "i_class",
               "i_category",
               "i_manufact",
               "i_size",
               "i_color"})
          .hashJoin(
              {"cs_bill_cdemo_sk"},
              {"cd_demo_sk"},
              custDemoNode,
              "",
              {"cs_quantity",
               "cs_list_price",
               "i_item_id",
               "i_item_desc",
               "i_current_price",
               "i_wholesale_cost",
               "i_brand",
               "i_class",
               "i_category",
               "i_manufact",
               "i_size",
               "i_color",
               "cd_gender",
               "cd_marital_status",
               "cd_education_status"})
          .partialAggregation(
              {"i_item_id",
               "i_item_desc",
               "i_current_price",
               "i_wholesale_cost",
               "i_brand",
               "i_class",
               "i_category",
               "i_manufact",
               "i_size",
               "i_color",
               "cd_gender",
               "cd_marital_status",
               "cd_education_status"},
              {"count(1) as cnt",
               "avg(cs_quantity) as agg1",
               "avg(cs_list_price) as agg2"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"cnt DESC"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[catalogSalesScanId] = getTableFilePaths(kCatalogSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[customerDemoScanId] =
      getTableFilePaths(kCustomerDemographics);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q18";
  return result;
}

// Q19: store_sales with item, customer, customer_address - regional brand
// analysis.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ19Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_item_sk", "ss_customer_sk", "ss_ext_sales_price"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> itemCols = {
      "i_item_sk", "i_brand_id", "i_brand", "i_manufact_id", "i_manager_id"};
  std::vector<std::string> customerCols = {
      "c_customer_sk", "c_current_addr_sk"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_zip"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto itemType = getRowType(kItem, itemCols);
  auto customerType = getRowType(kCustomer, customerCols);
  auto caType = getRowType(kCustomerAddress, caCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, dateDimScanId, itemScanId, customerScanId,
      caScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 1998", "d_moy = 11"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kItem,
                          itemType,
                          itemFileColumnNames,
                          {"i_manager_id = 8"})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto caNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
          .captureScanNodeId(caScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk", "ss_customer_sk", "ss_ext_sales_price"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_customer_sk",
               "ss_ext_sales_price",
               "i_brand_id",
               "i_brand",
               "i_manufact_id"})
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"ss_ext_sales_price",
               "i_brand_id",
               "i_brand",
               "i_manufact_id",
               "c_current_addr_sk"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"ss_ext_sales_price",
               "i_brand_id",
               "i_brand",
               "i_manufact_id",
               "ca_zip"})
          .partialAggregation(
              {"i_brand_id", "i_brand", "i_manufact_id"},
              {"sum(ss_ext_sales_price) as ext_price"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"ext_price DESC", "i_brand", "i_brand_id", "i_manufact_id"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q19";
  return result;
}

// Q42: date_dim, store_sales, item - category sales for a month.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ42Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"};
  std::vector<std::string> itemCols = {
      "i_item_sk", "i_category_id", "i_category", "i_manager_id"};

  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto itemType = getRowType(kItem, itemCols);

  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId dateDimScanId, storeSalesScanId, itemScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2000", "d_moy = 11"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kItem,
                          itemType,
                          itemFileColumnNames,
                          {"i_manager_id = 1"})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk", "ss_ext_sales_price", "d_year"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"d_year", "i_category_id", "i_category", "ss_ext_sales_price"})
          .partialAggregation(
              {"d_year", "i_category_id", "i_category"},
              {"sum(ss_ext_sales_price) as sum_sales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"sum_sales DESC", "d_year", "i_category_id", "i_category"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q42";
  return result;
}

// Q43: date_dim, store_sales, store - day of week sales by store.
// Note: "s_gmt_offset = -5" runs as a FilterNode with an explicit double cast
// (not a scan subfield filter) because s_gmt_offset is DECIMAL(5,2) in the
// parquet files and velox's subfield-filter conversion does not accept the
// decimal/integer comparison.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ43Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_day_name"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_store_sk", "ss_sales_price"};
  std::vector<std::string> storeCols = {
      "s_store_sk", "s_store_id", "s_store_name", "s_gmt_offset"};

  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId dateDimScanId, storeSalesScanId, storeScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2000"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .filter("cast(s_gmt_offset as double) = -5.0")
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_store_sk", "ss_sales_price", "d_day_name"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_sales_price", "d_day_name", "s_store_id", "s_store_name"})
          .partialAggregation(
              {"s_store_name", "s_store_id", "d_day_name"},
              {"sum(ss_sales_price) as day_sales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"s_store_name", "s_store_id", "d_day_name"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q43";
  return result;
}

// Q52: brand sales for a month (like Q3 but no item filter, item full build).
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ52Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_brand_id", "i_brand"};

  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto itemType = getRowType(kItem, itemCols);

  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId dateDimScanId, storeSalesScanId, itemScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 2000", "d_moy = 11"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk", "ss_ext_sales_price", "d_year"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"d_year", "i_brand_id", "i_brand", "ss_ext_sales_price"})
          .partialAggregation(
              {"d_year", "i_brand_id", "i_brand"},
              {"sum(ss_ext_sales_price) as ext_price"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"d_year", "ext_price DESC", "i_brand_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q52";
  return result;
}

// Q53: Sales by item manufacturer with store and date filters.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ53Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_month_seq"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_item_sk", "ss_store_sk", "ss_sales_price"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_manufact_id"};
  std::vector<std::string> storeCols = {
      "s_store_sk", "s_store_name", "s_gmt_offset"};

  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto itemType = getRowType(kItem, itemCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId dateDimScanId, storeSalesScanId, itemScanId, storeScanId;

  auto dateDimNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kDateDim,
              dateDimType,
              ddFileColumnNames,
              {"d_month_seq IN (1200, 1201, 1202, 1203, 1204, 1205, 1206, "
               "1207, 1208, 1209, 1210, 1211)"})
          .captureScanNodeId(dateDimScanId)
          .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kItem,
                          itemType,
                          itemFileColumnNames,
                          {"i_manufact_id IN (128, 129, 130, 131)"})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk", "ss_store_sk", "ss_sales_price", "d_month_seq"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_store_sk", "ss_sales_price", "d_month_seq", "i_manufact_id"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"ss_sales_price", "d_month_seq", "i_manufact_id", "s_store_name"})
          .partialAggregation(
              {"i_manufact_id", "d_month_seq"},
              {"sum(ss_sales_price) as sum_sales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"sum_sales", "i_manufact_id", "d_month_seq"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q53";
  return result;
}

// Q55: Simple brand sales in a time period.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ55Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_item_sk", "ss_ext_sales_price"};
  std::vector<std::string> itemCols = {
      "i_item_sk", "i_brand_id", "i_brand", "i_manager_id"};

  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto itemType = getRowType(kItem, itemCols);

  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId dateDimScanId, storeSalesScanId, itemScanId;

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 1999", "d_moy = 11"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kItem,
                          itemType,
                          itemFileColumnNames,
                          {"i_manager_id = 28"})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_item_sk", "ss_ext_sales_price"})
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_ext_sales_price", "i_brand_id", "i_brand"})
          .partialAggregation(
              {"i_brand_id", "i_brand"},
              {"sum(ss_ext_sales_price) as ext_price"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"ext_price DESC", "i_brand_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q55";
  return result;
}

// Q63: 4-way join item, store_sales, date_dim, store with aggregation
// SELECT i_manager_id, sum(ss_sales_price) as sum_sales
// FROM item, store_sales, date_dim, store
// WHERE ss_item_sk = i_item_sk AND ss_sold_date_sk = d_date_sk
//   AND ss_store_sk = s_store_sk AND d_month_seq BETWEEN 1176 AND 1187
// GROUP BY i_manager_id, d_moy
// ORDER BY i_manager_id, sum_sales
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ63Plan() const {
  std::vector<std::string> itemCols = {"i_item_sk", "i_manager_id"};
  std::vector<std::string> storeSalesCols = {
      "ss_item_sk", "ss_sold_date_sk", "ss_store_sk", "ss_sales_price"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_month_seq", "d_moy"};
  std::vector<std::string> storeCols = {"s_store_sk", "s_store_name"};

  auto itemType = getRowType(kItem, itemCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId itemScanId;
  core::PlanNodeId storeSalesScanId;
  core::PlanNodeId dateDimScanId;
  core::PlanNodeId storeScanId;

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_month_seq between 1176 and 1187"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_sold_date_sk", "ss_store_sk", "ss_sales_price", "i_manager_id"})
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_store_sk", "ss_sales_price", "i_manager_id", "d_moy"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"i_manager_id", "d_moy", "ss_sales_price"})
          .partialAggregation(
              {"i_manager_id", "d_moy"},
              {"sum(ss_sales_price) as sum_sales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"i_manager_id", "sum_sales"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q63";
  return result;
}

// Q89: 4-way join item, store_sales, date_dim, store with aggregation
// (similar to Q63)
// SELECT i_category, i_class, i_brand, s_store_name, s_company_name,
//        d_moy, sum(ss_sales_price) as sum_sales
// FROM item, store_sales, date_dim, store
// WHERE ss_item_sk = i_item_sk AND ss_sold_date_sk = d_date_sk
//   AND ss_store_sk = s_store_sk AND d_year = 1999
// GROUP BY i_category, i_class, i_brand, s_store_name, s_company_name, d_moy
// ORDER BY sum_sales, s_store_name
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ89Plan() const {
  std::vector<std::string> itemCols = {
      "i_item_sk", "i_category", "i_class", "i_brand"};
  std::vector<std::string> storeSalesCols = {
      "ss_item_sk", "ss_sold_date_sk", "ss_store_sk", "ss_sales_price"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year", "d_moy"};
  std::vector<std::string> storeCols = {
      "s_store_sk", "s_store_name", "s_company_name"};

  auto itemType = getRowType(kItem, itemCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId itemScanId;
  core::PlanNodeId storeSalesScanId;
  core::PlanNodeId dateDimScanId;
  core::PlanNodeId storeScanId;

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kDateDim,
                             dateDimType,
                             ddFileColumnNames,
                             {"d_year = 1999"})
                         .captureScanNodeId(dateDimScanId)
                         .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"ss_sold_date_sk",
               "ss_store_sk",
               "ss_sales_price",
               "i_category",
               "i_class",
               "i_brand"})
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"ss_store_sk",
               "ss_sales_price",
               "i_category",
               "i_class",
               "i_brand",
               "d_moy"})
          .hashJoin(
              {"ss_store_sk"},
              {"s_store_sk"},
              storeNode,
              "",
              {"i_category",
               "i_class",
               "i_brand",
               "s_store_name",
               "s_company_name",
               "d_moy",
               "ss_sales_price"})
          .partialAggregation(
              {"i_category",
               "i_class",
               "i_brand",
               "s_store_name",
               "s_company_name",
               "d_moy"},
              {"sum(ss_sales_price) as sum_sales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"sum_sales", "s_store_name"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q89";
  return result;
}

// Q96: count of store_sales in a time window with hd and store filters.
// NOTE (Bolt deviation): single-row global count with a trivial 1-row sort.
// LIMIT omitted (vs. spec, as everywhere in this file).
TpchPlan TpcdsQueryBuilder::getQ96Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_sold_time_sk", "ss_hdemo_sk", "ss_store_sk"};
  std::vector<std::string> timeDimCols = {"t_time_sk", "t_hour", "t_minute"};
  std::vector<std::string> hdCols = {"hd_demo_sk", "hd_dep_count"};
  std::vector<std::string> storeCols = {"s_store_sk", "s_store_name"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto timeDimType = getRowType(kTimeDim, timeDimCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& tdFileColumnNames = getFileColumnNames(kTimeDim);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, timeDimScanId, hdScanId, storeScanId;

  auto timeDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kTimeDim,
                             timeDimType,
                             tdFileColumnNames,
                             {"t_hour = 20", "t_minute >= 30"})
                         .captureScanNodeId(timeDimScanId)
                         .planNode();

  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics,
                        hdType,
                        hdFileColumnNames,
                        {"hd_dep_count = 7"})
                    .captureScanNodeId(hdScanId)
                    .planNode();

  auto storeNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(
                           kStore,
                           storeType,
                           storeFileColumnNames,
                           {"s_store_name = 'ese'"})
                       .captureScanNodeId(storeScanId)
                       .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_sold_time_sk"},
              {"t_time_sk"},
              timeDimNode,
              "",
              {"ss_hdemo_sk", "ss_store_sk"})
          .hashJoin({"ss_hdemo_sk"}, {"hd_demo_sk"}, hdNode, "", {"ss_store_sk"})
          .hashJoin({"ss_store_sk"}, {"s_store_sk"}, storeNode, "", {})
          .partialAggregation({}, {"count(1) as cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"cnt"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[timeDimScanId] = getTableFilePaths(kTimeDim);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q96";
  return result;
}

// Q74 (hand-built, not in Bolt): year-over-year customer sales ratio.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q74).
// year_total CTE = per-customer/year sum of ss_net_paid (store channel) and
// ws_net_paid (web channel) for d_year in (2001, 2002). The CTE's four
// filtered instances are built as four independent aggregation subtrees
// (velox plans are trees, not DAGs), joined on customer id with the
// year-over-year ratio filter web_ratio > store_ratio applied as a join
// filter on the last join. Totals are compared as doubles (duckdb's decimal
// division also yields double). ORDER BY customer_id.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ74Plan() const {
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& wsFileColumnNames = getFileColumnNames(kWebSales);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  TpchPlan result;

  // Builds one instance of the year_total CTE: per-customer sum of net_paid
  // for one channel (store or web) and one year. Output columns are renamed
  // with `prefix` ("ss1"/"ss2"/"ws1"/"ws2"); firstyear instances get the
  // year_total > 0 filter from the WHERE clause.
  auto makeYearTotal = [&](bool storeChannel,
                           int year,
                           bool withNames,
                           bool positiveOnly,
                           const std::string& prefix) -> core::PlanNodePtr {
    core::PlanNodeId factScanId, dateDimScanId, customerScanId;

    auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                           .filtersAsNode(filtersAsNode_)
                           .tableScan(
                               kDateDim,
                               dateDimType,
                               ddFileColumnNames,
                               {fmt::format("d_year = {}", year)})
                           .captureScanNodeId(dateDimScanId)
                           .planNode();

    std::vector<std::string> customerCols = {"c_customer_sk", "c_customer_id"};
    if (withNames) {
      customerCols.push_back("c_first_name");
      customerCols.push_back("c_last_name");
    }
    auto customerType = getRowType(kCustomer, customerCols);
    auto customerNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomer, customerType, custFileColumnNames, {})
            .captureScanNodeId(customerScanId)
            .planNode();

    const std::string dateKey =
        storeChannel ? "ss_sold_date_sk" : "ws_sold_date_sk";
    const std::string custKey =
        storeChannel ? "ss_customer_sk" : "ws_bill_customer_sk";
    const std::string netPaid = storeChannel ? "ss_net_paid" : "ws_net_paid";
    std::vector<std::string> factCols = {dateKey, custKey, netPaid};
    const auto* factTable = storeChannel ? kStoreSales : kWebSales;
    auto factType = getRowType(factTable, factCols);
    const auto& factFileColumnNames =
        storeChannel ? ssFileColumnNames : wsFileColumnNames;

    std::vector<std::string> groupKeys = {"c_customer_id"};
    std::vector<std::string> joinOutput = {netPaid, "c_customer_id"};
    if (withNames) {
      groupKeys.push_back("c_first_name");
      groupKeys.push_back("c_last_name");
      joinOutput.push_back("c_first_name");
      joinOutput.push_back("c_last_name");
    }

    std::vector<std::string> renames = {
        fmt::format("c_customer_id as {}_id", prefix),
        fmt::format("cast(year_total as double) as {}_total", prefix)};
    if (withNames) {
      renames.push_back(fmt::format("c_first_name as {}_first_name", prefix));
      renames.push_back(fmt::format("c_last_name as {}_last_name", prefix));
    }

    auto builder =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(factTable, factType, factFileColumnNames, {})
            .captureScanNodeId(factScanId)
            .hashJoin({dateKey}, {"d_date_sk"}, dateDimNode, "", {custKey, netPaid})
            .hashJoin({custKey}, {"c_customer_sk"}, customerNode, "", joinOutput)
            .partialAggregation(
                groupKeys, {fmt::format("sum({}) as year_total", netPaid)})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .project(renames);
    if (positiveOnly) {
      builder.filter(fmt::format("{}_total > 0.0", prefix));
    }
    auto node = builder.planNode();

    result.dataFiles[factScanId] = getTableFilePaths(factTable);
    result.dataFiles[dateDimScanId] = getTableFilePaths(kDateDim);
    result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
    return node;
  };

  // t_s_firstyear / t_w_firstyear (year_total > 0 per WHERE clause).
  auto sFirst = makeYearTotal(true, 2001, false, true, "ss1");
  auto wFirst = makeYearTotal(false, 2001, false, true, "ws1");
  // t_w_secyear.
  auto wSec = makeYearTotal(false, 2002, false, false, "ws2");

  // Probe side: t_s_secyear carries the customer id/name payload.
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .addNode([&](std::string /*id*/, core::PlanNodePtr /*input*/) {
            return makeYearTotal(true, 2002, true, false, "ss2");
          })
          .hashJoin(
              {"ss2_id"},
              {"ss1_id"},
              sFirst,
              "",
              {"ss2_id",
               "ss2_first_name",
               "ss2_last_name",
               "ss2_total",
               "ss1_total"})
          .hashJoin(
              {"ss2_id"},
              {"ws1_id"},
              wFirst,
              "",
              {"ss2_id",
               "ss2_first_name",
               "ss2_last_name",
               "ss2_total",
               "ss1_total",
               "ws1_total"})
          .hashJoin(
              {"ss2_id"},
              {"ws2_id"},
              wSec,
              "ws2_total / ws1_total > ss2_total / ss1_total",
              {"ss2_id", "ss2_first_name", "ss2_last_name"})
          .orderBy({"ss2_id"}, false)
          .planNode();

  result.plan = std::move(plan);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q74";
  return result;
}

// Q81 (hand-built, not in Bolt): catalog-returns analogue of Q30.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q81).
// CTE customer_total_return = per (returning customer, address state) sum of
// cr_return_amt_inc_tax over catalog_returns in d_year = 2000. Customers whose
// total return exceeds 1.2x the average total return of their state are joined
// with customer and their (GA-filtered) current address for a wide payload.
// The CTE is instantiated twice (velox plans are trees, not DAGs): once as the
// probe rows, once aggregated per state for the 1.2*avg threshold, joined on
// state with the threshold comparison as a join filter (matching the
// correlated subquery; totals compared as doubles, as duckdb does).
// ORDER BY all 16 output columns.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ81Plan() const {
  std::vector<std::string> catalogReturnsCols = {
      "cr_returned_date_sk",
      "cr_returning_customer_sk",
      "cr_returning_addr_sk",
      "cr_return_amt_inc_tax"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_year"};
  std::vector<std::string> ctrAddrCols = {"ca_address_sk", "ca_state"};
  std::vector<std::string> customerCols = {
      "c_customer_sk",
      "c_customer_id",
      "c_salutation",
      "c_first_name",
      "c_last_name",
      "c_current_addr_sk"};
  std::vector<std::string> gaAddrCols = {
      "ca_address_sk",
      "ca_street_number",
      "ca_street_name",
      "ca_street_type",
      "ca_suite_number",
      "ca_city",
      "ca_county",
      "ca_state",
      "ca_zip",
      "ca_country",
      "ca_gmt_offset",
      "ca_location_type"};

  auto catalogReturnsType = getRowType(kCatalogReturns, catalogReturnsCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  auto ctrAddrType = getRowType(kCustomerAddress, ctrAddrCols);
  auto customerType = getRowType(kCustomer, customerCols);
  auto gaAddrType = getRowType(kCustomerAddress, gaAddrCols);

  const auto& crFileColumnNames = getFileColumnNames(kCatalogReturns);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  TpchPlan result;

  // One instance of the customer_total_return CTE:
  // (ctr_customer_sk, ctr_state, ctr_total_return as double).
  auto makeCustomerTotalReturn = [&]() -> core::PlanNodePtr {
    core::PlanNodeId crScanId, ddScanId, caScanId;

    auto dateDimNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                           .filtersAsNode(filtersAsNode_)
                           .tableScan(
                               kDateDim,
                               dateDimType,
                               ddFileColumnNames,
                               {"d_year = 2000"})
                           .captureScanNodeId(ddScanId)
                           .planNode();

    auto ctrAddrNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomerAddress, ctrAddrType, caFileColumnNames, {})
            .captureScanNodeId(caScanId)
            .planNode();

    auto node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                kCatalogReturns, catalogReturnsType, crFileColumnNames, {})
            .captureScanNodeId(crScanId)
            .hashJoin(
                {"cr_returned_date_sk"},
                {"d_date_sk"},
                dateDimNode,
                "",
                {"cr_returning_customer_sk",
                 "cr_returning_addr_sk",
                 "cr_return_amt_inc_tax"})
            .hashJoin(
                {"cr_returning_addr_sk"},
                {"ca_address_sk"},
                ctrAddrNode,
                "",
                {"cr_returning_customer_sk",
                 "cr_return_amt_inc_tax",
                 "ca_state"})
            .partialAggregation(
                {"cr_returning_customer_sk", "ca_state"},
                {"sum(cr_return_amt_inc_tax) as ctr_total_return"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .project(
                {"cr_returning_customer_sk as ctr_customer_sk",
                 "ca_state as ctr_state",
                 "cast(ctr_total_return as double) as ctr_total_return"})
            .planNode();

    result.dataFiles[crScanId] = getTableFilePaths(kCatalogReturns);
    result.dataFiles[ddScanId] = getTableFilePaths(kDateDim);
    result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
    return node;
  };

  // Threshold side: avg(ctr_total_return) * 1.2 per state.
  auto thresholdNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .addNode([&](std::string /*id*/, core::PlanNodePtr /*input*/) {
            return makeCustomerTotalReturn();
          })
          .partialAggregation(
              {"ctr_state"}, {"avg(ctr_total_return) as avg_return"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"ctr_state as avg_state", "avg_return * 1.2 as threshold"})
          .planNode();

  core::PlanNodeId customerScanId, gaAddrScanId;
  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  auto gaAddrNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                        .filtersAsNode(filtersAsNode_)
                        .tableScan(
                            kCustomerAddress,
                            gaAddrType,
                            caFileColumnNames,
                            {"ca_state = 'GA'"})
                        .captureScanNodeId(gaAddrScanId)
                        .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .addNode([&](std::string /*id*/, core::PlanNodePtr /*input*/) {
            return makeCustomerTotalReturn();
          })
          .hashJoin(
              {"ctr_state"},
              {"avg_state"},
              thresholdNode,
              "ctr_total_return > threshold",
              {"ctr_customer_sk", "ctr_total_return"})
          .hashJoin(
              {"ctr_customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"ctr_total_return",
               "c_customer_id",
               "c_salutation",
               "c_first_name",
               "c_last_name",
               "c_current_addr_sk"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              gaAddrNode,
              "",
              {"c_customer_id",
               "c_salutation",
               "c_first_name",
               "c_last_name",
               "ca_street_number",
               "ca_street_name",
               "ca_street_type",
               "ca_suite_number",
               "ca_city",
               "ca_county",
               "ca_state",
               "ca_zip",
               "ca_country",
               "ca_gmt_offset",
               "ca_location_type",
               "ctr_total_return"})
          .orderBy(
              {"c_customer_id",
               "c_salutation",
               "c_first_name",
               "c_last_name",
               "ca_street_number",
               "ca_street_name",
               "ca_street_type",
               "ca_suite_number",
               "ca_city",
               "ca_county",
               "ca_state",
               "ca_zip",
               "ca_country",
               "ca_gmt_offset",
               "ca_location_type",
               "ctr_total_return"},
              false)
          .planNode();

  result.plan = std::move(plan);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[gaAddrScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q81";
  return result;
}

// Q93 (hand-built, not in Bolt): fact-to-fact store_sales LEFT OUTER JOIN
// store_returns on (item_sk, ticket_number) — the large-build join shape.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q93).
// Per spec, the reason join (WHERE sr_reason_sk = r_reason_sk AND
// r_reason_desc = 'reason 28') is applied AFTER the left outer join, so the
// plan keeps the honest spec shape: LEFT join against the FULL store_returns
// build (~29M rows at SF100), then an inner join with the 1-row reason
// build eliminates the null-extended rows. act_sales is computed as double
// (row counts are unaffected; only sort order of sums could differ at ties).
// ORDER BY sumsales, ss_customer_sk (NULLS FIRST, per spec).
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ93Plan() const {
  std::vector<std::string> storeSalesCols = {
      "ss_item_sk",
      "ss_ticket_number",
      "ss_customer_sk",
      "ss_quantity",
      "ss_sales_price"};
  std::vector<std::string> storeReturnsCols = {
      "sr_item_sk", "sr_ticket_number", "sr_reason_sk", "sr_return_quantity"};
  std::vector<std::string> reasonCols = {"r_reason_sk", "r_reason_desc"};

  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto storeReturnsType = getRowType(kStoreReturns, storeReturnsCols);
  auto reasonType = getRowType(kReason, reasonCols);

  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& srFileColumnNames = getFileColumnNames(kStoreReturns);
  const auto& reasonFileColumnNames = getFileColumnNames(kReason);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId storeSalesScanId, storeReturnsScanId, reasonScanId;

  // Full store_returns — the large build side.
  auto storeReturnsNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreReturns, storeReturnsType, srFileColumnNames, {})
          .captureScanNodeId(storeReturnsScanId)
          .planNode();

  auto reasonNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                        .filtersAsNode(filtersAsNode_)
                        .tableScan(
                            kReason,
                            reasonType,
                            reasonFileColumnNames,
                            {"r_reason_desc = 'reason 28'"})
                        .captureScanNodeId(reasonScanId)
                        .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(storeSalesScanId)
          .hashJoin(
              {"ss_item_sk", "ss_ticket_number"},
              {"sr_item_sk", "sr_ticket_number"},
              storeReturnsNode,
              "",
              {"ss_customer_sk",
               "ss_quantity",
               "ss_sales_price",
               "sr_reason_sk",
               "sr_return_quantity"},
              core::JoinType::kLeft)
          .hashJoin(
              {"sr_reason_sk"},
              {"r_reason_sk"},
              reasonNode,
              "",
              {"ss_customer_sk",
               "ss_quantity",
               "ss_sales_price",
               "sr_return_quantity"})
          .project(
              {"ss_customer_sk",
               "if(sr_return_quantity is not null, "
               "cast(ss_quantity - sr_return_quantity as double), "
               "cast(ss_quantity as double)) * cast(ss_sales_price as double) "
               "as act_sales"})
          .partialAggregation({"ss_customer_sk"}, {"sum(act_sales) as sumsales"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"sumsales ASC NULLS FIRST", "ss_customer_sk ASC NULLS FIRST"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[storeSalesScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[storeReturnsScanId] = getTableFilePaths(kStoreReturns);
  result.dataFiles[reasonScanId] = getTableFilePaths(kReason);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q93";
  return result;
}

// Q84 (hand-built, not in Bolt): customer id + name for customers in one city
// whose demographics fall in one income band, one row per store_returns row
// with a matching cdemo.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q84).
// The dimension chain customer -> customer_address (ca_city = 'Edgewood') ->
// household_demographics -> income_band (38128..88128) ->
// customer_demographics is assembled first and used as the BUILD of the final
// join; store_returns (28.8M rows at SF100) probes it on sr_cdemo_sk. The
// build carries the c_customer_id / c_first_name / c_last_name varchar
// payloads through every join — the wide-payload build shape.
// customername is built with concat(coalesce(...)) directly (velox has
// variadic concat), matching concat(concat(...)) in the reference.
// ORDER BY customer_id (c_customer_id) NULLS FIRST.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ84Plan() const {
  std::vector<std::string> customerCols = {
      "c_customer_id",
      "c_current_cdemo_sk",
      "c_current_hdemo_sk",
      "c_current_addr_sk",
      "c_first_name",
      "c_last_name"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_city"};
  std::vector<std::string> hdCols = {"hd_demo_sk", "hd_income_band_sk"};
  std::vector<std::string> ibCols = {
      "ib_income_band_sk", "ib_lower_bound", "ib_upper_bound"};
  std::vector<std::string> cdCols = {"cd_demo_sk"};
  std::vector<std::string> srCols = {"sr_cdemo_sk"};

  auto customerType = getRowType(kCustomer, customerCols);
  auto caType = getRowType(kCustomerAddress, caCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto ibType = getRowType(kIncomeBand, ibCols);
  auto cdType = getRowType(kCustomerDemographics, cdCols);
  auto srType = getRowType(kStoreReturns, srCols);

  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& ibFileColumnNames = getFileColumnNames(kIncomeBand);
  const auto& cdFileColumnNames = getFileColumnNames(kCustomerDemographics);
  const auto& srFileColumnNames = getFileColumnNames(kStoreReturns);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId customerScanId, caScanId, hdScanId, ibScanId, cdScanId,
      srScanId;

  auto caNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kCustomerAddress,
                        caType,
                        caFileColumnNames,
                        {"ca_city = 'Edgewood'"})
                    .captureScanNodeId(caScanId)
                    .planNode();

  auto hdNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kHouseholdDemographics, hdType, hdFileColumnNames, {})
          .captureScanNodeId(hdScanId)
          .planNode();

  // ib_lower_bound >= 38128 AND ib_upper_bound <= 38128 + 50000. Filter node
  // instead of pushed subfield filter: income_band's INTEGER columns get a
  // cast wrapper the subfield-filter converter rejects (20-row table, free).
  auto ibNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(kIncomeBand, ibType, ibFileColumnNames, {})
                    .captureScanNodeId(ibScanId)
                    .filter("ib_lower_bound >= 38128 and ib_upper_bound <= 88128")
                    .planNode();

  auto cdNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomerDemographics, cdType, cdFileColumnNames, {})
          .captureScanNodeId(cdScanId)
          .planNode();

  // Customer dimension chain — carries the wide name payload; final build.
  auto customerInfoNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"c_customer_id",
               "c_first_name",
               "c_last_name",
               "c_current_cdemo_sk",
               "c_current_hdemo_sk"})
          .hashJoin(
              {"c_current_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"c_customer_id",
               "c_first_name",
               "c_last_name",
               "c_current_cdemo_sk",
               "hd_income_band_sk"})
          .hashJoin(
              {"hd_income_band_sk"},
              {"ib_income_band_sk"},
              ibNode,
              "",
              {"c_customer_id",
               "c_first_name",
               "c_last_name",
               "c_current_cdemo_sk"})
          .hashJoin(
              {"c_current_cdemo_sk"},
              {"cd_demo_sk"},
              cdNode,
              "",
              {"c_customer_id", "c_first_name", "c_last_name", "cd_demo_sk"})
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreReturns, srType, srFileColumnNames, {})
          .captureScanNodeId(srScanId)
          .hashJoin(
              {"sr_cdemo_sk"},
              {"cd_demo_sk"},
              customerInfoNode,
              "",
              {"c_customer_id", "c_first_name", "c_last_name"})
          .project(
              {"c_customer_id as customer_id",
               "concat(coalesce(c_last_name, ''), ', ', "
               "coalesce(c_first_name, '')) as customername"})
          .orderBy({"customer_id ASC NULLS FIRST"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[ibScanId] = getTableFilePaths(kIncomeBand);
  result.dataFiles[cdScanId] = getTableFilePaths(kCustomerDemographics);
  result.dataFiles[srScanId] = getTableFilePaths(kStoreReturns);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q84";
  return result;
}

// Q40 (hand-built, not in Bolt): catalog_sales LEFT OUTER JOIN catalog_returns
// on (order_number, item_sk) — Q93's fact-to-fact shape, but the surrounding
// joins carry real payloads: cr_refunded_cash through the big left join,
// w_state from warehouse and i_item_id from item into the group-by keys, and
// d_date through the date join into the before/after CASE sums.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q40).
// Spec-honest order: LEFT join against the FULL catalog_returns build (14.4M
// rows at SF100) first, then date (60-day window), item (price band as a
// cast-to-double filter node, decimals compared as doubles), warehouse.
// Sums computed as doubles (row counts unaffected).
// ORDER BY w_state, i_item_id.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ40Plan() const {
  std::vector<std::string> catalogSalesCols = {
      "cs_sold_date_sk",
      "cs_item_sk",
      "cs_order_number",
      "cs_warehouse_sk",
      "cs_sales_price"};
  std::vector<std::string> catalogReturnsCols = {
      "cr_item_sk", "cr_order_number", "cr_refunded_cash"};
  std::vector<std::string> warehouseCols = {"w_warehouse_sk", "w_state"};
  std::vector<std::string> itemCols = {
      "i_item_sk", "i_item_id", "i_current_price"};
  std::vector<std::string> dateDimCols = {"d_date_sk", "d_date"};

  auto catalogSalesType = getRowType(kCatalogSales, catalogSalesCols);
  auto catalogReturnsType = getRowType(kCatalogReturns, catalogReturnsCols);
  auto warehouseType = getRowType(kWarehouse, warehouseCols);
  auto itemType = getRowType(kItem, itemCols);
  auto dateDimType = getRowType(kDateDim, dateDimCols);

  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& crFileColumnNames = getFileColumnNames(kCatalogReturns);
  const auto& wFileColumnNames = getFileColumnNames(kWarehouse);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId csScanId, crScanId, wScanId, itemScanId, ddScanId;

  // Full catalog_returns — the large fact build side of the left join.
  auto crNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogReturns, catalogReturnsType, crFileColumnNames, {})
          .captureScanNodeId(crScanId)
          .planNode();

  auto dateDimNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(
              kDateDim,
              dateDimType,
              ddFileColumnNames,
              {"d_date between '2000-02-10'::DATE and '2000-04-10'::DATE"})
          .captureScanNodeId(ddScanId)
          .planNode();

  // i_current_price BETWEEN 0.99 AND 1.49 — decimal compared as double
  // (same idiom as Q79's s_gmt_offset filter).
  auto itemNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kItem, itemType, itemFileColumnNames, {})
          .captureScanNodeId(itemScanId)
          .filter("cast(i_current_price as double) between 0.99 and 1.49")
          .planNode();

  auto warehouseNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kWarehouse, warehouseType, wFileColumnNames, {})
          .captureScanNodeId(wScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogSales, catalogSalesType, csFileColumnNames, {})
          .captureScanNodeId(csScanId)
          .hashJoin(
              {"cs_order_number", "cs_item_sk"},
              {"cr_order_number", "cr_item_sk"},
              crNode,
              "",
              {"cs_sold_date_sk",
               "cs_item_sk",
               "cs_warehouse_sk",
               "cs_sales_price",
               "cr_refunded_cash"},
              core::JoinType::kLeft)
          .hashJoin(
              {"cs_sold_date_sk"},
              {"d_date_sk"},
              dateDimNode,
              "",
              {"cs_item_sk",
               "cs_warehouse_sk",
               "cs_sales_price",
               "cr_refunded_cash",
               "d_date"})
          .hashJoin(
              {"cs_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"cs_warehouse_sk",
               "cs_sales_price",
               "cr_refunded_cash",
               "d_date",
               "i_item_id"})
          .hashJoin(
              {"cs_warehouse_sk"},
              {"w_warehouse_sk"},
              warehouseNode,
              "",
              {"cs_sales_price",
               "cr_refunded_cash",
               "d_date",
               "i_item_id",
               "w_state"})
          .project(
              {"w_state",
               "i_item_id",
               "if(d_date < cast('2000-03-11' as date), "
               "cast(cs_sales_price as double) - "
               "coalesce(cast(cr_refunded_cash as double), 0.0), 0.0) "
               "as sales_before_row",
               "if(d_date >= cast('2000-03-11' as date), "
               "cast(cs_sales_price as double) - "
               "coalesce(cast(cr_refunded_cash as double), 0.0), 0.0) "
               "as sales_after_row"})
          .partialAggregation(
              {"w_state", "i_item_id"},
              {"sum(sales_before_row) as sales_before",
               "sum(sales_after_row) as sales_after"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy({"w_state", "i_item_id"}, false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[csScanId] = getTableFilePaths(kCatalogSales);
  result.dataFiles[crScanId] = getTableFilePaths(kCatalogReturns);
  result.dataFiles[wScanId] = getTableFilePaths(kWarehouse);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[ddScanId] = getTableFilePaths(kDateDim);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q40";
  return result;
}

// Q54 (hand-built, not in Bolt): customer revenue segments.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q54).
// my_customers CTE = DISTINCT (c_customer_sk, c_current_addr_sk) over the
// catalog+web union (UNION ALL as a multi-source LocalPartition) restricted
// to Women/maternity items sold in 1998-12; it becomes the BUILD of the
// store_sales revenue join. The correlated scalar subqueries on d_month_seq
// resolve to the single value 1187 for (d_year=1998, d_moy=12), so the
// BETWEEN d_month_seq+1 AND d_month_seq+3 window is hardcoded as
// d_month_seq between 1188 and 1190 (same idiom as Bolt's hardcoded
// month_seq windows, e.g. Q46-style "d_month_seq between 1176 and 1187").
// The ca_county/ca_state = s_county/s_state join is a many-to-many equijoin
// per spec (each matching store multiplies revenue rows). Revenue is summed
// as decimal then cast to double before round(revenue/50) segmentation.
// ORDER BY segment, num_customers, segment_base.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ54Plan() const {
  std::vector<std::string> catalogSalesCols = {
      "cs_sold_date_sk", "cs_bill_customer_sk", "cs_item_sk"};
  std::vector<std::string> webSalesCols = {
      "ws_sold_date_sk", "ws_bill_customer_sk", "ws_item_sk"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_category", "i_class"};
  std::vector<std::string> dateDim1Cols = {"d_date_sk", "d_moy", "d_year"};
  std::vector<std::string> customerCols = {"c_customer_sk", "c_current_addr_sk"};
  std::vector<std::string> storeSalesCols = {
      "ss_sold_date_sk", "ss_customer_sk", "ss_ext_sales_price"};
  std::vector<std::string> dateDim2Cols = {"d_date_sk", "d_month_seq"};
  std::vector<std::string> caCols = {"ca_address_sk", "ca_county", "ca_state"};
  std::vector<std::string> storeCols = {"s_county", "s_state"};

  auto catalogSalesType = getRowType(kCatalogSales, catalogSalesCols);
  auto webSalesType = getRowType(kWebSales, webSalesCols);
  auto itemType = getRowType(kItem, itemCols);
  auto dateDim1Type = getRowType(kDateDim, dateDim1Cols);
  auto customerType = getRowType(kCustomer, customerCols);
  auto storeSalesType = getRowType(kStoreSales, storeSalesCols);
  auto dateDim2Type = getRowType(kDateDim, dateDim2Cols);
  auto caType = getRowType(kCustomerAddress, caCols);
  auto storeType = getRowType(kStore, storeCols);

  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& wsFileColumnNames = getFileColumnNames(kWebSales);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId csScanId, wsScanId, itemScanId, dd1ScanId, customerScanId,
      ssScanId, dd2ScanId, caScanId, storeScanId;

  // UNION ALL branches, renamed to the CTE's column names.
  auto csNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogSales, catalogSalesType, csFileColumnNames, {})
          .captureScanNodeId(csScanId)
          .project(
              {"cs_sold_date_sk as sold_date_sk",
               "cs_bill_customer_sk as customer_sk",
               "cs_item_sk as item_sk"})
          .planNode();

  auto wsNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(kWebSales, webSalesType, wsFileColumnNames, {})
                    .captureScanNodeId(wsScanId)
                    .project(
                        {"ws_sold_date_sk as sold_date_sk",
                         "ws_bill_customer_sk as customer_sk",
                         "ws_item_sk as item_sk"})
                    .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kItem,
                          itemType,
                          itemFileColumnNames,
                          {"i_category = 'Women'", "i_class = 'maternity'"})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto dateDim1Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDim1Type,
                              ddFileColumnNames,
                              {"d_moy = 12", "d_year = 1998"})
                          .captureScanNodeId(dd1ScanId)
                          .planNode();

  auto customerNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomer, customerType, custFileColumnNames, {})
          .captureScanNodeId(customerScanId)
          .planNode();

  // my_customers CTE: DISTINCT (c_customer_sk, c_current_addr_sk).
  auto myCustomersNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .localPartition({}, {csNode, wsNode})
          .hashJoin(
              {"item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"sold_date_sk", "customer_sk"})
          .hashJoin(
              {"sold_date_sk"}, {"d_date_sk"}, dateDim1Node, "", {"customer_sk"})
          .hashJoin(
              {"customer_sk"},
              {"c_customer_sk"},
              customerNode,
              "",
              {"c_customer_sk", "c_current_addr_sk"})
          .partialAggregation({"c_customer_sk", "c_current_addr_sk"}, {})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .planNode();

  // d_month_seq BETWEEN 1187+1 AND 1187+3 (see comment above).
  auto dateDim2Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDim2Type,
                              ddFileColumnNames,
                              {"d_month_seq between 1188 and 1190"})
                          .captureScanNodeId(dd2ScanId)
                          .planNode();

  auto caNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
          .captureScanNodeId(caScanId)
          .planNode();

  auto storeNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStore, storeType, storeFileColumnNames, {})
          .captureScanNodeId(storeScanId)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kStoreSales, storeSalesType, ssFileColumnNames, {})
          .captureScanNodeId(ssScanId)
          .hashJoin(
              {"ss_sold_date_sk"},
              {"d_date_sk"},
              dateDim2Node,
              "",
              {"ss_customer_sk", "ss_ext_sales_price"})
          .hashJoin(
              {"ss_customer_sk"},
              {"c_customer_sk"},
              myCustomersNode,
              "",
              {"ss_ext_sales_price", "c_customer_sk", "c_current_addr_sk"})
          .hashJoin(
              {"c_current_addr_sk"},
              {"ca_address_sk"},
              caNode,
              "",
              {"ss_ext_sales_price", "c_customer_sk", "ca_county", "ca_state"})
          .hashJoin(
              {"ca_county", "ca_state"},
              {"s_county", "s_state"},
              storeNode,
              "",
              {"ss_ext_sales_price", "c_customer_sk"})
          .partialAggregation(
              {"c_customer_sk"}, {"sum(ss_ext_sales_price) as revenue"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project(
              {"cast(round(cast(revenue as double) / 50.0) as integer) "
               "as segment"})
          .partialAggregation({"segment"}, {"count(1) as num_customers"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"segment", "num_customers", "segment * 50 as segment_base"})
          .orderBy(
              {"segment ASC NULLS FIRST",
               "num_customers ASC NULLS FIRST",
               "segment_base ASC"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[csScanId] = getTableFilePaths(kCatalogSales);
  result.dataFiles[wsScanId] = getTableFilePaths(kWebSales);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[dd1ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[customerScanId] = getTableFilePaths(kCustomer);
  result.dataFiles[ssScanId] = getTableFilePaths(kStoreSales);
  result.dataFiles[dd2ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[caScanId] = getTableFilePaths(kCustomerAddress);
  result.dataFiles[storeScanId] = getTableFilePaths(kStore);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q54";
  return result;
}

// Q80 (hand-built, not in Bolt): three parallel channel blocks — store,
// catalog and web sales each fact-to-fact LEFT-joined with their returns on
// (item, ticket/order), inner-joined date (30-day window), channel id table,
// item (price > 50 as cast-to-double filter node) and promotion
// (p_channel_tv = 'N') — aggregated per channel id, UNION ALL'd (multi-source
// LocalPartition), then ROLLUP(channel, id) via a GroupIdNode with grouping
// sets {channel,id}, {channel}, {} — true rollup, no deviation.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q80).
// The fact-to-fact left joins carry the return_amt/net_loss payloads and the
// probe carries sales_price/net_profit — the payload-bearing Q93 shape ×3.
// Money sums computed as doubles (group keys are strings; counts unaffected).
// ORDER BY channel NULLS FIRST, id NULLS FIRST.
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ80Plan() const {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  TpchPlan result;

  std::vector<std::string> dateDimCols = {"d_date_sk", "d_date"};
  auto dateDimType = getRowType(kDateDim, dateDimCols);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  std::vector<std::string> itemCols = {"i_item_sk", "i_current_price"};
  auto itemType = getRowType(kItem, itemCols);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  std::vector<std::string> promoCols = {"p_promo_sk", "p_channel_tv"};
  auto promoType = getRowType(kPromotion, promoCols);
  const auto& promoFileColumnNames = getFileColumnNames(kPromotion);

  // One channel block: sales LEFT JOIN returns, then date/id/item/promotion,
  // aggregate per id, project (channel, id, sales, returns_, profit).
  auto makeChannel = [&](const char* salesTable,
                         const char* returnsTable,
                         const char* idTable,
                         // sales-side columns
                         const std::string& soldDateCol,
                         const std::string& itemCol,
                         const std::string& idFkCol,
                         const std::string& promoCol,
                         const std::string& jk2Col,
                         const std::string& priceCol,
                         const std::string& profitCol,
                         // returns-side columns
                         const std::string& rItemCol,
                         const std::string& rJk2Col,
                         const std::string& rAmtCol,
                         const std::string& rLossCol,
                         // id table columns
                         const std::string& idKeyCol,
                         const std::string& idIdCol,
                         // output constants
                         const std::string& channelName,
                         const std::string& idPrefix) -> core::PlanNodePtr {
    core::PlanNodeId salesScanId, returnsScanId, ddScanId, idScanId,
        itemScanId, promoScanId;

    std::vector<std::string> salesCols = {
        soldDateCol, itemCol, idFkCol, promoCol, jk2Col, priceCol, profitCol};
    auto salesType = getRowType(salesTable, salesCols);
    std::vector<std::string> returnsCols = {
        rItemCol, rJk2Col, rAmtCol, rLossCol};
    auto returnsType = getRowType(returnsTable, returnsCols);
    std::vector<std::string> idCols = {idKeyCol, idIdCol};
    auto idType = getRowType(idTable, idCols);

    auto returnsNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                           .filtersAsNode(filtersAsNode_)
                           .tableScan(
                               returnsTable,
                               returnsType,
                               getFileColumnNames(returnsTable),
                               {})
                           .captureScanNodeId(returnsScanId)
                           .planNode();

    auto dateDimNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                kDateDim,
                dateDimType,
                ddFileColumnNames,
                {"d_date between '2000-08-23'::DATE and '2000-09-22'::DATE"})
            .captureScanNodeId(ddScanId)
            .planNode();

    auto idNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(idTable, idType, getFileColumnNames(idTable), {})
            .captureScanNodeId(idScanId)
            .planNode();

    // i_current_price > 50 — decimal compared as double.
    auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                        .filtersAsNode(filtersAsNode_)
                        .tableScan(kItem, itemType, itemFileColumnNames, {})
                        .captureScanNodeId(itemScanId)
                        .filter("cast(i_current_price as double) > 50.0")
                        .planNode();

    auto promoNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                         .filtersAsNode(filtersAsNode_)
                         .tableScan(
                             kPromotion,
                             promoType,
                             promoFileColumnNames,
                             {"p_channel_tv = 'N'"})
                         .captureScanNodeId(promoScanId)
                         .planNode();

    auto node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                salesTable, salesType, getFileColumnNames(salesTable), {})
            .captureScanNodeId(salesScanId)
            .hashJoin(
                {itemCol, jk2Col},
                {rItemCol, rJk2Col},
                returnsNode,
                "",
                {soldDateCol,
                 itemCol,
                 idFkCol,
                 promoCol,
                 priceCol,
                 profitCol,
                 rAmtCol,
                 rLossCol},
                core::JoinType::kLeft)
            .hashJoin(
                {soldDateCol},
                {"d_date_sk"},
                dateDimNode,
                "",
                {itemCol, idFkCol, promoCol, priceCol, profitCol, rAmtCol,
                 rLossCol})
            .hashJoin(
                {idFkCol},
                {idKeyCol},
                idNode,
                "",
                {itemCol, promoCol, priceCol, profitCol, rAmtCol, rLossCol,
                 idIdCol})
            .hashJoin(
                {itemCol},
                {"i_item_sk"},
                itemNode,
                "",
                {promoCol, priceCol, profitCol, rAmtCol, rLossCol, idIdCol})
            .hashJoin(
                {promoCol},
                {"p_promo_sk"},
                promoNode,
                "",
                {priceCol, profitCol, rAmtCol, rLossCol, idIdCol})
            .project(
                {idIdCol,
                 fmt::format("cast({} as double) as sales_row", priceCol),
                 fmt::format(
                     "coalesce(cast({} as double), 0.0) as returns_row",
                     rAmtCol),
                 fmt::format(
                     "cast({} as double) - coalesce(cast({} as double), 0.0) "
                     "as profit_row",
                     profitCol,
                     rLossCol)})
            .partialAggregation(
                {idIdCol},
                {"sum(sales_row) as sales",
                 "sum(returns_row) as returns_",
                 "sum(profit_row) as profit"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .project(
                {fmt::format("'{}' as channel", channelName),
                 fmt::format("concat('{}', {}) as id", idPrefix, idIdCol),
                 "sales",
                 "returns_",
                 "profit"})
            .planNode();

    result.dataFiles[salesScanId] = getTableFilePaths(salesTable);
    result.dataFiles[returnsScanId] = getTableFilePaths(returnsTable);
    result.dataFiles[ddScanId] = getTableFilePaths(kDateDim);
    result.dataFiles[idScanId] = getTableFilePaths(idTable);
    result.dataFiles[itemScanId] = getTableFilePaths(kItem);
    result.dataFiles[promoScanId] = getTableFilePaths(kPromotion);
    return node;
  };

  auto ssrNode = makeChannel(
      kStoreSales,
      kStoreReturns,
      kStore,
      "ss_sold_date_sk",
      "ss_item_sk",
      "ss_store_sk",
      "ss_promo_sk",
      "ss_ticket_number",
      "ss_ext_sales_price",
      "ss_net_profit",
      "sr_item_sk",
      "sr_ticket_number",
      "sr_return_amt",
      "sr_net_loss",
      "s_store_sk",
      "s_store_id",
      "store channel",
      "store");

  auto csrNode = makeChannel(
      kCatalogSales,
      kCatalogReturns,
      kCatalogPage,
      "cs_sold_date_sk",
      "cs_item_sk",
      "cs_catalog_page_sk",
      "cs_promo_sk",
      "cs_order_number",
      "cs_ext_sales_price",
      "cs_net_profit",
      "cr_item_sk",
      "cr_order_number",
      "cr_return_amount",
      "cr_net_loss",
      "cp_catalog_page_sk",
      "cp_catalog_page_id",
      "catalog channel",
      "catalog_page");

  auto wsrNode = makeChannel(
      kWebSales,
      kWebReturns,
      kWebSite,
      "ws_sold_date_sk",
      "ws_item_sk",
      "ws_web_site_sk",
      "ws_promo_sk",
      "ws_order_number",
      "ws_ext_sales_price",
      "ws_net_profit",
      "wr_item_sk",
      "wr_order_number",
      "wr_return_amt",
      "wr_net_loss",
      "web_site_sk",
      "web_site_id",
      "web channel",
      "web_site");

  // UNION ALL + ROLLUP(channel, id) + final sort.
  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .localPartition({}, {ssrNode, csrNode, wsrNode})
          .groupId(
              {"channel", "id"},
              {{"channel", "id"}, {"channel"}, {}},
              {"sales", "returns_", "profit"})
          .partialAggregation(
              {"channel", "id", "group_id"},
              {"sum(sales) as sales",
               "sum(returns_) as returns_",
               "sum(profit) as profit"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .project({"channel", "id", "sales", "returns_", "profit"})
          .orderBy({"channel ASC NULLS FIRST", "id ASC NULLS FIRST"}, false)
          .planNode();

  result.plan = std::move(plan);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q80";
  return result;
}

// Q72 (hand-built, not in Bolt): the join-VOLUME query — catalog_sales ⋈
// inventory ⋈ warehouse ⋈ item ⋈ customer_demographics ⋈
// household_demographics ⋈ date_dim ×3, LEFT promotion, LEFT catalog_returns.
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q72).
// Plan shape: the filtered catalog_sales side (d_year=1999, cd 'D', hd
// '>10000', ship-date > sold-date+5 as a join filter via date_add) with its
// LEFT joins is assembled first and used as the BUILD (~1M rows carrying the
// wide i_item_desc payload) of the final join; inventory (399M rows at SF100)
// ⋈ d2 ⋈ warehouse (w_warehouse_name payload) is the probe. The WHERE
// d1.d_week_seq = d2.d_week_seq equality is folded into the final join keys
// (inv_item_sk, d2_week_seq) = (cs_item_sk, d1_week_seq), with
// inv_quantity_on_hand < cs_quantity as the join filter — same rows as the
// spec's join+WHERE, joined on much more selective keys. The LEFT joins
// (promotion null-flag; catalog_returns row multiplication) commute with the
// inner joins they precede — identical multiplicity and null pattern.
// ORDER BY total_cnt DESC NULLS FIRST, i_item_desc, w_warehouse_name,
// d_week_seq (all NULLS FIRST).
// LIMIT omitted (vs. spec) so the final sort does full work — hybrid sort
// evaluation.
TpchPlan TpcdsQueryBuilder::getQ72Plan() const {
  std::vector<std::string> catalogSalesCols = {
      "cs_sold_date_sk",
      "cs_ship_date_sk",
      "cs_bill_cdemo_sk",
      "cs_bill_hdemo_sk",
      "cs_item_sk",
      "cs_promo_sk",
      "cs_order_number",
      "cs_quantity"};
  std::vector<std::string> inventoryCols = {
      "inv_date_sk", "inv_item_sk", "inv_warehouse_sk", "inv_quantity_on_hand"};
  std::vector<std::string> dateDim1Cols = {
      "d_date_sk", "d_date", "d_week_seq", "d_year"};
  std::vector<std::string> dateDim2Cols = {"d_date_sk", "d_week_seq"};
  std::vector<std::string> dateDim3Cols = {"d_date_sk", "d_date"};
  std::vector<std::string> cdCols = {"cd_demo_sk", "cd_marital_status"};
  std::vector<std::string> hdCols = {"hd_demo_sk", "hd_buy_potential"};
  std::vector<std::string> itemCols = {"i_item_sk", "i_item_desc"};
  std::vector<std::string> promoCols = {"p_promo_sk"};
  std::vector<std::string> crCols = {"cr_item_sk", "cr_order_number"};
  std::vector<std::string> warehouseCols = {
      "w_warehouse_sk", "w_warehouse_name"};

  auto catalogSalesType = getRowType(kCatalogSales, catalogSalesCols);
  auto inventoryType = getRowType(kInventory, inventoryCols);
  auto dateDim1Type = getRowType(kDateDim, dateDim1Cols);
  auto dateDim2Type = getRowType(kDateDim, dateDim2Cols);
  auto dateDim3Type = getRowType(kDateDim, dateDim3Cols);
  auto cdType = getRowType(kCustomerDemographics, cdCols);
  auto hdType = getRowType(kHouseholdDemographics, hdCols);
  auto itemType = getRowType(kItem, itemCols);
  auto promoType = getRowType(kPromotion, promoCols);
  auto crType = getRowType(kCatalogReturns, crCols);
  auto warehouseType = getRowType(kWarehouse, warehouseCols);

  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& invFileColumnNames = getFileColumnNames(kInventory);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& cdFileColumnNames = getFileColumnNames(kCustomerDemographics);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);
  const auto& promoFileColumnNames = getFileColumnNames(kPromotion);
  const auto& crFileColumnNames = getFileColumnNames(kCatalogReturns);
  const auto& wFileColumnNames = getFileColumnNames(kWarehouse);

  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();
  core::PlanNodeId csScanId, invScanId, dd1ScanId, dd2ScanId, dd3ScanId,
      cdScanId, hdScanId, itemScanId, promoScanId, crScanId, wScanId;

  auto dateDim1Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                          .filtersAsNode(filtersAsNode_)
                          .tableScan(
                              kDateDim,
                              dateDim1Type,
                              ddFileColumnNames,
                              {"d_year = 1999"})
                          .captureScanNodeId(dd1ScanId)
                          .project(
                              {"d_date_sk as d1_date_sk",
                               "d_date as d1_date",
                               "d_week_seq as d1_week_seq"})
                          .planNode();

  auto dateDim2Node =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kDateDim, dateDim2Type, ddFileColumnNames, {})
          .captureScanNodeId(dd2ScanId)
          .project({"d_date_sk as d2_date_sk", "d_week_seq as d2_week_seq"})
          .planNode();

  auto dateDim3Node =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kDateDim, dateDim3Type, ddFileColumnNames, {})
          .captureScanNodeId(dd3ScanId)
          .project({"d_date_sk as d3_date_sk", "d_date as d3_date"})
          .planNode();

  auto cdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kCustomerDemographics,
                        cdType,
                        cdFileColumnNames,
                        {"cd_marital_status = 'D'"})
                    .captureScanNodeId(cdScanId)
                    .planNode();

  auto hdNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                    .filtersAsNode(filtersAsNode_)
                    .tableScan(
                        kHouseholdDemographics,
                        hdType,
                        hdFileColumnNames,
                        {"hd_buy_potential = '>10000'"})
                    .captureScanNodeId(hdScanId)
                    .planNode();

  auto itemNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kItem, itemType, itemFileColumnNames, {})
                      .captureScanNodeId(itemScanId)
                      .planNode();

  auto promoNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kPromotion, promoType, promoFileColumnNames, {})
          .captureScanNodeId(promoScanId)
          .planNode();

  auto crNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogReturns, crType, crFileColumnNames, {})
          .captureScanNodeId(crScanId)
          .planNode();

  auto warehouseNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kWarehouse, warehouseType, wFileColumnNames, {})
          .captureScanNodeId(wScanId)
          .planNode();

  // Filtered catalog_sales side — becomes the build of the final join,
  // carrying the wide i_item_desc payload.
  auto csSideNode =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kCatalogSales, catalogSalesType, csFileColumnNames, {})
          .captureScanNodeId(csScanId)
          .hashJoin(
              {"cs_sold_date_sk"},
              {"d1_date_sk"},
              dateDim1Node,
              "",
              {"cs_ship_date_sk",
               "cs_bill_cdemo_sk",
               "cs_bill_hdemo_sk",
               "cs_item_sk",
               "cs_promo_sk",
               "cs_order_number",
               "cs_quantity",
               "d1_date",
               "d1_week_seq"})
          .hashJoin(
              {"cs_ship_date_sk"},
              {"d3_date_sk"},
              dateDim3Node,
              "d3_date > date_add('day', 5, d1_date)",
              {"cs_bill_cdemo_sk",
               "cs_bill_hdemo_sk",
               "cs_item_sk",
               "cs_promo_sk",
               "cs_order_number",
               "cs_quantity",
               "d1_week_seq"})
          .hashJoin(
              {"cs_bill_cdemo_sk"},
              {"cd_demo_sk"},
              cdNode,
              "",
              {"cs_bill_hdemo_sk",
               "cs_item_sk",
               "cs_promo_sk",
               "cs_order_number",
               "cs_quantity",
               "d1_week_seq"})
          .hashJoin(
              {"cs_bill_hdemo_sk"},
              {"hd_demo_sk"},
              hdNode,
              "",
              {"cs_item_sk",
               "cs_promo_sk",
               "cs_order_number",
               "cs_quantity",
               "d1_week_seq"})
          .hashJoin(
              {"cs_item_sk"},
              {"i_item_sk"},
              itemNode,
              "",
              {"cs_item_sk",
               "cs_promo_sk",
               "cs_order_number",
               "cs_quantity",
               "d1_week_seq",
               "i_item_desc"})
          .hashJoin(
              {"cs_promo_sk"},
              {"p_promo_sk"},
              promoNode,
              "",
              {"cs_item_sk",
               "cs_order_number",
               "cs_quantity",
               "d1_week_seq",
               "i_item_desc",
               "p_promo_sk"},
              core::JoinType::kLeft)
          .hashJoin(
              {"cs_item_sk", "cs_order_number"},
              {"cr_item_sk", "cr_order_number"},
              crNode,
              "",
              {"cs_item_sk",
               "cs_quantity",
               "d1_week_seq",
               "i_item_desc",
               "p_promo_sk"},
              core::JoinType::kLeft)
          .planNode();

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .tableScan(kInventory, inventoryType, invFileColumnNames, {})
          .captureScanNodeId(invScanId)
          .hashJoin(
              {"inv_date_sk"},
              {"d2_date_sk"},
              dateDim2Node,
              "",
              {"inv_item_sk",
               "inv_warehouse_sk",
               "inv_quantity_on_hand",
               "d2_week_seq"})
          .hashJoin(
              {"inv_warehouse_sk"},
              {"w_warehouse_sk"},
              warehouseNode,
              "",
              {"inv_item_sk",
               "inv_quantity_on_hand",
               "d2_week_seq",
               "w_warehouse_name"})
          .hashJoin(
              {"inv_item_sk", "d2_week_seq"},
              {"cs_item_sk", "d1_week_seq"},
              csSideNode,
              "inv_quantity_on_hand < cs_quantity",
              {"i_item_desc", "w_warehouse_name", "d1_week_seq", "p_promo_sk"})
          .project(
              {"i_item_desc",
               "w_warehouse_name",
               "d1_week_seq",
               "if(p_promo_sk is null, 1, 0) as no_promo_flag",
               "if(p_promo_sk is null, 0, 1) as promo_flag"})
          .partialAggregation(
              {"i_item_desc", "w_warehouse_name", "d1_week_seq"},
              {"sum(no_promo_flag) as no_promo",
               "sum(promo_flag) as promo",
               "count(1) as total_cnt"})
          .localPartition(std::vector<std::string>{})
          .finalAggregation()
          .orderBy(
              {"total_cnt DESC NULLS FIRST",
               "i_item_desc ASC NULLS FIRST",
               "w_warehouse_name ASC NULLS FIRST",
               "d1_week_seq ASC NULLS FIRST"},
              false)
          .planNode();

  TpchPlan result;
  result.plan = std::move(plan);
  result.dataFiles[csScanId] = getTableFilePaths(kCatalogSales);
  result.dataFiles[invScanId] = getTableFilePaths(kInventory);
  result.dataFiles[dd1ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[dd2ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[dd3ScanId] = getTableFilePaths(kDateDim);
  result.dataFiles[cdScanId] = getTableFilePaths(kCustomerDemographics);
  result.dataFiles[hdScanId] = getTableFilePaths(kHouseholdDemographics);
  result.dataFiles[itemScanId] = getTableFilePaths(kItem);
  result.dataFiles[promoScanId] = getTableFilePaths(kPromotion);
  result.dataFiles[crScanId] = getTableFilePaths(kCatalogReturns);
  result.dataFiles[wScanId] = getTableFilePaths(kWarehouse);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q72";
  return result;
}

// Q64 (hand-built, not in Bolt): cross_sales CTE joined with itself across
// consecutive years — the widest output in TPC-DS (21 columns, 11 varchar).
// Semantic reference: TPC-DS qualification SQL (duckdb tpcds extension q64).
// cs_ui CTE = catalog fact-to-fact inner join (catalog_sales ⋈
// catalog_returns on item/order) aggregated per item with the HAVING
// sale > 2*refund filter (decimal sums compared exactly via refund+refund).
// cross_sales = an 18-table star join around store_sales ⋈ store_returns
// (fact-to-fact on item/ticket) carrying product/store/address varchar
// payloads into a 15-key group-by. Velox plans are trees, so cross_sales is
// instantiated twice (like Q74/Q81), with the final WHERE cs1.syear = 1999 /
// cs2.syear = 2000 pushed into each instance's d1 scan (exact: syear is the
// d1.d_year group key). The item price band (BETWEEN 64 AND 74 intersected
// with BETWEEN 65 AND 79 = 65..74) is a cast-to-double filter node; the
// cd1/cd2 marital-status inequality is a join filter. Measures are summed as
// doubles (group keys unaffected).
// ORDER BY product_name, store_name, cs2.cnt, cs1.s1, cs2.s1. Spec Q64 has
// no LIMIT; the final ORDER BY does full work — hybrid sort evaluation.
//
// KNOWN HYBRID BUG (2026-08-10, unresolved): with --hybrid_join_enabled=true
// this query nondeterministically returns 477 rows instead of 472 (baseline
// and duckdb agree on 472). The 5 extra rows are exactly the rows duckdb
// produces when cs1.store_name/store_zip = cs2.store_name/store_zip is
// replaced with IS NOT DISTINCT FROM — i.e. the hybrid join sometimes lets
// build/probe rows whose nullable VARCHAR group-by keys are NULL match each
// other on the final self-join. Never loses rows, never emits other rows;
// when wrong it is always exactly this null-key row set. Observed across
// coalesced and scattered modes and both reorder settings (flaky run to
// run); not observed with --num_drivers=1 or with hybrid off. The other
// hand-built queries are unaffected (their join keys are never NULL on both
// sides). Exclude Q64 from hybrid-arm campaigns until the hybrid null-key
// path is fixed.
TpchPlan TpcdsQueryBuilder::getQ64Plan() const {
  auto planNodeIdGenerator = std::make_shared<core::PlanNodeIdGenerator>();

  TpchPlan result;

  const auto& csFileColumnNames = getFileColumnNames(kCatalogSales);
  const auto& crFileColumnNames = getFileColumnNames(kCatalogReturns);
  const auto& ssFileColumnNames = getFileColumnNames(kStoreSales);
  const auto& srFileColumnNames = getFileColumnNames(kStoreReturns);
  const auto& ddFileColumnNames = getFileColumnNames(kDateDim);
  const auto& storeFileColumnNames = getFileColumnNames(kStore);
  const auto& custFileColumnNames = getFileColumnNames(kCustomer);
  const auto& cdFileColumnNames = getFileColumnNames(kCustomerDemographics);
  const auto& promoFileColumnNames = getFileColumnNames(kPromotion);
  const auto& hdFileColumnNames = getFileColumnNames(kHouseholdDemographics);
  const auto& caFileColumnNames = getFileColumnNames(kCustomerAddress);
  const auto& ibFileColumnNames = getFileColumnNames(kIncomeBand);
  const auto& itemFileColumnNames = getFileColumnNames(kItem);

  // cs_ui CTE: items whose catalog sale total exceeds twice the refund total.
  auto makeCsUi = [&]() -> core::PlanNodePtr {
    core::PlanNodeId csScanId, crScanId;
    std::vector<std::string> csCols = {
        "cs_item_sk", "cs_order_number", "cs_ext_list_price"};
    auto csType = getRowType(kCatalogSales, csCols);
    std::vector<std::string> crCols = {
        "cr_item_sk",
        "cr_order_number",
        "cr_refunded_cash",
        "cr_reversed_charge",
        "cr_store_credit"};
    auto crType = getRowType(kCatalogReturns, crCols);

    auto crNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kCatalogReturns, crType, crFileColumnNames, {})
                      .captureScanNodeId(crScanId)
                      .planNode();

    auto node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCatalogSales, csType, csFileColumnNames, {})
            .captureScanNodeId(csScanId)
            .hashJoin(
                {"cs_item_sk", "cs_order_number"},
                {"cr_item_sk", "cr_order_number"},
                crNode,
                "",
                {"cs_item_sk",
                 "cs_ext_list_price",
                 "cr_refunded_cash",
                 "cr_reversed_charge",
                 "cr_store_credit"})
            .project(
                {"cs_item_sk",
                 "cs_ext_list_price",
                 "cr_refunded_cash + cr_reversed_charge + cr_store_credit "
                 "as refund_row"})
            .partialAggregation(
                {"cs_item_sk"},
                {"sum(cs_ext_list_price) as sale",
                 "sum(refund_row) as refund"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .filter("sale > refund + refund")
            .project({"cs_item_sk as csui_item_sk"})
            .planNode();

    result.dataFiles[csScanId] = getTableFilePaths(kCatalogSales);
    result.dataFiles[crScanId] = getTableFilePaths(kCatalogReturns);
    return node;
  };

  // One cross_sales instance, with the final query's syear predicate pushed
  // into the d1 scan. Output columns carry `prefix` ("cs1"/"cs2").
  auto makeCrossSales = [&](int year,
                            const std::string& prefix) -> core::PlanNodePtr {
    core::PlanNodeId ssScanId, srScanId, d1ScanId, d2ScanId, d3ScanId,
        storeScanId, custScanId, cd1ScanId, cd2ScanId, promoScanId, hd1ScanId,
        hd2ScanId, ad1ScanId, ad2ScanId, ib1ScanId, ib2ScanId, itemScanId;

    std::vector<std::string> ssCols = {
        "ss_sold_date_sk",
        "ss_item_sk",
        "ss_customer_sk",
        "ss_cdemo_sk",
        "ss_hdemo_sk",
        "ss_addr_sk",
        "ss_store_sk",
        "ss_promo_sk",
        "ss_ticket_number",
        "ss_wholesale_cost",
        "ss_list_price",
        "ss_coupon_amt"};
    auto ssType = getRowType(kStoreSales, ssCols);
    std::vector<std::string> srCols = {"sr_item_sk", "sr_ticket_number"};
    auto srType = getRowType(kStoreReturns, srCols);
    std::vector<std::string> ddCols = {"d_date_sk", "d_year"};
    auto ddType = getRowType(kDateDim, ddCols);
    std::vector<std::string> storeCols = {
        "s_store_sk", "s_store_name", "s_zip"};
    auto storeType = getRowType(kStore, storeCols);
    std::vector<std::string> custCols = {
        "c_customer_sk",
        "c_current_cdemo_sk",
        "c_current_hdemo_sk",
        "c_current_addr_sk",
        "c_first_sales_date_sk",
        "c_first_shipto_date_sk"};
    auto custType = getRowType(kCustomer, custCols);
    std::vector<std::string> cdCols = {"cd_demo_sk", "cd_marital_status"};
    auto cdType = getRowType(kCustomerDemographics, cdCols);
    std::vector<std::string> promoCols = {"p_promo_sk"};
    auto promoType = getRowType(kPromotion, promoCols);
    std::vector<std::string> hdCols = {"hd_demo_sk", "hd_income_band_sk"};
    auto hdType = getRowType(kHouseholdDemographics, hdCols);
    std::vector<std::string> caCols = {
        "ca_address_sk", "ca_street_number", "ca_street_name", "ca_city",
        "ca_zip"};
    auto caType = getRowType(kCustomerAddress, caCols);
    std::vector<std::string> ibCols = {"ib_income_band_sk"};
    auto ibType = getRowType(kIncomeBand, ibCols);
    std::vector<std::string> itemCols = {
        "i_item_sk", "i_product_name", "i_color", "i_current_price"};
    auto itemType = getRowType(kItem, itemCols);

    auto srNode = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(kStoreReturns, srType, srFileColumnNames, {})
                      .captureScanNodeId(srScanId)
                      .planNode();

    auto csUiNode = makeCsUi();

    auto d1Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                      .filtersAsNode(filtersAsNode_)
                      .tableScan(
                          kDateDim,
                          ddType,
                          ddFileColumnNames,
                          {fmt::format("d_year = {}", year)})
                      .captureScanNodeId(d1ScanId)
                      .project(
                          {"d_date_sk as d1_date_sk", "d_year as syear"})
                      .planNode();

    auto d2Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kDateDim, ddType, ddFileColumnNames, {})
            .captureScanNodeId(d2ScanId)
            .project({"d_date_sk as d2_date_sk", "d_year as fsyear"})
            .planNode();

    auto d3Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kDateDim, ddType, ddFileColumnNames, {})
            .captureScanNodeId(d3ScanId)
            .project({"d_date_sk as d3_date_sk", "d_year as s2year"})
            .planNode();

    auto storeNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kStore, storeType, storeFileColumnNames, {})
            .captureScanNodeId(storeScanId)
            .planNode();

    auto custNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomer, custType, custFileColumnNames, {})
            .captureScanNodeId(custScanId)
            .planNode();

    auto cd1Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomerDemographics, cdType, cdFileColumnNames, {})
            .captureScanNodeId(cd1ScanId)
            .project(
                {"cd_demo_sk as cd1_demo_sk",
                 "cd_marital_status as cd1_marital_status"})
            .planNode();

    auto cd2Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomerDemographics, cdType, cdFileColumnNames, {})
            .captureScanNodeId(cd2ScanId)
            .project(
                {"cd_demo_sk as cd2_demo_sk",
                 "cd_marital_status as cd2_marital_status"})
            .planNode();

    auto promoNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kPromotion, promoType, promoFileColumnNames, {})
            .captureScanNodeId(promoScanId)
            .planNode();

    auto hd1Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kHouseholdDemographics, hdType, hdFileColumnNames, {})
            .captureScanNodeId(hd1ScanId)
            .project(
                {"hd_demo_sk as hd1_demo_sk",
                 "hd_income_band_sk as hd1_income_band_sk"})
            .planNode();

    auto hd2Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kHouseholdDemographics, hdType, hdFileColumnNames, {})
            .captureScanNodeId(hd2ScanId)
            .project(
                {"hd_demo_sk as hd2_demo_sk",
                 "hd_income_band_sk as hd2_income_band_sk"})
            .planNode();

    auto ad1Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
            .captureScanNodeId(ad1ScanId)
            .project(
                {"ca_address_sk as ad1_address_sk",
                 "ca_street_number as b_street_number",
                 "ca_street_name as b_street_name",
                 "ca_city as b_city",
                 "ca_zip as b_zip"})
            .planNode();

    auto ad2Node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kCustomerAddress, caType, caFileColumnNames, {})
            .captureScanNodeId(ad2ScanId)
            .project(
                {"ca_address_sk as ad2_address_sk",
                 "ca_street_number as c_street_number",
                 "ca_street_name as c_street_name",
                 "ca_city as c_city",
                 "ca_zip as c_zip"})
            .planNode();

    auto ib1Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(kIncomeBand, ibType, ibFileColumnNames, {})
                       .captureScanNodeId(ib1ScanId)
                       .project({"ib_income_band_sk as ib1_income_band_sk"})
                       .planNode();

    auto ib2Node = PlanBuilder(planNodeIdGenerator, pool_.get())
                       .filtersAsNode(filtersAsNode_)
                       .tableScan(kIncomeBand, ibType, ibFileColumnNames, {})
                       .captureScanNodeId(ib2ScanId)
                       .project({"ib_income_band_sk as ib2_income_band_sk"})
                       .planNode();

    // i_color IN (...) pushed; the price band (65..74, the intersection of
    // the two spec BETWEENs) as a cast-to-double filter node.
    auto itemNode =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(
                kItem,
                itemType,
                itemFileColumnNames,
                {"i_color in ('purple', 'burlywood', 'indian', 'spring', "
                 "'floral', 'medium')"})
            .captureScanNodeId(itemScanId)
            .filter("cast(i_current_price as double) between 65.0 and 74.0")
            .planNode();

    auto node =
        PlanBuilder(planNodeIdGenerator, pool_.get())
            .filtersAsNode(filtersAsNode_)
            .tableScan(kStoreSales, ssType, ssFileColumnNames, {})
            .captureScanNodeId(ssScanId)
            .hashJoin(
                {"ss_item_sk", "ss_ticket_number"},
                {"sr_item_sk", "sr_ticket_number"},
                srNode,
                "",
                {"ss_sold_date_sk", "ss_item_sk", "ss_customer_sk",
                 "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk",
                 "ss_promo_sk", "ss_wholesale_cost", "ss_list_price",
                 "ss_coupon_amt"})
            .hashJoin(
                {"ss_item_sk"},
                {"csui_item_sk"},
                csUiNode,
                "",
                {"ss_sold_date_sk", "ss_item_sk", "ss_customer_sk",
                 "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk",
                 "ss_promo_sk", "ss_wholesale_cost", "ss_list_price",
                 "ss_coupon_amt"})
            .hashJoin(
                {"ss_item_sk"},
                {"i_item_sk"},
                itemNode,
                "",
                {"ss_sold_date_sk", "ss_item_sk", "ss_customer_sk",
                 "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk",
                 "ss_promo_sk", "ss_wholesale_cost", "ss_list_price",
                 "ss_coupon_amt", "i_product_name"})
            .hashJoin(
                {"ss_sold_date_sk"},
                {"d1_date_sk"},
                d1Node,
                "",
                {"ss_item_sk", "ss_customer_sk", "ss_cdemo_sk", "ss_hdemo_sk",
                 "ss_addr_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear"})
            .hashJoin(
                {"ss_customer_sk"},
                {"c_customer_sk"},
                custNode,
                "",
                {"ss_item_sk", "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk",
                 "ss_store_sk", "ss_promo_sk", "ss_wholesale_cost",
                 "ss_list_price", "ss_coupon_amt", "i_product_name", "syear",
                 "c_current_cdemo_sk", "c_current_hdemo_sk",
                 "c_current_addr_sk", "c_first_sales_date_sk",
                 "c_first_shipto_date_sk"})
            .hashJoin(
                {"ss_cdemo_sk"},
                {"cd1_demo_sk"},
                cd1Node,
                "",
                {"ss_item_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk",
                 "ss_promo_sk", "ss_wholesale_cost", "ss_list_price",
                 "ss_coupon_amt", "i_product_name", "syear",
                 "c_current_cdemo_sk", "c_current_hdemo_sk",
                 "c_current_addr_sk", "c_first_sales_date_sk",
                 "c_first_shipto_date_sk", "cd1_marital_status"})
            .hashJoin(
                {"c_current_cdemo_sk"},
                {"cd2_demo_sk"},
                cd2Node,
                "cd1_marital_status <> cd2_marital_status",
                {"ss_item_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk",
                 "ss_promo_sk", "ss_wholesale_cost", "ss_list_price",
                 "ss_coupon_amt", "i_product_name", "syear",
                 "c_current_hdemo_sk", "c_current_addr_sk",
                 "c_first_sales_date_sk", "c_first_shipto_date_sk"})
            .hashJoin(
                {"ss_hdemo_sk"},
                {"hd1_demo_sk"},
                hd1Node,
                "",
                {"ss_item_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_current_hdemo_sk",
                 "c_current_addr_sk", "c_first_sales_date_sk",
                 "c_first_shipto_date_sk", "hd1_income_band_sk"})
            .hashJoin(
                {"hd1_income_band_sk"},
                {"ib1_income_band_sk"},
                ib1Node,
                "",
                {"ss_item_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_current_hdemo_sk",
                 "c_current_addr_sk", "c_first_sales_date_sk",
                 "c_first_shipto_date_sk"})
            .hashJoin(
                {"c_current_hdemo_sk"},
                {"hd2_demo_sk"},
                hd2Node,
                "",
                {"ss_item_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_current_addr_sk",
                 "c_first_sales_date_sk", "c_first_shipto_date_sk",
                 "hd2_income_band_sk"})
            .hashJoin(
                {"hd2_income_band_sk"},
                {"ib2_income_band_sk"},
                ib2Node,
                "",
                {"ss_item_sk", "ss_addr_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_current_addr_sk",
                 "c_first_sales_date_sk", "c_first_shipto_date_sk"})
            .hashJoin(
                {"ss_addr_sk"},
                {"ad1_address_sk"},
                ad1Node,
                "",
                {"ss_item_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_current_addr_sk",
                 "c_first_sales_date_sk", "c_first_shipto_date_sk",
                 "b_street_number", "b_street_name", "b_city", "b_zip"})
            .hashJoin(
                {"c_current_addr_sk"},
                {"ad2_address_sk"},
                ad2Node,
                "",
                {"ss_item_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_first_sales_date_sk",
                 "c_first_shipto_date_sk", "b_street_number", "b_street_name",
                 "b_city", "b_zip", "c_street_number", "c_street_name",
                 "c_city", "c_zip"})
            .hashJoin(
                {"c_first_sales_date_sk"},
                {"d2_date_sk"},
                d2Node,
                "",
                {"ss_item_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "c_first_shipto_date_sk",
                 "b_street_number", "b_street_name", "b_city", "b_zip",
                 "c_street_number", "c_street_name", "c_city", "c_zip",
                 "fsyear"})
            .hashJoin(
                {"c_first_shipto_date_sk"},
                {"d3_date_sk"},
                d3Node,
                "",
                {"ss_item_sk", "ss_store_sk", "ss_promo_sk",
                 "ss_wholesale_cost", "ss_list_price", "ss_coupon_amt",
                 "i_product_name", "syear", "b_street_number",
                 "b_street_name", "b_city", "b_zip", "c_street_number",
                 "c_street_name", "c_city", "c_zip", "fsyear", "s2year"})
            .hashJoin(
                {"ss_store_sk"},
                {"s_store_sk"},
                storeNode,
                "",
                {"ss_item_sk", "ss_promo_sk", "ss_wholesale_cost",
                 "ss_list_price", "ss_coupon_amt", "i_product_name", "syear",
                 "b_street_number", "b_street_name", "b_city", "b_zip",
                 "c_street_number", "c_street_name", "c_city", "c_zip",
                 "fsyear", "s2year", "s_store_name", "s_zip"})
            .hashJoin(
                {"ss_promo_sk"},
                {"p_promo_sk"},
                promoNode,
                "",
                {"ss_item_sk", "ss_wholesale_cost", "ss_list_price",
                 "ss_coupon_amt", "i_product_name", "syear",
                 "b_street_number", "b_street_name", "b_city", "b_zip",
                 "c_street_number", "c_street_name", "c_city", "c_zip",
                 "fsyear", "s2year", "s_store_name", "s_zip"})
            .project(
                {"i_product_name", "ss_item_sk", "s_store_name", "s_zip",
                 "b_street_number", "b_street_name", "b_city", "b_zip",
                 "c_street_number", "c_street_name", "c_city", "c_zip",
                 "syear", "fsyear", "s2year",
                 "cast(ss_wholesale_cost as double) as s1_row",
                 "cast(ss_list_price as double) as s2_row",
                 "cast(ss_coupon_amt as double) as s3_row"})
            .partialAggregation(
                {"i_product_name", "ss_item_sk", "s_store_name", "s_zip",
                 "b_street_number", "b_street_name", "b_city", "b_zip",
                 "c_street_number", "c_street_name", "c_city", "c_zip",
                 "syear", "fsyear", "s2year"},
                {"count(1) as cnt",
                 "sum(s1_row) as s1",
                 "sum(s2_row) as s2",
                 "sum(s3_row) as s3"})
            .localPartition(std::vector<std::string>{})
            .finalAggregation()
            .project(
                {fmt::format("i_product_name as {}_product_name", prefix),
                 fmt::format("ss_item_sk as {}_item_sk", prefix),
                 fmt::format("s_store_name as {}_store_name", prefix),
                 fmt::format("s_zip as {}_store_zip", prefix),
                 fmt::format("b_street_number as {}_b_street_number", prefix),
                 fmt::format("b_street_name as {}_b_street_name", prefix),
                 fmt::format("b_city as {}_b_city", prefix),
                 fmt::format("b_zip as {}_b_zip", prefix),
                 fmt::format("c_street_number as {}_c_street_number", prefix),
                 fmt::format("c_street_name as {}_c_street_name", prefix),
                 fmt::format("c_city as {}_c_city", prefix),
                 fmt::format("c_zip as {}_c_zip", prefix),
                 fmt::format("syear as {}_syear", prefix),
                 fmt::format("fsyear as {}_fsyear", prefix),
                 fmt::format("s2year as {}_s2year", prefix),
                 fmt::format("cnt as {}_cnt", prefix),
                 fmt::format("s1 as {}_s1", prefix),
                 fmt::format("s2 as {}_s2", prefix),
                 fmt::format("s3 as {}_s3", prefix)})
            .planNode();

    result.dataFiles[ssScanId] = getTableFilePaths(kStoreSales);
    result.dataFiles[srScanId] = getTableFilePaths(kStoreReturns);
    result.dataFiles[d1ScanId] = getTableFilePaths(kDateDim);
    result.dataFiles[d2ScanId] = getTableFilePaths(kDateDim);
    result.dataFiles[d3ScanId] = getTableFilePaths(kDateDim);
    result.dataFiles[storeScanId] = getTableFilePaths(kStore);
    result.dataFiles[custScanId] = getTableFilePaths(kCustomer);
    result.dataFiles[cd1ScanId] = getTableFilePaths(kCustomerDemographics);
    result.dataFiles[cd2ScanId] = getTableFilePaths(kCustomerDemographics);
    result.dataFiles[promoScanId] = getTableFilePaths(kPromotion);
    result.dataFiles[hd1ScanId] = getTableFilePaths(kHouseholdDemographics);
    result.dataFiles[hd2ScanId] = getTableFilePaths(kHouseholdDemographics);
    result.dataFiles[ad1ScanId] = getTableFilePaths(kCustomerAddress);
    result.dataFiles[ad2ScanId] = getTableFilePaths(kCustomerAddress);
    result.dataFiles[ib1ScanId] = getTableFilePaths(kIncomeBand);
    result.dataFiles[ib2ScanId] = getTableFilePaths(kIncomeBand);
    result.dataFiles[itemScanId] = getTableFilePaths(kItem);
    return node;
  };

  auto cs2Node = makeCrossSales(2000, "cs2");

  auto plan =
      PlanBuilder(planNodeIdGenerator, pool_.get())
          .filtersAsNode(filtersAsNode_)
          .addNode([&](std::string /*id*/, core::PlanNodePtr /*input*/) {
            return makeCrossSales(1999, "cs1");
          })
          .hashJoin(
              {"cs1_item_sk", "cs1_store_name", "cs1_store_zip"},
              {"cs2_item_sk", "cs2_store_name", "cs2_store_zip"},
              cs2Node,
              "cs2_cnt <= cs1_cnt",
              {"cs1_product_name", "cs1_store_name", "cs1_store_zip",
               "cs1_b_street_number", "cs1_b_street_name", "cs1_b_city",
               "cs1_b_zip", "cs1_c_street_number", "cs1_c_street_name",
               "cs1_c_city", "cs1_c_zip", "cs1_syear", "cs1_cnt", "cs1_s1",
               "cs1_s2", "cs1_s3", "cs2_s1", "cs2_s2", "cs2_s3", "cs2_syear",
               "cs2_cnt"})
          .orderBy(
              {"cs1_product_name", "cs1_store_name", "cs2_cnt", "cs1_s1",
               "cs2_s1"},
              false)
          .planNode();

  result.plan = std::move(plan);
  result.dataFileFormat = format_;
  result.planName = "tpcds_q64";
  return result;
}

const std::vector<std::string> TpcdsQueryBuilder::kTableNames_ = {
    kStoreSales,
    kStoreReturns,
    kCatalogSales,
    kCatalogReturns,
    kWebSales,
    kWebReturns,
    kInventory,
    kCustomer,
    kCustomerAddress,
    kCustomerDemographics,
    kDateDim,
    kTimeDim,
    kItem,
    kStore,
    kPromotion,
    kHouseholdDemographics,
    kWarehouse,
    kShipMode,
    kReason,
    kIncomeBand,
    kCallCenter,
    kCatalogPage,
    kWebPage,
    kWebSite};

const std::unordered_map<std::string, std::vector<std::string>>
    TpcdsQueryBuilder::kTables_ = {
        // Fact tables
        {"store_sales",
         {"ss_sold_date_sk", "ss_sold_time_sk", "ss_item_sk", "ss_customer_sk",
          "ss_cdemo_sk", "ss_hdemo_sk", "ss_addr_sk", "ss_store_sk",
          "ss_promo_sk", "ss_ticket_number", "ss_quantity", "ss_wholesale_cost",
          "ss_list_price", "ss_sales_price", "ss_ext_discount_amt",
          "ss_ext_sales_price", "ss_ext_wholesale_cost", "ss_ext_list_price",
          "ss_ext_tax", "ss_coupon_amt", "ss_net_paid", "ss_net_paid_inc_tax",
          "ss_net_profit"}},
        {"store_returns",
         {"sr_returned_date_sk", "sr_return_time_sk", "sr_item_sk",
          "sr_customer_sk", "sr_cdemo_sk", "sr_hdemo_sk", "sr_addr_sk",
          "sr_store_sk", "sr_reason_sk", "sr_ticket_number",
          "sr_return_quantity", "sr_return_amt", "sr_return_tax",
          "sr_return_amt_inc_tax", "sr_fee", "sr_return_ship_cost",
          "sr_refunded_cash", "sr_reversed_charge", "sr_store_credit",
          "sr_net_loss"}},
        {"catalog_sales",
         {"cs_sold_date_sk", "cs_sold_time_sk", "cs_ship_date_sk",
          "cs_bill_customer_sk", "cs_bill_cdemo_sk", "cs_bill_hdemo_sk",
          "cs_bill_addr_sk", "cs_ship_customer_sk", "cs_ship_cdemo_sk",
          "cs_ship_hdemo_sk", "cs_ship_addr_sk", "cs_call_center_sk",
          "cs_catalog_page_sk", "cs_ship_mode_sk", "cs_warehouse_sk",
          "cs_item_sk", "cs_promo_sk", "cs_order_number", "cs_quantity",
          "cs_wholesale_cost", "cs_list_price", "cs_sales_price",
          "cs_ext_discount_amt", "cs_ext_sales_price", "cs_ext_wholesale_cost",
          "cs_ext_list_price", "cs_ext_tax", "cs_coupon_amt", "cs_ext_ship_cost",
          "cs_net_paid", "cs_net_paid_inc_tax", "cs_net_paid_inc_ship",
          "cs_net_paid_inc_ship_tax", "cs_net_profit"}},
        {"catalog_returns",
         {"cr_returned_date_sk", "cr_returned_time_sk", "cr_item_sk",
          "cr_refunded_customer_sk", "cr_refunded_cdemo_sk",
          "cr_refunded_hdemo_sk", "cr_refunded_addr_sk",
          "cr_returning_customer_sk", "cr_returning_cdemo_sk",
          "cr_returning_hdemo_sk", "cr_returning_addr_sk", "cr_call_center_sk",
          "cr_catalog_page_sk", "cr_ship_mode_sk", "cr_warehouse_sk",
          "cr_reason_sk", "cr_order_number", "cr_return_quantity",
          "cr_return_amount", "cr_return_tax", "cr_return_amt_inc_tax",
          "cr_fee", "cr_return_ship_cost", "cr_refunded_cash",
          "cr_reversed_charge", "cr_store_credit", "cr_net_loss"}},
        {"web_sales",
         {"ws_sold_date_sk", "ws_sold_time_sk", "ws_ship_date_sk", "ws_item_sk",
          "ws_bill_customer_sk", "ws_bill_cdemo_sk", "ws_bill_hdemo_sk",
          "ws_bill_addr_sk", "ws_ship_customer_sk", "ws_ship_cdemo_sk",
          "ws_ship_hdemo_sk", "ws_ship_addr_sk", "ws_web_page_sk",
          "ws_web_site_sk", "ws_ship_mode_sk", "ws_warehouse_sk", "ws_promo_sk",
          "ws_order_number", "ws_quantity", "ws_wholesale_cost", "ws_list_price",
          "ws_sales_price", "ws_ext_discount_amt", "ws_ext_sales_price",
          "ws_ext_wholesale_cost", "ws_ext_list_price", "ws_ext_tax",
          "ws_coupon_amt", "ws_ext_ship_cost", "ws_net_paid",
          "ws_net_paid_inc_tax", "ws_net_paid_inc_ship",
          "ws_net_paid_inc_ship_tax", "ws_net_profit"}},
        {"web_returns",
         {"wr_returned_date_sk", "wr_returned_time_sk", "wr_item_sk",
          "wr_refunded_customer_sk", "wr_refunded_cdemo_sk",
          "wr_refunded_hdemo_sk", "wr_refunded_addr_sk",
          "wr_returning_customer_sk", "wr_returning_cdemo_sk",
          "wr_returning_hdemo_sk", "wr_returning_addr_sk", "wr_web_page_sk",
          "wr_reason_sk", "wr_order_number", "wr_return_quantity",
          "wr_return_amt", "wr_return_tax", "wr_return_amt_inc_tax", "wr_fee",
          "wr_return_ship_cost", "wr_refunded_cash", "wr_reversed_charge",
          "wr_account_credit", "wr_net_loss"}},
        {"inventory",
         {"inv_date_sk", "inv_item_sk", "inv_warehouse_sk",
          "inv_quantity_on_hand"}},
        // Dimension tables
        {"customer",
         {"c_customer_sk", "c_customer_id", "c_current_cdemo_sk",
          "c_current_hdemo_sk", "c_current_addr_sk", "c_first_shipto_date_sk",
          "c_first_sales_date_sk", "c_salutation", "c_first_name",
          "c_last_name", "c_preferred_cust_flag", "c_birth_day",
          "c_birth_month", "c_birth_year", "c_birth_country", "c_login",
          "c_email_address", "c_last_review_date_sk"}},
        {"customer_address",
         {"ca_address_sk", "ca_address_id", "ca_street_number",
          "ca_street_name", "ca_street_type", "ca_suite_number", "ca_city",
          "ca_county", "ca_state", "ca_zip", "ca_country", "ca_gmt_offset",
          "ca_location_type"}},
        {"customer_demographics",
         {"cd_demo_sk", "cd_gender", "cd_marital_status", "cd_education_status",
          "cd_purchase_estimate", "cd_credit_rating", "cd_dep_count",
          "cd_dep_employed_count", "cd_dep_college_count"}},
        {"date_dim",
         {"d_date_sk", "d_date_id", "d_date", "d_month_seq", "d_week_seq",
          "d_quarter_seq", "d_year", "d_dow", "d_moy", "d_dom", "d_qoy",
          "d_fy_year", "d_fy_quarter_seq", "d_fy_week_seq", "d_day_name",
          "d_quarter_name", "d_holiday", "d_weekend", "d_following_holiday",
          "d_first_dom", "d_last_dom", "d_same_day_ly", "d_same_day_lq",
          "d_current_day", "d_current_week", "d_current_month",
          "d_current_quarter", "d_current_year"}},
        {"time_dim",
         {"t_time_sk", "t_time_id", "t_time", "t_hour", "t_minute", "t_second",
          "t_am_pm", "t_shift", "t_sub_shift", "t_meal_time"}},
        {"item",
         {"i_item_sk", "i_item_id", "i_rec_start_date", "i_rec_end_date",
          "i_item_desc", "i_current_price", "i_wholesale_cost", "i_brand_id",
          "i_brand", "i_class_id", "i_class", "i_category_id", "i_category",
          "i_manufact_id", "i_manufact", "i_size", "i_formulation", "i_color",
          "i_units", "i_container", "i_manager_id", "i_product_name"}},
        {"store",
         {"s_store_sk", "s_store_id", "s_rec_start_date", "s_rec_end_date",
          "s_closed_date_sk", "s_store_name", "s_number_employees",
          "s_floor_space", "s_hours", "s_manager", "s_market_id",
          "s_geography_class", "s_market_desc", "s_market_manager",
          "s_division_id", "s_division_name", "s_company_id", "s_company_name",
          "s_street_number", "s_street_name", "s_street_type", "s_suite_number",
          "s_city", "s_county", "s_state", "s_zip", "s_country", "s_gmt_offset",
          "s_tax_percentage"}},
        {"promotion",
         {"p_promo_sk", "p_promo_id", "p_start_date_sk", "p_end_date_sk",
          "p_item_sk", "p_cost", "p_response_target", "p_promo_name",
          "p_channel_dmail", "p_channel_email", "p_channel_catalog",
          "p_channel_tv", "p_channel_radio", "p_channel_press",
          "p_channel_event", "p_channel_demo", "p_channel_details", "p_purpose",
          "p_discount_active"}},
        {"household_demographics",
         {"hd_demo_sk", "hd_income_band_sk", "hd_buy_potential", "hd_dep_count",
          "hd_vehicle_count"}},
        {"warehouse",
         {"w_warehouse_sk", "w_warehouse_id", "w_warehouse_name",
          "w_warehouse_sq_ft", "w_street_number", "w_street_name",
          "w_street_type", "w_suite_number", "w_city", "w_county", "w_state",
          "w_zip", "w_country", "w_gmt_offset"}},
        {"ship_mode",
         {"sm_ship_mode_sk", "sm_ship_mode_id", "sm_type", "sm_code",
          "sm_carrier", "sm_contract"}},
        {"reason", {"r_reason_sk", "r_reason_id", "r_reason_desc"}},
        {"income_band",
         {"ib_income_band_sk", "ib_lower_bound", "ib_upper_bound"}},
        {"call_center",
         {"cc_call_center_sk", "cc_call_center_id", "cc_rec_start_date",
          "cc_rec_end_date", "cc_closed_date_sk", "cc_open_date_sk", "cc_name",
          "cc_class", "cc_employees", "cc_sq_ft", "cc_hours", "cc_manager",
          "cc_mkt_id", "cc_mkt_class", "cc_mkt_desc", "cc_market_manager",
          "cc_division", "cc_division_name", "cc_company", "cc_company_name",
          "cc_street_number", "cc_street_name", "cc_street_type",
          "cc_suite_number", "cc_city", "cc_county", "cc_state", "cc_zip",
          "cc_country", "cc_gmt_offset", "cc_tax_percentage"}},
        {"catalog_page",
         {"cp_catalog_page_sk", "cp_catalog_page_id", "cp_start_date_sk",
          "cp_end_date_sk", "cp_department", "cp_catalog_number",
          "cp_catalog_page_number", "cp_description", "cp_type"}},
        {"web_page",
         {"wp_web_page_sk", "wp_web_page_id", "wp_rec_start_date",
          "wp_rec_end_date", "wp_creation_date_sk", "wp_access_date_sk",
          "wp_autogen_flag", "wp_customer_sk", "wp_url", "wp_type",
          "wp_char_count", "wp_link_count", "wp_image_count",
          "wp_max_ad_count"}},
        {"web_site",
         {"web_site_sk", "web_site_id", "web_rec_start_date",
          "web_rec_end_date", "web_name", "web_open_date_sk",
          "web_close_date_sk", "web_class", "web_manager", "web_mkt_id",
          "web_mkt_class", "web_mkt_desc", "web_market_manager",
          "web_company_id", "web_company_name", "web_street_number",
          "web_street_name", "web_street_type", "web_suite_number", "web_city",
          "web_county", "web_state", "web_zip", "web_country", "web_gmt_offset",
          "web_tax_percentage"}}};

} // namespace facebook::velox::exec::test
