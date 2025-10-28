#include <iostream>
#include <fstream>

#include "velox/substrait/SubstraitToVeloxPlan.h"
#include "JsonToProtoConverter.h"

#include "velox/common/base/Fs.h"
#include "velox/common/file/FileSystems.h"
#include "velox/common/memory/Memory.h"
#include "velox/connectors/hive/HiveConnector.h"
#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/dwio/common/FileSink.h"
#include "velox/dwio/dwrf/RegisterDwrfReader.h"
#include "velox/dwio/dwrf/RegisterDwrfWriter.h"
#include "velox/dwio/parquet/RegisterParquetReader.h"
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#include "velox/exec/Task.h"
#include "velox/exec/tests/utils/HiveConnectorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"


#include <folly/init/Init.h>
#include <algorithm>

#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

#include "velox/connectors/hive/HiveConnectorSplit.h"
#include "velox/vector/VectorPrinter.h"
#include "velox/exec/HashPartitionFunction.h"
#include "velox/exec/Exchange.h"
#include "velox/exec/ExchangeSource.h"
#include "velox/exec/tests/utils/LocalExchangeSource.h"

using namespace facebook::velox;

std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>>
  makeSplits(
      const facebook::velox::substrait::SubstraitVeloxPlanConverter& converter,
      std::shared_ptr<const core::PlanNode> planNode, const std::string& dataDir) {
    const auto& splitInfos = converter.splitInfos();
    auto leafPlanNodeIds = planNode->leafPlanNodeIds();
    // Only one leaf node is expected here.
    const auto& splitInfo = splitInfos.at(*leafPlanNodeIds.begin());

    const auto& paths = splitInfo->paths;
    const auto& starts = splitInfo->starts;
    const auto& lengths = splitInfo->lengths;
    const auto fileFormat = splitInfo->format;

    std::vector<std::shared_ptr<connector::hive::HiveConnectorSplit>>
        splits;
    splits.reserve(paths.size());
    std::cout << "=== Splits ===" << std::endl;
    for (int i = 0; i < paths.size(); i++) {
      // auto path = fmt::format("{}{}", dataDir, paths[i]);
      auto path = std::filesystem::absolute(fmt::format("{}{}", dataDir, paths[i]));
      auto start = starts[i];
      auto length = lengths[i];
      std::cout << path << " " << start << " " << length << std::endl;
      auto hiveSplit = std::make_shared<connector::hive::HiveConnectorSplit>(
          "test-hive", // connector ID (must match registration)
          path.string(),        // file path
          fileFormat,  // dwio::common::FileFormat
          start,       // start offset
          length       // length
      );
      splits.emplace_back(hiveSplit);
    }
    return splits;
  }

  exec::Split remoteSplit(const std::string& taskId) {
    return exec::Split(std::make_shared<exec::RemoteConnectorSplit>(taskId));
  }

  void addRemoteSplits(
      std::shared_ptr<exec::Task> task,
      const std::string &planNodeId,
      const std::string &remoteTaskId) {
    task->addSplit(planNodeId, remoteSplit(remoteTaskId));
    task->noMoreSplits(planNodeId);
  }

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << "<data_root> <num_drivers> <substrait_plan_1.json> <substrait_plan_2.json>" << std::endl;
    return 1;
  }

  std::string plan1 = argv[3];
  std::string plan2 = argv[4];
  std::string dataRoot = argv[1];
  int num_driver = std::stoi(argv[2]);

  memory::MemoryManager::initialize({});
  filesystems::registerLocalFileSystem();
  auto pool = memory::memoryManager()->addLeafPool();

  // Register functions

  functions::prestosql::registerAllScalarFunctions();


  // Register Hive connector manually
  const std::string kHiveConnectorId = "test-hive";

  // Register the Hive Connector Factory.
  connector::registerConnectorFactory(
      std::make_shared<connector::hive::HiveConnectorFactory>());
  // Create a new connector instance from the connector factory and register
  // it:
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector(
              kHiveConnectorId,
              std::make_shared<config::ConfigBase>(
                  std::unordered_map<std::string, std::string>()));
  connector::registerConnector(hiveConnector);

  // To be able to read local files, we need to register the local file
  // filesystem. We also need to register the dwrf reader factory as well as a
  // write protocol, in this case commit is not required:
  filesystems::registerLocalFileSystem();
  dwio::common::registerFileSinks();
  // dwrf::registerDwrfReaderFactory();
  // dwrf::registerDwrfWriterFactory();
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
  serializer::presto::PrestoVectorSerde::registerNamedVectorSerde();


  // Init executor
  std::shared_ptr<folly::Executor> executor(
  std::make_shared<folly::CPUThreadPoolExecutor>(
        std::thread::hardware_concurrency()));
  
  // Init ExchangeSource

  exec::ExchangeSource::factories().clear();
  exec::ExchangeSource::registerFactory(exec::test::createLocalExchangeSource);

  // Parse JSON to Substrait plan
  ::substrait::Plan substraitPlan1;
  JsonToProtoConverter::readFromFile(plan1, substraitPlan1);
  

  // Task 1: TableScan->Filter->PartitionedOut


  // Convert to Velox plan
  facebook::velox::substrait::SubstraitVeloxPlanConverter converter(pool.get());
  auto planNode1 = converter.toVeloxPlan(substraitPlan1);

  

  // PartitionedOutNode
  auto outputType = planNode1->outputType();
  auto exprSet = std::make_shared<core::FieldAccessTypedExpr>(outputType->childAt(0), outputType->nameOf(0));
  auto hashFunctionSpec = std::make_shared<exec::HashPartitionFunctionSpec>(outputType, std::vector<column_index_t>{0});
  planNode1 = std::make_shared<core::PartitionedOutputNode>(
    planNode1->id() + ".ShufflePartition",
    core::PartitionedOutputNode::Kind::kPartitioned,
    std::vector<core::TypedExprPtr>{exprSet},
    2, 
    false, 
    hashFunctionSpec,
    planNode1->outputType(), 
    VectorSerde::Kind::kPresto,
    planNode1);

  std::cout << "=== PlanNode1 ===\n" << planNode1->toString(true, true) << std::endl;
  
  // Create tasks
  core::PlanFragment planFragment1{planNode1};
    
  auto task1 = exec::Task::create("local://task1", planFragment1, 0, core::QueryCtx::create(executor.get()),
  exec::Task::ExecutionMode::kParallel);

  auto leafIds = planNode1->leafPlanNodeIds();
  VELOX_CHECK_EQ(leafIds.size(), 1, "Expected exactly one leaf node");
  auto scanNodeId = *leafIds.begin();
  
  // Add splits
  auto splits = makeSplits(converter, planNode1, dataRoot);
  for (const auto& split : splits) {
    task1->addSplit(scanNodeId, exec::Split{split});
  }
  task1->noMoreSplits(scanNodeId);

  



  // Task 2: Exchange->TableWriter->PartitionedOut(dummy)

  // ExchangeNode
  core::PlanNodePtr planNode2 = std::make_shared<core::ExchangeNode>(
    "2.Exchange",
    planNode1->outputType(),
    VectorSerde::Kind::kPresto
  );

  // TableWriteNode
  outputType = planNode2->outputType();
  std::vector<std::shared_ptr<const connector::hive::HiveColumnHandle>> columnHandles;
  for (int i = 0; i < outputType->size(); ++i) {
    columnHandles.push_back(std::make_shared<connector::hive::HiveColumnHandle>(
        outputType->nameOf(i),
        connector::hive::HiveColumnHandle::ColumnType::kRegular,
        outputType->childAt(i),
        outputType->childAt(i)));
  }

  auto dataDir = std::filesystem::absolute(dataRoot);
  auto locationHandle = std::make_shared<connector::hive::LocationHandle>(
      dataDir, dataDir, connector::hive::LocationHandle::TableType::kNew);

  auto hiveHandle = std::make_shared<connector::hive::HiveInsertTableHandle>(
      columnHandles,
      locationHandle,
      dwio::common::FileFormat::PARQUET);
  
  auto rowCountType = ROW({"rows"}, {BIGINT()});
  auto insertHandle = std::make_shared<core::InsertTableHandle>(kHiveConnectorId, hiveHandle);
  planNode2 = std::make_shared<core::TableWriteNode>(
      planNode2->id() + ".tableWriter",
      outputType,
      outputType->names(),
      nullptr,
      insertHandle,
      true,
      rowCountType,
      connector::CommitStrategy::kNoCommit,
      planNode2);

  // PartitionedOutNode (dummy)
  outputType = planNode2->outputType();
  planNode2 = core::PartitionedOutputNode::single(
    planNode2->id() + ".dummyPartition",
    planNode2->outputType(), 
    VectorSerde::Kind::kPresto,
    planNode2);


  std::cout << "=== PlanNode2 ===\n" << planNode2->toString(true, true) << std::endl;

  // Create tasks
  
  core::PlanFragment planFragment2{planNode2};
    
  auto task2 = exec::Task::create("local://task2", planFragment2, 1, core::QueryCtx::create(executor.get()),
  exec::Task::ExecutionMode::kParallel);

  leafIds = planNode2->leafPlanNodeIds();
  VELOX_CHECK_EQ(leafIds.size(), 1, "Expected exactly one leaf node");
  auto exchangeId = *leafIds.begin();

  // Add split

  addRemoteSplits(task2, exchangeId, task1->taskId());


  // Task 3:
  // TableScan->Filter->PartitionedOut (single)

  // Parse JSON to Substrait plan
  ::substrait::Plan substraitPlan2;
  JsonToProtoConverter::readFromFile(plan2, substraitPlan2);

  // Convert to Velox plan
  auto planNode3 = converter.toVeloxPlan(substraitPlan2);
  
  planNode3 = core::PartitionedOutputNode::single(
    "3.dummyPartition",
    planNode3->outputType(),
    VectorSerde::Kind::kPresto,
    planNode3);
  
  std::cout << "=== PlanNode3 ===\n" << planNode3->toString(true, true) << std::endl;

  // Create tasks
  
  core::PlanFragment planFragment3{planNode3};
    
  auto task3 = exec::Task::create("local://task3", planFragment3, 0, core::QueryCtx::create(executor.get()),
  exec::Task::ExecutionMode::kParallel);
  
  leafIds = planNode3->leafPlanNodeIds();
  VELOX_CHECK_EQ(leafIds.size(), 1, "Expected exactly one leaf node");
  scanNodeId = *leafIds.begin();


  splits = makeSplits(converter, planNode3, dataRoot);
  for (const auto& split : splits) {
    task3->addSplit(scanNodeId, exec::Split{split});
  }
  task3->noMoreSplits(scanNodeId);




  // Task 4:
  // Task1->LocalExchange->Task3->LocalExchange ->Join->TableWrite(for verify results) PartitionedOut(dummy)
  
  core::PlanNodePtr exchangeForBuild = std::make_shared<core::ExchangeNode>(
    "3.ExchangeForBuild",
    planNode1->outputType(),
    VectorSerde::Kind::kPresto
  );

  core::PlanNodePtr exchangeForProbe = std::make_shared<core::ExchangeNode>(
    "3.ExchangeForProbe",
    planNode3->outputType(),
    VectorSerde::Kind::kPresto
  );
  

  // Output type of join
  auto leftType = exchangeForBuild->outputType();
  auto rightType = exchangeForProbe->outputType();

  std::vector<std::string> outputNames;
  std::vector<TypePtr> outputTypes;
  
  for (int i = 0; i < leftType->size(); ++i) {
    outputNames.push_back(leftType->nameOf(i));
    outputTypes.push_back(leftType->childAt(i));
  }
  for (int i = 0; i < rightType->size(); ++i) {
    outputNames.push_back(rightType->nameOf(i));
    outputTypes.push_back(rightType->childAt(i));
  }
  
  auto joinOutputType = ROW(std::move(outputNames), std::move(outputTypes));
  
  // Join keys

  std::vector<core::FieldAccessTypedExprPtr> leftKeys{
    std::make_shared<core::FieldAccessTypedExpr>(leftType->childAt(0), leftType->nameOf(0))
  };

  std::vector<core::FieldAccessTypedExprPtr> rightKeys{
    std::make_shared<core::FieldAccessTypedExpr>(rightType->childAt(0), rightType->nameOf(0))
  };

  // Join node
  core::PlanNodePtr planNode4 = std::make_shared<core::HashJoinNode>(
    "3.HashJoin",
    core::JoinType::kInner,
    false,
    leftKeys,
    rightKeys,
    nullptr,
    exchangeForBuild,
    exchangeForProbe,
    joinOutputType
  );

  auto joinOutput = planNode4->outputType();

  // TableWriter for verify
  columnHandles.clear();
  for (int i = 0; i < joinOutput->size(); ++i) {
    columnHandles.push_back(std::make_shared<connector::hive::HiveColumnHandle>(
        joinOutput->nameOf(i),
        connector::hive::HiveColumnHandle::ColumnType::kRegular,
        joinOutput->childAt(i),
        joinOutput->childAt(i)));
  }
  auto joinHiveHandle = std::make_shared<connector::hive::HiveInsertTableHandle>(
      columnHandles,
      locationHandle,
      dwio::common::FileFormat::PARQUET);
  
  auto joinInsertHandle = std::make_shared<core::InsertTableHandle>(kHiveConnectorId, joinHiveHandle);

  planNode4 = std::make_shared<core::TableWriteNode>(
      "3.tableWriter",
      joinOutput,
      joinOutput->names(),
      nullptr,
      joinInsertHandle,
      true,
      rowCountType,
      connector::CommitStrategy::kNoCommit,
      planNode4);

  // PartitionedOutNode (dummy)
  planNode4 = core::PartitionedOutputNode::single(
    "3.dummyPartition",
    planNode4->outputType(),
    VectorSerde::Kind::kPresto,
    planNode4);
  
  
  std::cout << "=== PlanNode4 ===\n" << planNode4->toString(true, true) << std::endl;

  // Create tasks
  
  core::PlanFragment planFragment4{planNode4};
    
  auto task4 = exec::Task::create("local://task4", planFragment4, 0, core::QueryCtx::create(executor.get()),
  exec::Task::ExecutionMode::kParallel);

  // Add split
  addRemoteSplits(task4, exchangeForBuild->id(), task1->taskId());
  addRemoteSplits(task4, exchangeForProbe->id(), task3->taskId());
  

  // Execute
  task4->start(num_driver);
  std::this_thread::sleep_for(std::chrono::seconds(5));
  task3->start(num_driver);
  task2->start(num_driver);
  task1->start(num_driver);
  std::cout << "After start all" << std::endl;
  // while(true) {
  //   if (std::filesystem::exists(dataRoot + "/test2.parquet")) {
  //     std::cout << "file exists!" << std::endl;
  //     task3->start(num_driver);
  //     break;
  //   }
  // }

  return 0;
}
