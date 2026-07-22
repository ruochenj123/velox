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
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/RowHashJoin.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/expression/AstExpression.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"
#include "velox/experimental/cudf/expression/JitExpression.h"

#include "folly/Conv.h"
#include "velox/exec/Driver.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Values.h"

#include <cudf/detail/nvtx/ranges.hpp>

#include <cuda.h>

#include <iostream>

static const std::string kCudfAdapterName = "cuDF";

namespace facebook::velox::cudf_velox {

namespace {

template <class... Deriveds, class Base>
bool isAnyOf(const Base* p) {
  return ((dynamic_cast<const Deriveds*>(p) != nullptr) || ...);
}

} // namespace

core::PlanNodePtr CompileState::getPlanNode(const core::PlanNodeId& id) const {
  auto& nodes = driverFactory_.planNodes;
  auto it = std::find_if(nodes.cbegin(), nodes.cend(), [&id](const auto& node) {
    return node->id() == id;
  });
  if (it != nodes.end()) {
    return *it;
  }
  VELOX_CHECK(driverFactory_.consumerNode->id() == id);
  return driverFactory_.consumerNode;
}

bool CompileState::compile(bool allowCpuFallback) {
  auto operators = driver_.operators();

  // ---- Fused multi-way probe pre-pass ----
  // When enabled, replace a maximal run of >=2 consecutive inner-join
  // HashProbe operators on this pipeline with a single FusedRowHashJoinProbe
  // that walks each probe tuple through all N build hash tables in one kernel.
  // Runs before the per-operator adaptation loop so those HashProbe operators
  // are consumed here (the normal RowHashJoinProbeAdapter never sees them).
  if (CudfConfig::getInstance().benchmarkFusedProbe &&
      CudfConfig::getInstance().benchmarkRowWiseGather) {
    auto* preCtx = driver_.driverCtx();
    auto isValidId = [](const core::PlanNodeId& id) {
      return !id.empty() && id != "N/A";
    };
    auto isFusibleProbe = [&](exec::Operator* op)
        -> std::shared_ptr<const core::HashJoinNode> {
      if (dynamic_cast<const exec::HashProbe*>(op) == nullptr) {
        return nullptr;
      }
      if (!isValidId(op->planNodeId())) {
        return nullptr;
      }
      auto node = std::dynamic_pointer_cast<const core::HashJoinNode>(
          getPlanNode(op->planNodeId()));
      if (!node || node->joinType() != core::JoinType::kInner ||
          node->filter()) {
        return nullptr;
      }
      return node;
    };

    for (int32_t i = 0; i < static_cast<int32_t>(operators.size());) {
      std::vector<std::shared_ptr<const core::HashJoinNode>> runNodes;
      int32_t j = i;
      while (j < static_cast<int32_t>(operators.size())) {
        auto node = isFusibleProbe(operators[j]);
        if (!node) {
          break;
        }
        runNodes.push_back(node);
        ++j;
      }
      // Normally the fused probe only replaces a RUN of >=2 joins (a single
      // join has nothing to fuse). benchmarkFuseSingleJoin lets it fire on a
      // single join too: that yields a FusedRowHashJoinProbe with numSteps=1,
      // i.e. our ROW-NATIVE join primitive (row build store + device-probeable
      // GpuHashTable + probe-and-materialize in one kernel) standing in for
      // cudf::inner_join. It isolates the row-layout join from fusion, so we can
      // ask whether the primitive is good BEFORE asking whether chaining it wins.
      const size_t minRun =
          CudfConfig::getInstance().benchmarkFuseSingleJoin ? 1 : 2;
      if (runNodes.size() >= minRun) {
        std::vector<std::unique_ptr<exec::Operator>> fused;

        // GPU re-batching is orthogonal to the layout: give the fused probe the
        // same large batches CudfBatchConcat gives the columnar probe, so the
        // comparison isolates the layout rather than the batch size. The fused
        // probe replaced the WHOLE chain, so a single concat in front of it is
        // all there is -- unlike column mode, which needs one before every
        // probe because each join re-materialises a (shrinking) intermediate.
        if (CudfConfig::getInstance().benchmarkConcatBeforeJoin) {
          fused.push_back(
              std::make_unique<CudfBatchConcat>(
                  operators[i]->operatorId(),
                  preCtx,
                  runNodes.front(),
                  runNodes.front()->sources()[0]->outputType()));
        }

        // Record which joins are fused so RowHashJoinBuild builds only the
        // device map for them (and skips the dead cudf::hash_join). Done before
        // the move below consumes runNodes.
        for (const auto& rn : runNodes) {
          CudfConfig::getInstance().markProbeFused(rn->id());
        }
        fused.push_back(std::make_unique<FusedRowHashJoinProbe>(
            operators[i]->operatorId(), preCtx, std::move(runNodes)));
        // Replace the half-open range [i, j) with the single fused operator.
        driverFactory_.replaceOperators(driver_, i, j, std::move(fused));
        operators = driver_.operators();
        ++i; // step past the fused operator
      } else {
        i = (j > i) ? j : i + 1;
      }
    }
  }

  // Cache debug flag to avoid repeated getInstance() calls
  const bool debugEnabled = CudfConfig::getInstance().debugEnabled;

  // Cache "before" operator descriptions so we can print before/after together.
  std::vector<std::pair<int32_t, std::string>> beforeOperators;
  if (debugEnabled) {
    for (const auto& op : operators) {
      beforeOperators.emplace_back(op->operatorId(), op->toString());
    }
  }

  bool replacementsMade = false;
  auto ctx = driver_.driverCtx();

  // Helper to check if planNodeId is valid (some operators like CallbackSink
  // have "N/A")
  auto isValidPlanNodeId = [](const core::PlanNodeId& id) {
    return !id.empty() && id != "N/A";
  };

  // Use adapter registry for GPU Operator Replacement
  auto& registry = OperatorAdapterRegistry::getInstance();

  // Cached operator properties including adapter pointer.
  struct OperatorProperties : OperatorAdapter::Properties {
    const OperatorAdapter* adapter = nullptr;
  };

  auto getOperatorProperties =
      [&registry, this, &isValidPlanNodeId, ctx](const exec::Operator* op) {
        OperatorProperties props;
        auto adapter = registry.findAdapter(op);
        props.adapter = adapter;
        if (adapter && isValidPlanNodeId(op->planNodeId())) {
          static_cast<OperatorAdapter::Properties&>(props) =
              adapter->properties(op, getPlanNode(op->planNodeId()), ctx);
        }
        if (isAnyOf<CudfOperator>(op)) {
          // CudfOperator is always fully GPU compatible
          // (runs on GPU, accepts GPU input, produces GPU output).
          props.canRunOnGPU = true;
          props.acceptsGpuInput = true;
          props.producesGpuOutput = true;
        }
        if (isAnyOf<FusedRowHashJoinProbe, CudfBatchConcat>(op)) {
          // Both are inserted by the fused PRE-PASS, before this loop runs, so
          // unlike operators created inside an adapter's createReplacements they
          // ARE re-examined here -- and no adapter matches either of them.
          //
          // Note CudfOperatorBase does NOT inherit CudfOperator (they are
          // unrelated classes), so the isAnyOf<CudfOperator> test above does not
          // catch them. Without this, they are judged CPU operators and a
          // CudfToVelox is inserted in front, handing CudfBatchConcat a Velox
          // RowVector -- which trips its "expects CudfVector input" check.
          //
          // Both consume GPU input (CudfVector / RowStoreVector) and produce GPU
          // output, so mark them fully GPU compatible.
          props.canRunOnGPU = true;
          props.acceptsGpuInput = true;
          props.producesGpuOutput = true;
        }
        return props;
      };

  // caching operator properties
  std::vector<OperatorProperties> opProps(operators.size());
  std::transform(
      operators.begin(),
      operators.end(),
      opProps.begin(),
      getOperatorProperties);

  int32_t operatorsOffset = 0;
  for (int32_t operatorIndex = 0; operatorIndex < operators.size();
       ++operatorIndex) {
    std::vector<std::unique_ptr<exec::Operator>> replaceOp;

    exec::Operator* oper = operators[operatorIndex];
    auto replacingOperatorIndex = operatorIndex + operatorsOffset;
    VELOX_CHECK(oper);
    const auto& thisOpProps =
        opProps[operatorIndex]; // cached operator properties

    const bool previousOperatorIsNotGpu =
        operatorIndex > 0 and !opProps[operatorIndex - 1].producesGpuOutput;
    const bool nextOperatorIsNotGpu = (operatorIndex < operators.size() - 1) and
        !opProps[operatorIndex + 1].acceptsGpuInput;
    const bool isLastOperatorOfTask =
        driverFactory_.outputDriver and operatorIndex == operators.size() - 1;

    auto id = oper->operatorId();

    // Cache planNode for this operator (avoid multiple lookups)
    core::PlanNodePtr planNode = nullptr;
    if (isValidPlanNodeId(oper->planNodeId())) {
      planNode = getPlanNode(oper->planNodeId());
    }

    if (previousOperatorIsNotGpu and thisOpProps.acceptsGpuInput and planNode) {
      replaceOp.push_back(
          std::make_unique<CudfFromVelox>(
              id, planNode->outputType(), ctx, planNode->id() + "-from-velox"));
    }
    if (not replaceOp.empty()) {
      // from-velox only, because need to inserted before current operator.
      operatorsOffset += replaceOp.size();
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_,
          replacingOperatorIndex,
          replacingOperatorIndex,
          std::move(replaceOp));
      replacingOperatorIndex = operatorIndex + operatorsOffset;
      replaceOp.clear();
      replacementsMade = true;
    }

    // Use adapter registry to handle operator replacement
    auto keepOperator = 1; // Default: keep original
    const auto& adapter = thisOpProps.adapter;
    bool isPureCpuOperator = true;

    if (adapter) {
      keepOperator = adapter->keepOperator();
      if (keepOperator == 0) {
        if (planNode && thisOpProps.canRunOnGPU) {
          auto replacements =
              adapter->createReplacements(oper, planNode, ctx, id);
          for (auto& r : replacements) {
            replaceOp.push_back(std::move(r));
          }
          isPureCpuOperator = false;
        } else {
          // This is the CPU fallback case.
          isPureCpuOperator = true;
        }
      } else {
        // adapter is present and keepOperator is 1, so this is GPU compatible
        // operator. so this CPU operators is allowed even if fallback is
        // disabled.
        isPureCpuOperator = false;
      }
    } else {
      // special case for CudfOperator
      //
      // CudfBatchConcat must be listed for the same reason FusedRowHashJoinProbe
      // is: the fused pre-pass inserts BOTH into the operator list before this
      // loop, so both are re-examined here, and neither has an adapter nor
      // inherits CudfOperator. Omitting it makes the concat a "pure CPU"
      // operator, which aborts the task outright when allowCpuFallback=false.
      // See the matching isAnyOf<> in the properties lambda above -- these two
      // lists have to agree.
      if (isAnyOf<CudfOperator>(oper) ||
          isAnyOf<FusedRowHashJoinProbe, CudfBatchConcat>(oper)) {
        isPureCpuOperator = false;
      } else {
        // CPU operator without adapter
        isPureCpuOperator = true;
      }
    }

    if (thisOpProps.producesGpuOutput and
        (nextOperatorIsNotGpu or isLastOperatorOfTask) and planNode) {
      replaceOp.push_back(
          std::make_unique<CudfToVelox>(
              id, planNode->outputType(), ctx, planNode->id() + "-to-velox"));
    }

    if (debugEnabled) {
      VLOG(1) << "Operator: ID " << oper->operatorId() << ": "
              << oper->toString() << ", keepOperator = " << keepOperator
              << ", isPureCpuOperator = " << isPureCpuOperator
              << ", replaceOp.size() = " << replaceOp.size()
              << ", previousOperatorIsNotGpu = " << previousOperatorIsNotGpu
              << ", nextOperatorIsNotGpu = " << nextOperatorIsNotGpu
              << ", isLastOperatorOfTask = " << isLastOperatorOfTask
              << ", canRunOnGPU[" << operatorIndex
              << "] = " << thisOpProps.canRunOnGPU << ", acceptsGpuInput["
              << operatorIndex << "] = " << thisOpProps.acceptsGpuInput
              << ", producesGpuOutput[" << operatorIndex
              << "] = " << thisOpProps.producesGpuOutput
              << ", planNode = " << bool(planNode);
    }
    if (!allowCpuFallback) {
      // condition is if GPU replacement success or if CPU operators itself is
      // GPU compatible. or if specific CPU operator is allowed even when
      // fallback is disabled.
      VELOX_CHECK(!isPureCpuOperator, "Replacement with cuDF operator failed");
    } else if (isPureCpuOperator) {
      LOG(WARNING)
          << "Replacement with cuDF operator failed. Falling back to CPU execution";
      LOG(WARNING) << "Replacement Failed Operator: " << oper->toString();
      auto planNode = getPlanNode(oper->planNodeId());
      LOG(WARNING) << "Replacement Failed PlanNode: "
                   << planNode->toString(true, false);
    }

    if (not replaceOp.empty()) {
      // ReplaceOp, to-velox.
      operatorsOffset += replaceOp.size() - 1 + keepOperator;
      [[maybe_unused]] auto replaced = driverFactory_.replaceOperators(
          driver_,
          replacingOperatorIndex + keepOperator,
          replacingOperatorIndex + 1,
          std::move(replaceOp));
      replacementsMade = true;
    }
  }

  if (debugEnabled) {
    // Print before/after together for easy comparison.
    LOG(INFO) << "Operators " << "before adapting for cuDF"
              << ": count [" << beforeOperators.size() << "]";
    for (const auto& [id, desc] : beforeOperators) {
      LOG(INFO) << "  Operator: ID " << id << ": " << desc;
    }
    LOG(INFO) << "allowCpuFallback = " << allowCpuFallback;

    operators = driver_.operators();
    LOG(INFO) << "Operators " << "after adapting for cuDF"
              << ": count [" << operators.size() << "]";
    for (const auto& op : operators) {
      LOG(INFO) << "  Operator: ID " << op->operatorId() << ": "
                << op->toString();
    }
  }

  return replacementsMade;
}

struct CudfDriverAdapter {
  CudfDriverAdapter(bool allowCpuFallback)
      : allowCpuFallback_{allowCpuFallback} {}

  // Call operator needed by DriverAdapter
  bool operator()(const exec::DriverFactory& factory, exec::Driver& driver) {
    if (!driver.driverCtx()->queryConfig().get<bool>(
            CudfConfig::kCudfEnabled, CudfConfig::getInstance().enabled) &&
        allowCpuFallback_) {
      return false;
    }
    auto state = CompileState(factory, driver);
    auto res = state.compile(allowCpuFallback_);
    return res;
  }

 private:
  bool allowCpuFallback_;
};

static bool isCudfRegistered = false;

bool cudfIsRegistered() {
  return isCudfRegistered;
}

void registerCudf() {
  if (cudfIsRegistered()) {
    return;
  }

  // Register operator adapters
  registerAllOperatorAdapters();

  auto prefix = CudfConfig::getInstance().functionNamePrefix;
  registerBuiltinFunctions(prefix);
  registerStepAwareBuiltinAggregationFunctions(prefix);

  CUDF_FUNC_RANGE();
  cudaFree(nullptr); // Initialize CUDA context at startup

  const std::string mrMode = CudfConfig::getInstance().memoryResource;
  auto mr = cudf_velox::createMemoryResource(
      mrMode, CudfConfig::getInstance().memoryPercent);
  cudf::set_current_device_resource(mr.get());
  mr_ = mr;

  const auto& outputMrMode = CudfConfig::getInstance().outputMemoryResource;
  if (!outputMrMode.empty() && outputMrMode != mrMode) {
    output_mr_ = cudf_velox::createMemoryResource(
        outputMrMode, CudfConfig::getInstance().memoryPercent);
  } else {
    output_mr_ = mr_;
  }

  if (CudfConfig::getInstance().benchmarkRowWiseGather) {
    exec::Operator::registerOperator(
        std::make_unique<RowHashJoinBridgeTranslator>());
  } else {
    exec::Operator::registerOperator(
        std::make_unique<CudfHashJoinBridgeTranslator>());
  }
  CudfDriverAdapter cda{CudfConfig::getInstance().allowCpuFallback};
  exec::DriverAdapter cudfAdapter{kCudfAdapterName, {}, cda};
  exec::DriverFactory::registerAdapter(cudfAdapter);

  if (CudfConfig::getInstance().astExpressionEnabled) {
    registerAstEvaluator(CudfConfig::getInstance().astExpressionPriority);
  }

  if (CudfConfig::getInstance().jitExpressionEnabled) {
    registerJitEvaluator(CudfConfig::getInstance().jitExpressionPriority);
  }

  isCudfRegistered = true;
}

void unregisterCudf() {
  output_mr_ = nullptr;
  mr_ = nullptr;
  exec::DriverFactory::adapters.erase(
      std::remove_if(
          exec::DriverFactory::adapters.begin(),
          exec::DriverFactory::adapters.end(),
          [](const exec::DriverAdapter& adapter) {
            return adapter.label == kCudfAdapterName;
          }),
      exec::DriverFactory::adapters.end());

  isCudfRegistered = false;
}

CudfConfig& CudfConfig::getInstance() {
  static CudfConfig instance;
  return instance;
}

void CudfConfig::markProbeFused(const std::string& joinNodeId) {
  std::lock_guard<std::mutex> l(fusedProbeMutex_);
  fusedProbeJoinIds_.insert(joinNodeId);
}

bool CudfConfig::isProbeFused(const std::string& joinNodeId) const {
  std::lock_guard<std::mutex> l(fusedProbeMutex_);
  return fusedProbeJoinIds_.count(joinNodeId) > 0;
}

void CudfConfig::initialize(
    std::unordered_map<std::string, std::string>&& config) {
  if (config.find(kCudfEnabled) != config.end()) {
    enabled = folly::to<bool>(config[kCudfEnabled]);
  }
  if (config.find(kCudfDebugEnabled) != config.end()) {
    debugEnabled = folly::to<bool>(config[kCudfDebugEnabled]);
  }
  if (config.find(kCudfMemoryResource) != config.end()) {
    memoryResource = config[kCudfMemoryResource];
  }
  if (config.find(kCudfMemoryPercent) != config.end()) {
    memoryPercent = folly::to<int32_t>(config[kCudfMemoryPercent]);
  }
  if (config.find(kCudfOutputMr) != config.end()) {
    outputMemoryResource = config[kCudfOutputMr];
  }
  if (config.find(kCudfBatchSizeMinThreshold) != config.end()) {
    batchSizeMinThreshold =
        folly::to<int32_t>(config[kCudfBatchSizeMinThreshold]);
  }
  if (config.find(kCudfBatchSizeMaxThreshold) != config.end()) {
    batchSizeMaxThreshold =
        folly::to<int32_t>(config[kCudfBatchSizeMaxThreshold]);
  }
  if (config.find(kCudfConcatOptimizationEnabled) != config.end()) {
    concatOptimizationEnabled =
        folly::to<bool>(config[kCudfConcatOptimizationEnabled]);
  }
  if (config.find(kCudfFunctionNamePrefix) != config.end()) {
    functionNamePrefix = config[kCudfFunctionNamePrefix];
  }
  if (config.find(kCudfAstExpressionEnabled) != config.end()) {
    astExpressionEnabled = folly::to<bool>(config[kCudfAstExpressionEnabled]);
  }
  if (config.find(kCudfJitExpressionEnabled) != config.end()) {
    jitExpressionEnabled = folly::to<bool>(config[kCudfJitExpressionEnabled]);
  }
  if (config.find(kCudfAstExpressionPriority) != config.end()) {
    astExpressionPriority =
        folly::to<int32_t>(config[kCudfAstExpressionPriority]);
  }
  if (config.find(kCudfAllowCpuFallback) != config.end()) {
    allowCpuFallback = folly::to<bool>(config[kCudfAllowCpuFallback]);
  }
  if (config.find(kCudfLogFallback) != config.end()) {
    logFallback = folly::to<bool>(config[kCudfLogFallback]);
  }
  if (config.find(kCudfTopNBatchSize) != config.end()) {
    topNBatchSize = folly::to<int32_t>(config[kCudfTopNBatchSize]);
  }
  if (config.find(kCudfTimestampUnit) != config.end()) {
    const auto& unit = config[kCudfTimestampUnit];
    if (unit == "s") {
      timestampUnit = cudf::type_id::TIMESTAMP_SECONDS;
    } else if (unit == "ms") {
      timestampUnit = cudf::type_id::TIMESTAMP_MILLISECONDS;
    } else if (unit == "us") {
      timestampUnit = cudf::type_id::TIMESTAMP_MICROSECONDS;
    } else if (unit == "ns") {
      timestampUnit = cudf::type_id::TIMESTAMP_NANOSECONDS;
    } else {
      VELOX_FAIL(
          "Invalid timestamp unit: {}. Valid values are: s, ms, us, ns", unit);
    }
  }
}

} // namespace facebook::velox::cudf_velox
