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
#include "velox/experimental/cudf/connectors/hive/CudfHiveConnector.h"
#include "velox/experimental/cudf/exec/CudfAssignUniqueId.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/CudfEnforceSingleRow.h"
#include "velox/experimental/cudf/exec/CudfFilterProject.h"
#include "velox/experimental/cudf/exec/CudfHashAggregation.h"
#include "velox/experimental/cudf/exec/CudfHashJoin.h"
#include "velox/experimental/cudf/exec/CudfLimit.h"
#include "velox/experimental/cudf/exec/RowHashJoin.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/CudfMarkDistinct.h"
#include "velox/experimental/cudf/exec/CudfOrderBy.h"
#include "velox/experimental/cudf/exec/CudfTopN.h"
#include "velox/experimental/cudf/exec/OperatorAdapters.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/Validation.h"
#include "velox/experimental/cudf/expression/ExpressionEvaluator.h"

#include "velox/connectors/ConnectorRegistry.h"
#include "velox/exec/AssignUniqueId.h"
#include "velox/exec/CallbackSink.h"
#include "velox/exec/EnforceSingleRow.h"
#include "velox/exec/FilterProject.h"
#include "velox/exec/HashAggregation.h"
#include "velox/exec/HashBuild.h"
#include "velox/exec/HashProbe.h"
#include "velox/exec/Limit.h"
#include "velox/exec/LocalPartition.h"
#include "velox/exec/MarkDistinct.h"
#include "velox/exec/OrderBy.h"
#include "velox/exec/StreamingAggregation.h"
#include "velox/exec/TableScan.h"
#include "velox/exec/Task.h"
#include "velox/exec/TopN.h"
#include "velox/exec/Values.h"

namespace facebook::velox::cudf_velox {

/// OperatorAdapterRegistry Implementation
OperatorAdapterRegistry& OperatorAdapterRegistry::getInstance() {
  static OperatorAdapterRegistry instance;
  return instance;
}

void OperatorAdapterRegistry::registerAdapter(
    std::unique_ptr<OperatorAdapter> adapter) {
  adapters_.push_back(std::move(adapter));
}

const OperatorAdapter* OperatorAdapterRegistry::findAdapter(
    const exec::Operator* op) const {
  for (const auto& adapter : adapters_) {
    if (adapter->canHandle(op)) {
      return adapter.get();
    }
  }
  // Note: It is possible to have priority based adapter search.
  // But this is not implemented because it is not needed for now.
  return nullptr;
}

const std::vector<std::unique_ptr<OperatorAdapter>>&
OperatorAdapterRegistry::getAdapters() const {
  return adapters_;
}

void OperatorAdapterRegistry::clear() {
  adapters_.clear();
}

/// TableScanAdapter - Keeps original operator (produces GPU output)
class TableScanAdapter : public OperatorAdapter {
 public:
  TableScanAdapter() : OperatorAdapter("TableScan") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TableScan*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto tableScanNode =
        std::dynamic_pointer_cast<const core::TableScanNode>(planNode);
    if (!tableScanNode) {
      LOG_FALLBACK(
          "TableScan planNode is not TableScanNode, PlanNode id: {}",
          planNode->id());
      return false;
    }
    auto const& connector = velox::connector::ConnectorRegistry::tryGet(
        tableScanNode->tableHandle()->connectorId());
    auto cudfHiveConnector = std::dynamic_pointer_cast<
        facebook::velox::cudf_velox::connector::hive::CudfHiveConnector>(
        connector);
    if (!cudfHiveConnector) {
      LOG_FALLBACK(
          "TableScan connector is not CudfHiveConnector, PlanNode id: {}",
          planNode->id());
    }
    return cudfHiveConnector != nullptr;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// Approximate width of one row of `type`, in bytes.
///
/// Used to decide whether a scan-adjacent projection actually REDUCES what has
/// to cross PCIe (see FilterProjectAdapter::canRunOnGPU). Only the comparison
/// matters, not the absolute value, so fixed-width types use their real widths
/// and VARCHAR gets a nominal charge -- enough for "did this projection drop a
/// string column?" to come out the right way.
static int64_t estimateRowBytes(const RowTypePtr& type) {
  int64_t bytes = 0;
  for (auto i = 0; i < type->size(); i++) {
    switch (type->childAt(i)->kind()) {
      case TypeKind::BOOLEAN:
      case TypeKind::TINYINT:
        bytes += 1;
        break;
      case TypeKind::SMALLINT:
        bytes += 2;
        break;
      case TypeKind::INTEGER: // includes DATE (logical type over INTEGER)
      case TypeKind::REAL:
        bytes += 4;
        break;
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        bytes += 16; // nominal: strings are the thing we most want to drop
        break;
      default: // BIGINT, DOUBLE, TIMESTAMP, ...
        bytes += 8;
        break;
    }
  }
  return bytes;
}

/// FilterProjectAdapter - Replaces with CudfFilterProject
class FilterProjectAdapter : public OperatorAdapter {
 public:
  FilterProjectAdapter() : OperatorAdapter("FilterProject") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::FilterProject*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    auto filterProjectOp = dynamic_cast<const exec::FilterProject*>(op);
    if (!filterProjectOp) {
      LOG_FALLBACK(
          "FilterProjectAdapter operator is not FilterProject, PlanNode id: {}",
          planNode->id());
      return false;
    }

    auto projectPlanNode =
        std::dynamic_pointer_cast<const core::ProjectNode>(planNode);
    auto filterNode = filterProjectOp->filterNode();

    // [Benchmark] Keep this FilterProject on the CPU, moving the Velox->cuDF
    // (H2D) boundary to *after* it, so that fewer bytes cross PCIe.
    //
    // This is a PER-NODE decision, not a global switch. canRunOnGPU is called
    // once per FilterProject operator with that operator's own plan node, so we
    // answer differently for each. Two conditions must BOTH hold:
    //
    //   (1) SCAN-ADJACENT. The operator's input must come straight from a
    //       TableScan, i.e. it sits BEFORE the H2D boundary already. A
    //       mid-pipeline FilterProject is on the GPU; forcing it to the CPU
    //       would insert a D2H/H2D round trip (e.g. TPC-H Q7's post-join
    //       filter, Q8's post-aggregation division).
    //
    //   (2) IT MUST ACTUALLY REDUCE TRANSFERRED DATA. Either it filters (fewer
    //       ROWS cross PCIe) or its output row is narrower than its input row
    //       (fewer BYTES per row). A widening projection on the CPU sends MORE
    //       data, defeating the purpose -- e.g. year(o_orderdate) turns a
    //       4-byte DATE into an 8-byte BIGINT, so it belongs on the GPU, while
    //       l_extendedprice*(1-l_discount) folds two 8-byte doubles into one
    //       and belongs on the CPU.
    //
    // Together this is a cost-based placement of the H2D boundary rather than a
    // benchmark flag: put the boundary after any scan-adjacent projection that
    // shrinks what has to be transferred.
    if (CudfConfig::getInstance().benchmarkKeepProjectOnCpu) {
      // The operator's true upstream: when a Filter and a Project are fused
      // into one FilterProject, planNode is the ProjectNode and its source is
      // the FilterNode, so step through the filter to find the real source.
      core::PlanNodePtr srcNode;
      if (filterNode) {
        srcNode = filterNode->sources()[0];
      } else if (projectPlanNode) {
        srcNode = projectPlanNode->sources()[0];
      }

      const bool scanAdjacent =
          srcNode != nullptr &&
          std::dynamic_pointer_cast<const core::TableScanNode>(srcNode) !=
              nullptr;

      bool reducesTransfer = (filterNode != nullptr); // drops rows
      if (!reducesTransfer && srcNode != nullptr) {
        reducesTransfer = estimateRowBytes(planNode->outputType()) <
            estimateRowBytes(srcNode->outputType());
      }
      // Row-wise benchmark pipelines pack rows on the CPU (CudfFromVelox
      // pinned pack / keys-only pack), so a scan-adjacent projection — e.g.
      // computed join-key columns for the multi-key / duplicate-key matcher
      // benchmarks — must stay UPSTREAM of the pack even when it widens the
      // row: the row/boundary join operators consume its output directly and
      // a GPU CudfFilterProject between pack and join would break the
      // RowStoreVector format contract.
      if (CudfConfig::getInstance().benchmarkRowWiseGather) {
        reducesTransfer = true;
      }

      if (scanAdjacent && reducesTransfer) {
        LOG_FALLBACK(
            "FilterProject kept on CPU (scan-adjacent and reduces H2D), "
            "PlanNode id: {}",
            planNode->id());
        return false;
      }
      // Otherwise fall through: run it on the GPU like any other operator.
    }

    if (projectPlanNode) {
      if (projectPlanNode->sources()[0]->outputType()->size() == 0) {
        if (filterNode || !projectPlanNode->projections().empty()) {
          LOG_FALLBACK(
              "FilterProject empty input type with filter or projections, PlanNode id: {}",
              planNode->id());
          return false;
        }
      }
    }

    // Check filter separately
    if (filterNode) {
      if (!canBeEvaluatedByCudf(
              {filterNode->filter()}, ctx->task->queryCtx().get())) {
        LOG_FALLBACK(
            "FilterProject filter cannot be evaluated by cuDF, PlanNode id: {}",
            planNode->id());
        return false;
      }
    }

    // Check projects separately
    if (projectPlanNode) {
      if (!canBeEvaluatedByCudf(
              projectPlanNode->projections(), ctx->task->queryCtx().get())) {
        LOG_FALLBACK(
            "FilterProject projections cannot be evaluated by cuDF, PlanNode id: {}",
            planNode->id());
        return false;
      }
    }
    return true;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto filterProjectOp = dynamic_cast<const exec::FilterProject*>(op);
    auto projectPlanNode =
        std::dynamic_pointer_cast<const core::ProjectNode>(planNode);
    auto filterPlanNode = filterProjectOp->filterNode();

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfFilterProject>(
            operatorId, ctx, filterPlanNode, projectPlanNode));
    return result;
  }
};

/// AggregationAdapter - Replaces with CudfHashAggregation
class AggregationAdapter : public OperatorAdapter {
 public:
  AggregationAdapter() : OperatorAdapter("Aggregation") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashAggregation*>(op) != nullptr ||
        dynamic_cast<const exec::StreamingAggregation*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      LOG_FALLBACK(
          "Aggregation op is not HashAggregation or StreamingAggregation, PlanNode id: {}",
          planNode->id());
      return false;
    }

    auto aggregationPlanNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(planNode);
    if (!aggregationPlanNode) {
      LOG_FALLBACK(
          "Aggregation planNode is not AggregationNode, PlanNode id: {}",
          planNode->id());
      return false;
    }

    bool canEvaluate =
        canBeEvaluatedByCudf(*aggregationPlanNode, ctx->task->queryCtx().get());
    if (!canEvaluate) {
      LOG_FALLBACK(
          "Aggregation aggregation cannot be evaluated by cuDF, PlanNode id: {}",
          planNode->id());
    }
    return canEvaluate;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto aggregationPlanNode =
        std::dynamic_pointer_cast<const core::AggregationNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    if (CudfConfig::getInstance().concatOptimizationEnabled) {
      result.push_back(
          std::make_unique<CudfBatchConcat>(
              operatorId, ctx, aggregationPlanNode));
    }
    result.push_back(
        std::make_unique<CudfHashAggregation>(
            operatorId, ctx, aggregationPlanNode));
    return result;
  }
};

class CudfHashJoinBaseAdapter : public OperatorAdapter {
 public:
  using OperatorAdapter::OperatorAdapter;

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx) const override {
    if (!canHandle(op)) {
      LOG_FALLBACK(
          "HashJoin operator is not HashBuild or HashProbe, PlanNode id: {}",
          planNode->id());
      return false;
    }

    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    if (!joinPlanNode) {
      LOG_FALLBACK(
          "HashJoin planNode is not HashJoinNode, PlanNode id: {}",
          planNode->id());
      return false;
    }

    if (!CudfHashJoinProbe::isSupportedJoinType(joinPlanNode->joinType())) {
      LOG_FALLBACK(
          "HashJoin unsupported join type, PlanNode id: {}", planNode->id());
      return false;
    }

    // Disabling null-aware anti join with filter until we implement it right
    if (joinPlanNode->joinType() == core::JoinType::kAnti &&
        joinPlanNode->isNullAware() && joinPlanNode->filter()) {
      LOG_FALLBACK(
          "HashJoin null-aware anti join with filter not implemented, PlanNode id: {}",
          planNode->id());
      return false;
    }

    // Null-aware LEFT SEMI PROJECT with filter requires tracking per-row
    // NULL vs no-match state during filter evaluation, which is not yet
    // implemented. The no-filter case is supported.
    if (joinPlanNode->joinType() == core::JoinType::kLeftSemiProject &&
        joinPlanNode->isNullAware() && joinPlanNode->filter()) {
      return false;
    }

    if (joinPlanNode->filter()) {
      if (!canBeEvaluatedByCudf(
              {joinPlanNode->filter()}, ctx->task->queryCtx().get())) {
        LOG_FALLBACK(
            "HashJoin join filter cannot be evaluated by cuDF, PlanNode id: {}",
            planNode->id());
        return false;
      }
    }
    return true;
  }
};

/// HashJoinBuildAdapter - Replaces with CudfHashJoinBuild
class HashJoinBuildAdapter : public CudfHashJoinBaseAdapter {
 public:
  HashJoinBuildAdapter() : CudfHashJoinBaseAdapter("HashJoinBuild") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashBuild*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfHashJoinBuild>(operatorId, ctx, joinPlanNode));
    return result;
  }
};

/// HashJoinProbeAdapter - Replaces with CudfHashJoinProbe
class HashJoinProbeAdapter : public CudfHashJoinBaseAdapter {
 public:
  HashJoinProbeAdapter() : CudfHashJoinBaseAdapter("HashJoinProbe") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashProbe*>(op) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    if (CudfConfig::getInstance().benchmarkConcatBeforeJoin) {
      // Accumulate probe batches to batchSizeMinThreshold rows before probing.
      // The concat's schema is the probe's *input* type -- sources()[0] is the
      // probe side of a HashJoinNode -- not the join's output type.
      result.push_back(
          std::make_unique<CudfBatchConcat>(
              operatorId,
              ctx,
              joinPlanNode,
              joinPlanNode->sources()[0]->outputType()));
    }
    result.push_back(
        std::make_unique<CudfHashJoinProbe>(operatorId, ctx, joinPlanNode));
    return result;
  }
};

/// RowHashJoinBuildAdapter - Replaces with RowHashJoinBuild (row-wise gather)
class RowHashJoinBuildAdapter : public OperatorAdapter {
 public:
  RowHashJoinBuildAdapter() : OperatorAdapter("RowHashJoinBuild") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashBuild*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    if (!canHandle(op)) return false;
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    if (!joinPlanNode) return false;
    // Row-wise only supports inner join, no filter
    return joinPlanNode->joinType() == core::JoinType::kInner &&
           !joinPlanNode->filter();
  }

  bool acceptsGpuInput() const override { return true; }
  bool producesGpuOutput() const override { return false; }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<RowHashJoinBuild>(operatorId, ctx, joinPlanNode));
    return result;
  }
};

/// RowHashJoinProbeAdapter - Replaces with RowHashJoinProbe (row-wise gather)
class RowHashJoinProbeAdapter : public OperatorAdapter {
 public:
  RowHashJoinProbeAdapter() : OperatorAdapter("RowHashJoinProbe") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::HashProbe*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    if (!canHandle(op)) return false;
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    if (!joinPlanNode) return false;
    return joinPlanNode->joinType() == core::JoinType::kInner &&
           !joinPlanNode->filter();
  }

  bool acceptsGpuInput() const override { return true; }
  bool producesGpuOutput() const override { return true; }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto joinPlanNode =
        std::dynamic_pointer_cast<const core::HashJoinNode>(planNode);
    std::vector<std::unique_ptr<exec::Operator>> result;

    // GPU re-batching is ORTHOGONAL to the row/column layout choice -- it is a
    // knob, not a rival design (exactly like concatOptimizationEnabled before
    // the aggregation). Applying it to the row path too lets us ask the fair
    // question: with BOTH given large probe batches, does the layout still win?
    //
    // HEAD OF THE CHAIN ONLY. CudfBatchConcat requires CudfVector input
    // (VELOX_CHECK_NOT_NULL in doAddInput). Consecutive RowHashJoinProbes pass
    // RowStoreVector to each other -- that format stickiness is the whole point
    // of the row path -- so a concat in front of a downstream row probe would
    // crash. One concat at the head is also sufficient: the batch size it
    // establishes propagates down the chain.
    const bool isChainHead =
        std::dynamic_pointer_cast<const core::HashJoinNode>(
            joinPlanNode->sources()[0]) == nullptr;
    if (CudfConfig::getInstance().benchmarkConcatBeforeJoin && isChainHead) {
      result.push_back(
          std::make_unique<CudfBatchConcat>(
              operatorId,
              ctx,
              joinPlanNode,
              joinPlanNode->sources()[0]->outputType()));
    }

    result.push_back(
        std::make_unique<RowHashJoinProbe>(operatorId, ctx, joinPlanNode));
    return result;
  }
};

/// OrderByAdapter - Replaces with CudfOrderBy
class OrderByAdapter : public OperatorAdapter {
 public:
  OrderByAdapter() : OperatorAdapter("OrderBy") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::OrderBy*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::OrderByNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto orderByPlanNode =
        std::dynamic_pointer_cast<const core::OrderByNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfOrderBy>(operatorId, ctx, orderByPlanNode));
    return result;
  }
};

/// TopNAdapter - Replaces with CudfTopN
class TopNAdapter : public OperatorAdapter {
 public:
  TopNAdapter() : OperatorAdapter("TopN") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::TopN*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::TopNNode>(planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto topNPlanNode =
        std::dynamic_pointer_cast<const core::TopNNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(std::make_unique<CudfTopN>(operatorId, ctx, topNPlanNode));
    return result;
  }
};

/// LimitAdapter - Replaces with CudfLimit
class LimitAdapter : public OperatorAdapter {
 public:
  LimitAdapter() : OperatorAdapter("Limit") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Limit*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::LimitNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto limitPlanNode =
        std::dynamic_pointer_cast<const core::LimitNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfLimit>(operatorId, ctx, limitPlanNode));
    return result;
  }
};

/// LocalPartitionAdapter - Conditionally replaces with CudfLocalPartition
class LocalPartitionAdapter : public OperatorAdapter {
 public:
  LocalPartitionAdapter() : OperatorAdapter("LocalPartition") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::LocalPartition*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* op,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    auto localPartitionPlanNode =
        std::dynamic_pointer_cast<const core::LocalPartitionNode>(planNode);
    bool canRun = canHandle(op) && localPartitionPlanNode &&
        CudfLocalPartition::shouldReplace(localPartitionPlanNode);
    if (!canRun) {
      LOG_FALLBACK(
          "LocalPartitionAdapter {}, PlanNode id: {}",
          !canHandle(op) ? "operator is not LocalPartition"
              : !localPartitionPlanNode
              ? "planNode is not LocalPartitionNode"
              : "CudfLocalPartition::shouldReplace returned false",
          planNode->id());
    }
    return canRun;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto localPartitionPlanNode =
        std::dynamic_pointer_cast<const core::LocalPartitionNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfLocalPartition>(
            operatorId, ctx, localPartitionPlanNode));
    return result;
  }

  bool keepOperator() const override {
    return false;
  }
};

/// LocalExchangeAdapter - Keeps original operator
class LocalExchangeAdapter : public OperatorAdapter {
 public:
  LocalExchangeAdapter() : OperatorAdapter("LocalExchange") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::LocalExchange*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/) const override {
    return true;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// AssignUniqueIdAdapter - Replaces with CudfAssignUniqueId
class AssignUniqueIdAdapter : public OperatorAdapter {
 public:
  AssignUniqueIdAdapter() : OperatorAdapter("AssignUniqueId") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::AssignUniqueId*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(
               planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto assignUniqueIdPlanNode =
        std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfAssignUniqueId>(
            operatorId,
            ctx,
            assignUniqueIdPlanNode,
            assignUniqueIdPlanNode->taskUniqueId(),
            assignUniqueIdPlanNode->uniqueIdCounter()));
    return result;
  }
};

/// ValuesAdapter - Keeps the original operator, but recognises GPU-RESIDENT input.
///
/// A ValuesNode holds RowVectorPtrs -- and CudfVector IS a RowVector. So a plan
/// can be given vectors whose data already lives in GPU memory, and exec::Values
/// will hand them straight to the next operator. That is how you benchmark the
/// "data hot in GPU" regime that GPU-database papers actually measure: no I/O,
/// no parquet decode, no H2D. The query is nothing but the query.
///
/// Without the check below this does not work. properties() AND-gates
/// producesGpuOutput with canRunOnGPU, so returning canRunOnGPU=false (as this
/// adapter used to, unconditionally) forces producesGpuOutput=false. ToCudf then
/// inserts a CudfFromVelox after Values -- which would try to Arrow-convert data
/// that is ALREADY on the device, i.e. a pointless D2H followed by an H2D.
///
/// We still keep the original exec::Values operator (keepOperator + empty
/// createReplacements); all that changes is that we tell ToCudf its output is
/// already on the GPU, so no conversion boundary is inserted.
class ValuesAdapter : public OperatorAdapter {
 public:
  ValuesAdapter() : OperatorAdapter("Values") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::Values*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    // GPU-resident iff every value the node holds is a CudfVector.
    auto valuesNode =
        std::dynamic_pointer_cast<const core::ValuesNode>(planNode);
    if (valuesNode != nullptr && !valuesNode->values().empty()) {
      bool allGpu = true;
      for (const auto& v : valuesNode->values()) {
        if (std::dynamic_pointer_cast<CudfVector>(v) == nullptr) {
          allGpu = false;
          break;
        }
      }
      if (allGpu) {
        return true;
      }
    }
    LOG_FALLBACK(
        "Values operator not supported on cuDF (values are not CudfVectors), "
        "PlanNode id: {}",
        planNode->id());
    return false;
  }

  bool acceptsGpuInput() const override {
    return false; // it is a source; it has no input
  }

  bool producesGpuOutput() const override {
    // Only consulted when canRunOnGPU is true, i.e. the values ARE CudfVectors.
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// MarkDistinctAdapter - Replaces with CudfMarkDistinct
class MarkDistinctAdapter : public OperatorAdapter {
 public:
  MarkDistinctAdapter() : OperatorAdapter("MarkDistinct") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::MarkDistinct*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::MarkDistinctNode>(planNode) !=
        nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto markDistinctPlanNode =
        std::dynamic_pointer_cast<const core::MarkDistinctNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfMarkDistinct>(
            operatorId, ctx, markDistinctPlanNode));
    return result;
  }
};

/// EnforceSingleRowAdapter - Replaces with CudfEnforceSingleRow
class EnforceSingleRowAdapter : public OperatorAdapter {
 public:
  EnforceSingleRowAdapter() : OperatorAdapter("EnforceSingleRow") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::EnforceSingleRow*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    return std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(
               planNode) != nullptr;
  }

  bool acceptsGpuInput() const override {
    return true;
  }

  bool producesGpuOutput() const override {
    return true;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* ctx,
      int32_t operatorId) const override {
    auto enforceSingleRowPlanNode =
        std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(planNode);

    std::vector<std::unique_ptr<exec::Operator>> result;
    result.push_back(
        std::make_unique<CudfEnforceSingleRow>(
            operatorId, ctx, enforceSingleRowPlanNode));
    return result;
  }
};

/// CallbackSinkAdapter - Keeps original operator
class CallbackSinkAdapter : public OperatorAdapter {
 public:
  CallbackSinkAdapter() : OperatorAdapter("CallbackSink") {}

  bool canHandle(const exec::Operator* op) const override {
    return dynamic_cast<const exec::CallbackSink*>(op) != nullptr;
  }

  bool canRunOnGPU(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& planNode,
      exec::DriverCtx* /*ctx*/) const override {
    LOG_FALLBACK(
        "CallbackSink operator not supported on cuDF, PlanNode id: {}",
        planNode->id());
    return false;
  }

  bool acceptsGpuInput() const override {
    return false;
  }

  bool producesGpuOutput() const override {
    return false;
  }

  std::vector<std::unique_ptr<exec::Operator>> createReplacements(
      const exec::Operator* /*op*/,
      const core::PlanNodePtr& /*planNode*/,
      exec::DriverCtx* /*ctx*/,
      int32_t /*operatorId*/) const override {
    return {}; // Keep original operator
  }

  bool keepOperator() const override {
    return true;
  }
};

/// Registration Function
void registerAllOperatorAdapters() {
  auto& registry = OperatorAdapterRegistry::getInstance();

  // Clear any existing adapters
  registry.clear();

  // Register all adapters
  registry.registerAdapter(std::make_unique<TableScanAdapter>());
  registry.registerAdapter(std::make_unique<FilterProjectAdapter>());
  registry.registerAdapter(std::make_unique<AggregationAdapter>());
  if (CudfConfig::getInstance().benchmarkRowWiseGather) {
    registry.registerAdapter(std::make_unique<RowHashJoinBuildAdapter>());
    registry.registerAdapter(std::make_unique<RowHashJoinProbeAdapter>());
  } else {
    registry.registerAdapter(std::make_unique<HashJoinBuildAdapter>());
    registry.registerAdapter(std::make_unique<HashJoinProbeAdapter>());
  }
  registry.registerAdapter(std::make_unique<OrderByAdapter>());
  registry.registerAdapter(std::make_unique<TopNAdapter>());
  registry.registerAdapter(std::make_unique<LimitAdapter>());
  registry.registerAdapter(std::make_unique<LocalPartitionAdapter>());
  registry.registerAdapter(std::make_unique<LocalExchangeAdapter>());
  registry.registerAdapter(std::make_unique<AssignUniqueIdAdapter>());
  registry.registerAdapter(std::make_unique<MarkDistinctAdapter>());
  registry.registerAdapter(std::make_unique<EnforceSingleRowAdapter>());
  registry.registerAdapter(std::make_unique<ValuesAdapter>());
  registry.registerAdapter(std::make_unique<CallbackSinkAdapter>());
}

} // namespace facebook::velox::cudf_velox
