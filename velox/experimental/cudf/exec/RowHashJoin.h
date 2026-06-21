/*
 * RowHashJoin.h
 *
 * Row-wise hash join operators for Velox cudf integration.
 *
 * These operators implement the same join semantics as CudfHashJoin but
 * use a fixed-stride row store for materialization (gather) instead of
 * columnar cudf::gather(). This enables comparing row-wise vs column-wise
 * gather performance within the Velox execution framework.
 *
 * Pipeline:
 *   CPU TableScan → Filter → Project → [CudfFromVelox] →
 *     RowHashJoinBuild (GPU: col→row transpose, extract keys, build HT)
 *     RowHashJoinProbe (GPU: col→row transpose, extract keys, probe, row gather)
 *
 * Restrictions (benchmark-only):
 *   - Inner join only
 *   - No filter expressions
 *   - Fixed-width types only (no VARCHAR)
 *   - Single join key (first column = join key)
 */

#pragma once

#include "velox/experimental/cudf/exec/CudfHashJoin.h" // CudfHashJoinBridge
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/JoinBridge.h"
#include "velox/exec/Operator.h"

#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

// ============================================================================
// Bridge: extends CudfHashJoinBridge to also carry row-store data
// ============================================================================

class RowHashJoinBridge : public exec::JoinBridge {
 public:
  struct BuildData {
    // Row store on GPU (entire build table in row layout)
    GpuFixedRowStore gpuRowStore;
    // Ownership of device memory
    rmm::device_buffer rowBuffer;
    rmm::device_buffer fieldsBuffer;
    // Hash join object built from key column
    std::shared_ptr<cudf::hash_join> hashJoin;
    // Key column on GPU (extracted from row store)
    rmm::device_buffer keyBuffer;
    int64_t numRows;
    int32_t rowWidth;
  };

  void setBuildData(std::shared_ptr<BuildData> data);
  std::shared_ptr<BuildData> dataOrFuture(ContinueFuture* future);

  void setBuildStream(rmm::cuda_stream_view stream);
  std::optional<rmm::cuda_stream_view> getBuildStream();

 private:
  std::shared_ptr<BuildData> buildData_;
  std::optional<rmm::cuda_stream_view> buildStream_;
};

// ============================================================================
// Build operator: col→row transpose + extract keys + build hash table
// ============================================================================

class RowHashJoinBuild : public CudfOperatorBase {
 public:
  RowHashJoinBuild(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  std::vector<RowVectorPtr> inputs_;  // RowStoreVector or CudfVector
  ContinueFuture future_{ContinueFuture::makeEmpty()};
};

// ============================================================================
// Probe operator: col→row transpose + extract keys + probe + row gather
// ============================================================================

class RowHashJoinProbe : public CudfOperatorBase {
 public:
  RowHashJoinProbe(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::HashJoinNode> joinNode);

  bool needsInput() const override;
  exec::BlockingReason isBlocked(ContinueFuture* future) override;
  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;

 private:
  std::shared_ptr<const core::HashJoinNode> joinNode_;
  ContinueFuture future_{ContinueFuture::makeEmpty()};

  // Build-side data from bridge
  std::shared_ptr<RowHashJoinBridge::BuildData> buildData_;
  std::optional<rmm::cuda_stream_view> buildStream_;

  // Per-probe-batch state
  RowVectorPtr input_;

  // Row layout metadata (computed once from output type)
  std::vector<FieldDesc> probeFields_;
  int32_t probeRowWidth_ = 0;
  std::vector<FieldDesc> outputFields_;
  int32_t outputRowWidth_ = 0;

  // Column index mappings
  std::vector<cudf::size_type> leftKeyIndices_;
  std::vector<cudf::size_type> leftColumnIndicesToGather_;
  std::vector<int> leftColumnOutputIndices_;
  std::vector<cudf::size_type> rightColumnIndicesToGather_;
  std::vector<int> rightColumnOutputIndices_;

  // Pre-allocated device buffers (reused per batch)
  rmm::device_buffer probeRowBuffer_;      // probe rows on GPU
  rmm::device_buffer probeFieldsBuffer_;   // FieldDesc on GPU
  rmm::device_buffer probeKeyBuffer_;      // extracted probe keys (RowStoreVector path)
  rmm::device_buffer probeGatherBuffer_;   // gathered probe rows
  rmm::device_buffer buildGatherBuffer_;   // gathered build rows

  // Probe key column pointer (CudfVector path: points into input CudfVector)
  const void* probeKeyData_ = nullptr;
  int32_t probeKeyWidth_ = 0;
  int64_t probeRowCapacity_ = 0;
  int64_t gatherCapacity_ = 0;
  bool fieldsUploaded_ = false;

  bool initialized_ = false;
  bool finished_ = false;
};

// ============================================================================
// Bridge translator: connects plan node to row-wise operators
// ============================================================================

class RowHashJoinBridgeTranslator : public exec::Operator::PlanNodeTranslator {
 public:
  std::unique_ptr<exec::Operator> toOperator(
      exec::DriverCtx* ctx,
      int32_t id,
      const core::PlanNodePtr& node) override;

  std::unique_ptr<exec::JoinBridge> toJoinBridge(
      const core::PlanNodePtr& node) override;

  exec::OperatorSupplier toOperatorSupplier(
      const core::PlanNodePtr& node) override;
};

} // namespace facebook::velox::cudf_velox
