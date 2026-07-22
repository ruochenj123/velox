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

#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/exec/RowStoreVector.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Operator.h"

#include <queue>

namespace facebook::velox::cudf_velox {

class CudfBatchConcat : public CudfOperatorBase {
 public:
  /// `outputType` is the schema of the batches flowing through this operator.
  /// When null it defaults to planNode->outputType(), which is correct only
  /// when the concat sits directly in front of an operator whose output type
  /// equals its input type. Before a hash-join probe it does not, so the caller
  /// must pass the probe's input type (joinNode->sources()[0]->outputType()).
  CudfBatchConcat(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::PlanNode> planNode,
      RowTypePtr outputType = nullptr);

  bool needsInput() const override {
    return !noMoreInput_ && outputQueue_.empty() &&
        currentNumRows_ < targetRows_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;

 private:
  // Row-wise concat, used when the input is a RowStoreVector rather than a
  // CudfVector (i.e. the row-ingest path in CudfFromVelox is active).
  //
  // This is MUCH simpler than the columnar case, and that simplicity is itself
  // the point of the row layout: a RowStoreVector is one contiguous, fixed-
  // stride buffer, so concatenating N of them is N plain device-to-device
  // memcpys end-to-end. No per-column tables, no null masks, no cudf::concatenate
  // -- the columnar path's cost is dominated by exactly those per-column
  // allocations and null-mask memsets.
  RowVectorPtr concatRowStore();

  exec::DriverCtx* const driverCtx_;
  std::vector<CudfVectorPtr> buffer_;
  std::vector<std::shared_ptr<RowStoreVector>> rowBuffer_;
  std::queue<std::unique_ptr<cudf::table>> outputQueue_;
  rmm::cuda_stream_view outputQueueStream_{rmm::cuda_stream_default};
  size_t currentNumRows_{0};
  const size_t targetRows_{0};
};

} // namespace facebook::velox::cudf_velox
