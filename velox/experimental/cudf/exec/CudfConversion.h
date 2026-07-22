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
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h" // FieldDesc
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <deque>
#include <memory>
#include <vector>

namespace facebook::velox::cudf_velox {

class CudfFromVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kGpuBatchSizeRows =
      "velox.cudf.gpu_batch_size_rows";

  CudfFromVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  bool needsInput() const override {
    return !finished_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  const std::optional<std::string> timestampTimeZone_;
  std::vector<RowVectorPtr> inputs_;
  std::size_t currentOutputSize_ = 0;
  bool finished_ = false;

  // ---- Row-wise ingest (benchmarkCpuColToRow) ----
  //
  // Packs Velox's columnar batch straight into a fixed-stride ROW buffer on the
  // host and sends it as ONE contiguous H2D copy, producing a RowStoreVector.
  //
  // WHY THIS CAN BEAT THE COLUMNAR PATH ON THE CRITICAL PATH: E2E here is pinned
  // to ingest cost, not GPU cost. Columnar ingest produces N separate device
  // column buffers -- N allocations, N H2D copies, and N NULL-MASK MEMSETS. In
  // any nsys profile of this system cudaMemsetAsync is 54-79% of ALL CPU-side
  // CUDA time, and it is exactly those per-column null masks. A row buffer has
  // no null masks and no per-column anything: one allocation, one copy. It also
  // makes the GPU-side transposeToRows unnecessary, since the data arrives in
  // row layout already.
  //
  // The original implementation of this path was, in its own words, "deliberately
  // naive": it malloc'd AND ZEROED a fresh std::vector every batch, copied field
  // by field with scalar memcpy, transferred from PAGEABLE memory (which makes
  // cudaMemcpyAsync synchronous and ~2-3x slower), stream.synchronize()'d on
  // every batch, and re-uploaded the constant field descriptors each time. All of
  // that is fixed below; the state cached here is what makes it possible.
  struct PinnedSlot {
    uint8_t* host = nullptr; // cudaHostAlloc'd
    int64_t capacity = 0;
    cudaEvent_t done = nullptr; // recorded after this slot's H2D
    bool inUse = false;
  };
  // Two slots, ping-ponged: while slot A's copy is in flight we pack into slot
  // B. We wait on a slot's event before reusing it -- by then, one full batch
  // later, it has long since completed, so the wait costs nothing. Without this
  // the operator must synchronize before reusing the host buffer, which parks
  // the driver thread on every batch (the same bug the fused probe had).
  PinnedSlot pinned_[2];
  int pinnedSlot_ = 0;

  // Whether this operator's DOWNSTREAM consumer can actually accept a
  // RowStoreVector. Row ingest is only legal when it can: a CudfFilterProject
  // (e.g. TPC-H Q9/Q8's year(o_orderdate) projection, which correctly runs on
  // the GPU because it WIDENS the row) accepts only CudfVector and would
  // otherwise die on "cudfInput != nullptr". Determined once, on the first
  // getOutput, by inspecting the next operator in the driver -- the same trick
  // RowHashJoinProbe uses for emitColumnar_.
  //   -1 = not yet determined, 0 = emit CudfVector, 1 = emit RowStoreVector
  int emitRowStore_ = -1;

  // ONE stream for every RowStoreVector this operator produces.
  //
  // The columnar path takes a fresh stream from the pool on each doGetOutput,
  // which is fine there because a CudfVector carries its own stream and cuDF
  // handles the ordering. It is NOT fine for rows: CudfBatchConcat merges N
  // RowStoreVectors with device-to-device copies issued on ONE stream, and then
  // releases the sources. If the sources were produced on DIFFERENT streams,
  // those copies read buffers with no cross-stream ordering, and the sources are
  // freed on their own streams while the copies may still be in flight -- a race
  // that shows up as nondeterministically wrong query results.
  //
  // Pinning the row path to a single per-operator stream makes every
  // RowStoreVector it emits stream-ordered against every other, which is exactly
  // the invariant concatRowStore() relies on.
  std::optional<rmm::cuda_stream_view> rowStream_;

  // Row layout is a property of the input schema, so compute it once.
  std::vector<FieldDesc> rowFields_;
  int32_t rowWidth_ = 0;
  bool rowLayoutReady_ = false;
  rmm::device_buffer rowFieldsDevice_; // uploaded once, not per batch
};

class CudfToVelox : public CudfOperatorBase {
 public:
  static constexpr const char* kPassthroughMode =
      "velox.cudf.to_velox.passthrough_mode";

  CudfToVelox(
      int32_t operatorId,
      RowTypePtr outputType,
      exec::DriverCtx* driverCtx,
      std::string planNodeId);

  bool needsInput() const override {
    return !finished_;
  }

  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }

  bool isFinished() override {
    return finished_;
  }

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doClose() override;

 private:
  bool isPassthroughMode() const;
  std::optional<uint64_t> averageRowSize();
  // Convert inputs_.front() to Velox once; slice it CPU-side per batch.
  RowVectorPtr convertFrontToVelox();
  std::optional<uint64_t> averageRowSize_;
  std::deque<CudfVectorPtr> inputs_;
  // Non-CudfVector inputs that pass through without D2H conversion
  std::deque<RowVectorPtr> passthroughInputs_;
  // Converted CPU-side buffer being drained by successive doGetOutput() calls.
  RowVectorPtr veloxBuffer_;
  // Current offset into veloxBuffer_ for the next slice.
  vector_size_t veloxOffset_{0};
  bool finished_ = false;
};

} // namespace facebook::velox::cudf_velox
