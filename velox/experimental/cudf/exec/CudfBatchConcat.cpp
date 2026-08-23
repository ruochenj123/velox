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
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfBatchConcat.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/GpuRowOps.cuh"
#include "velox/experimental/cudf/exec/RowStoreVector.h"

namespace facebook::velox::cudf_velox {

CudfBatchConcat::CudfBatchConcat(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::PlanNode> planNode,
    RowTypePtr outputType)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType ? outputType : planNode->outputType(),
          planNode->id(),
          "CudfBatchConcat",
          nvtx3::rgb{211, 211, 211}, /* LightGrey */
          NvtxMethodFlag::kAll,
          std::nullopt,
          planNode),
      driverCtx_(driverCtx),
      targetRows_(CudfConfig::getInstance().batchSizeMinThreshold) {}

void CudfBatchConcat::doAddInput(RowVectorPtr input) {
  // Row-wise input (the row-ingest path produces RowStoreVector). Concatenating
  // these is trivially cheap -- see concatRowStore().
  if (auto rowVec = std::dynamic_pointer_cast<RowStoreVector>(input)) {
    currentNumRows_ += rowVec->size();
    rowBuffer_.push_back(std::move(rowVec));
    return;
  }

  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(
      cudfVector,
      "CudfBatchConcat expects CudfVector or RowStoreVector input");

  // Push input cudf table to buffer
  currentNumRows_ += cudfVector->getTableView().num_rows();
  buffer_.push_back(std::move(cudfVector));
}

// ============================================================================
// Row-wise concatenation.
//
// A RowStoreVector is ONE contiguous, fixed-stride row buffer. So merging N of
// them is N device-to-device memcpys, end to end, into a single allocation:
// no per-column tables, no null masks, no cudf::concatenate. The columnar path
// has to allocate and zero a null mask PER COLUMN per batch, and in every nsys
// profile of this system cudaMemsetAsync (which is exactly those null masks) is
// 54-79% of all CPU-side CUDA time. Here it is zero.
// ============================================================================
RowVectorPtr CudfBatchConcat::concatRowStore() {
  VELOX_CHECK(!rowBuffer_.empty());

  const auto& first = rowBuffer_.front();
  const int32_t rowWidth = first->rowWidth();
  auto stream = first->stream();
  const auto& fields = first->hostFields();

  int64_t totalRows = 0;
  for (const auto& v : rowBuffer_) {
    VELOX_CHECK_EQ(
        v->rowWidth(), rowWidth, "CudfBatchConcat: row width mismatch");
    // The device-to-device copies below are issued on ONE stream and the sources
    // are released immediately afterwards. That is only safe if every source was
    // produced on that same stream (CudfFromVelox pins its row path to a single
    // per-operator stream for exactly this reason). If it ever is not, the copies
    // would read unordered buffers and the sources would be freed underneath
    // them -- a race that surfaces as nondeterministically wrong results, so fail
    // loudly instead.
    VELOX_CHECK(
        v->stream().value() == stream.value(),
        "CudfBatchConcat: RowStoreVector inputs must share one stream");
    totalRows += v->size();
  }

  // Null sidecar: if ANY input carries one, the merged store carries one
  // (inputs without contribute zeroed = all-valid bytes). Strings (pointer
  // slots, 2026-08-23): slots stay valid across the copy; the merged store
  // just retains the inputs whose buffers they reference.
  int32_t nullStride = 0;
  bool anyStringRefs = false;
  for (const auto& v : rowBuffer_) {
    if (v->nullStride() > 0) {
      VELOX_CHECK(
          nullStride == 0 || nullStride == v->nullStride(),
          "CudfBatchConcat: row store null strides disagree");
      nullStride = v->nullStride();
    }
    anyStringRefs = anyStringRefs || v->hasStringRefs();
  }
  const int64_t rowBytes = static_cast<int64_t>(totalRows) * rowWidth;
  const int64_t nullBytes = static_cast<int64_t>(totalRows) * nullStride;
  rmm::device_buffer merged(rowBytes + nullBytes, stream);
  auto* dst = static_cast<uint8_t*>(merged.data());
  auto* nullDst = dst + rowBytes;
  for (const auto& v : rowBuffer_) {
    const int64_t bytes = v->gpuRowBytes();
    if (bytes > 0) {
      cudaMemcpyAsync(
          dst, v->gpuRowData(), bytes, cudaMemcpyDeviceToDevice, stream.value());
      dst += bytes;
    }
    if (nullStride > 0 && v->size() > 0) {
      const int64_t nb = static_cast<int64_t>(v->size()) * nullStride;
      if (v->nullStride() > 0) {
        cudaMemcpyAsync(
            nullDst, v->gpuRowData() + v->gpuRowBytes(), nb,
            cudaMemcpyDeviceToDevice, stream.value());
      } else {
        cudaMemsetAsync(nullDst, 0, nb, stream.value());
      }
      nullDst += nb;
    }
  }

  rmm::device_buffer fieldsBuf(
      fields.data(), fields.size() * sizeof(FieldDesc), stream);
  auto fieldsHost = fields;

  // Collect keep-alives BEFORE releasing the inputs.
  std::vector<std::shared_ptr<void>> keepAlive;
  if (anyStringRefs) {
    for (const auto& v : rowBuffer_) {
      if (v->hasStringRefs()) {
        for (const auto& o : v->stringKeepAlive()) {
          keepAlive.push_back(o);
        }
      }
    }
  }
  rowBuffer_.clear();
  currentNumRows_ = 0;

  // Stream-ordered: the sources stay alive until their copies complete because
  // rmm frees on the same stream, so no synchronize is needed here.
  auto out = std::make_shared<RowStoreVector>(
      pool(),
      outputType_,
      totalRows,
      std::move(merged),
      std::move(fieldsBuf),
      std::move(fieldsHost),
      rowWidth,
      stream);
  if (nullStride > 0) {
    out->setNullSidecar(nullStride);
  }
  if (anyStringRefs) {
    out->addStringKeepAlives(keepAlive);
  }
  return out;
}

RowVectorPtr CudfBatchConcat::doGetOutput() {
  // ---- Row-wise path ----
  if (!rowBuffer_.empty() &&
      (currentNumRows_ >= targetRows_ || noMoreInput_)) {
    return concatRowStore();
  }

  // Drain the queue if there is any output to be flushed
  if (!outputQueue_.empty()) {
    auto table = std::move(outputQueue_.front());
    auto rowCount = table->num_rows();
    outputQueue_.pop();
    return std::make_shared<CudfVector>(
        pool(), outputType_, rowCount, std::move(table), outputQueueStream_);
  }

  // Merge tables if there are enough rows
  if (!buffer_.empty() && (currentNumRows_ >= targetRows_ || noMoreInput_)) {
    // Use stream from existing buffer vectors
    outputQueueStream_ = buffer_[0]->stream();
    auto tables = getConcatenatedTableBatched(
        std::exchange(buffer_, {}),
        outputType_,
        outputQueueStream_,
        get_output_mr());

    currentNumRows_ = 0;

    for (auto it = tables.begin(); it + 1 != tables.end(); ++it) {
      outputQueue_.push(std::move(*it));
    }

    // If last table is a smaller batch and we still expect more input and keep
    // it in buffer.
    auto& last = tables.back();
    auto rowCount = last->num_rows();

    if (!noMoreInput_ && rowCount < targetRows_) {
      currentNumRows_ = rowCount;
      buffer_.push_back(
          std::make_shared<CudfVector>(
              pool(),
              outputType_,
              rowCount,
              std::move(last),
              outputQueueStream_));
    } else {
      outputQueue_.push(std::move(last));
    }

    // Return the first batch from the new queue
    if (!outputQueue_.empty()) {
      auto table = std::move(outputQueue_.front());
      auto rowCount = table->num_rows();
      outputQueue_.pop();
      return std::make_shared<CudfVector>(
          pool(), outputType_, rowCount, std::move(table), outputQueueStream_);
    }
  }

  return nullptr;
}

bool CudfBatchConcat::isFinished() {
  return noMoreInput_ && buffer_.empty() && rowBuffer_.empty() &&
      outputQueue_.empty();
}

} // namespace facebook::velox::cudf_velox
