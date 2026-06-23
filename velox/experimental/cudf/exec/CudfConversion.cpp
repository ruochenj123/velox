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
#include "velox/experimental/cudf/BenchmarkTimelineFlag.h"
#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/exec/RowStoreVector.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda_runtime.h>

#include <chrono>

namespace facebook::velox::cudf_velox {

namespace {
// Concatenate multiple RowVectors into a single RowVector.
// Copied from AggregationFuzzer.cpp.
RowVectorPtr mergeRowVectors(
    const std::vector<RowVectorPtr>& results,
    velox::memory::MemoryPool* pool) {
  VELOX_NVTX_FUNC_RANGE();
  if (results.size() == 1) {
    return results[0];
  }
  vector_size_t totalCount = 0;
  for (const auto& result : results) {
    totalCount += result->size();
  }
  auto copy =
      BaseVector::create<RowVector>(results[0]->type(), totalCount, pool);
  auto copyCount = 0;
  for (const auto& result : results) {
    copy->copy(result.get(), copyCount, 0, result->size());
    copyCount += result->size();
  }
  return copy;
}

cudf::size_type preferredGpuBatchSizeRows(
    const facebook::velox::core::QueryConfig& queryConfig) {
  constexpr cudf::size_type kDefaultGpuBatchSizeRows = 100000;
  const auto batchSize = queryConfig.get<int32_t>(
      CudfFromVelox::kGpuBatchSizeRows, kDefaultGpuBatchSizeRows);
  VELOX_CHECK_GT(batchSize, 0, "velox.cudf.gpu_batch_size_rows must be > 0");
  VELOX_CHECK_LE(
      batchSize,
      std::numeric_limits<vector_size_t>::max(),
      "velox.cudf.gpu_batch_size_rows must be <= max(vector_size_t)");
  return batchSize;
}
} // namespace

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfFromVelox",
          nvtx3::rgb{255, 140, 0}, // Orange
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt),
      timestampTimeZone_(driverCtx->queryConfig().get<std::string>(
          facebook::velox::core::QueryConfig::kSessionTimezone)) {}

void CudfFromVelox::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    // Materialize lazy vectors
    for (auto& child : input->children()) {
      child->loadedVector();
    }
    input->loadedVector();

    // Timeline: initialize thread_local scan state on first addInput call
    // (this runs on the driver's actual thread, unlike the constructor).
    if (inputs_.empty() && CudfConfig::getInstance().benchmarkLogTimeline &&
        operatorCtx_->driverCtx()->driverId == 0) {
      auto& scanState = threadScanTraceState();
      if (scanState.driverId == -1) {
        scanState.driverId = operatorCtx_->driverCtx()->driverId;
        scanState.pipelineId = operatorCtx_->driverCtx()->pipelineId;
        benchmarkTimelineEnabled().store(true, std::memory_order_relaxed);
      }
    }

    // Accumulate inputs
    inputs_.push_back(input);
    currentOutputSize_ += input->size();
  }
}

RowVectorPtr CudfFromVelox::doGetOutput() {
  const auto targetOutputSize =
      preferredGpuBatchSizeRows(operatorCtx_->driverCtx()->queryConfig());

  finished_ = noMoreInput_ && inputs_.empty();

  if (finished_ or
      (currentOutputSize_ < targetOutputSize and not noMoreInput_) or
      inputs_.empty()) {
    return nullptr;
  }

  // Timeline: initialize per-thread scan state and signal scan phase complete
  const bool logTimeline = CudfConfig::getInstance().benchmarkLogTimeline;
  const auto driverId = operatorCtx_->driverCtx()->driverId;
  const auto pipelineId = operatorCtx_->driverCtx()->pipelineId;
  auto timeNow = []() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d FromVelox start %lld\n",
           (long long)timeNow(), pipelineId, (long long)inputs_[0]->size());
  }

  auto tpStart = std::chrono::steady_clock::now();

  // Select inputs that don't exceed the max vector size limit
  std::vector<RowVectorPtr> selectedInputs;
  vector_size_t totalSize = 0;
  auto const maxVectorSize = std::numeric_limits<vector_size_t>::max();

  for (const auto& input : inputs_) {
    if (totalSize + input->size() <= maxVectorSize) {
      selectedInputs.push_back(input);
      totalSize += input->size();
    } else {
      break;
    }
  }

  auto tpBeforeMerge = std::chrono::steady_clock::now();

  // Combine selected RowVectors into a single RowVector
  auto input = mergeRowVectors(selectedInputs, inputs_[0]->pool());

  auto tpAfterMerge = std::chrono::steady_clock::now();

  // Remove processed inputs
  inputs_.erase(inputs_.begin(), inputs_.begin() + selectedInputs.size());
  currentOutputSize_ -= totalSize;

  // Early return if no input
  if (input->size() == 0) {
    return nullptr;
  }

  // Get a stream from the global stream pool
  auto stream = cudfGlobalStreamPool().get_stream();
  const bool logH2D = CudfConfig::getInstance().benchmarkLogGatherTime;
  const bool rowWiseMode = CudfConfig::getInstance().benchmarkRowWiseGather;



  // ===== Row-wise path: CPU col→row + cudaMemcpy =====
  // CPU does col→row transpose, then cudaMemcpyAsync sends row buffer to GPU.
  // Outputs RowStoreVector (flows directly to RowHashJoinProbe/Build).
  // Only enabled when both rowWiseMode AND cpuColToRow are set.
  const bool cpuColToRow = CudfConfig::getInstance().benchmarkCpuColToRow;
  if (rowWiseMode && cpuColToRow) {
    auto numRows = static_cast<int64_t>(input->size());
    auto rowType = std::dynamic_pointer_cast<const RowType>(input->type());
    VELOX_CHECK_NOT_NULL(rowType);

    // Compute field layout from the RowType
    std::vector<FieldDesc> fields;
    int32_t offset = 0;
    for (int i = 0; i < rowType->size(); i++) {
      FieldDesc fd;
      fd.offset = offset;
      auto kind = rowType->childAt(i)->kind();
      switch (kind) {
        case TypeKind::BOOLEAN:
        case TypeKind::TINYINT:
          fd.byte_width = 1; break;
        case TypeKind::SMALLINT:
          fd.byte_width = 2; break;
        case TypeKind::INTEGER:
          fd.byte_width = 4; break;
        case TypeKind::BIGINT:
        case TypeKind::DOUBLE:
          fd.byte_width = 8; break;
        case TypeKind::REAL:
          fd.byte_width = 4; break;
        default:
          VELOX_FAIL("RowCudfFromVelox: unsupported TypeKind {}", (int)kind);
      }
      fields.push_back(fd);
      offset += fd.byte_width;
    }
    int32_t rowWidth = (offset + 7) & ~7; // 8-byte aligned

    // CPU col→row conversion: pack columns into a contiguous row buffer
    std::chrono::steady_clock::time_point cpuConvStart;
    if (logH2D) {
      cpuConvStart = std::chrono::steady_clock::now();
    }

    int64_t totalBytes = numRows * rowWidth;
    // Use a host buffer (pinned or regular) for the row data
    std::vector<uint8_t> hostRowBuf(totalBytes, 0);

    // For each column, copy data row-by-row into the packed layout
    for (int col = 0; col < rowType->size(); col++) {
      auto child = input->childAt(col);
      auto rawData = child->valuesAsVoid();
      VELOX_CHECK_NOT_NULL(rawData, "RowCudfFromVelox: null raw data for col {}", col);
      int32_t fieldOffset = fields[col].offset;
      int32_t fieldWidth = fields[col].byte_width;
      const uint8_t* src = static_cast<const uint8_t*>(rawData);

      for (int64_t row = 0; row < numRows; row++) {
        std::memcpy(
            hostRowBuf.data() + row * rowWidth + fieldOffset,
            src + row * fieldWidth,
            fieldWidth);
      }
    }

    if (logH2D) {
      auto cpuConvEnd = std::chrono::steady_clock::now();
      auto cpuConvNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
          cpuConvEnd - cpuConvStart).count();
      addRuntimeStat(
          "cpuColToRowNanos",
          RuntimeCounter(cpuConvNanos, RuntimeCounter::Unit::kNanos));
    }

    // H2D transfer
    std::chrono::steady_clock::time_point h2dStart;
    if (logH2D) {
      h2dStart = std::chrono::steady_clock::now();
    }

    rmm::device_buffer gpuRowBuf(totalBytes, stream);
    cudaMemcpyAsync(
        gpuRowBuf.data(), hostRowBuf.data(), totalBytes,
        cudaMemcpyHostToDevice, stream.value());

    // Upload field descriptors
    rmm::device_buffer gpuFieldsBuf(
        fields.data(), fields.size() * sizeof(FieldDesc), stream);

    // Sync to ensure transfer complete before returning
    stream.synchronize();

    if (logH2D) {
      auto h2dEnd = std::chrono::steady_clock::now();
      auto h2dNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
          h2dEnd - h2dStart).count();
      addRuntimeStat(
          "h2dWallNanos",
          RuntimeCounter(h2dNanos, RuntimeCounter::Unit::kNanos));
      addRuntimeStat(
          "h2dRows",
          RuntimeCounter(static_cast<int64_t>(numRows)));
      addRuntimeStat(
          "h2dBytes",
          RuntimeCounter(totalBytes, RuntimeCounter::Unit::kBytes));
    }

    auto result = std::make_shared<RowStoreVector>(
        input->pool(), outputType_, numRows,
        std::move(gpuRowBuf), std::move(gpuFieldsBuf),
        std::move(fields), rowWidth, stream);
    if (logTimeline && driverId == 0) {
      printf("OPTRACE %lld d0 p%d FromVelox end %lld\n",
             (long long)timeNow(), pipelineId, (long long)numRows);
      threadScanTraceState().scanTraceNeeded = true;
    }
    return result;
  }

  // ===== Standard columnar path: Arrow → cudf =====

  // Convert RowVector to cudf table.  toCudfTable synchronizes the stream
  // internally before releasing Arrow host buffers, so no additional sync
  // is needed here.
  auto tpBeforeToCudf = std::chrono::steady_clock::now();

  auto tbl = with_arrow::toCudfTable(
      input, input->pool(), stream, get_output_mr(), timestampTimeZone_);

  auto tpAfterToCudf = std::chrono::steady_clock::now();

  // Always emit detailed breakdown stats
  {
    auto nanos = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    addRuntimeStat(
        "fromVeloxSelectNanos",
        RuntimeCounter(nanos(tpStart, tpBeforeMerge), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxMergeNanos",
        RuntimeCounter(nanos(tpBeforeMerge, tpAfterMerge), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxToCudfNanos",
        RuntimeCounter(nanos(tpBeforeToCudf, tpAfterToCudf), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxTotalNanos",
        RuntimeCounter(nanos(tpStart, tpAfterToCudf), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "fromVeloxBatchesMerged",
        RuntimeCounter(static_cast<int64_t>(selectedInputs.size())));
    addRuntimeStat(
        "fromVeloxOutputRows",
        RuntimeCounter(static_cast<int64_t>(input->size())));
  }

  if (logH2D) {
    addRuntimeStat(
        "h2dRows",
        RuntimeCounter(static_cast<int64_t>(input->size())));

    // Compute actual bytes on GPU (sum of column data sizes)
    int64_t h2dBytes = 0;
    auto tblView = tbl->view();
    for (int c = 0; c < tblView.num_columns(); c++) {
      auto const& col = tblView.column(c);
      h2dBytes += static_cast<int64_t>(col.size()) *
                  cudf::size_of(col.type());
    }
    addRuntimeStat(
        "h2dBytes",
        RuntimeCounter(h2dBytes, RuntimeCounter::Unit::kBytes));
  }

  VELOX_CHECK_NOT_NULL(tbl);

  // Return a CudfVector that owns the cudf table
  const auto size = tbl->num_rows();
  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d FromVelox end %lld\n",
           (long long)timeNow(), pipelineId, (long long)size);
    threadScanTraceState().scanTraceNeeded = true;
  }
  return std::make_shared<CudfVector>(
      input->pool(), outputType_, size, std::move(tbl), stream);
}

void CudfFromVelox::doClose() {
  // TODO(kn): Remove default stream after redesign of CudfFromVelox
  cudf::get_default_stream(cudf::allow_default_stream).synchronize();
  Operator::close();
  inputs_.clear();
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          outputType,
          planNodeId,
          "CudfToVelox",
          nvtx3::rgb{148, 0, 211}, // Purple
          NvtxMethodFlag::kAll,
          std::nullopt,
          std::nullopt) {}

bool CudfToVelox::isPassthroughMode() const {
  return operatorCtx_->driverCtx()->queryConfig().get<bool>(
      kPassthroughMode, true);
}

void CudfToVelox::doAddInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
    if (cudfInput) {
      inputs_.push_back(std::move(cudfInput));
    } else {
      // Non-CudfVector (e.g. skip_output dummy RowVector) — pass through
      passthroughInputs_.push_back(std::move(input));
    }
  }
}

std::optional<uint64_t> CudfToVelox::averageRowSize() {
  if (!averageRowSize_) {
    averageRowSize_ =
        inputs_.front()->estimateFlatSize() / inputs_.front()->size();
  }
  return averageRowSize_;
}

// Pop inputs_.front(), convert its GPU table to a Velox RowVector via a
// single to_arrow_host + synchronize, and return it.  The caller is
// responsible for any further slicing.
RowVectorPtr CudfToVelox::convertFrontToVelox() {
  auto cudfVector = std::move(inputs_.front());
  inputs_.pop_front();
  auto stream = cudfVector->stream();
  auto tableView = cudfVector->getTableView();
  auto output = with_arrow::toVeloxColumn(
      tableView, pool(), outputType_, "", stream, get_temp_mr());
  stream.synchronize();
  output->setType(outputType_);
  return output;
}

// Output batching strategy
// ========================
// The key constraint is minimising D->H (device-to-host) transfers.
// Each call to toVeloxColumn / to_arrow_host triggers one D->H copy per
// column, so calling it once per output batch (rather than once per row
// or once per input batch) is critical for performance.
//
// Two cases arise depending on the size of the front GPU input relative
// to targetBatchSize:
//
//  (A) Front input >= targetBatchSize  (e.g. CudfOrderBy: one large sorted
//      table).  We convert the whole input to Velox in one shot and then
//      slice it purely on the CPU using BaseVector::slice().  Subsequent
//      getOutput() calls return successive CPU slices with no additional
//      D->H work until veloxBuffer_ is exhausted.
//
//  (B) Front input < targetBatchSize  (e.g. CudfFilterProject with high
//      selectivity: many small GPU batches).  We concatenate inputs on device
//      until we accumulate targetBatchSize rows, then convert the concat
//      result to Velox in one shot.  This preserves the GPU-side merge
//      that avoids emitting many undersized Velox batches downstream.
//
// In both cases exactly one toVeloxColumn + stream.synchronize() is issued
// per output batch, regardless of how many GPU inputs were consumed.
RowVectorPtr CudfToVelox::doGetOutput() {
  if (finished_) {
    return nullptr;
  }

  // Drain passthrough inputs first (e.g. skip_output CPU RowVectors)
  if (!passthroughInputs_.empty()) {
    auto result = std::move(passthroughInputs_.front());
    passthroughInputs_.pop_front();
    finished_ = noMoreInput_ && inputs_.empty() && passthroughInputs_.empty();
    return result;
  }

  if (outputType_->size() == 0) {
    // cuDF zero-column tables do not have a row count, so we sum the sizes
    // of all CudfVectors in the inputs_, to maintain the logical count.
    // This is necessary to ensure correct behavior for e.g. `count` operators.
    vector_size_t totalSize = 0;
    while (!inputs_.empty()) {
      totalSize += inputs_.front()->size();
      inputs_.pop_front();
    }
    finished_ = noMoreInput_ && inputs_.empty();
    if (totalSize == 0) {
      return nullptr;
    }
    return BaseVector::create<RowVector>(outputType_, totalSize, pool());
  }

  // Drain veloxBuffer_ (populated on a previous call) before consuming
  // more GPU inputs.
  if (!veloxBuffer_) {
    if (inputs_.empty()) {
      finished_ = noMoreInput_ && passthroughInputs_.empty();
      return nullptr;
    }

    // Passthrough mode: emit each GPU input as a single Velox batch with no
    // re-batching.  Used when the caller knows the batch size is already
    // correct (e.g. default pipeline without explicit batch-size overrides).
    if (isPassthroughMode()) {
      auto output = convertFrontToVelox();
      finished_ = noMoreInput_ && inputs_.empty();
      if (output->size() == 0) {
        return nullptr;
      }
      return output;
    }

    const auto targetBatchSize = outputBatchRows(averageRowSize());

    if (static_cast<vector_size_t>(inputs_.front()->size()) >=
        targetBatchSize) {
      // Case A: large input.  Convert once; subsequent calls slice CPU-side.
      veloxBuffer_ = convertFrontToVelox();
      veloxOffset_ = 0;
      averageRowSize_ = std::nullopt; // recompute from next input
    } else {
      // Case B: small inputs.  GPU-concat until we reach targetBatchSize,
      // then convert the merged table in one D->H transfer.
      auto stream = inputs_.front()->stream();
      std::vector<CudfVectorPtr> toConcat;
      vector_size_t accumulated = 0;
      while (!inputs_.empty() && accumulated < targetBatchSize) {
        accumulated += static_cast<vector_size_t>(inputs_.front()->size());
        toConcat.push_back(std::move(inputs_.front()));
        inputs_.pop_front();
      }
      VELOX_CHECK_LE(
          accumulated,
          std::numeric_limits<cudf::size_type>::max(),
          "Accumulated row count exceeds cudf int32 limit");
      auto concatTable = getConcatenatedTable(
          std::move(toConcat), outputType_, stream, get_temp_mr());
      auto tableView = concatTable->view();
      veloxBuffer_ = with_arrow::toVeloxColumn(
          tableView, pool(), outputType_, "", stream, get_temp_mr());
      stream.synchronize();
      veloxBuffer_->setType(outputType_);
      veloxOffset_ = 0;
      averageRowSize_ = std::nullopt;
    }
  }

  // Slice veloxBuffer_ on the CPU to produce the next output batch.
  const auto totalRows = static_cast<vector_size_t>(veloxBuffer_->size());
  if (veloxOffset_ >= totalRows) {
    veloxBuffer_.reset();
    finished_ = noMoreInput_ && inputs_.empty();
    return nullptr;
  }

  const auto targetBatchSize = outputBatchRows(
      veloxBuffer_->estimateFlatSize() /
      static_cast<uint64_t>(std::max<vector_size_t>(totalRows, 1)));
  const auto take = std::min(targetBatchSize, totalRows - veloxOffset_);

  auto slice = std::dynamic_pointer_cast<RowVector>(
      veloxBuffer_->slice(veloxOffset_, take));
  VELOX_CHECK_NOT_NULL(slice);
  veloxOffset_ += take;

  if (veloxOffset_ >= totalRows) {
    veloxBuffer_.reset();
    finished_ = noMoreInput_ && inputs_.empty();
  }

  return slice;
}

void CudfToVelox::doClose() {
  Operator::close();
  inputs_.clear();
  veloxBuffer_.reset();
}

} // namespace facebook::velox::cudf_velox
