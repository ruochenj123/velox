/*
 * RowHashJoin.cpp
 *
 * Row-wise hash join operators for comparing row vs column gather in Velox.
 * See RowHashJoin.h for design overview.
 */

#include "velox/experimental/cudf/exec/RowHashJoin.h"
#include "velox/experimental/cudf/exec/GpuRowOps.cuh"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/RowStoreVector.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/CudfConfig.h"

#include "velox/exec/Task.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>

namespace facebook::velox::cudf_velox {

// ============================================================================
// Helpers
// ============================================================================

namespace {

/// Compute FieldDesc array and row width from a Velox RowType (fixed-width only).
std::pair<std::vector<FieldDesc>, int32_t> computeRowLayout(
    const RowTypePtr& type) {
  std::vector<FieldDesc> fields;
  int32_t offset = 0;
  for (int i = 0; i < type->size(); i++) {
    FieldDesc fd;
    fd.offset = offset;
    auto kind = type->childAt(i)->kind();
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
        VELOX_FAIL("RowHashJoin: unsupported TypeKind {}", (int)kind);
    }
    fields.push_back(fd);
    offset += fd.byte_width;
  }
  int32_t rowWidth = (offset + 7) & ~7; // 8-byte align
  return {fields, rowWidth};
}

/// Transpose a cudf::table_view (already on GPU) into a row buffer on GPU.
/// Returns the row buffer as rmm::device_buffer.
void transposeToRows(
    const cudf::table_view& table,
    const std::vector<FieldDesc>& fields,
    int32_t rowWidth,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  int32_t numRows = table.num_rows();
  int32_t numCols = table.num_columns();
  if (numRows == 0 || numCols == 0) return;

  // Collect column data pointers
  std::vector<const uint8_t*> colPtrs(numCols);
  for (int i = 0; i < numCols; i++) {
    colPtrs[i] = static_cast<const uint8_t*>(table.column(i).head());
  }

  columnsToRows(
      colPtrs.data(), fields.data(), numCols, numRows, rowWidth,
      d_row_buffer, stream);
}

/// Map cudf type to byte width.
int32_t cudfTypeWidth(cudf::type_id id) {
  switch (id) {
    case cudf::type_id::INT8:
    case cudf::type_id::BOOL8:
      return 1;
    case cudf::type_id::INT16:
      return 2;
    case cudf::type_id::INT32:
    case cudf::type_id::TIMESTAMP_DAYS:
      return 4;
    case cudf::type_id::INT64:
    case cudf::type_id::FLOAT64:
    case cudf::type_id::TIMESTAMP_SECONDS:
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return 8;
    case cudf::type_id::FLOAT32:
      return 4;
    default:
      VELOX_FAIL("RowHashJoin: unsupported cudf type_id {}", (int)id);
  }
}

/// Compute FieldDesc from a cudf::table_view's column types.
std::pair<std::vector<FieldDesc>, int32_t> computeRowLayoutFromTable(
    const cudf::table_view& table) {
  std::vector<FieldDesc> fields;
  int32_t offset = 0;
  for (int i = 0; i < table.num_columns(); i++) {
    FieldDesc fd;
    fd.offset = offset;
    fd.byte_width = cudfTypeWidth(table.column(i).type().id());
    fields.push_back(fd);
    offset += fd.byte_width;
  }
  int32_t rowWidth = (offset + 7) & ~7;
  return {fields, rowWidth};
}

/// Compute FieldDesc from a cudf::table_view, EXCLUDING one column.
/// The excluded column (e.g. join key) is kept columnar to avoid
/// a redundant transpose + extract cycle.
std::pair<std::vector<FieldDesc>, int32_t> computeRowLayoutFromTableExcluding(
    const cudf::table_view& table, int32_t skipColIdx) {
  std::vector<FieldDesc> fields;
  int32_t offset = 0;
  for (int i = 0; i < table.num_columns(); i++) {
    if (i == skipColIdx) continue;
    FieldDesc fd;
    fd.offset = offset;
    fd.byte_width = cudfTypeWidth(table.column(i).type().id());
    fields.push_back(fd);
    offset += fd.byte_width;
  }
  int32_t rowWidth = (offset + 7) & ~7;
  return {fields, rowWidth};
}

/// Transpose a cudf::table_view into a row buffer, EXCLUDING one column.
void transposeToRowsExcluding(
    const cudf::table_view& table,
    const std::vector<FieldDesc>& fields,
    int32_t rowWidth,
    int32_t skipColIdx,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  int32_t numRows = table.num_rows();
  int32_t numCols = table.num_columns() - 1;
  if (numRows == 0 || numCols <= 0) return;

  std::vector<const uint8_t*> colPtrs;
  colPtrs.reserve(numCols);
  for (int i = 0; i < table.num_columns(); i++) {
    if (i == skipColIdx) continue;
    colPtrs.push_back(static_cast<const uint8_t*>(table.column(i).head()));
  }

  columnsToRows(
      colPtrs.data(), fields.data(), numCols, numRows, rowWidth,
      d_row_buffer, stream);
}

} // namespace

// ============================================================================
// RowHashJoinBridge
// ============================================================================

void RowHashJoinBridge::setBuildData(std::shared_ptr<BuildData> data) {
  std::vector<ContinuePromise> promises;
  {
    std::lock_guard<std::mutex> l(mutex_);
    VELOX_CHECK(buildData_ == nullptr, "Build data already set");
    buildData_ = std::move(data);
    promises = std::move(promises_);
  }
  notify(std::move(promises));
}

std::shared_ptr<RowHashJoinBridge::BuildData>
RowHashJoinBridge::dataOrFuture(ContinueFuture* future) {
  std::lock_guard<std::mutex> l(mutex_);
  if (buildData_) {
    return buildData_;  // All probes share the same data
  }
  promises_.emplace_back("RowHashJoinBridge::dataOrFuture");
  *future = promises_.back().getSemiFuture();
  return nullptr;
}

void RowHashJoinBridge::setBuildStream(rmm::cuda_stream_view stream) {
  std::lock_guard<std::mutex> l(mutex_);
  buildStream_ = stream;
}

std::optional<rmm::cuda_stream_view> RowHashJoinBridge::getBuildStream() {
  std::lock_guard<std::mutex> l(mutex_);
  return buildStream_;
}

// ============================================================================
// RowHashJoinBuild
// ============================================================================

RowHashJoinBuild::RowHashJoinBuild(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          nullptr,
          joinNode->id(),
          "RowHashJoinBuild",
          nvtx3::rgb{139, 69, 19}, // Saddle Brown
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(joinNode) {}

void RowHashJoinBuild::doAddInput(RowVectorPtr input) {
  if (input->size() > 0) {
    inputs_.push_back(std::move(input));
  }
}

bool RowHashJoinBuild::needsInput() const {
  return !noMoreInput_;
}

RowVectorPtr RowHashJoinBuild::doGetOutput() {
  return nullptr;
}

void RowHashJoinBuild::doNoMoreInput() {
  Operator::noMoreInput();
  std::vector<ContinuePromise> promises;
  std::vector<std::shared_ptr<exec::Driver>> peers;
  if (!operatorCtx_->task()->allPeersFinished(
          planNodeId(), operatorCtx_->driver(), &future_, promises, peers)) {
    return;
  }

  // Collect all peers' inputs
  for (auto& peer : peers) {
    auto op = peer->findOperator(planNodeId());
    auto* build = dynamic_cast<RowHashJoinBuild*>(op);
    VELOX_CHECK_NOT_NULL(build);
    inputs_.insert(inputs_.end(), build->inputs_.begin(), build->inputs_.end());
  }

  SCOPE_EXIT {
    peers.clear();
    for (auto& promise : promises) {
      promise.setValue();
    }
  };

  auto stream = cudfGlobalStreamPool().get_stream();

  // Check if inputs are RowStoreVectors (already in row layout on GPU)
  auto firstRowStore = std::dynamic_pointer_cast<RowStoreVector>(inputs_[0]);

  int64_t numRows = 0;
  int32_t rowWidth = 0;
  std::vector<FieldDesc> fields;
  rmm::device_buffer rowBuffer;
  rmm::device_buffer fieldsBuffer;
  rmm::device_buffer keyBuffer;
  std::shared_ptr<cudf::hash_join> hashJoin;

  if (firstRowStore) {
    // ---- RowStoreVector path: concatenate GPU row buffers ----
    rowWidth = firstRowStore->rowWidth();
    fields = firstRowStore->hostFields();

    // Compute total rows
    for (auto& inp : inputs_) {
      numRows += inp->size();
    }

    // Allocate combined buffer and copy each chunk
    int64_t totalBytes = numRows * rowWidth;
    rowBuffer = rmm::device_buffer(totalBytes, stream);
    int64_t offset = 0;
    for (auto& inp : inputs_) {
      auto rsv = std::dynamic_pointer_cast<RowStoreVector>(inp);
      VELOX_CHECK_NOT_NULL(rsv);
      int64_t chunkBytes = rsv->gpuRowBytes();
      if (chunkBytes > 0) {
        cudaMemcpyAsync(
            static_cast<uint8_t*>(rowBuffer.data()) + offset,
            rsv->gpuRowData(), chunkBytes,
            cudaMemcpyDeviceToDevice, stream.value());
        offset += chunkBytes;
      }
    }
    inputs_.clear();

    // Upload field descriptors
    fieldsBuffer = rmm::device_buffer(
        fields.data(), fields.size() * sizeof(FieldDesc), stream);

    // RowStoreVector path: key is embedded in rows, need to extract
    auto rightKeys = joinNode_->rightKeys();
    auto rightType = joinNode_->sources()[1]->outputType();
    int keyColIdx = rightType->getChildIdx(rightKeys[0]->name());
    int32_t keyOffset = fields[keyColIdx].offset;
    int32_t keyWidth = fields[keyColIdx].byte_width;

    GpuFixedRowStore tmpStore;
    tmpStore.row_buffer = static_cast<uint8_t*>(rowBuffer.data());
    tmpStore.row_width = rowWidth;
    tmpStore.num_rows = numRows;
    tmpStore.num_fields = fields.size();
    tmpStore.fields = static_cast<const FieldDesc*>(fieldsBuffer.data());

    keyBuffer = rmm::device_buffer(numRows * keyWidth, stream);
    extractKeysFromRows(
        tmpStore, keyOffset, keyWidth,
        keyBuffer.data(), stream.value());

    auto keyCol = cudf::column_view(
        cudf::data_type{keyWidth == 8 ? cudf::type_id::INT64 : cudf::type_id::INT32},
        static_cast<cudf::size_type>(numRows),
        keyBuffer.data(), nullptr, 0);
    auto keyTable = cudf::table_view({keyCol});
    hashJoin = std::make_shared<cudf::hash_join>(
        keyTable, cudf::null_equality::UNEQUAL, stream);
  } else {
    // ---- CudfVector path: concatenate + transpose (excluding key) ----
    auto buildType = joinNode_->sources()[1]->outputType();
    std::vector<CudfVectorPtr> cudfInputs;
    for (auto& inp : inputs_) {
      auto cv = std::dynamic_pointer_cast<CudfVector>(inp);
      VELOX_CHECK_NOT_NULL(cv);
      cudfInputs.push_back(std::move(cv));
    }
    inputs_.clear();

    auto concatenated = getConcatenatedTable(
        std::move(cudfInputs), buildType, stream, get_output_mr());
    VELOX_CHECK_NOT_NULL(concatenated);
    auto buildView = concatenated->view();
    numRows = buildView.num_rows();

    // Identify key column — keep it columnar (avoid transpose + extract)
    auto rightKeys = joinNode_->rightKeys();
    auto rightType = joinNode_->sources()[1]->outputType();
    int keyColIdx = rightType->getChildIdx(rightKeys[0]->name());
    auto keyColView = buildView.column(keyColIdx);
    int32_t keyWidth = cudfTypeWidth(keyColView.type().id());

    // Save key column via direct D2D copy (columnar → columnar, no extract)
    keyBuffer = rmm::device_buffer(numRows * keyWidth, stream);
    cudaMemcpyAsync(
        keyBuffer.data(), keyColView.head(),
        static_cast<size_t>(numRows) * keyWidth,
        cudaMemcpyDeviceToDevice, stream.value());

    // Compute row layout EXCLUDING key column (narrower rows)
    auto [f, rw] = computeRowLayoutFromTableExcluding(buildView, keyColIdx);
    fields = std::move(f);
    rowWidth = rw;

    int64_t rowBytes = numRows * rowWidth;
    rowBuffer = rmm::device_buffer(rowBytes, stream);

    cudaEvent_t txStart, txEnd;
    const bool timeTx = CudfConfig::getInstance().benchmarkLogGatherTime;
    if (timeTx) {
      cudaEventCreate(&txStart);
      cudaEventCreate(&txEnd);
      cudaEventRecord(txStart, stream.value());
    }

    // Transpose only non-key columns (key stays columnar)
    transposeToRowsExcluding(
        buildView, fields, rowWidth, keyColIdx,
        static_cast<uint8_t*>(rowBuffer.data()), stream.value());

    if (timeTx) {
      cudaEventRecord(txEnd, stream.value());
      cudaEventSynchronize(txEnd);
      float ms = 0;
      cudaEventElapsedTime(&ms, txStart, txEnd);
      auto nanos = static_cast<int64_t>(ms * 1e6);
      addRuntimeStat(
          "gpuTransposeWallNanos",
          RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
      addRuntimeStat(
          "gpuTransposeRows",
          RuntimeCounter(static_cast<int64_t>(numRows)));
      cudaEventDestroy(txStart);
      cudaEventDestroy(txEnd);
    }

    fieldsBuffer = rmm::device_buffer(
        fields.data(), fields.size() * sizeof(FieldDesc), stream);

    // Build hash table from columnar key (no extractKeysFromRows needed!)
    auto keyCol = cudf::column_view(
        cudf::data_type{keyWidth == 8 ? cudf::type_id::INT64 : cudf::type_id::INT32},
        static_cast<cudf::size_type>(numRows),
        keyBuffer.data(), nullptr, 0);
    auto keyTable = cudf::table_view({keyCol});
    hashJoin = std::make_shared<cudf::hash_join>(
        keyTable, cudf::null_equality::UNEQUAL, stream);
  }

  // Build GpuFixedRowStore handle
  GpuFixedRowStore gpuStore;
  gpuStore.row_buffer = static_cast<uint8_t*>(rowBuffer.data());
  gpuStore.row_width = rowWidth;
  gpuStore.num_rows = numRows;
  gpuStore.num_fields = fields.size();
  gpuStore.fields = static_cast<const FieldDesc*>(fieldsBuffer.data());

  stream.synchronize();

  // Push to bridge
  RowHashJoinBridge::BuildData bd;
  bd.gpuRowStore = gpuStore;
  bd.rowBuffer = std::move(rowBuffer);
  bd.fieldsBuffer = std::move(fieldsBuffer);
  bd.hashJoin = std::move(hashJoin);
  bd.keyBuffer = std::move(keyBuffer);
  bd.numRows = numRows;
  bd.rowWidth = rowWidth;

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto rowBridge = std::dynamic_pointer_cast<RowHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(rowBridge);
  rowBridge->setBuildStream(stream);
  rowBridge->setBuildData(std::make_shared<RowHashJoinBridge::BuildData>(std::move(bd)));
}

exec::BlockingReason RowHashJoinBuild::isBlocked(ContinueFuture* future) {
  if (!future_.valid()) {
    return exec::BlockingReason::kNotBlocked;
  }
  *future = std::move(future_);
  return exec::BlockingReason::kWaitForJoinBuild;
}

bool RowHashJoinBuild::isFinished() {
  return !future_.valid() && noMoreInput_;
}

// ============================================================================
// RowHashJoinProbe
// ============================================================================

RowHashJoinProbe::RowHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::HashJoinNode> joinNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNode->outputType(),
          joinNode->id(),
          "RowHashJoinProbe",
          nvtx3::rgb{184, 134, 11}, // Dark Goldenrod
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNode),
      joinNode_(joinNode) {
  // Resolve key column indices
  auto leftType = joinNode_->sources()[0]->outputType();
  auto leftKeys = joinNode_->leftKeys();
  for (auto& key : leftKeys) {
    leftKeyIndices_.push_back(
        static_cast<cudf::size_type>(leftType->getChildIdx(key->name())));
  }

  // Resolve which columns to gather and their output positions
  auto outputType = joinNode_->outputType();
  auto rightType = joinNode_->sources()[1]->outputType();

  for (int i = 0; i < outputType->size(); i++) {
    auto name = outputType->nameOf(i);
    auto leftIdx = leftType->getChildIdxIfExists(name);
    if (leftIdx.has_value()) {
      leftColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(leftIdx.value()));
      leftColumnOutputIndices_.push_back(i);
    } else {
      auto rightIdx = rightType->getChildIdx(name);
      rightColumnIndicesToGather_.push_back(
          static_cast<cudf::size_type>(rightIdx));
      rightColumnOutputIndices_.push_back(i);
    }
  }
}

bool RowHashJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

exec::BlockingReason RowHashJoinProbe::isBlocked(ContinueFuture* future) {
  if (buildData_) {
    return exec::BlockingReason::kNotBlocked;
  }

  auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
      operatorCtx_->driverCtx()->splitGroupId, planNodeId());
  auto rowBridge = std::dynamic_pointer_cast<RowHashJoinBridge>(joinBridge);
  VELOX_CHECK_NOT_NULL(rowBridge);

  auto data = rowBridge->dataOrFuture(future);
  if (data) {
    buildData_ = std::move(data);
    buildStream_ = rowBridge->getBuildStream();
    return exec::BlockingReason::kNotBlocked;
  }
  return exec::BlockingReason::kWaitForJoinBuild;
}

void RowHashJoinProbe::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

RowVectorPtr RowHashJoinProbe::doGetOutput() {
  if (!input_) {
    if (noMoreInput_) {
      finished_ = true;
    }
    return nullptr;
  }

  // Timeline logging
  const bool logTimeline = CudfConfig::getInstance().benchmarkLogTimeline;
  const auto driverId = operatorCtx_->driverCtx()->driverId;
  const auto pipelineId = operatorCtx_->driverCtx()->pipelineId;
  auto timeNow = []() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d RowJoinProbe start %lld\n",
           (long long)timeNow(), pipelineId, (long long)input_->size());
  }

  VELOX_CHECK_NOT_NULL(buildData_);
  auto& bd = *buildData_;
  int32_t buildRowWidth = bd.rowWidth;

  // Dual-path: detect RowStoreVector (pre-transposed) vs CudfVector (needs transpose)
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  int32_t probeRows = 0;
  GpuFixedRowStore probeGpuStore;

  auto rowStoreInput = std::dynamic_pointer_cast<RowStoreVector>(input_);
  if (rowStoreInput) {
    // ---- Path A: RowStoreVector (already row-layout on GPU) ----
    stream = rowStoreInput->stream();
    probeRows = rowStoreInput->size();

    if (!initialized_) {
      probeFields_ = rowStoreInput->hostFields();
      probeRowWidth_ = rowStoreInput->rowWidth();
      initialized_ = true;
    }

    // Upload fields descriptor if not done
    if (!fieldsUploaded_) {
      probeFieldsBuffer_ = rmm::device_buffer(
          probeFields_.data(), probeFields_.size() * sizeof(FieldDesc), stream);
      fieldsUploaded_ = true;
    }

    // Ensure key buffer is large enough
    if (probeRows > probeRowCapacity_) {
      probeRowCapacity_ = probeRows;
      probeKeyBuffer_ = rmm::device_buffer(probeRows * 8, stream);
    }

    probeGpuStore.row_buffer = static_cast<uint8_t*>(rowStoreInput->gpuRowData());
    probeGpuStore.row_width = probeRowWidth_;
    probeGpuStore.num_rows = probeRows;
    probeGpuStore.num_fields = probeFields_.size();
    probeGpuStore.fields = static_cast<const FieldDesc*>(probeFieldsBuffer_.data());

    addRuntimeStat(
        "probeTransposeSkipped",
        RuntimeCounter(static_cast<int64_t>(probeRows)));
  } else {
    // ---- Path B: CudfVector (transpose non-key columns, keep key columnar) ----
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
    VELOX_CHECK_NOT_NULL(cudfInput, "RowHashJoinProbe: input must be RowStoreVector or CudfVector");
    stream = cudfInput->stream();
    auto probeView = cudfInput->getTableView();
    probeRows = probeView.num_rows();

    // Save probe key column reference (columnar, no copy needed — input_ stays alive)
    auto probeKeyColView = probeView.column(leftKeyIndices_[0]);
    int32_t keyWidth = cudfTypeWidth(probeKeyColView.type().id());
    probeKeyData_ = probeKeyColView.head();
    probeKeyWidth_ = keyWidth;

    if (!initialized_) {
      // Compute row layout EXCLUDING key column (narrower rows)
      auto [pf, prw] = computeRowLayoutFromTableExcluding(probeView, leftKeyIndices_[0]);
      probeFields_ = std::move(pf);
      probeRowWidth_ = prw;
      initialized_ = true;
    }

    int64_t probeRowBytes = (int64_t)probeRows * probeRowWidth_;
    if (probeRows > probeRowCapacity_) {
      probeRowCapacity_ = probeRows;
      probeRowBuffer_ = rmm::device_buffer(probeRowBytes, stream);
      if (!fieldsUploaded_) {
        probeFieldsBuffer_ = rmm::device_buffer(
            probeFields_.data(), probeFields_.size() * sizeof(FieldDesc), stream);
        fieldsUploaded_ = true;
      }
    }
    cudaEvent_t txStart, txEnd;
    const bool timeTx = CudfConfig::getInstance().benchmarkLogGatherTime;
    if (timeTx) {
      cudaEventCreate(&txStart);
      cudaEventCreate(&txEnd);
      cudaEventRecord(txStart, stream.value());
    }

    // Transpose only non-key columns (key stays columnar)
    transposeToRowsExcluding(
        probeView, probeFields_, probeRowWidth_, leftKeyIndices_[0],
        static_cast<uint8_t*>(probeRowBuffer_.data()), stream.value());

    if (timeTx) {
      cudaEventRecord(txEnd, stream.value());
      cudaEventSynchronize(txEnd);
      float ms = 0;
      cudaEventElapsedTime(&ms, txStart, txEnd);
      auto nanos = static_cast<int64_t>(ms * 1e6);
      addRuntimeStat(
          "gpuTransposeWallNanos",
          RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
      addRuntimeStat(
          "gpuTransposeRows",
          RuntimeCounter(static_cast<int64_t>(probeRows)));
    }

    probeGpuStore.row_buffer = static_cast<uint8_t*>(probeRowBuffer_.data());
    probeGpuStore.row_width = probeRowWidth_;
    probeGpuStore.num_rows = probeRows;
    probeGpuStore.num_fields = probeFields_.size();
    probeGpuStore.fields = static_cast<const FieldDesc*>(probeFieldsBuffer_.data());
  }

  // ---- 2. Prepare probe key for hash join ----
  const void* keyData = nullptr;
  int32_t keyWidth = 0;
  if (probeKeyData_) {
    // CudfVector path: key is already columnar, use directly
    keyData = probeKeyData_;
    keyWidth = probeKeyWidth_;
    probeKeyData_ = nullptr;  // reset for next batch
  } else {
    // RowStoreVector path: extract key from row buffer
    int32_t keyOffset = probeFields_[leftKeyIndices_[0]].offset;
    keyWidth = probeFields_[leftKeyIndices_[0]].byte_width;
    extractKeysFromRows(
        probeGpuStore, keyOffset, keyWidth,
        probeKeyBuffer_.data(), stream.value());
    keyData = probeKeyBuffer_.data();
  }

  // ---- 3. Hash probe ----
  auto probeKeyCol = cudf::column_view(
      cudf::data_type{keyWidth == 8 ? cudf::type_id::INT64 : cudf::type_id::INT32},
      static_cast<cudf::size_type>(probeRows),
      keyData, nullptr, 0);
  auto probeKeyTable = cudf::table_view({probeKeyCol});

  auto [leftIndices, rightIndices] = bd.hashJoin->inner_join(
      probeKeyTable, std::nullopt, stream, get_temp_mr());

  int32_t numMatches = leftIndices->size();

  // ---- 4. Row gather ----

  // Benchmark timing via cuda events
  cudaEvent_t gatherStart, gatherEnd;
  const bool timeGather = CudfConfig::getInstance().benchmarkLogGatherTime;
  if (timeGather) {
    cudaEventCreate(&gatherStart);
    cudaEventCreate(&gatherEnd);
    cudaEventRecord(gatherStart, stream.value());
  }

  if (numMatches > 0) {
    // Ensure gather buffers are large enough
    int64_t neededProbe = (int64_t)numMatches * probeRowWidth_;
    int64_t neededBuild = (int64_t)numMatches * buildRowWidth;
    if (numMatches > gatherCapacity_) {
      gatherCapacity_ = numMatches;
      probeGatherBuffer_ = rmm::device_buffer(neededProbe, stream);
      buildGatherBuffer_ = rmm::device_buffer(neededBuild, stream);
    }

    // Gather probe rows
    gatherRowsWarp(
        probeGpuStore,
        reinterpret_cast<const int32_t*>(leftIndices->data()),
        numMatches,
        static_cast<uint8_t*>(probeGatherBuffer_.data()),
        stream.value());

    // Gather build rows
    gatherRowsWarp(
        bd.gpuRowStore,
        reinterpret_cast<const int32_t*>(rightIndices->data()),
        numMatches,
        static_cast<uint8_t*>(buildGatherBuffer_.data()),
        stream.value());
  }

  if (timeGather) {
    cudaEventRecord(gatherEnd, stream.value());
  }

  stream.synchronize();

  // Record gather time as operator stat
  if (timeGather) {
    float ms = 0;
    cudaEventSynchronize(gatherEnd);
    cudaEventElapsedTime(&ms, gatherStart, gatherEnd);
    auto nanos = static_cast<int64_t>(ms * 1e6);
    addRuntimeStat(
        "rowGatherWallNanos",
        RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "rowGatherOutputRows",
        RuntimeCounter(static_cast<int64_t>(numMatches)));
    cudaEventDestroy(gatherStart);
    cudaEventDestroy(gatherEnd);
  }

  // Release input
  rowStoreInput.reset();
  input_.reset();
  finished_ = noMoreInput_;

  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d RowJoinProbe end %lld\n",
           (long long)timeNow(), pipelineId, (long long)numMatches);
  }

  // Return dummy output (1 row) to indicate progress
  // This matches the skip_output behavior of CudfHashJoinProbe.
  auto outputType = outputType_;
  if (CudfConfig::getInstance().benchmarkSkipOutput && numMatches > 0) {
    // Create a minimal 1-row output
    std::vector<VectorPtr> children(outputType->size());
    for (int i = 0; i < outputType->size(); i++) {
      children[i] = BaseVector::createNullConstant(
          outputType->childAt(i), 1, pool());
    }
    return std::make_shared<RowVector>(
        pool(), outputType, nullptr, 1, std::move(children));
  } else if (numMatches == 0) {
    return nullptr;
  }

  // Non-skip path: also return 1 row (we don't implement full D2H for rows)
  std::vector<VectorPtr> children(outputType->size());
  for (int i = 0; i < outputType->size(); i++) {
    children[i] = BaseVector::createNullConstant(
        outputType->childAt(i), 1, pool());
  }
  return std::make_shared<RowVector>(
      pool(), outputType, nullptr, 1, std::move(children));
}

void RowHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

bool RowHashJoinProbe::isFinished() {
  return finished_;
}

// ============================================================================
// RowHashJoinBridgeTranslator
// ============================================================================

std::unique_ptr<exec::Operator> RowHashJoinBridgeTranslator::toOperator(
    exec::DriverCtx* ctx,
    int32_t id,
    const core::PlanNodePtr& node) {
  auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
  if (joinNode) {
    return std::make_unique<RowHashJoinProbe>(id, ctx, joinNode);
  }
  return nullptr;
}

std::unique_ptr<exec::JoinBridge>
RowHashJoinBridgeTranslator::toJoinBridge(const core::PlanNodePtr& node) {
  auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
  if (joinNode) {
    return std::make_unique<RowHashJoinBridge>();
  }
  return nullptr;
}

exec::OperatorSupplier RowHashJoinBridgeTranslator::toOperatorSupplier(
    const core::PlanNodePtr& node) {
  auto joinNode = std::dynamic_pointer_cast<const core::HashJoinNode>(node);
  if (joinNode) {
    return [joinNode](int32_t operatorId, exec::DriverCtx* ctx) {
      return std::make_unique<RowHashJoinBuild>(operatorId, ctx, joinNode);
    };
  }
  return nullptr;
}

} // namespace facebook::velox::cudf_velox
