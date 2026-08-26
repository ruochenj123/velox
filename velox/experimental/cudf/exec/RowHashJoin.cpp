/*
 * RowHashJoin.cpp
 *
 * Row-wise hash join operators for comparing row vs column gather in Velox.
 * See RowHashJoin.h for design overview.
 */

#include "velox/experimental/cudf/exec/HostRowVector.h"
#include "velox/experimental/cudf/exec/RowHashJoin.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/DedicatedStream.h"
#include "velox/experimental/cudf/exec/DeferralStats.h"
#include "velox/experimental/cudf/exec/GpuRowOps.cuh"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/RowStoreVector.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/CudfConfig.h"

#include "velox/exec/Task.h"
#include "velox/exec/Driver.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <rmm/device_buffer.hpp>

#include <cstdlib>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::cudf_velox {

// ============================================================================
// Helpers
// ============================================================================

namespace {

// Align `off` up to `w` (a power of two: 1/2/4/8). EVERY row/accumulator layout
// here must natural-align its fields: the fused kernel copies them with
// `*reinterpret_cast<uint64_t*>` / `<uint32_t*>`, which faults
// (cudaErrorMisalignedAddress) on an unaligned address. Packing tightly is only
// safe when every field is 8 bytes (Q5/Q8) -- the moment a 4-byte column
// precedes an 8-byte one (any mixed-width schema), the 8-byte field lands on a
// 4-byte boundary and the kernel crashes. Natural alignment costs a few padding
// bytes and fixes it.
static inline int32_t alignUp(int32_t off, int32_t w) {
  return (off + w - 1) & ~(w - 1);
}

/// Compute FieldDesc array and row width from a Velox RowType (fixed-width only).
std::pair<std::vector<FieldDesc>, int32_t> computeRowLayout(
    const RowTypePtr& type) {
  std::vector<FieldDesc> fields;
  int32_t offset = 0;
  for (int i = 0; i < type->size(); i++) {
    FieldDesc fd;
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
    offset = alignUp(offset, fd.byte_width);
    fd.offset = offset;
    fields.push_back(fd);
    offset += fd.byte_width;
  }
  int32_t rowWidth = (offset + 7) & ~7; // 8-byte align
  return {fields, rowWidth};
}

/// Fixed-stride row stores have no null representation. Every table_view
/// that is transposed into a row store MUST be null-free; reading head()
/// under a null mask silently fabricates values (observed on TPC-DS
/// wr_return_amt: identical row counts, diverging content checksums -- see
/// REVIEW-ROUND2.md, 2026-08-17). Hard-fail per the row path's own
/// unsupported-batch rule; the columnar cudf path remains the fallback for
/// nullable data.
void checkTransposeInputNullFree(
    const cudf::table_view& table,
    const char* site) {
  for (int i = 0; i < table.num_columns(); i++) {
    VELOX_CHECK_EQ(
        table.column(i).null_count(),
        0,
        "RowHashJoin {}: column {} has nulls; the fixed-stride row store "
        "cannot represent them (null support tracked in REVIEW-ROUND2.md)",
        site,
        i);
  }
}

/// Transpose a cudf::table_view (already on GPU) into a row buffer on GPU.
/// Returns the row buffer as rmm::device_buffer.
/// Returns the out-of-line string heap (empty when the table has no string
/// columns). String columns' chars are concatenated into one heap and their
/// slots written by stringsToSlots; fixed columns go through columnsToRows.
rmm::device_buffer transposeToRows(
    const cudf::table_view& table,
    const std::vector<FieldDesc>& fields,
    int32_t rowWidth,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  int32_t numRows = table.num_rows();
  int32_t numCols = table.num_columns();
  if (numRows == 0 || numCols == 0) return rmm::device_buffer{};
  checkTransposeInputNullFree(table, "transposeToRows");

  // Fixed-width columns: pointer + FieldDesc subset.
  std::vector<const uint8_t*> colPtrs;
  std::vector<FieldDesc> fixedFields;
  std::vector<int> strCols;
  for (int i = 0; i < numCols; i++) {
    if (fields[i].kind == kFieldString) {
      strCols.push_back(i);
      continue;
    }
    colPtrs.push_back(static_cast<const uint8_t*>(table.column(i).head()));
    fixedFields.push_back(fields[i]);
  }
  if (!fixedFields.empty()) {
    columnsToRows(
        colPtrs.data(), fixedFields.data(), fixedFields.size(), numRows,
        rowWidth, d_row_buffer, stream);
  }
  if (strCols.empty()) {
    return rmm::device_buffer{};
  }
  rmm::cuda_stream_view sv(stream);
  int64_t total = 0;
  std::vector<int64_t> bases(strCols.size());
  for (size_t k = 0; k < strCols.size(); k++) {
    cudf::strings_column_view scv(table.column(strCols[k]));
    bases[k] = total;
    total += scv.chars_size(sv);
  }
  rmm::device_buffer heap(total, sv);
  for (size_t k = 0; k < strCols.size(); k++) {
    const auto col = table.column(strCols[k]);
    cudf::strings_column_view scv(col);
    auto offsets = scv.offsets();
    const bool off64 = offsets.type().id() == cudf::type_id::INT64;
    // offsets() is the UNSLICED child; offsets are absolute into the chars
    // buffer, so for a sliced column skip col.offset() entries and subtract
    // the first offset when relocating bytes into the combined heap.
    const uint8_t* offBase = static_cast<const uint8_t*>(offsets.head()) +
        static_cast<int64_t>(col.offset()) * (off64 ? 8 : 4);
    int64_t firstOff = 0;
    if (col.offset() != 0) {
      if (off64) {
        cudaMemcpyAsync(&firstOff, offBase, 8, cudaMemcpyDeviceToHost, stream);
      } else {
        int32_t f32 = 0;
        cudaMemcpyAsync(&f32, offBase, 4, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        firstOff = f32;
      }
      cudaStreamSynchronize(stream);
    }
    const int64_t bytes = scv.chars_size(sv);
    const uint8_t* chars =
        reinterpret_cast<const uint8_t*>(scv.chars_begin(sv));
    if (bytes > 0) {
      cudaMemcpyAsync(
          static_cast<uint8_t*>(heap.data()) + bases[k],
          chars + firstOff, bytes, cudaMemcpyDeviceToDevice, stream);
    }
    stringsToSlots(
        offBase,
        off64,
        chars,
        bases[k] - firstOff,
        numRows,
        d_row_buffer,
        rowWidth,
        fields[strCols[k]].offset,
        stream);
  }
  return heap;
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

/// Extract all join-key columns from a fixed-stride row store into separate
/// device buffers and append a matching cudf::column_view for each. Supports
/// composite (multi-column) keys: cudf::hash_join compares row-wise across all
/// key columns. Buffers are appended to `outBuffers` (ownership) and views to
/// `outViews`; the caller must keep `outBuffers` alive as long as the views /
/// any hash_join built from them are used (the build key table is referenced
/// by cudf::hash_join for its lifetime).
void extractKeyColumns(
    const GpuFixedRowStore& store,
    const std::vector<FieldDesc>& fields,
    const std::vector<cudf::size_type>& keyColIndices,
    int64_t numRows,
    rmm::cuda_stream_view stream,
    std::vector<rmm::device_buffer>& outBuffers,
    std::vector<cudf::column_view>& outViews) {
  for (auto colIdx : keyColIndices) {
    VELOX_CHECK(
        fields[colIdx].kind == kFieldFixed,
        "RowHashJoin: string join keys are not supported by the row path");
    int32_t keyOffset = fields[colIdx].offset;
    int32_t keyWidth = fields[colIdx].byte_width;
    rmm::device_buffer buf(numRows * keyWidth, stream);
    extractKeysFromRows(
        store, keyOffset, keyWidth, buf.data(), stream.value());
    // View captures buf.data() before the move; rmm move preserves the device
    // pointer, so the view stays valid once buf is moved into outBuffers.
    outViews.emplace_back(
        cudf::data_type{
            keyWidth == 8 ? cudf::type_id::INT64 : cudf::type_id::INT32},
        static_cast<cudf::size_type>(numRows),
        buf.data(), nullptr, 0);
    outBuffers.push_back(std::move(buf));
  }
}

/// Compute FieldDesc from a cudf::table_view's column types.
std::pair<std::vector<FieldDesc>, int32_t> computeRowLayoutFromTable(
    const cudf::table_view& table) {
  std::vector<FieldDesc> fields;
  int32_t offset = 0;
  for (int i = 0; i < table.num_columns(); i++) {
    FieldDesc fd;
    if (table.column(i).type().id() == cudf::type_id::STRING) {
      // Out-of-line strings (2026-08-21): 16B slot, 8-aligned.
      fd.byte_width = kRowStrSlotBytes;
      fd.kind = kFieldString;
      offset = alignUp(offset, 8);
    } else {
      fd.byte_width = cudfTypeWidth(table.column(i).type().id());
      offset = alignUp(offset, fd.byte_width);
    }
    fd.offset = offset;
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
    fd.byte_width = cudfTypeWidth(table.column(i).type().id());
    offset = alignUp(offset, fd.byte_width);
    fd.offset = offset;
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
  checkTransposeInputNullFree(table, "transposeToRowsSkip");

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

/// Compute a KEY-ONLY row layout: FieldDesc for just the key columns (in
/// keyColIndices order), packed contiguously. Gives the fingerprint verify a
/// row-native build key while the payload is fetched columnar.
std::pair<std::vector<FieldDesc>, int32_t> computeKeyRowLayout(
    const cudf::table_view& table,
    const std::vector<cudf::size_type>& keyColIndices) {
  std::vector<FieldDesc> fields;
  int32_t offset = 0;
  for (auto idx : keyColIndices) {
    FieldDesc fd;
    fd.byte_width = cudfTypeWidth(table.column(idx).type().id());
    offset = alignUp(offset, fd.byte_width);
    fd.offset = offset;
    fields.push_back(fd);
    offset += fd.byte_width;
  }
  int32_t rowWidth = (offset + 7) & ~7;
  return {fields, rowWidth};
}

/// Transpose ONLY the key columns (keyColIndices order) into a contiguous
/// key-row buffer laid out per `fields`.
void transposeKeyColumnsToRows(
    const cudf::table_view& table,
    const std::vector<cudf::size_type>& keyColIndices,
    const std::vector<FieldDesc>& fields,
    int32_t rowWidth,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  int32_t numRows = table.num_rows();
  int32_t numCols = static_cast<int32_t>(keyColIndices.size());
  if (numRows == 0 || numCols == 0) return;
  checkTransposeInputNullFree(table, "transposeKeyColumnsToRows");
  std::vector<const uint8_t*> colPtrs;
  colPtrs.reserve(numCols);
  for (auto idx : keyColIndices) {
    colPtrs.push_back(static_cast<const uint8_t*>(table.column(idx).head()));
  }
  columnsToRows(
      colPtrs.data(), fields.data(), numCols, numRows, rowWidth, d_row_buffer,
      stream);
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

  // Dedicated build stream (see DedicatedStream.h): the build's
  // synchronize() must not wait behind other drivers' pooled work.
  auto stream = acquireDedicatedStream();

  // EMPTY build (no batch reached any build driver): publish an empty
  // BuildData so the probe short-circuits to zero matches instead of
  // indexing inputs_[0] on an empty vector.
  if (inputs_.empty()) {
    RowHashJoinBridge::BuildData bd;
    bd.numRows = 0;
    bd.rowWidth = 0;
    bd.boundaryHybrid = CudfConfig::getInstance().benchmarkBoundaryHybrid;
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, planNodeId());
    auto rowBridge = std::dynamic_pointer_cast<RowHashJoinBridge>(joinBridge);
    VELOX_CHECK_NOT_NULL(rowBridge);
    rowBridge->setBuildStream(stream);
    rowBridge->setBuildData(
        std::make_shared<RowHashJoinBridge::BuildData>(std::move(bd)));
    return;
  }

  // Check if inputs are RowStoreVectors (already in row layout on GPU)
  auto firstRowStore = std::dynamic_pointer_cast<RowStoreVector>(inputs_[0]);

  int64_t numRows = 0;
  int32_t rowWidth = 0;
  std::vector<FieldDesc> fields;
  rmm::device_buffer rowBuffer;
  rmm::device_buffer fieldsBuffer;
  // Null sidecar for the combined build store (2026-08-17 null support).
  rmm::device_buffer buildNullBuffer;
  rmm::device_buffer buildCharsBuffer; // out-of-line string heap
  std::vector<std::string> buildFieldNames; // pruned-pack names (by-name keys)
  int32_t buildNullStride = 0;
  std::vector<rmm::device_buffer> keyBuffers;
  std::shared_ptr<cudf::hash_join> hashJoin;

  // ---- Row-native N:M matcher (CudfConfig::benchmarkRowTable) ----
  // Build the chained multimap DIRECTLY over the key rows of the row store
  // (keys-only store under boundary-hybrid; full row store otherwise). In
  // this mode neither extractKeyColumns nor cudf::hash_join runs at all:
  // keyBuffers stay empty and hashJoin stays null.
  const bool useRowTable = CudfConfig::getInstance().benchmarkRowTable;
  RowNativeHashTable rowTable{};
  rmm::device_buffer rowTableHeads;
  rmm::device_buffer rowTableNextFp;
  rmm::device_buffer rowTablePackedKeys;
  bool hasRowTable = false;
  auto buildRowTable = [&](const GpuFixedRowStore& store,
                           const std::vector<FieldDesc>& storeFields,
                           const std::vector<cudf::size_type>& keyIdx) {
    VELOX_CHECK_LE(
        keyIdx.size(),
        static_cast<size_t>(kRowTableMaxKeys),
        "row-table matcher: too many join key columns");
    RowKeyLayout kl{};
    kl.numKeys = static_cast<int32_t>(keyIdx.size());
    for (int32_t k = 0; k < kl.numKeys; k++) {
      kl.offset[k] = storeFields[keyIdx[k]].offset;
      kl.width[k] = storeFields[keyIdx[k]].byte_width;
      VELOX_CHECK(
          kl.width[k] == 4 || kl.width[k] == 8,
          "row-table matcher: fixed-width 4/8-byte keys only");
    }
    const int32_t cap = rowTableCapacity(store.num_rows);
    rowTableHeads = rmm::device_buffer(
        static_cast<size_t>(cap) * sizeof(int32_t), stream);
    rowTableNextFp = rmm::device_buffer(
        static_cast<size_t>(std::max<int32_t>(store.num_rows, 1)) *
            sizeof(uint64_t),
        stream);
    rowTable.heads = static_cast<int32_t*>(rowTableHeads.data());
    rowTable.nextFp = static_cast<uint64_t*>(rowTableNextFp.data());
    rowTable.capacity = cap;
    rowTable.numRows = store.num_rows;
    rowTable.buildStore = store;
    rowTable.keys = kl;

    // ---- Composite-key range packing (benchmarkRowTablePackKeys) ----
    // One min/max pass over the build keys, one 2K-word D2H at build finalize
    // (the only sync this adds, once per join). Pack engages iff the summed
    // per-key bit-widths fit one uint64; otherwise the table runs unpacked.
    RowKeyPack pack{};
    if (CudfConfig::getInstance().benchmarkRowTablePackKeys &&
        kl.numKeys >= 2 && store.num_rows > 0) {
      const size_t mmWords = 2 * static_cast<size_t>(kl.numKeys);
      rmm::device_buffer dMinmax(mmWords * sizeof(uint64_t), stream);
      auto* mm = static_cast<uint64_t*>(dMinmax.data());
      // mins pre-filled with 0xFF (UINT64_MAX), maxs with 0x00.
      cudaMemsetAsync(
          mm, 0xFF, static_cast<size_t>(kl.numKeys) * sizeof(uint64_t),
          stream.value());
      cudaMemsetAsync(
          mm + kl.numKeys, 0x00,
          static_cast<size_t>(kl.numKeys) * sizeof(uint64_t), stream.value());
      rowTableComputeKeyMinMax(store, kl, mm, stream.value());
      std::vector<uint64_t> hostMinmax(mmWords);
      cudaMemcpyAsync(
          hostMinmax.data(), mm, mmWords * sizeof(uint64_t),
          cudaMemcpyDeviceToHost, stream.value());
      stream.synchronize();
      int32_t totalBits = 0;
      for (int32_t k = 0; k < kl.numKeys; k++) {
        const uint64_t mn = hostMinmax[k];
        const uint64_t mx = hostMinmax[kl.numKeys + k];
        const uint64_t span = mx - mn;
        pack.mins[k] = mn;
        pack.spans[k] = span;
        // Cap at 63: a zero-span key contributes 0 bits and rel is always 0,
        // but a shift amount of 64 would be UB.
        pack.shifts[k] = std::min<int32_t>(totalBits, 63);
        totalBits += span == 0 ? 0 : 64 - __builtin_clzll(span);
      }
      if (totalBits <= 64) {
        pack.enabled = 1;
        // <= 32 bits: the fingerprint IS the packed key -> verify-free walk,
        // no stored-key array needed. Otherwise store packedKeys[numRows] so
        // verify is one dense 8-byte load instead of K strided row loads.
        pack.exactFp = totalBits <= 32 ? 1 : 0;
        if (!pack.exactFp) {
          rowTablePackedKeys = rmm::device_buffer(
              static_cast<size_t>(store.num_rows) * sizeof(uint64_t), stream);
          rowTable.packedKeys =
              static_cast<const uint64_t*>(rowTablePackedKeys.data());
        }
        addRuntimeStat(
            "rowTablePackedKeyBits", RuntimeCounter(totalBits));
      }
    }
    rowTable.pack = pack;

    buildRowNativeHashTable(rowTable, stream.value());
    hasRowTable = true;
  };
  // Matcher-build phase timing (row table build vs extract+cudf::hash_join),
  // reported as "matcherBuildWallNanos" when benchmarkLogGatherTime is set.
  cudaEvent_t mbStart, mbEnd;
  const bool timeMatcherBuild =
      CudfConfig::getInstance().benchmarkLogGatherTime;
  bool matcherBuildTimed = false;
  auto matcherBuildBegin = [&]() {
    if (timeMatcherBuild) {
      cudaEventCreate(&mbStart);
      cudaEventCreate(&mbEnd);
      cudaEventRecord(mbStart, stream.value());
      matcherBuildTimed = true;
    }
  };
  auto matcherBuildEnd = [&]() {
    if (matcherBuildTimed) {
      cudaEventRecord(mbEnd, stream.value());
      cudaEventSynchronize(mbEnd);
      float ms = 0;
      cudaEventElapsedTime(&ms, mbStart, mbEnd);
      addRuntimeStat(
          "matcherBuildWallNanos",
          RuntimeCounter(
              static_cast<int64_t>(ms * 1e6), RuntimeCounter::Unit::kNanos));
      cudaEventDestroy(mbStart);
      cudaEventDestroy(mbEnd);
      matcherBuildTimed = false;
    }
  };

  // ---- Boundary-hybrid build (CudfConfig::benchmarkBoundaryHybrid) ----
  // The upstream CudfFromVelox packed ONLY the join-key columns to the GPU
  // (keys-only row layout, join-key order) and attached the full host
  // batches to each RowStoreVector. Here the keys concatenate exactly like
  // the normal RowStoreVector path (so cudf::hash_join's build row ids are
  // global over the concatenation), while the payload batches are fed — in
  // the SAME order — into a host-side BoundaryHostStore whose prefix sums
  // invert global row id -> (batchId, rowInBatch). Build payload bytes never
  // touch PCIe.
  const bool boundaryBuild =
      firstRowStore != nullptr && firstRowStore->boundaryKeysOnly();
  std::shared_ptr<BoundaryHostStore> boundaryStore;

  if (firstRowStore) {
    // ---- RowStoreVector path: concatenate GPU row buffers ----
    rowWidth = firstRowStore->rowWidth();
    fields = firstRowStore->hostFields();

    // Out-of-line strings: the combined store gets ONE heap; each chunk's
    // heap is appended and its slots rebased by the chunk's heap base.
    int64_t totalCharsBuild = 0;
    for (auto& inp : inputs_) {
      if (auto r = std::dynamic_pointer_cast<RowStoreVector>(inp)) {
        totalCharsBuild += r->charsBytes();
      }
    }
    rmm::device_buffer strFieldOffsetsDev;
    int32_t numStrFields = 0;
    if (totalCharsBuild > 0) {
      buildCharsBuffer = rmm::device_buffer(totalCharsBuild, stream);
      auto strOffs = firstRowStore->stringFieldOffsets();
      numStrFields = static_cast<int32_t>(strOffs.size());
      strFieldOffsetsDev = rmm::device_buffer(
          strOffs.data(), strOffs.size() * sizeof(int32_t), stream);
    }
    int64_t charsOffset = 0;

    // CROSS-STREAM ORDERING (2026-08-21 fix): each input row store was
    // filled by an H2D copy on ITS PRODUCER'S stream (CudfFromVelox's
    // per-operator row stream, async mode on the build side). The D2D
    // concatenation below runs on THIS build's stream, so without an
    // explicit dependency it can read buffers whose H2D is still in
    // flight: right row counts, partially stale contents -> lost matches,
    // nondeterministic, scaling with batch size and driver count (found at
    // SF30 / 24 drivers / 100K batches; invisible at SF1). Make the build
    // stream wait on every distinct producer stream via events.
    {
      std::vector<cudaStream_t> waited;
      for (auto& inp : inputs_) {
        auto r = std::dynamic_pointer_cast<RowStoreVector>(inp);
        if (r == nullptr) {
          continue;
        }
        cudaStream_t ps = r->stream().value();
        if (ps == stream.value() ||
            std::find(waited.begin(), waited.end(), ps) != waited.end()) {
          continue;
        }
        waited.push_back(ps);
        cudaEvent_t ev;
        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, ps);
        cudaStreamWaitEvent(stream.value(), ev, 0);
        cudaEventDestroy(ev);
      }
    }

    // Compute total rows
    for (auto& inp : inputs_) {
      numRows += inp->size();
    }

    // Null sidecar (2026-08-17): if ANY input carries one, the combined
    // store needs one; inputs without contribute zeroed (all-valid) bytes.
    for (auto& inp : inputs_) {
      if (auto r = std::dynamic_pointer_cast<RowStoreVector>(inp)) {
        if (r->nullStride() > 0) {
          VELOX_CHECK(
              buildNullStride == 0 || buildNullStride == r->nullStride(),
              "row store null strides disagree across build batches");
          buildNullStride = r->nullStride();
        }
      }
    }
    if (buildNullStride > 0) {
      buildNullBuffer =
          rmm::device_buffer(numRows * buildNullStride, stream);
    }

    // Allocate combined buffer and copy each chunk
    int64_t totalBytes = numRows * rowWidth;
    rowBuffer = rmm::device_buffer(totalBytes, stream);
    int64_t offset = 0;
    int64_t nullOffset = 0;
    for (auto& inp : inputs_) {
      auto rsv = std::dynamic_pointer_cast<RowStoreVector>(inp);
      VELOX_CHECK_NOT_NULL(rsv);
      int64_t chunkBytes = rsv->gpuRowBytes();
      if (chunkBytes > 0) {
        cudaMemcpyAsync(
            static_cast<uint8_t*>(rowBuffer.data()) + offset,
            rsv->gpuRowData(), chunkBytes,
            cudaMemcpyDeviceToDevice, stream.value());
        if (rsv->charsBytes() > 0) {
          cudaMemcpyAsync(
              static_cast<uint8_t*>(buildCharsBuffer.data()) + charsOffset,
              rsv->charsData(), rsv->charsBytes(),
              cudaMemcpyDeviceToDevice, stream.value());
          rebaseStringOffsets(
              static_cast<uint8_t*>(rowBuffer.data()) + offset,
              rsv->size(), rowWidth,
              static_cast<const int32_t*>(strFieldOffsetsDev.data()),
              numStrFields, charsOffset, stream.value());
          charsOffset += rsv->charsBytes();
        }
        offset += chunkBytes;
      }
      if (buildNullStride > 0 && rsv->size() > 0) {
        const int64_t nb =
            static_cast<int64_t>(rsv->size()) * buildNullStride;
        if (rsv->nullStride() > 0) {
          cudaMemcpyAsync(
              static_cast<uint8_t*>(buildNullBuffer.data()) + nullOffset,
              rsv->gpuRowData() + rsv->gpuRowBytes(),
              nb, cudaMemcpyDeviceToDevice, stream.value());
        } else {
          cudaMemsetAsync(
              static_cast<uint8_t*>(buildNullBuffer.data()) + nullOffset,
              0, nb, stream.value());
        }
        nullOffset += nb;
      }
      if (boundaryBuild) {
        // Host retention IN GPU-CONCAT ORDER: this loop appends host batches
        // in exactly the order their key rows were appended above, which is
        // what makes hostStore->idForGlobalRow() the inverse of the GPU's
        // global build row id.
        VELOX_CHECK(
            rsv->boundaryKeysOnly(),
            "boundary-hybrid build: mixed keys-only and full-row inputs");
        const auto& hostBatches = rsv->boundaryHostBatches();
        VELOX_CHECK(
            rsv->size() == 0 || !hostBatches.empty(),
            "boundary-hybrid build: input lost its host payload");
        if (boundaryStore == nullptr && !hostBatches.empty()) {
          auto hostType =
              std::dynamic_pointer_cast<const RowType>(hostBatches[0]->type());
          VELOX_CHECK_NOT_NULL(hostType);
          boundaryStore =
              std::make_shared<BoundaryHostStore>(hostType, pool());
        }
        for (const auto& hb : hostBatches) {
          boundaryStore->addBatch(hb);
        }
        rsv->clearBoundaryPayload(); // store holds the refs now
      }
    }
    if (boundaryBuild && boundaryStore != nullptr) {
      VELOX_CHECK_EQ(
          boundaryStore->totalRows(),
          numRows,
          "boundary-hybrid build: host retention rows != GPU key rows");
    }
    inputs_.clear();

    // Upload field descriptors
    fieldsBuffer = rmm::device_buffer(
        fields.data(), fields.size() * sizeof(FieldDesc), stream);

    // RowStoreVector path: keys are embedded in rows, need to extract.
    auto rightKeys = joinNode_->rightKeys();
    auto rightType = joinNode_->sources()[1]->outputType();
    std::vector<cudf::size_type> keyColIndices;
    buildFieldNames = firstRowStore->fieldNames();
    if (boundaryBuild) {
      // Keys-only layout: field k IS rightKeys[k] (CudfFromVelox packed the
      // key columns in join-key order), so the key indices are simply 0..K-1
      // — NOT the child indices in rightType.
      VELOX_CHECK_EQ(
          fields.size(),
          rightKeys.size(),
          "boundary-hybrid build: keys-only layout width mismatch");
      for (size_t k = 0; k < rightKeys.size(); k++) {
        keyColIndices.push_back(static_cast<cudf::size_type>(k));
      }
    } else if (!buildFieldNames.empty()) {
      // Pruned subset pack: resolve keys BY NAME against the store's own
      // field names (channel indices of the full type do not apply).
      for (auto& k : rightKeys) {
        const auto fi = firstRowStore->fieldIndexOf(k->name());
        VELOX_CHECK_GE(
            fi, 0, "row build: key '{}' not in pruned pack", k->name());
        keyColIndices.push_back(static_cast<cudf::size_type>(fi));
      }
    } else {
      for (auto& k : rightKeys) {
        keyColIndices.push_back(
            static_cast<cudf::size_type>(rightType->getChildIdx(k->name())));
      }
    }

    GpuFixedRowStore tmpStore;
    tmpStore.row_buffer = static_cast<uint8_t*>(rowBuffer.data());
    tmpStore.row_width = rowWidth;
    tmpStore.num_rows = numRows;
    tmpStore.num_fields = fields.size();
    tmpStore.fields = static_cast<const FieldDesc*>(fieldsBuffer.data());

    matcherBuildBegin();
    if (useRowTable) {
      // Row-native matcher: chained multimap straight over the key rows.
      // No extract_keys, no cudf::hash_join.
      buildRowTable(tmpStore, fields, keyColIndices);
    } else {
      std::vector<cudf::column_view> keyViews;
      extractKeyColumns(
          tmpStore, fields, keyColIndices, numRows, stream,
          keyBuffers, keyViews);

      auto keyTable = cudf::table_view(keyViews);
      hashJoin = std::make_shared<cudf::hash_join>(
          keyTable, cudf::null_equality::UNEQUAL, stream);
    }
    matcherBuildEnd();
  } else {
    // ---- CudfVector path: concatenate + transpose ----
    // Per-phase GPU timing (VELOX_CUDF_BUILD_PROFILE): stamp the stream at each phase
    // boundary; one sync at the end reads all deltas. Answers "where does the
    // build wall go" -- concat vs transpose(all cols) vs key-extract vs map-build.
    const bool profBuild = (std::getenv("VELOX_CUDF_BUILD_PROFILE") != nullptr);
    cudaEvent_t pe[6];
    if (profBuild) {
      for (auto& e : pe) {
        cudaEventCreate(&e);
      }
    }
    auto stamp = [&](int i) {
      if (profBuild) {
        cudaEventRecord(pe[i], stream.value());
      }
    };
    stamp(0);

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
    stamp(1);

    // Identify key columns (composite keys supported)
    auto rightKeys = joinNode_->rightKeys();
    auto rightType = joinNode_->sources()[1]->outputType();
    std::vector<cudf::size_type> keyColIndices;
    for (auto& k : rightKeys) {
      keyColIndices.push_back(
          static_cast<cudf::size_type>(rightType->getChildIdx(k->name())));
    }

      auto [f, rw] = computeRowLayoutFromTable(buildView);
      fields = std::move(f);
      rowWidth = rw;

      int64_t rowBytes = numRows * rowWidth;
      rowBuffer = rmm::device_buffer(rowBytes, stream);

      buildCharsBuffer = transposeToRows(
          buildView, fields, rowWidth,
          static_cast<uint8_t*>(rowBuffer.data()), stream.value());
      stamp(2);

      fieldsBuffer = rmm::device_buffer(
          fields.data(), fields.size() * sizeof(FieldDesc), stream);

      // Extract keys from rows for hash table construction
      GpuFixedRowStore tmpStore;
      tmpStore.row_buffer = static_cast<uint8_t*>(rowBuffer.data());
      tmpStore.row_width = rowWidth;
      tmpStore.num_rows = numRows;
      tmpStore.num_fields = fields.size();
      tmpStore.fields = static_cast<const FieldDesc*>(fieldsBuffer.data());

      if (useRowTable) {
        // Row-native matcher: build straight from the transposed row store;
        // no key extraction, no cudf::hash_join.
        matcherBuildBegin();
        buildRowTable(tmpStore, fields, keyColIndices);
        matcherBuildEnd();
        stamp(3);
        stamp(4);
        stamp(5);
      } else {
        matcherBuildBegin();
        std::vector<cudf::column_view> keyViews;
        extractKeyColumns(
            tmpStore, fields, keyColIndices, numRows, stream,
            keyBuffers, keyViews);
        stamp(3);

        auto keyTable = cudf::table_view(keyViews);
        hashJoin = std::make_shared<cudf::hash_join>(
            keyTable, cudf::null_equality::UNEQUAL, stream);
        matcherBuildEnd();
        stamp(4);
        stamp(5);
      }
    if (profBuild) {
      cudaEventSynchronize(pe[5]);
      float tConcat = 0, tTranspose = 0, tExtract = 0, tHashJoin = 0, tFused = 0;
      cudaEventElapsedTime(&tConcat, pe[0], pe[1]);
      cudaEventElapsedTime(&tTranspose, pe[1], pe[2]);
      cudaEventElapsedTime(&tExtract, pe[2], pe[3]);
      cudaEventElapsedTime(&tHashJoin, pe[3], pe[4]);
      cudaEventElapsedTime(&tFused, pe[4], pe[5]);
      fprintf(
          stderr,
          "[BUILD_PROFILE] nRows=%ld cols=%d rowWidth=%d "
          "| concat=%.2f transpose=%.2f extractKeys=%.2f "
          "hashJoin=%.2f (unused)=%.2f | GPUtotal=%.2f ms\n",
          (long)numRows,
          (int)fields.size(),
          (int)rowWidth,
          tConcat,
          tTranspose,
          tExtract,
          tHashJoin,
          tFused,
          tConcat + tTranspose + tExtract + tHashJoin + tFused);
      for (auto& e : pe) {
        cudaEventDestroy(e);
      }
    }
  }

  // Build GpuFixedRowStore handle
  GpuFixedRowStore gpuStore;
  gpuStore.row_buffer = static_cast<uint8_t*>(rowBuffer.data());
  gpuStore.row_width = rowWidth;
  gpuStore.num_rows = numRows;
  gpuStore.num_fields = fields.size();
  gpuStore.fields = static_cast<const FieldDesc*>(fieldsBuffer.data());
  if (buildNullStride > 0) {
    gpuStore.null_bytes =
        static_cast<const uint8_t*>(buildNullBuffer.data());
    gpuStore.null_stride = buildNullStride;
  }
  if (buildCharsBuffer.size() > 0) {
    gpuStore.chars = static_cast<const uint8_t*>(buildCharsBuffer.data());
    gpuStore.chars_bytes = static_cast<int64_t>(buildCharsBuffer.size());
  }
  stream.synchronize();

  // Push to bridge
  RowHashJoinBridge::BuildData bd;
  bd.gpuRowStore = gpuStore;
  bd.rowBuffer = std::move(rowBuffer);
  bd.fieldsBuffer = std::move(fieldsBuffer);
  bd.hashJoin = std::move(hashJoin);
  bd.keyBuffers = std::move(keyBuffers);
  bd.numRows = numRows;
  bd.rowWidth = rowWidth;
  bd.hostFields = std::move(fields);
  bd.hostFieldNames = std::move(buildFieldNames);
  bd.nullBuffer = std::move(buildNullBuffer);
  bd.nullStride = buildNullStride;
  bd.charsBuffer = std::move(buildCharsBuffer);
  // Boundary-hybrid: hand the probe the host payload store; gpuRowStore then
  // holds keys only and is used solely for hash-table construction above.
  bd.boundaryHybrid = boundaryBuild;
  bd.hostStore = std::move(boundaryStore);
  // Row-native matcher: hand the probe the chained multimap (device pointers
  // stay valid — rmm::device_buffer moves preserve the allocation).
  bd.hasRowTable = hasRowTable;
  bd.rowTable = rowTable;
  bd.rowTableHeads = std::move(rowTableHeads);
  bd.rowTableNextFp = std::move(rowTableNextFp);
  bd.rowTablePackedKeys = std::move(rowTablePackedKeys);

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

  // Host-side phase timers (only emitted under benchmarkLogGatherTime):
  // prep (input/layout) -> matcher (incl. count readback) -> gather (incl.
  // string compaction + sync) -> output construction (row store or
  // row->col + strings column). Sum == this call's wall.
  const auto tpEnter = std::chrono::steady_clock::now();
  auto hostNanos = [](auto a, auto b) {
    return static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
  };

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

  if (bd.numRows == 0) {
    // Empty build: inner join yields nothing. Consume the batch.
    input_.reset();
    finished_ = noMoreInput_;
    return nullptr;
  }

  // ---- Boundary-hybrid probe path ----
  // Keys-only input + host-retained payload: GPU does keys-in / id-pairs-out
  // only; the output RowVector is materialized on the CPU. See
  // boundaryProbe() for the full flow.
  if (bd.boundaryHybrid) {
    auto boundaryInput = std::dynamic_pointer_cast<RowStoreVector>(input_);
    VELOX_CHECK(
        boundaryInput != nullptr && boundaryInput->boundaryKeysOnly(),
        "boundary-hybrid probe requires a keys-only RowStoreVector input "
        "with host payload attached (produced by CudfFromVelox's keys-only "
        "pinned pack)");
    auto out = boundaryProbe(boundaryInput, boundaryInput->stream());
    boundaryInput.reset();
    input_.reset();
    finished_ = noMoreInput_;
    if (logTimeline && driverId == 0) {
      printf("OPTRACE %lld d0 p%d RowJoinProbe end %lld\n",
             (long long)timeNow(), pipelineId,
             (long long)(out ? out->size() : 0));
    }
    return out;
  }

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

    // Batch-level adaptive deferral: the upstream pack may switch layouts
    // mid-query (pruned-eager -> crossing-set). Detect the new signature
    // and recompute both the input capture and the output layout. The
    // switch is one-way, so this fires at most once per operator.
    if (initialized_ &&
        (rowStoreInput->rowWidth() != probeRowWidth_ ||
         rowStoreInput->fieldNames() != probeFieldNames_)) {
      initialized_ = false;
      fieldsUploaded_ = false;
      outputLayoutComputed_ = false;
      outputFields_.clear();
      outputStringFields_.clear();
      deferredOutputCols_.clear();
      probeGatherMappings_.clear();
      buildGatherMappings_.clear();
      outputRowIdField_ = -1;
      addRuntimeStat("probeLayoutSwitch", RuntimeCounter(1));
    }
    if (!initialized_) {
      probeFields_ = rowStoreInput->hostFields();
      probeRowWidth_ = rowStoreInput->rowWidth();
      // Spine deferral v2: a crossing-set input records its column names;
      // fields (incl. join keys) resolve BY NAME against them, and the
      // hidden __rowid field indexes the batch's retained host rows.
      probeFieldNames_ = rowStoreInput->fieldNames();
      if (!probeFieldNames_.empty()) {
        std::vector<cudf::size_type> remapped;
        for (const auto& key : joinNode_->leftKeys()) {
          const auto fi = rowStoreInput->fieldIndexOf(key->name());
          VELOX_CHECK_GE(
              fi, 0, "spine probe: key '{}' not in crossing set", key->name());
          remapped.push_back(static_cast<cudf::size_type>(fi));
        }
        leftKeyIndices_ = std::move(remapped);
      }
      initialized_ = true;
    }
    // Per-batch provenance (each emitted pack batch has its own store).
    probeProvStore_ = rowStoreInput->provenanceStore();
    probeRowIdField_ = rowStoreInput->rowIdField();

    // Upload fields descriptor if not done
    if (!fieldsUploaded_) {
      probeFieldsBuffer_ = rmm::device_buffer(
          probeFields_.data(), probeFields_.size() * sizeof(FieldDesc), stream);
      fieldsUploaded_ = true;
    }

    probeGpuStore.row_buffer = static_cast<uint8_t*>(rowStoreInput->gpuRowData());
    probeGpuStore.row_width = probeRowWidth_;
    probeGpuStore.num_rows = probeRows;
    probeGpuStore.num_fields = probeFields_.size();
    probeGpuStore.fields = static_cast<const FieldDesc*>(probeFieldsBuffer_.data());
    if (rowStoreInput->charsBytes() > 0) {
      probeGpuStore.chars = rowStoreInput->charsData();
      probeGpuStore.chars_bytes = rowStoreInput->charsBytes();
    }

    addRuntimeStat(
        "probeTransposeSkipped",
        RuntimeCounter(static_cast<int64_t>(probeRows)));
  } else {
    // ---- Path B: CudfVector (transpose all columns to rows, extract key) ----
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
    VELOX_CHECK_NOT_NULL(cudfInput, "RowHashJoinProbe: input must be RowStoreVector or CudfVector");
    stream = cudfInput->stream();
    auto probeView = cudfInput->getTableView();
    probeRows = probeView.num_rows();
    // No provenance on a GPU-columnar input.
    probeProvStore_.reset();
    probeRowIdField_ = -1;

    if (!initialized_) {
      // Include ALL columns in row layout (key embedded)
      auto [pf, prw] = computeRowLayoutFromTable(probeView);
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

    // Transpose ALL columns (including key) to rows
    probeCharsBuffer_ = transposeToRows(
        probeView, probeFields_, probeRowWidth_,
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
    if (probeCharsBuffer_.size() > 0) {
      probeGpuStore.chars =
          static_cast<const uint8_t*>(probeCharsBuffer_.data());
      probeGpuStore.chars_bytes =
          static_cast<int64_t>(probeCharsBuffer_.size());
    }
  }

  // ---- Compute output row layout (once) ----
  // Determines which fields from probe/build appear in the query output,
  // and their destination offsets. Column order matches outputType.
  if (!outputLayoutComputed_) {
    auto outType = joinNode_->outputType();
    auto leftType = joinNode_->sources()[0]->outputType();
    auto rightType = joinNode_->sources()[1]->outputType();
    const auto& buildHostFields = buildData_->hostFields;

    // Per-mapping FIELD indices for the null sidecar copy (2026-08-17).
    std::vector<int32_t> pSrcField, pDstField, bSrcField, bDstField;
    int32_t dstOffset = 0;
    for (int i = 0; i < outType->size(); i++) {
      auto name = outType->nameOf(i);
      // Probe-side resolution: BY NAME against the crossing set when the
      // input is a subset pack (spine deferral v2), else by child index.
      std::optional<uint32_t> leftIdx;
      if (!probeFieldNames_.empty()) {
        for (size_t f = 0; f < probeFieldNames_.size(); f++) {
          if (probeFieldNames_[f] == name) {
            leftIdx = static_cast<uint32_t>(f);
            break;
          }
        }
        // A probe-side column NOT in the crossing set is DEFERRED: it will
        // be materialized from the provenance store, not gathered on GPU.
        if (!leftIdx.has_value() &&
            leftType->getChildIdxIfExists(name).has_value()) {
          VELOX_CHECK_NOT_NULL(
              probeProvStore_,
              "spine probe: column '{}' absent from crossing set and no "
              "provenance store",
              name);
          deferredOutputCols_.push_back(i);
          outputFields_.push_back({-1, 0, kFieldFixed}); // placeholder
          continue;
        }
      } else {
        leftIdx = leftType->getChildIdxIfExists(name);
      }
      int32_t byteWidth;
      int32_t kind;
      if (leftIdx.has_value()) {
        int srcCol = leftIdx.value();
        byteWidth = probeFields_[srcCol].byte_width;
        kind = probeFields_[srcCol].kind;
        if (kind == kFieldString) {
          dstOffset = alignUp(dstOffset, 8);
        }
        probeGatherMappings_.push_back(
            {probeFields_[srcCol].offset, dstOffset, byteWidth});
        pSrcField.push_back(srcCol);
        pDstField.push_back(i);
      } else {
        int srcCol = -1;
        if (!buildData_->hostFieldNames.empty()) {
          for (size_t f = 0; f < buildData_->hostFieldNames.size(); f++) {
            if (buildData_->hostFieldNames[f] == name) {
              srcCol = static_cast<int>(f);
              break;
            }
          }
          VELOX_CHECK_GE(
              srcCol, 0, "row join: output column '{}' not packed", name);
        } else {
          srcCol = rightType->getChildIdx(name);
        }
        byteWidth = buildHostFields[srcCol].byte_width;
        kind = buildHostFields[srcCol].kind;
        if (kind == kFieldString) {
          dstOffset = alignUp(dstOffset, 8);
        }
        buildGatherMappings_.push_back(
            {buildHostFields[srcCol].offset, dstOffset, byteWidth});
        bSrcField.push_back(srcCol);
        bDstField.push_back(i);
      }
      FieldDesc ofd;
      ofd.offset = dstOffset;
      ofd.byte_width = byteWidth;
      ofd.kind = kind;
      outputFields_.push_back(ofd);
      if (kind == kFieldString) {
        // Slots are copied verbatim by the gather; compactStrings then
        // relocates their bytes from the side's heap (0 = probe, 1 =
        // build) into the output heap.
        outputStringFields_.push_back(
            {dstOffset, leftIdx.has_value() ? 0 : 1});
      }
      dstOffset += byteWidth;
    }
    // Spine deferral v2: chained/materialized outputs carry the hidden
    // __rowid as an extra trailing field (gathered like any 8B field; not
    // part of outputType).
    if (probeProvStore_ != nullptr) {
      dstOffset = alignUp(dstOffset, 8);
      outputRowIdField_ = static_cast<int32_t>(outputFields_.size());
      probeGatherMappings_.push_back(
          {probeFields_[probeRowIdField_].offset, dstOffset, 8});
      pSrcField.push_back(probeRowIdField_);
      pDstField.push_back(0); // rowid is never null; bit 0 never set by it
      outputFields_.push_back({dstOffset, 8, kFieldFixed});
      dstOffset += 8;
    }
    outputRowWidth_ = (dstOffset + 7) & ~7;
    if (!outputStringFields_.empty()) {
      outputStringFieldsBuffer_ = rmm::device_buffer(
          outputStringFields_.data(),
          outputStringFields_.size() * sizeof(StringGatherField), stream);
      std::vector<int32_t> strOffs;
      for (const auto& f : outputStringFields_) {
        strOffs.push_back(f.dst_offset);
      }
      outputStrOffsetsBuffer_ = rmm::device_buffer(
          strOffs.data(), strOffs.size() * sizeof(int32_t), stream);
    }

    // Upload field mappings to GPU (constant across all batches)
    if (!probeGatherMappings_.empty()) {
      probeGatherMappingsBuffer_ = rmm::device_buffer(
          probeGatherMappings_.data(),
          probeGatherMappings_.size() * sizeof(FieldMapping), stream);
    }
    if (!buildGatherMappings_.empty()) {
      buildGatherMappingsBuffer_ = rmm::device_buffer(
          buildGatherMappings_.data(),
          buildGatherMappings_.size() * sizeof(FieldMapping), stream);
    }
    if (!pSrcField.empty()) {
      probeGatherSrcFieldBuffer_ = rmm::device_buffer(
          pSrcField.data(), pSrcField.size() * sizeof(int32_t), stream);
      probeGatherDstFieldBuffer_ = rmm::device_buffer(
          pDstField.data(), pDstField.size() * sizeof(int32_t), stream);
    }
    if (!bSrcField.empty()) {
      buildGatherSrcFieldBuffer_ = rmm::device_buffer(
          bSrcField.data(), bSrcField.size() * sizeof(int32_t), stream);
      buildGatherDstFieldBuffer_ = rmm::device_buffer(
          bDstField.data(), bDstField.size() * sizeof(int32_t), stream);
    }

    addRuntimeStat("outputGatherProbeFields",
        RuntimeCounter(static_cast<int64_t>(probeGatherMappings_.size())));
    addRuntimeStat("outputGatherBuildFields",
        RuntimeCounter(static_cast<int64_t>(buildGatherMappings_.size())));
    addRuntimeStat("outputRowWidthBytes",
        RuntimeCounter(static_cast<int64_t>(outputRowWidth_)));

    // Upload output field descriptors to GPU (for RowStoreVector output)
    outputFieldsBuffer_ = rmm::device_buffer(
        outputFields_.data(),
        outputFields_.size() * sizeof(FieldDesc), stream);

    outputLayoutComputed_ = true;
  }

  // ---- 2+3. MATCHER: probe keys -> (probe, build) id pair lists ----
  // Key k lives at probeFields_[leftKeyIndices_[k]] in the full row layout.
  std::vector<FieldDesc> matcherKeyDescs;
  matcherKeyDescs.reserve(leftKeyIndices_.size());
  for (auto ki : leftKeyIndices_) {
    matcherKeyDescs.push_back(probeFields_[ki]);
  }
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndices;
  const auto tpMatcher = std::chrono::steady_clock::now();
  int32_t numMatches = runMatcher(
      probeGpuStore, matcherKeyDescs, probeRows, stream,
      leftIndices, rightIndices);
  const auto tpGather = std::chrono::steady_clock::now();

  // ---- Determine (once) whether this is the terminal row-mode join ----
  // A probe is terminal if the operator immediately downstream in the driver
  // pipeline is NOT another RowHashJoinProbe (i.e. it's a columnar cudf op
  // such as aggregation/orderBy). Terminal probes must emit CudfVector so the
  // downstream columnar operator can consume it; chained probes emit
  // RowStoreVector so the next row-join skips the col->row transpose.
  if (emitColumnar_ < 0) {
    emitColumnar_ = 1; // default: assume terminal (safe: always consumable)
    auto* driver = operatorCtx_->driver();
    if (driver != nullptr) {
      const auto ops = driver->operators();
      // Find self, then inspect the next operator in the pipeline.
      for (size_t i = 0; i + 1 < ops.size(); i++) {
        if (ops[i] == this) {
          if (dynamic_cast<RowHashJoinProbe*>(ops[i + 1]) != nullptr) {
            emitColumnar_ = 0; // next op is a row-join -> keep row layout
          } else if (dynamic_cast<RowHashJoinBuild*>(ops[i + 1]) != nullptr) {
            // Our output is the BUILD side of a later join (bushy plan).
            // The build takes ONE layout for all its inputs, so we hand
            // over rows only when this probe can never switch to deferred
            // (deferral off); under --boundary_hybrid a build-feeding probe
            // stays columnar (deferred batches must be materialized, and
            // the adaptive switch would otherwise mix rows and columns).
            nextIsBuild_ = true;
            emitColumnar_ =
                CudfConfig::getInstance().benchmarkBoundaryHybrid ? 1 : 0;
          } else if (dynamic_cast<CudfToVelox*>(ops[i + 1]) != nullptr) {
            // CPU exit: with provenance, emit a HOST RowVector directly
            // (CudfToVelox passes non-CudfVector inputs through) -- the
            // deferred payload never touches the GPU (paper case (a)).
            hostExit_ = true;
          }
          break;
        }
      }
    }
    addRuntimeStat(
        "rowJoinEmitColumnar",
        RuntimeCounter(static_cast<int64_t>(emitColumnar_)));
  }

  // Batch-level adaptive deferral: only the chain's TERMINAL probe reports
  // survivors, keyed by its own join id -- the same id the spine pack
  // resolved as the chain endpoint (DeferralPlan probe-side walk). A CPU
  // exit reports 0: with direct host emission, deferring is always right.
  // Recorded before the numMatches==0 early-return so empty batches count.
  if (emitColumnar_ == 1 || nextIsBuild_) {
    DeferralStats::instance().recordSurvived(
        joinNode_->id(), hostExit_ ? 0 : numMatches);
  }

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
    // Selective gather + concatenate: only gather columns that appear in
    // the output type, laid out in outputType column order.
    int64_t neededOutput = (int64_t)numMatches * outputRowWidth_;
    if (numMatches > gatherCapacity_) {
      gatherCapacity_ = numMatches;
      probeGatherBuffer_ = rmm::device_buffer(neededOutput, stream);
    } else if (neededOutput > static_cast<int64_t>(probeGatherBuffer_.size())) {
      probeGatherBuffer_ = rmm::device_buffer(neededOutput, stream);
    }

    // Null sidecar for the output (2026-08-17): needed iff either input
    // store carries one. Zeroed each batch; the gather ORs source bits in.
    uint8_t* outNullBytes = nullptr;
    if (probeGpuStore.null_stride > 0 || bd.gpuRowStore.null_stride > 0) {
      outputNullStride_ =
          (static_cast<int32_t>(outputType_->size()) + 7) / 8;
      const int64_t nb =
          static_cast<int64_t>(numMatches) * outputNullStride_;
      if (nb > static_cast<int64_t>(outputNullBuffer_.size())) {
        outputNullBuffer_ = rmm::device_buffer(nb, stream);
      }
      cudaMemsetAsync(outputNullBuffer_.data(), 0, nb, stream.value());
      outNullBytes = static_cast<uint8_t*>(outputNullBuffer_.data());
    } else {
      outputNullStride_ = 0;
    }

    selectiveGatherAndConcat(
        probeGpuStore,
        reinterpret_cast<const int32_t*>(leftIndices->data()),
        static_cast<const FieldMapping*>(probeGatherMappingsBuffer_.data()),
        static_cast<int32_t>(probeGatherMappings_.size()),
        bd.gpuRowStore,
        reinterpret_cast<const int32_t*>(rightIndices->data()),
        static_cast<const FieldMapping*>(buildGatherMappingsBuffer_.data()),
        static_cast<int32_t>(buildGatherMappings_.size()),
        numMatches,
        outputRowWidth_,
        static_cast<uint8_t*>(probeGatherBuffer_.data()),
        stream.value(),
        static_cast<const int32_t*>(probeGatherSrcFieldBuffer_.data()),
        static_cast<const int32_t*>(probeGatherDstFieldBuffer_.data()),
        static_cast<const int32_t*>(buildGatherSrcFieldBuffer_.data()),
        static_cast<const int32_t*>(buildGatherDstFieldBuffer_.data()),
        outNullBytes,
        outputNullStride_);

    // ---- Out-of-line string compaction ----
    // Size pass + scan gives each output row its heap base (one sync for
    // the total), then one kernel copies the bytes and rewrites offsets.
    outputCharsBuffer_ = rmm::device_buffer{};
    if (!outputStringFields_.empty() &&
        (probeGpuStore.chars_bytes > 0 || bd.gpuRowStore.chars_bytes > 0)) {
      const size_t baseBytes =
          (static_cast<size_t>(numMatches) + 1) * sizeof(int64_t);
      if (baseBytes > outputRowBaseBuffer_.size()) {
        outputRowBaseBuffer_ = rmm::device_buffer(baseBytes, stream);
      }
      const int64_t heapBytes = stringHeapLayout(
          static_cast<const uint8_t*>(probeGatherBuffer_.data()),
          numMatches,
          outputRowWidth_,
          static_cast<const int32_t*>(outputStrOffsetsBuffer_.data()),
          static_cast<int32_t>(outputStringFields_.size()),
          static_cast<int64_t*>(outputRowBaseBuffer_.data()),
          stream.value());
      if (heapBytes > 0) {
        outputCharsBuffer_ = rmm::device_buffer(heapBytes, stream);
        compactStrings(
            static_cast<uint8_t*>(probeGatherBuffer_.data()),
            numMatches,
            outputRowWidth_,
            static_cast<const StringGatherField*>(
                outputStringFieldsBuffer_.data()),
            static_cast<int32_t>(outputStringFields_.size()),
            probeGpuStore.chars,
            bd.gpuRowStore.chars,
            static_cast<const int64_t*>(outputRowBaseBuffer_.data()),
            static_cast<uint8_t*>(outputCharsBuffer_.data()),
            stream.value());
      }
    }
  }

  if (timeGather) {
    cudaEventRecord(gatherEnd, stream.value());
  }

  // NOTE: No stream.synchronize() here. The gather kernel above and the
  // input-buffer free below (rowStoreInput.reset()/input_.reset()) are ordered
  // on the same `stream`. The RMM memory resource in use (pool/async/arena/
  // managed_* — see createMemoryResource in GpuResources.cpp) is
  // stream-ordered: deallocate() returns the block to the freeing stream's
  // free list and any cross-stream reuse waits on a recorded event, so the
  // input buffer cannot be handed out until this stream's gather completes.
  // The sync is therefore redundant for correctness. We keep it only when
  // measuring gather time in isolation.
  if (timeGather) {
    stream.synchronize();
  }

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

  const auto tpOutput = std::chrono::steady_clock::now();
  struct PhaseReport {
    RowHashJoinProbe* op;
    bool on;
    std::chrono::steady_clock::time_point e, m, g, o;
    decltype(hostNanos) f;
    ~PhaseReport() {
      if (!on) return;
      auto now = std::chrono::steady_clock::now();
      op->addRuntimeStat("probePrepHostNanos",
          RuntimeCounter(f(e, m), RuntimeCounter::Unit::kNanos));
      op->addRuntimeStat("probeMatcherHostNanos",
          RuntimeCounter(f(m, g), RuntimeCounter::Unit::kNanos));
      op->addRuntimeStat("probeGatherHostNanos",
          RuntimeCounter(f(g, o), RuntimeCounter::Unit::kNanos));
      op->addRuntimeStat("probeOutputHostNanos",
          RuntimeCounter(f(o, now), RuntimeCounter::Unit::kNanos));
    }
  } phaseReport{this, timeGather, tpEnter, tpMatcher, tpGather, tpOutput,
                hostNanos};

  if (logTimeline && driverId == 0) {
    printf("OPTRACE %lld d0 p%d RowJoinProbe end %lld\n",
           (long long)timeNow(), pipelineId, (long long)numMatches);
  }

  if (numMatches == 0) {
    return nullptr;
  }

  // skip_output path: return dummy 1-row to indicate progress
  if (CudfConfig::getInstance().benchmarkSkipOutput) {
    std::vector<VectorPtr> children(outputType_->size());
    for (int i = 0; i < outputType_->size(); i++) {
      children[i] = BaseVector::createNullConstant(
          outputType_->childAt(i), 1, pool());
    }
    return std::make_shared<RowVector>(
        pool(), outputType_, nullptr, 1, std::move(children));
  }

  // ---- Terminal join: transpose row buffer -> cudf columns (CudfVector) ----
  // A probe feeding a BUILD emits rows for eager batches only.
  if (emitColumnar_ == 1 || (nextIsBuild_ && !deferredOutputCols_.empty())) {
    if (hostExit_ &&
        (probeProvStore_ != nullptr ||
         CudfConfig::getInstance().benchmarkRowOutputNative)) {
      return makeHostOutput(numMatches, stream);
    }
    return makeColumnarOutput(numMatches, stream);
  }

  // Normal path: return a RowStoreVector wrapping the gathered output buffer.
  // The buffer layout matches outputType column order (described by outputFields_).
  // This is structurally correct for chained joins — downstream operators
  // can consume it as RowStoreVector and index fields by outputType column order.

  // Clone the output fields buffer (each RowStoreVector needs its own copy
  // since the probe operator reuses outputFieldsBuffer_ across batches)
  rmm::device_buffer outFieldsBuf(
      outputFieldsBuffer_.data(),
      outputFieldsBuffer_.size(), stream);

  // Transfer ownership of the gathered row data to the output vector.
  // Allocate a fresh probeGatherBuffer_ for the next batch.
  rmm::device_buffer outputRowData = std::move(probeGatherBuffer_);
  gatherCapacity_ = 0;  // force realloc on next batch
  if (outputNullStride_ > 0) {
    // RowStoreVector expects the sidecar right after the rows; the gather
    // wrote it to a separate buffer, so repack (rows + nulls) for the
    // chained consumer. Only taken when a nullable column is present.
    const int64_t rb = static_cast<int64_t>(numMatches) * outputRowWidth_;
    const int64_t nb = static_cast<int64_t>(numMatches) * outputNullStride_;
    rmm::device_buffer packed(rb + nb, stream);
    cudaMemcpyAsync(packed.data(), outputRowData.data(), rb,
        cudaMemcpyDeviceToDevice, stream.value());
    cudaMemcpyAsync(static_cast<uint8_t*>(packed.data()) + rb,
        outputNullBuffer_.data(), nb, cudaMemcpyDeviceToDevice, stream.value());
    outputRowData = std::move(packed);
  }

  auto out = std::make_shared<RowStoreVector>(
      pool(),
      outputType_,
      static_cast<int64_t>(numMatches),
      std::move(outputRowData),
      std::move(outFieldsBuf),
      outputFields_,
      outputRowWidth_,
      stream);
  if (outputNullStride_ > 0) {
    out->setNullSidecar(outputNullStride_);
  }
  if (outputCharsBuffer_.size() > 0) {
    out->setCharsBuffer(std::move(outputCharsBuffer_));
    outputCharsBuffer_ = rmm::device_buffer{};
  }
  // Spine deferral v2: name the fields (deferred slots get "" so they never
  // resolve) and pass the provenance through; the __rowid rides as the
  // trailing hidden field.
  if (probeProvStore_ != nullptr) {
    std::vector<std::string> names;
    names.reserve(outputFields_.size());
    for (int i = 0; i < static_cast<int>(outputType_->size()); i++) {
      names.push_back(
          outputFields_[i].offset < 0 ? std::string() : outputType_->nameOf(i));
    }
    names.push_back("__rowid");
    out->setFieldNames(std::move(names));
    out->setProvenance(probeProvStore_, outputRowIdField_);
  }
  return out;
}

// ============================================================================
// makeColumnarOutput — row->col transpose at the terminal join
// ============================================================================
// Converts the gathered fixed-stride row buffer (probeGatherBuffer_, layout
// described by outputFields_ / outputRowWidth_) into a column-major cudf::table
// wrapped in a CudfVector, so that downstream columnar cudf operators
// (aggregation, orderBy, ...) can consume it.
// ============================================================================
// makeHostOutput -- direct CPU-exit emission (spine deferral, paper case a)
// ============================================================================
// The consumer is CudfToVelox (results leave the GPU anyway): D2H the
// gathered fixed rows once, extract fields host-side, gather DEFERRED
// columns from the provenance store by the __rowid values embedded in the
// rows. Deferred payload is never uploaded.
RowVectorPtr RowHashJoinProbe::makeHostOutput(
    int32_t numMatches,
    rmm::cuda_stream_view stream) {
  const int32_t numCols = outputType_->size();
  std::vector<VectorPtr> children(numCols);
  if (!deferredOutputCols_.empty()) {
    // DEFERRED exit: uniformly COLUMNAR, and the rows never cross. The
    // whole output row buffer is transposed on the device (with __rowid as
    // one more column), the GPU-side columns come over through the arrow
    // path, the ids are read from the transposed column on the host, and
    // the deferred payload is gathered from the retained batch.
    VELOX_CHECK_NOT_NULL(probeProvStore_);
    auto gpuCols = transposeGpuOutputColumns(numMatches, stream, true);
    auto rowIdCol = std::move(gpuCols.back());
    gpuCols.pop_back();
    std::vector<std::unique_ptr<cudf::column>> subset;
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    std::vector<int32_t> subsetIdx;
    for (int i = 0; i < numCols; i++) {
      if (gpuCols[i] != nullptr) {
        subset.push_back(std::move(gpuCols[i]));
        names.push_back(outputType_->nameOf(i));
        types.push_back(outputType_->childAt(i));
        subsetIdx.push_back(i);
      }
    }
    subset.push_back(std::move(rowIdCol));
    names.push_back("__rowid");
    types.push_back(BIGINT());
    auto tbl = std::make_unique<cudf::table>(std::move(subset));
    auto host = with_arrow::toVeloxColumn(
        tbl->view(),
        pool(),
        std::static_pointer_cast<const Type>(ROW(std::move(names), std::move(types))),
        stream,
        get_output_mr());
    for (size_t k = 0; k < subsetIdx.size(); k++) {
      children[subsetIdx[k]] = host->childAt(k);
    }
    const auto* ids64 =
        host->childAt(subsetIdx.size())->asFlatVector<int64_t>()->rawValues();
    materializeIds_.resize(numMatches);
    for (int32_t r = 0; r < numMatches; r++) {
      materializeIds_[r] = static_cast<int32_t>(ids64[r]);
    }
    probeProvStore_->idsForGlobalRows(
        materializeIds_.data(), numMatches, materializeRowIds_);
    const auto& storeType = probeProvStore_->rowType();
    for (auto outIdx : deferredOutputCols_) {
      auto col = BaseVector::create(
          outputType_->childAt(outIdx), numMatches, pool());
      probeProvStore_->gather(
          static_cast<int32_t>(
              storeType->getChildIdx(outputType_->nameOf(outIdx))),
          materializeRowIds_,
          col,
          materializeSentinelScratch_);
      children[outIdx] = std::move(col);
    }
    outputCharsBuffer_ = rmm::device_buffer{};
    addRuntimeStat(
        "hostExitRows", RuntimeCounter(static_cast<int64_t>(numMatches)));
    addRuntimeStat(
        "deferredMaterializeRows",
        RuntimeCounter(static_cast<int64_t>(numMatches)));
    return std::make_shared<RowVector>(
        pool(), outputType_, nullptr, numMatches, std::move(children));
  }

  // EAGER exit: rows. D2H the row buffer (+ string heap, null sidecar).
  const int64_t rowsBytes =
      static_cast<int64_t>(numMatches) * outputRowWidth_;
  hostRowsScratch_.resize(rowsBytes);
  cudaMemcpyAsync(
      hostRowsScratch_.data(),
      probeGatherBuffer_.data(),
      rowsBytes,
      cudaMemcpyDeviceToHost,
      stream.value());
  // Strings' compacted heap, if any GPU-gathered string column exists.
  if (outputCharsBuffer_.size() > 0) {
    hostCharsScratch_.resize(outputCharsBuffer_.size());
    cudaMemcpyAsync(
        hostCharsScratch_.data(),
        outputCharsBuffer_.data(),
        outputCharsBuffer_.size(),
        cudaMemcpyDeviceToHost,
        stream.value());
  }
  // Null sidecar of the output rows, if present.
  if (outputNullStride_ > 0) {
    hostNullsScratch_.resize(
        static_cast<int64_t>(numMatches) * outputNullStride_);
    cudaMemcpyAsync(
        hostNullsScratch_.data(),
        outputNullBuffer_.data(),
        hostNullsScratch_.size(),
        cudaMemcpyDeviceToHost,
        stream.value());
  }
  stream.synchronize();

  const uint8_t* rows = hostRowsScratch_.data();
  std::vector<FieldDesc> fields(
      outputFields_.begin(), outputFields_.begin() + numCols);
  const bool hasChars = outputCharsBuffer_.size() > 0;
  outputCharsBuffer_ = rmm::device_buffer{};
  addRuntimeStat(
      "hostExitRows", RuntimeCounter(static_cast<int64_t>(numMatches)));
  if (CudfConfig::getInstance().benchmarkRowOutputNative) {
    // Native row output: hand the D2H'd rows (+ heap, null sidecar) and the
    // host-gathered deferred columns to the consumer as-is. No transpose.
    addRuntimeStat(
        "hostExitNativeBytes",
        RuntimeCounter(rowsBytes, RuntimeCounter::Unit::kBytes));
    return std::make_shared<HostRowVector>(
        pool(),
        outputType_,
        numMatches,
        std::move(hostRowsScratch_),
        outputRowWidth_,
        std::move(fields),
        std::make_shared<const std::vector<uint8_t>>(
            hasChars ? std::move(hostCharsScratch_) : std::vector<uint8_t>{}),
        outputNullStride_ > 0 ? std::move(hostNullsScratch_)
                              : std::vector<uint8_t>{},
        outputNullStride_);
  }
  // GPU-gathered columns: strided host extraction (shared with
  // HostRowVector::materialize).
  return extractHostRows(
      pool(),
      outputType_,
      numMatches,
      rows,
      outputRowWidth_,
      fields,
      hostCharsScratch_.data(),
      outputNullStride_ > 0 ? hostNullsScratch_.data() : nullptr,
      outputNullStride_);
}

// Device transpose of the GPU-gathered output rows into cudf columns, in
// output-column order; deferred columns (offset < 0) are left nullptr for
// the caller to supply (uploaded for a GPU consumer, host-gathered for a
// CPU exit). Shared by makeColumnarOutput and makeHostOutput.
std::vector<std::unique_ptr<cudf::column>>
RowHashJoinProbe::transposeGpuOutputColumns(
    int32_t numMatches,
    rmm::cuda_stream_view stream,
    bool withRowId) {
  const int32_t numCols = outputType_->size();
  std::vector<std::unique_ptr<rmm::device_buffer>> colBuffers(numCols);
  std::vector<uint8_t*> fixedPtrs;
  std::vector<FieldDesc> fixedFields;
  for (int i = 0; i < numCols; i++) {
    if (outputFields_[i].kind == kFieldString ||
        outputFields_[i].offset < 0 /* deferred: materialized above */) {
      continue;
    }
    int64_t bytes = static_cast<int64_t>(numMatches) *
        outputFields_[i].byte_width;
    colBuffers[i] = std::make_unique<rmm::device_buffer>(bytes, stream);
    fixedPtrs.push_back(static_cast<uint8_t*>(colBuffers[i]->data()));
    fixedFields.push_back(outputFields_[i]);
  }

  // Scatter row buffer -> columns.
  std::unique_ptr<rmm::device_buffer> rowIdBuf;
  if (withRowId) {
    VELOX_CHECK_GE(outputRowIdField_, 0);
    rowIdBuf = std::make_unique<rmm::device_buffer>(
        static_cast<int64_t>(numMatches) * 8, stream);
    fixedPtrs.push_back(static_cast<uint8_t*>(rowIdBuf->data()));
    fixedFields.push_back(outputFields_[outputRowIdField_]);
  }
  if (!fixedFields.empty()) {
    rowsToColumns(
        static_cast<const uint8_t*>(probeGatherBuffer_.data()),
        fixedFields.data(),
        fixedPtrs.data(),
        static_cast<int32_t>(fixedFields.size()),
        numMatches,
        outputRowWidth_,
        stream.value());
  }

  // Wrap each device buffer as a cudf::column with the right type.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numCols);
  for (int i = 0; i < numCols; i++) {
    if (outputFields_[i].offset < 0) {
      columns.push_back(nullptr); // deferred: supplied by the caller
      continue;
    }
    auto cudfType = veloxToCudfDataType(outputType_->childAt(i));
    rmm::device_buffer mask{};
    cudf::size_type nullCount = 0;
    if (outputNullStride_ > 0) {
      // Convert this column's sidecar bits into an Arrow validity mask
      // (2026-08-17 null support). UNKNOWN_NULL_COUNT defers counting.
      mask = rmm::device_buffer(
          cudf::bitmask_allocation_size_bytes(numMatches), stream);
      sidecarToMask(
          static_cast<const uint8_t*>(outputNullBuffer_.data()),
          outputNullStride_,
          i,
          numMatches,
          static_cast<uint32_t*>(mask.data()),
          stream.value());
      nullCount = cudf::null_count(
          static_cast<const cudf::bitmask_type*>(mask.data()),
          0,
          static_cast<cudf::size_type>(numMatches),
          stream);
    }
    if (outputFields_[i].kind == kFieldString) {
      // Slots (inline or heap) -> cudf strings column: lengths, exclusive
      // scan (one sync for the total), then a byte copy per row.
      const size_t offBytes =
          (static_cast<size_t>(numMatches) + 1) * sizeof(int64_t);
      if (offBytes > outputRowBaseBuffer_.size()) {
        outputRowBaseBuffer_ = rmm::device_buffer(offBytes, stream);
      }
      const int64_t totalChars = stringFieldOffsets(
          static_cast<const uint8_t*>(probeGatherBuffer_.data()),
          numMatches,
          outputRowWidth_,
          outputFields_[i].offset,
          static_cast<int64_t*>(outputRowBaseBuffer_.data()),
          stream.value());
      VELOX_CHECK_LE(
          totalChars,
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
          "row join: string column exceeds int32 offsets");
      auto offsetsCol = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT32},
          numMatches + 1,
          cudf::mask_state::UNALLOCATED,
          stream,
          get_output_mr());
      rmm::device_buffer chars(totalChars, stream);
      stringFieldToChars(
          static_cast<const uint8_t*>(probeGatherBuffer_.data()),
          numMatches,
          outputRowWidth_,
          outputFields_[i].offset,
          static_cast<const uint8_t*>(outputCharsBuffer_.data()),
          static_cast<const int64_t*>(outputRowBaseBuffer_.data()),
          offsetsCol->mutable_view().data<int32_t>(),
          static_cast<uint8_t*>(chars.data()),
          stream.value());
      columns.push_back(cudf::make_strings_column(
          numMatches,
          std::move(offsetsCol),
          std::move(chars),
          nullCount,
          std::move(mask)));
      continue;
    }
    columns.push_back(std::make_unique<cudf::column>(
        cudfType,
        static_cast<cudf::size_type>(numMatches),
        std::move(*colBuffers[i]),      // data buffer (ownership moved)
        std::move(mask),
        nullCount));
  }

  if (withRowId) {
    // The hidden __rowid as one more INT64 column (no null mask).
    columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        static_cast<cudf::size_type>(numMatches),
        std::move(*rowIdBuf),
        rmm::device_buffer{},
        0));
  }
  return columns;
}

RowVectorPtr RowHashJoinProbe::makeColumnarOutput(
    int32_t numMatches,
    rmm::cuda_stream_view stream) {
  const int32_t numCols = outputType_->size();
  VELOX_CHECK_LE(static_cast<size_t>(numCols), outputFields_.size());

  // Spine deferral v2: materialize DEFERRED probe-side columns from the
  // provenance store -- extract the hidden __rowid field of the gathered
  // rows, read the ids back, host-gather survivors, upload, and splice the
  // columns into the output table below.
  // Transpose FIRST (review round 6): the whole output row buffer, with the
  // hidden __rowid as one more column, goes rows -> columns once on the
  // device; the ids are read from that column. No separate id pass.
  const bool hasDeferred = !deferredOutputCols_.empty() && numMatches > 0;
  auto columns = transposeGpuOutputColumns(numMatches, stream, hasDeferred);
  std::vector<std::unique_ptr<cudf::column>> deferredCols(numCols);
  if (hasDeferred) {
    VELOX_CHECK_NOT_NULL(probeProvStore_);
    auto tpDefer = std::chrono::steady_clock::now();
    auto rowIdCol = std::move(columns.back());
    columns.pop_back();
    std::vector<int64_t> ids64(numMatches);
    cudaMemcpyAsync(
        ids64.data(),
        rowIdCol->view().data<int64_t>(),
        static_cast<size_t>(numMatches) * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream.value());
    stream.synchronize();
    materializeIds_.resize(numMatches);
    for (int32_t i = 0; i < numMatches; i++) {
      materializeIds_[i] = static_cast<int32_t>(ids64[i]);
    }
    probeProvStore_->idsForGlobalRows(
        materializeIds_.data(), numMatches, materializeRowIds_);
    const auto& storeType = probeProvStore_->rowType();
    std::vector<VectorPtr> hostCols;
    std::vector<std::string> hostNames;
    for (auto outIdx : deferredOutputCols_) {
      const auto& name = outputType_->nameOf(outIdx);
      auto col = BaseVector::create(
          outputType_->childAt(outIdx), numMatches, pool());
      probeProvStore_->gather(
          static_cast<int32_t>(storeType->getChildIdx(name)),
          materializeRowIds_,
          col,
          materializeSentinelScratch_);
      hostCols.push_back(std::move(col));
      hostNames.push_back(name);
    }
    // Upload as one table; column order == deferredOutputCols_ order.
    auto hostRow = std::make_shared<RowVector>(
        pool(),
        ROW(std::move(hostNames),
            [&] {
              std::vector<TypePtr> ts;
              for (auto outIdx : deferredOutputCols_) {
                ts.push_back(outputType_->childAt(outIdx));
              }
              return ts;
            }()),
        nullptr,
        numMatches,
        std::move(hostCols));
    auto tbl = with_arrow::toCudfTable(
        hostRow, pool(), stream, get_output_mr());
    auto cols = tbl->release();
    for (size_t k = 0; k < deferredOutputCols_.size(); k++) {
      deferredCols[deferredOutputCols_[k]] = std::move(cols[k]);
    }
    addRuntimeStat(
        "deferredMaterializeRows",
        RuntimeCounter(static_cast<int64_t>(numMatches)));
    addRuntimeStat(
        "deferredMaterializeNanos",
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tpDefer)
                .count(),
            RuntimeCounter::Unit::kNanos));
  }

  // Allocate one device buffer per FIXED output column and remember its
  // base ptr; string columns (2026-08-21) are materialized separately below.
  for (int i = 0; i < numCols; i++) {
    if (outputFields_[i].offset < 0) {
      VELOX_CHECK_NOT_NULL(deferredCols[i]);
      columns[i] = std::move(deferredCols[i]);
    }
  }

  auto table = std::make_unique<cudf::table>(std::move(columns));
  outputCharsBuffer_ = rmm::device_buffer{};

  // Gather buffer is consumed by the per-column copies above; force realloc.
  gatherCapacity_ = 0;
  probeGatherBuffer_ = rmm::device_buffer{};

  // NOTE: No stream.synchronize() here. rowsToColumns above and the
  // probeGatherBuffer_ free are ordered on the same `stream`, and the RMM
  // resource is stream-ordered (see createMemoryResource in GpuResources.cpp),
  // so the gather buffer cannot be reused until the transpose completes. The
  // returned CudfVector carries `stream`, so downstream operators observe the
  // scattered columns via the standard stream-ordered consumption contract.

  addRuntimeStat(
      "rowToColOutputRows",
      RuntimeCounter(static_cast<int64_t>(numMatches)));

  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      static_cast<vector_size_t>(numMatches),
      std::move(table),
      stream);
}

void RowHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

bool RowHashJoinProbe::isFinished() {
  return finished_;
}

// ============================================================================
// Boundary-hybrid probe (CudfConfig::benchmarkBoundaryHybrid)
// ============================================================================
//
// The paper's §3 single-join design — hybrid layout AT the device boundary:
//   1. ONLY the probe keys crossed PCIe (keys-only RowStoreVector); the
//      payload columns of this batch are host-resident (attached batches).
//   2. Extract the key column(s) from the keys-only row store, probe the
//      build hash table on the GPU (cudf::hash_join::inner_join over the
//      keys-only build store — the build payload also never left the host).
//   3. Read back ONLY the surviving id pairs: (probe row within this batch,
//      global build row id). Pinned staging + one stream sync — the same
//      count-readback barrier the row join already pays, extended to the two
//      id arrays.
//   4. Materialize the output RowVector on the CPU: probe-side columns
//      gather from THIS batch's retained host batches (ids are batch-local
//      because we process batch-at-a-time); build-side columns gather
//      through the build's global-id -> (batchId, rowInBatch) map into the
//      build's retained batches. Both gathers run HybridContainer's
//      scattered extraction (see BoundaryHostStore.h).
//
// Boundary traffic per probe row: K key bytes H2D + 8 bytes/survivor D2H,
// versus the full-row pack's 50-100 B/row H2D — the mode's entire point.

void RowHashJoinProbe::PinnedIdBuffer::ensure(int64_t n) {
  if (n <= capacity) {
    return;
  }
  if (data != nullptr) {
    cudaFreeHost(data);
    data = nullptr;
  }
  // Grow-by-doubling: cudaHostAlloc globally serializes the driver, so
  // amortize (same rationale as the PinnedPackSlot pool).
  capacity = std::max<int64_t>(n, capacity * 2);
  cudaError_t err = cudaHostAlloc(
      reinterpret_cast<void**>(&data),
      capacity * sizeof(int32_t),
      cudaHostAllocDefault);
  VELOX_CHECK(
      err == cudaSuccess,
      "boundary-hybrid: cudaHostAlloc({} B) failed: {}",
      capacity * sizeof(int32_t),
      cudaGetErrorString(err));
}

RowHashJoinProbe::PinnedIdBuffer::~PinnedIdBuffer() {
  if (data != nullptr) {
    cudaFreeHost(data);
  }
}

// ============================================================================
// Row-native matcher probe (CudfConfig::benchmarkRowTable)
// ============================================================================
// Two-pass N:M probe mirroring cudf's count+retrieve, but with the keys read
// straight from the probe ROW store and compared against the build KEY rows:
//   pass 1: per-probe-row match counts (chain walk + fingerprint verify)
//   scan  : cub exclusive sum -> per-row output offsets + total
//   pass 2: refill (probeIdx, buildIdx) at the scanned offsets
// One stream sync to read the total — the same per-batch barrier
// cudf::hash_join::inner_join pays internally to size its output.
int32_t RowHashJoinProbe::rowTableProbe(
    const GpuFixedRowStore& probeStore,
    const RowKeyLayout& probeKeys,
    int32_t probeRows,
    rmm::cuda_stream_view stream,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>& leftIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>& rightIndices) {
  auto& bd = *buildData_;
  VELOX_CHECK(bd.hasRowTable, "rowTableProbe: build did not produce a table");
  if (probeRows == 0) {
    return 0;
  }

  if (probeRows > rowTableProbeCapacity_) {
    rowTableProbeCapacity_ = probeRows;
    rowTableCounts_ =
        rmm::device_buffer((size_t)probeRows * sizeof(int32_t), stream);
    rowTableOffsets_ =
        rmm::device_buffer((size_t)probeRows * sizeof(int32_t), stream);
    const size_t tb = rowTableScanTempBytes(probeRows);
    if (tb > rowTableScanTempBytes_) {
      rowTableScanTempBytes_ = tb;
      rowTableScanTemp_ = rmm::device_buffer(tb, stream);
    }
  }
  if (rowTableTotal_.size() == 0) {
    rowTableTotal_ = rmm::device_buffer(sizeof(int32_t), stream);
  }

  auto* counts = static_cast<int32_t*>(rowTableCounts_.data());
  auto* offsets = static_cast<int32_t*>(rowTableOffsets_.data());
  auto* total = static_cast<int32_t*>(rowTableTotal_.data());

  rowTableProbeCount(
      bd.rowTable, probeStore, probeKeys, probeRows, counts, stream.value());
  rowTableScanOffsets(
      counts,
      offsets,
      probeRows,
      rowTableScanTemp_.data(),
      rowTableScanTempBytes_,
      total,
      stream.value());

  int32_t numMatches = 0;
  cudaMemcpyAsync(
      &numMatches, total, sizeof(int32_t), cudaMemcpyDeviceToHost,
      stream.value());
  stream.synchronize();

  if (numMatches > 0) {
    leftIndices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
        numMatches, stream, get_temp_mr());
    rightIndices = std::make_unique<rmm::device_uvector<cudf::size_type>>(
        numMatches, stream, get_temp_mr());
    rowTableProbeFill(
        bd.rowTable,
        probeStore,
        probeKeys,
        probeRows,
        offsets,
        leftIndices->data(),
        rightIndices->data(),
        stream.value());
  }
  return numMatches;
}

// The shared MATCHER front half of the regular and boundary probe paths:
// probe keys in, (probe, build) id pair lists out.
//   cudf arm            : extract the key column(s) from the row store, then
//                         cudf::hash_join::inner_join (cuco);
//   row arm (hasRowTable): the row-native chained multimap, probed with keys
//                         read straight from the row store — the extract_keys
//                         step does not run at all.
// keyDescs[k] locates join key k inside probeStore's rows; the two callers
// differ ONLY in how they resolve it (full layout vs keys-only layout).
int32_t RowHashJoinProbe::runMatcher(
    const GpuFixedRowStore& probeStore,
    const std::vector<FieldDesc>& keyDescs,
    int32_t probeRows,
    rmm::cuda_stream_view stream,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>& leftIndices,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>>& rightIndices) {
  auto& bd = *buildData_;
  const int numKeys = static_cast<int>(keyDescs.size());
  int32_t numMatches = 0;

  // Matcher-phase device timing ("matcherWallNanos"): covers extract+probe
  // for the cudf arm and count+scan+refill for the row arm — gather excluded.
  cudaEvent_t mStart, mEnd;
  const bool timeMatcher = CudfConfig::getInstance().benchmarkLogGatherTime;
  if (timeMatcher) {
    cudaEventCreate(&mStart);
    cudaEventCreate(&mEnd);
    cudaEventRecord(mStart, stream.value());
  }

  if (bd.hasRowTable) {
    RowKeyLayout pk{};
    pk.numKeys = numKeys;
    for (int k = 0; k < numKeys; k++) {
      pk.offset[k] = keyDescs[k].offset;
      pk.width[k] = keyDescs[k].byte_width;
    }
    numMatches = rowTableProbe(
        probeStore, pk, probeRows, stream, leftIndices, rightIndices);
  } else {
    if (static_cast<int>(probeKeyBuffers_.size()) != numKeys) {
      probeKeyBuffers_.clear();
      probeKeyBuffers_.resize(numKeys);
      probeKeyCapacity_ = 0;
    }
    if (probeRows > probeKeyCapacity_) {
      probeKeyCapacity_ = probeRows;
      for (int k = 0; k < numKeys; k++) {
        probeKeyBuffers_[k] = rmm::device_buffer(
            (int64_t)probeRows * keyDescs[k].byte_width, stream);
      }
    }
    std::vector<cudf::column_view> probeKeyViews;
    probeKeyViews.reserve(numKeys);
    for (int k = 0; k < numKeys; k++) {
      extractKeysFromRows(
          probeStore,
          keyDescs[k].offset,
          keyDescs[k].byte_width,
          probeKeyBuffers_[k].data(),
          stream.value());
      probeKeyViews.emplace_back(
          cudf::data_type{
              keyDescs[k].byte_width == 8 ? cudf::type_id::INT64
                                          : cudf::type_id::INT32},
          static_cast<cudf::size_type>(probeRows),
          probeKeyBuffers_[k].data(),
          nullptr,
          0);
    }

    auto probeKeyTable = cudf::table_view(probeKeyViews);
    auto [l, r] = bd.hashJoin->inner_join(
        probeKeyTable, std::nullopt, stream, get_temp_mr());
    leftIndices = std::move(l);
    rightIndices = std::move(r);
    numMatches = static_cast<int32_t>(leftIndices->size());
  }

  if (timeMatcher) {
    cudaEventRecord(mEnd, stream.value());
    cudaEventSynchronize(mEnd);
    float mMs = 0;
    cudaEventElapsedTime(&mMs, mStart, mEnd);
    addRuntimeStat(
        "matcherWallNanos",
        RuntimeCounter(
            static_cast<int64_t>(mMs * 1e6), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "matcherMatches", RuntimeCounter(static_cast<int64_t>(numMatches)));
    cudaEventDestroy(mStart);
    cudaEventDestroy(mEnd);
  }
  return numMatches;
}

RowVectorPtr RowHashJoinProbe::boundaryProbe(
    const std::shared_ptr<RowStoreVector>& rowStoreInput,
    rmm::cuda_stream_view stream) {
  auto& bd = *buildData_;
  VELOX_CHECK(
      bd.hasRowTable || bd.hashJoin != nullptr,
      "boundary-hybrid probe: build produced neither a hash_join nor a "
      "row-native table");
  // NOTE: bd.hostStore may be null when the build side was EMPTY (no host
  // batches were ever attached); every probe then yields zero matches and we
  // return before touching it. Checked below, after the zero-match exit.

  const int32_t probeRows = rowStoreInput->size();
  const auto& keyFields = rowStoreInput->hostFields(); // keys-only, key order
  const int numKeys = static_cast<int>(leftKeyIndices_.size());
  VELOX_CHECK_EQ(
      static_cast<int>(keyFields.size()),
      numKeys,
      "boundary-hybrid probe: input is not in keys-only layout");

  // ---- 1+2. MATCHER: keys in, id pairs out ----
  // Field k IS leftKeys[k] in the keys-only layout (packed in join-key
  // order), so the key descriptors are just the input's hostFields.
  GpuFixedRowStore probeStore = rowStoreInput->getGpuRowStore();
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndices;
  int32_t numMatches = runMatcher(
      probeStore, keyFields, probeRows, stream, leftIndices, rightIndices);

  // ---- 3. Survivor-id readback: 2 * numMatches * 4 B, pinned, one sync ----
  if (numMatches > 0) {
    boundaryProbeIds_.ensure(numMatches);
    boundaryBuildIds_.ensure(numMatches);
    cudaMemcpyAsync(
        boundaryProbeIds_.data,
        leftIndices->data(),
        (size_t)numMatches * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.value());
    cudaMemcpyAsync(
        boundaryBuildIds_.data,
        rightIndices->data(),
        (size_t)numMatches * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.value());
  }
  // One barrier per batch: the row join pays an equivalent per-batch barrier
  // for its match-count readback, so this adds no extra serialization.
  stream.synchronize();

  addRuntimeStat(
      "boundaryProbeRows", RuntimeCounter(static_cast<int64_t>(probeRows)));
  addRuntimeStat(
      "boundaryMatches", RuntimeCounter(static_cast<int64_t>(numMatches)));
  addRuntimeStat(
      "boundaryIdReadbackBytes",
      RuntimeCounter(
          2LL * numMatches * sizeof(int32_t), RuntimeCounter::Unit::kBytes));

  if (numMatches == 0) {
    return nullptr;
  }
  VELOX_CHECK_NOT_NULL(
      bd.hostStore, "boundary-hybrid probe: build has no host payload store");

  auto tpGatherStart = std::chrono::steady_clock::now();

  // ---- 4a. Retain THIS batch's host payload (batch-local ids) ----
  const auto& hostBatches = rowStoreInput->boundaryHostBatches();
  VELOX_CHECK(
      !hostBatches.empty(),
      "boundary-hybrid probe: input lost its host payload");
  if (boundaryProbeStore_ == nullptr) {
    auto hostType =
        std::dynamic_pointer_cast<const RowType>(hostBatches[0]->type());
    VELOX_CHECK_NOT_NULL(hostType);
    boundaryProbeStore_ =
        std::make_unique<BoundaryHostStore>(hostType, pool());
  } else {
    boundaryProbeStore_->clearBatches(); // previous batch's payload released
  }
  for (const auto& hb : hostBatches) {
    boundaryProbeStore_->addBatch(hb);
  }
  VELOX_CHECK_EQ(
      boundaryProbeStore_->totalRows(),
      static_cast<int64_t>(probeRows),
      "boundary-hybrid probe: host retention rows != GPU key rows");

  // ---- 4b. Output column -> source mapping (once) ----
  // Every output column resolves against a retained-batch child: probe-side
  // columns (INCLUDING the key — probe side is the simplest key source, per
  // the design) from this batch's host batches, build-side columns from the
  // build's host store.
  if (!boundaryLayoutReady_) {
    const auto& probeType = boundaryProbeStore_->rowType();
    const auto& buildType = bd.hostStore->rowType();
    for (int j = 0; j < outputType_->size(); j++) {
      const auto& name = outputType_->nameOf(j);
      auto probeIdx = probeType->getChildIdxIfExists(name);
      if (probeIdx.has_value()) {
        boundaryOutFromProbe_.push_back(
            {j, static_cast<int32_t>(probeIdx.value())});
      } else {
        boundaryOutFromBuild_.push_back(
            {j, static_cast<int32_t>(buildType->getChildIdx(name))});
      }
    }
    boundaryLayoutReady_ = true;
  }

  // ---- 4c. Id decode: batch-local probe ids and global build ids become
  // scattered (batchId, rowInBatch) HybridRowIds. ----
  boundaryProbeStore_->idsForGlobalRows(
      boundaryProbeIds_.data, numMatches, boundaryProbeRowIds_);
  bd.hostStore->idsForGlobalRows(
      boundaryBuildIds_.data, numMatches, boundaryBuildRowIds_);

  // ---- 4d. Host gather: one scattered extraction pass per output column ----
  std::vector<VectorPtr> children(outputType_->size());
  for (const auto& [outIdx, childIdx] : boundaryOutFromProbe_) {
    auto col =
        BaseVector::create(outputType_->childAt(outIdx), numMatches, pool());
    boundaryProbeStore_->gather(
        childIdx, boundaryProbeRowIds_, col, boundarySentinelScratch_);
    children[outIdx] = std::move(col);
  }
  for (const auto& [outIdx, childIdx] : boundaryOutFromBuild_) {
    auto col =
        BaseVector::create(outputType_->childAt(outIdx), numMatches, pool());
    bd.hostStore->gather(
        childIdx, boundaryBuildRowIds_, col, boundarySentinelScratch_);
    children[outIdx] = std::move(col);
  }

  // This batch's host payload is no longer needed (output copied out).
  boundaryProbeStore_->clearBatches();
  rowStoreInput->clearBoundaryPayload();

  {
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::steady_clock::now() - tpGatherStart)
                     .count();
    addRuntimeStat(
        "boundaryHostGatherNanos",
        RuntimeCounter(nanos, RuntimeCounter::Unit::kNanos));
  }

  // skip_output parity with the other arms: they run the (GPU) gather and
  // then drop the batch; boundary runs its (host) gather — the cost under
  // study — and drops the assembled batch the same way.
  if (CudfConfig::getInstance().benchmarkSkipOutput) {
    std::vector<VectorPtr> dummy(outputType_->size());
    for (int i = 0; i < outputType_->size(); i++) {
      dummy[i] =
          BaseVector::createNullConstant(outputType_->childAt(i), 1, pool());
    }
    return std::make_shared<RowVector>(
        pool(), outputType_, nullptr, 1, std::move(dummy));
  }

  // CPU RowVector out. Downstream CudfToVelox passes non-CudfVector inputs
  // through untouched, so this needs no D2H conversion stage.
  return std::make_shared<RowVector>(
      pool(),
      outputType_,
      nullptr,
      static_cast<vector_size_t>(numMatches),
      std::move(children));
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
  // Only claim join nodes the row path implements (mirrors the adapter
  // predicate in OperatorAdapters.cpp); others fall through to the next
  // registered translator (CudfHashJoinBridgeTranslator) -- 2026-08-21.
  if (joinNode && joinNode->joinType() == core::JoinType::kInner &&
      !joinNode->filter()) {
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
