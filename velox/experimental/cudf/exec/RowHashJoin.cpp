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
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/CudfConfig.h"

#include "velox/exec/Task.h"
#include "velox/exec/Driver.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
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
void transposeToRows(
    const cudf::table_view& table,
    const std::vector<FieldDesc>& fields,
    int32_t rowWidth,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  int32_t numRows = table.num_rows();
  int32_t numCols = table.num_columns();
  if (numRows == 0 || numCols == 0) return;
  checkTransposeInputNullFree(table, "transposeToRows");

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
    fd.byte_width = cudfTypeWidth(table.column(i).type().id());
    offset = alignUp(offset, fd.byte_width);
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

  auto stream = cudfGlobalStreamPool().get_stream();

  // Check if inputs are RowStoreVectors (already in row layout on GPU)
  auto firstRowStore = std::dynamic_pointer_cast<RowStoreVector>(inputs_[0]);

  int64_t numRows = 0;
  int32_t rowWidth = 0;
  std::vector<FieldDesc> fields;
  rmm::device_buffer rowBuffer;
  rmm::device_buffer fieldsBuffer;
  // Null sidecar for the combined build store (2026-08-17 null support).
  rmm::device_buffer buildNullBuffer2;
  int32_t buildNullStride2 = 0;
  std::vector<rmm::device_buffer> keyBuffers;
  std::shared_ptr<cudf::hash_join> hashJoin;

  // Fused-probe device hash table (key -> build row index). Built from the
  // already-extracted COLUMNAR key columns (coalesced key load), and only when
  // fused -- in which case the cudf::hash_join below is dead work and skipped.
  GpuHashTable deviceMap{};
  rmm::device_buffer mapSlots;
  bool hasDeviceMap = false;
  // COLUMNAR build fetch (VELOX_CUDF_COL_BUILD): skip the row-store transpose for
  // combined (non-fingerprint) fused builds; the probe fetches build payload
  // straight from these columns. Worth it when each build row is fetched < ~1x.
  bool columnarBuild = false;
  std::shared_ptr<cudf::table> columnarBuildTablePtr;
  // Fingerprint slots hold only a hash, not the key, so the probe re-packs the
  // build key from a row to verify. When payload is fetched columnar, transpose
  // ONLY the key columns into this dedicated key-row store -- decoupling key
  // layout (always row-native) from payload layout (columnar vs full row store).
  rmm::device_buffer keyRowBuffer;
  rmm::device_buffer keyFieldsBuffer;
  std::vector<FieldDesc> keyRowFields;
  int32_t keyRowWidth = 0;
  bool hasKeyRowStore = false;
  // Build EXACTLY ONE hash table, decided per join by whether THIS join's probe
  // was fused. The fused pre-pass recorded that during driver adaptation (before
  // any build runs), so there is no hedging and no dead work:
  //   probe fused     -> FusedRowHashJoinProbe reads deviceMap; build only it.
  //   probe not fused -> RowHashJoinProbe reads cudf::hash_join; build only it.
  // (In a bushy plan like Q8, main-chain joins are fused -> deviceMap; build-side
  // sub-joins are not -> cudf::hash_join.)
  const bool wantFusedMap =
      CudfConfig::getInstance().isProbeFused(joinNode_->id());
  const bool skipHashJoin = wantFusedMap;

  // ---- Row-native N:M matcher (CudfConfig::benchmarkRowTable) ----
  // Build the chained multimap DIRECTLY over the key rows of the row store
  // (keys-only store under boundary-hybrid; full row store otherwise). In
  // this mode neither extractKeyColumns nor cudf::hash_join runs at all:
  // keyBuffers stay empty and hashJoin stays null.
  const bool useRowTable = CudfConfig::getInstance().benchmarkRowTable;
  VELOX_CHECK(
      !(useRowTable && wantFusedMap),
      "benchmarkRowTable is the single-join row matcher; it does not combine "
      "with benchmarkFusedProbe (which keeps its own N:1 device map)");
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
  bool packEnabled = false;
  std::vector<int64_t> packMin, packMax;
  std::vector<int32_t> packShift;
  uint64_t packSentinel = 0;
  auto buildFusedMap = [&](const std::vector<cudf::column_view>& keyViews,
                           int64_t nRows) {
    const int32_t numKeyWords = static_cast<int32_t>(keyViews.size());
    const int32_t cap = fusedTableCapacity(static_cast<int32_t>(nRows));

    std::vector<ColSeedField> keyCols(numKeyWords);
    for (int32_t k = 0; k < numKeyWords; k++) {
      const auto& kv = keyViews[k];
      const int32_t w = cudfTypeWidth(kv.type().id());
      keyCols[k].data = static_cast<const uint8_t*>(kv.head<uint8_t>()) +
          (int64_t)kv.offset() * w;
      keyCols[k].byte_width = w;
      keyCols[k].acc_offset = 0; // unused for keys
    }
    rmm::device_buffer dKeyCols(
        keyCols.data(), keyCols.size() * sizeof(ColSeedField), stream);

    // ---- Decide composite-key PACKING (single-word fast path) ----
    // Pack the K key columns into one uint64 when their combined bit-width fits
    // in <=63 bits. Requires per-column min/max (two reductions each). Only for
    // fixed-width integer keys (width 4/8, assumed non-negative).
    auto scalarI64 = [&](const std::unique_ptr<cudf::scalar>& s) -> int64_t {
      if (s->type().id() == cudf::type_id::INT64) {
        return static_cast<cudf::numeric_scalar<int64_t>*>(s.get())->value(
            stream);
      }
      return static_cast<int64_t>(
          static_cast<cudf::numeric_scalar<int32_t>*>(s.get())->value(stream));
    };
    packEnabled = (numKeyWords >= 1 && numKeyWords <= kMaxKeyCols);
    packMin.assign(numKeyWords, 0);
    packMax.assign(numKeyWords, 0);
    packShift.assign(numKeyWords, 0);
    int32_t totalBits = 0;
    for (int32_t k = 0; k < numKeyWords && packEnabled; k++) {
      const auto id = keyViews[k].type().id();
      if (id != cudf::type_id::INT32 && id != cudf::type_id::INT64) {
        packEnabled = false;
        break;
      }
      auto mn = cudf::reduce(
          keyViews[k], *cudf::make_min_aggregation<cudf::reduce_aggregation>(),
          keyViews[k].type(), stream, get_output_mr());
      auto mx = cudf::reduce(
          keyViews[k], *cudf::make_max_aggregation<cudf::reduce_aggregation>(),
          keyViews[k].type(), stream, get_output_mr());
      const int64_t lo = scalarI64(mn), hi = scalarI64(mx);
      const uint64_t range = static_cast<uint64_t>(hi - lo);
      const int32_t bits = (range == 0) ? 1 : (64 - __builtin_clzll(range));
      packMin[k] = lo;
      packMax[k] = hi;
      packShift[k] = totalBits;
      totalBits += bits;
    }
    if (totalBits > 63) {
      packEnabled = false;
    }

    if (packEnabled) {
      packSentinel = (totalBits >= 64) ? 0ULL : (1ULL << totalBits);
      rmm::device_buffer dMins(
          packMin.data(), packMin.size() * sizeof(int64_t), stream);
      rmm::device_buffer dShifts(
          packShift.data(), packShift.size() * sizeof(int32_t), stream);
      // COMBINED slots (cuco-style 8-byte (key,index) slot): if the packed key
      // plus the row index fit in <=63 bits, store [key | index<<keyBits] in ONE
      // word -- one atomicCAS per insert, half the scattered writes. This is the
      // build's dominant cost (latency-bound random atomics); the packed 2-word
      // path is the fallback when they don't fit.
      const int32_t indexBits = (nRows <= 1)
          ? 1
          : (64 - __builtin_clzll(static_cast<uint64_t>(nRows - 1)));
      const bool combined = (totalBits + indexBits <= 63);
      if (combined) {
        mapSlots = rmm::device_buffer((int64_t)cap * sizeof(uint64_t), stream);
        buildHashTableFromColumnsCombined(
            static_cast<const ColSeedField*>(dKeyCols.data()),
            numKeyWords,
            static_cast<const int64_t*>(dMins.data()),
            static_cast<const int32_t*>(dShifts.data()),
            totalBits,
            static_cast<int32_t>(nRows),
            static_cast<uint64_t*>(mapSlots.data()),
            cap,
            stream.value());
        deviceMap =
            GpuHashTable{static_cast<uint64_t*>(mapSlots.data()), cap, 1};
        deviceMap.combinedKeyBits = totalBits;
      } else {
        // Exact key + index don't fit in one 63-bit word. Rather than a 16-byte
        // 2-word slot (cuco's slow cas_dependent_write path), store cuco's
        // 8-byte fingerprint slot [fp32 | index<<32] -- one atomicCAS. The probe
        // verifies the real key on a fingerprint hit (build row is cache-hot).
        const int32_t fpBits = 32;
        mapSlots = rmm::device_buffer((int64_t)cap * sizeof(uint64_t), stream);
        buildHashTableFromColumnsHashed(
            static_cast<const ColSeedField*>(dKeyCols.data()),
            numKeyWords,
            static_cast<const int64_t*>(dMins.data()),
            static_cast<const int32_t*>(dShifts.data()),
            fpBits,
            static_cast<int32_t>(nRows),
            static_cast<uint64_t*>(mapSlots.data()),
            cap,
            stream.value());
        deviceMap =
            GpuHashTable{static_cast<uint64_t*>(mapSlots.data()), cap, 1};
        deviceMap.combinedKeyBits = fpBits;
        deviceMap.fingerprintSlot = 1;
      }
      stream.synchronize(); // dMins/dShifts must outlive the launch
    } else {
      mapSlots = rmm::device_buffer(
          (int64_t)cap * (numKeyWords + 1) * sizeof(uint64_t), stream);
      buildHashTableFromColumns(
          static_cast<const ColSeedField*>(dKeyCols.data()),
          numKeyWords,
          static_cast<int32_t>(nRows),
          static_cast<uint64_t*>(mapSlots.data()),
          cap,
          stream.value());
      deviceMap = GpuHashTable{
          static_cast<uint64_t*>(mapSlots.data()), cap, numKeyWords};
    }
    hasDeviceMap = true;
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
              buildNullStride2 == 0 || buildNullStride2 == r->nullStride(),
              "row store null strides disagree across build batches");
          buildNullStride2 = r->nullStride();
        }
      }
    }
    if (buildNullStride2 > 0) {
      buildNullBuffer2 =
          rmm::device_buffer(numRows * buildNullStride2, stream);
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
        offset += chunkBytes;
      }
      if (buildNullStride2 > 0 && rsv->size() > 0) {
        const int64_t nb =
            static_cast<int64_t>(rsv->size()) * buildNullStride2;
        if (rsv->nullStride() > 0) {
          cudaMemcpyAsync(
              static_cast<uint8_t*>(buildNullBuffer2.data()) + nullOffset,
              rsv->gpuRowData() + rsv->gpuRowBytes(),
              nb, cudaMemcpyDeviceToDevice, stream.value());
        } else {
          cudaMemsetAsync(
              static_cast<uint8_t*>(buildNullBuffer2.data()) + nullOffset,
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
    if (boundaryBuild) {
      // Keys-only layout: field k IS rightKeys[k] (CudfFromVelox packed the
      // key columns in join-key order), so the key indices are simply 0..K-1
      // — NOT the child indices in rightType.
      VELOX_CHECK_EQ(
          fields.size(),
          rightKeys.size(),
          "boundary-hybrid build: keys-only layout width mismatch");
      VELOX_CHECK(
          !wantFusedMap,
          "boundary-hybrid does not support the fused probe (single-join "
          "design); unset benchmarkFusedProbe");
      for (size_t k = 0; k < rightKeys.size(); k++) {
        keyColIndices.push_back(static_cast<cudf::size_type>(k));
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

      if (!skipHashJoin) {
        auto keyTable = cudf::table_view(keyViews);
        hashJoin = std::make_shared<cudf::hash_join>(
            keyTable, cudf::null_equality::UNEQUAL, stream);
      }
      if (wantFusedMap) {
        buildFusedMap(keyViews, numRows);
      }
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

    // COLUMNAR-BUILD path: for a fused build (no sub-join hashJoin), build the
    // hash table straight from the ORIGINAL columns -- no transpose, no key
    // re-extract. If the slot is a plain combined slot (no fingerprint verify),
    // skip the row-store transpose entirely and let the probe fetch build payload
    // columnar. Fingerprint builds fall back to the row store (verify needs it).
    const bool colBuildFlag =
        (std::getenv("VELOX_CUDF_COL_BUILD") != nullptr) && wantFusedMap &&
        skipHashJoin;
    bool fusedMapBuilt = false;
    if (colBuildFlag) {
      std::vector<cudf::column_view> directKeys;
      for (auto idx : keyColIndices) {
        directKeys.push_back(buildView.column(idx));
      }
      buildFusedMap(directKeys, numRows);
      fusedMapBuilt = true;
      // Columnar payload for EVERY key regime. Combined/multi-word keep the key
      // in the slot; fingerprint keeps it in a dedicated key-row store (below).
      columnarBuild = true;
    }

    if (columnarBuild) {
      // Compute the row LAYOUT (host-side: field offsets/widths, needed by the
      // probe's acc-layout width lookups) but SKIP the payload transpose and row
      // buffer. The probe fetches build payload straight from the columns; keep
      // the concatenated table alive for it.
      auto [f, rw] = computeRowLayoutFromTable(buildView);
      fields = std::move(f);
      rowWidth = rw;
      columnarBuildTablePtr =
          std::shared_ptr<cudf::table>(std::move(concatenated));
      // Fingerprint verify re-packs the build key from a row: build a KEY-ONLY
      // row store so the key stays row-native even though the payload is
      // columnar. (Combined/multi-word carry the key in the slot -- no store.)
      if (deviceMap.fingerprintSlot) {
        auto [kf, krw] = computeKeyRowLayout(buildView, keyColIndices);
        keyRowFields = std::move(kf);
        keyRowWidth = krw;
        keyRowBuffer =
            rmm::device_buffer((int64_t)numRows * keyRowWidth, stream);
        transposeKeyColumnsToRows(
            buildView, keyColIndices, keyRowFields, keyRowWidth,
            static_cast<uint8_t*>(keyRowBuffer.data()), stream.value());
        keyFieldsBuffer = rmm::device_buffer(
            keyRowFields.data(), keyRowFields.size() * sizeof(FieldDesc),
            stream);
        hasKeyRowStore = true;
      }
      stamp(2);
      stamp(3);
      stamp(4);
      stamp(5);
    } else {
      // Row-store path: transpose ALL columns to rows, extract keys, build.
      auto [f, rw] = computeRowLayoutFromTable(buildView);
      fields = std::move(f);
      rowWidth = rw;

      int64_t rowBytes = numRows * rowWidth;
      rowBuffer = rmm::device_buffer(rowBytes, stream);

      transposeToRows(
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

        // hashJoin feeds non-fused RowHashJoinProbe (sub-joins); skipHashJoin
        // is set only when EVERY join is fused, so it is then dead work.
        // deviceMap is built from the columnar key columns (coalesced key
        // load).
        if (!skipHashJoin) {
          auto keyTable = cudf::table_view(keyViews);
          hashJoin = std::make_shared<cudf::hash_join>(
              keyTable, cudf::null_equality::UNEQUAL, stream);
        }
        matcherBuildEnd();
        stamp(4);
        if (wantFusedMap && !fusedMapBuilt) {
          buildFusedMap(keyViews, numRows);
        }
        stamp(5);
      }
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
          "[BUILD_PROFILE] nRows=%ld cols=%d rowWidth=%d skipHashJoin=%d "
          "wantFusedMap=%d | concat=%.2f transpose=%.2f extractKeys=%.2f "
          "hashJoin=%.2f fusedMap=%.2f | GPUtotal=%.2f ms\n",
          (long)numRows,
          (int)fields.size(),
          (int)rowWidth,
          (int)skipHashJoin,
          (int)wantFusedMap,
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
  if (buildNullStride2 > 0) {
    gpuStore.null_bytes =
        static_cast<const uint8_t*>(buildNullBuffer2.data());
    gpuStore.null_stride = buildNullStride2;
  }

  // (The fused-probe device hash table `deviceMap` was built above, inside the
  // input-specific branch, from the columnar key columns -- coalesced key load,
  // and no dead cudf::hash_join.)

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
  bd.deviceMap = deviceMap;
  bd.mapSlots = std::move(mapSlots);
  bd.hasDeviceMap = hasDeviceMap;
  bd.packEnabled = packEnabled;
  bd.packMin = std::move(packMin);
  bd.packMax = std::move(packMax);
  bd.packShift = std::move(packShift);
  bd.packSentinel = packSentinel;
  bd.columnarBuild = columnarBuild;
  bd.buildTable = std::move(columnarBuildTablePtr);
  bd.nullBuffer = std::move(buildNullBuffer2);
  bd.nullStride = buildNullStride2;
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

  // Key-only row store (fingerprint verify under columnar payload).
  if (hasKeyRowStore) {
    GpuFixedRowStore keyStore;
    keyStore.row_buffer = static_cast<uint8_t*>(keyRowBuffer.data());
    keyStore.row_width = keyRowWidth;
    keyStore.num_rows = numRows;
    keyStore.num_fields = static_cast<int32_t>(keyRowFields.size());
    keyStore.fields = static_cast<const FieldDesc*>(keyFieldsBuffer.data());
    bd.keyRowStore = keyStore;
    bd.keyRowBuffer = std::move(keyRowBuffer);
    bd.keyFieldsBuffer = std::move(keyFieldsBuffer);
    bd.keyRowFields = std::move(keyRowFields);
    bd.keyRowWidth = keyRowWidth;
    bd.hasKeyRowStore = true;
  }

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

    probeGpuStore.row_buffer = static_cast<uint8_t*>(rowStoreInput->gpuRowData());
    probeGpuStore.row_width = probeRowWidth_;
    probeGpuStore.num_rows = probeRows;
    probeGpuStore.num_fields = probeFields_.size();
    probeGpuStore.fields = static_cast<const FieldDesc*>(probeFieldsBuffer_.data());

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
    transposeToRows(
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
      auto leftIdx = leftType->getChildIdxIfExists(name);
      int32_t byteWidth;
      if (leftIdx.has_value()) {
        int srcCol = leftIdx.value();
        byteWidth = probeFields_[srcCol].byte_width;
        probeGatherMappings_.push_back(
            {probeFields_[srcCol].offset, dstOffset, byteWidth});
        pSrcField.push_back(srcCol);
        pDstField.push_back(i);
      } else {
        int srcCol = rightType->getChildIdx(name);
        byteWidth = buildHostFields[srcCol].byte_width;
        buildGatherMappings_.push_back(
            {buildHostFields[srcCol].offset, dstOffset, byteWidth});
        bSrcField.push_back(srcCol);
        bDstField.push_back(i);
      }
      outputFields_.push_back({dstOffset, byteWidth});
      dstOffset += byteWidth;
    }
    outputRowWidth_ = (dstOffset + 7) & ~7;

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
  // Two interchangeable matchers behind one contract:
  //   cudf arm            : extract the key column(s) from the probe row
  //                         store, then cudf::hash_join::inner_join (cuco);
  //   row arm (hasRowTable): the row-native chained multimap, probed with
  //                         keys read straight from the probe row store —
  //                         the extract_keys step does not run at all.
  const int numKeys = static_cast<int>(leftKeyIndices_.size());
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndices;
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
      pk.offset[k] = probeFields_[leftKeyIndices_[k]].offset;
      pk.width[k] = probeFields_[leftKeyIndices_[k]].byte_width;
    }
    numMatches = rowTableProbe(
        probeGpuStore, pk, probeRows, stream, leftIndices, rightIndices);
  } else {
    if (static_cast<int>(probeKeyBuffers_.size()) != numKeys) {
      probeKeyBuffers_.clear();
      probeKeyBuffers_.resize(numKeys);
      probeKeyCapacity_ = 0;
    }
    if (probeRows > probeKeyCapacity_) {
      probeKeyCapacity_ = probeRows;
      for (int k = 0; k < numKeys; k++) {
        int32_t kw = probeFields_[leftKeyIndices_[k]].byte_width;
        probeKeyBuffers_[k] =
            rmm::device_buffer((int64_t)probeRows * kw, stream);
      }
    }

    std::vector<cudf::column_view> probeKeyViews;
    probeKeyViews.reserve(numKeys);
    for (int k = 0; k < numKeys; k++) {
      int32_t keyOffset = probeFields_[leftKeyIndices_[k]].offset;
      int32_t keyWidth = probeFields_[leftKeyIndices_[k]].byte_width;
      extractKeysFromRows(
          probeGpuStore, keyOffset, keyWidth,
          probeKeyBuffers_[k].data(), stream.value());
    probeKeyViews.emplace_back(
        cudf::data_type{
            keyWidth == 8 ? cudf::type_id::INT64 : cudf::type_id::INT32},
        static_cast<cudf::size_type>(probeRows),
        probeKeyBuffers_[k].data(), nullptr, 0);
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
          }
          break;
        }
      }
    }
    addRuntimeStat(
        "rowJoinEmitColumnar",
        RuntimeCounter(static_cast<int64_t>(emitColumnar_)));
  }

  // ---- Terminal join: transpose row buffer -> cudf columns (CudfVector) ----
  if (emitColumnar_ == 1) {
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

  return std::make_shared<RowStoreVector>(
      pool(),
      outputType_,
      static_cast<int64_t>(numMatches),
      std::move(outputRowData),
      std::move(outFieldsBuf),
      outputFields_,
      outputRowWidth_,
      stream);
}

// ============================================================================
// makeColumnarOutput — row->col transpose at the terminal join
// ============================================================================
// Converts the gathered fixed-stride row buffer (probeGatherBuffer_, layout
// described by outputFields_ / outputRowWidth_) into a column-major cudf::table
// wrapped in a CudfVector, so that downstream columnar cudf operators
// (aggregation, orderBy, ...) can consume it.
RowVectorPtr RowHashJoinProbe::makeColumnarOutput(
    int32_t numMatches,
    rmm::cuda_stream_view stream) {
  const int32_t numCols = outputType_->size();
  VELOX_CHECK_EQ(static_cast<size_t>(numCols), outputFields_.size());

  // Allocate one device buffer per output column and remember its base ptr.
  std::vector<std::unique_ptr<rmm::device_buffer>> colBuffers;
  colBuffers.reserve(numCols);
  std::vector<uint8_t*> colPtrs(numCols);
  for (int i = 0; i < numCols; i++) {
    int64_t bytes = static_cast<int64_t>(numMatches) *
        outputFields_[i].byte_width;
    auto buf = std::make_unique<rmm::device_buffer>(bytes, stream);
    colPtrs[i] = static_cast<uint8_t*>(buf->data());
    colBuffers.push_back(std::move(buf));
  }

  // Scatter row buffer -> columns.
  rowsToColumns(
      static_cast<const uint8_t*>(probeGatherBuffer_.data()),
      outputFields_.data(),
      colPtrs.data(),
      numCols,
      numMatches,
      outputRowWidth_,
      stream.value());

  // Wrap each device buffer as a cudf::column with the right type.
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numCols);
  for (int i = 0; i < numCols; i++) {
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
    columns.push_back(std::make_unique<cudf::column>(
        cudfType,
        static_cast<cudf::size_type>(numMatches),
        std::move(*colBuffers[i]),      // data buffer (ownership moved)
        std::move(mask),
        nullCount));
  }

  auto table = std::make_unique<cudf::table>(std::move(columns));

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
  // (Field k is leftKeys[k]; the build key rows were packed in rightKeys
  // order, so column k lines up on both sides.)
  //   cudf arm            : extract key columns from the keys-only row store,
  //                         then cudf::hash_join::inner_join;
  //   row arm (hasRowTable): probe the row-native chained multimap with keys
  //                         read straight from the keys-only row store — no
  //                         extraction at all.
  GpuFixedRowStore probeStore = rowStoreInput->getGpuRowStore();
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> leftIndices;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> rightIndices;
  int32_t numMatches = 0;

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
      pk.offset[k] = keyFields[k].offset;
      pk.width[k] = keyFields[k].byte_width;
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
            (int64_t)probeRows * keyFields[k].byte_width, stream);
      }
    }
    std::vector<cudf::column_view> probeKeyViews;
    probeKeyViews.reserve(numKeys);
    for (int k = 0; k < numKeys; k++) {
      extractKeysFromRows(
          probeStore,
          keyFields[k].offset,
          keyFields[k].byte_width,
          probeKeyBuffers_[k].data(),
          stream.value());
      probeKeyViews.emplace_back(
          cudf::data_type{
              keyFields[k].byte_width == 8 ? cudf::type_id::INT64
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
// FusedRowHashJoinProbe
// ============================================================================

namespace {
/// Byte width of a fixed-width Velox scalar type (fused acc / output layout).
int32_t fusedTypeWidth(const TypePtr& t) {
  switch (t->kind()) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
      return 1;
    case TypeKind::SMALLINT:
      return 2;
    case TypeKind::INTEGER:
    case TypeKind::REAL:
      return 4;
    case TypeKind::BIGINT:
    case TypeKind::DOUBLE:
      return 8;
    default:
      VELOX_FAIL(
          "FusedRowHashJoinProbe: unsupported TypeKind {}", (int)t->kind());
  }
}
} // namespace

FusedRowHashJoinProbe::FusedRowHashJoinProbe(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::vector<std::shared_ptr<const core::HashJoinNode>> joinNodes)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          joinNodes.back()->outputType(),
          joinNodes.back()->id(),
          "FusedRowHashJoinProbe",
          nvtx3::rgb{178, 34, 34}, // Firebrick
          NvtxMethodFlag::kAll,
          std::nullopt,
          joinNodes.back()),
      joinNodes_(std::move(joinNodes)),
      numSteps_(static_cast<int32_t>(joinNodes_.size())) {
  builds_.resize(numSteps_);
  buildFutures_.resize(numSteps_);
}

bool FusedRowHashJoinProbe::needsInput() const {
  return !noMoreInput_ && !finished_ && input_ == nullptr;
}

exec::BlockingReason FusedRowHashJoinProbe::isBlocked(ContinueFuture* future) {
  // Fetch build data from all N bridges; block on the first that isn't ready.
  for (int32_t s = 0; s < numSteps_; s++) {
    if (builds_[s]) {
      continue;
    }
    auto joinBridge = operatorCtx_->task()->getCustomJoinBridge(
        operatorCtx_->driverCtx()->splitGroupId, joinNodes_[s]->id());
    auto rowBridge = std::dynamic_pointer_cast<RowHashJoinBridge>(joinBridge);
    VELOX_CHECK_NOT_NULL(rowBridge);
    auto data = rowBridge->dataOrFuture(future);
    if (!data) {
      return exec::BlockingReason::kWaitForJoinBuild;
    }
    VELOX_CHECK(
        data->hasDeviceMap,
        "FusedRowHashJoinProbe requires device hash tables; "
        "set benchmarkFusedProbe before build");
    builds_[s] = std::move(data);
  }
  return exec::BlockingReason::kNotBlocked;
}

void FusedRowHashJoinProbe::doAddInput(RowVectorPtr input) {
  input_ = std::move(input);
}

void FusedRowHashJoinProbe::setupLayout(
    const std::vector<FieldDesc>& probeFields,
    rmm::cuda_stream_view stream) {
  // ---- Accumulator layout: union of all columns across C_0..C_N ----
  std::unordered_map<std::string, int32_t> accOffset;
  std::unordered_map<std::string, int32_t> accWidthByName;
  int32_t off = 0;
  auto addCol = [&](const std::string& n, int32_t w) {
    if (accOffset.count(n)) {
      return;
    }
    off = alignUp(off, w); // natural-align: the kernel reads acc with reinterpret_cast
    accOffset[n] = off;
    accWidthByName[n] = w;
    off += w;
  };

  // Lazy payload: only KEY probe columns enter the accumulator; pure-payload
  // probe columns are read from the probe at output time (survivors only).
  // Requires columnar seed (need the probe columns) and columnar output.
  {
    const auto& cfg = CudfConfig::getInstance();
    lazyPayload_ = cfg.benchmarkFusedLazyPayload && cfg.fusedColumnarSeed &&
        cfg.fusedColumnarOutput;
  }
  std::unordered_set<std::string> probeKeyNames;
  for (int32_t s = 0; s < numSteps_; s++) {
    for (auto& key : joinNodes_[s]->leftKeys()) {
      probeKeyNames.insert(key->name());
    }
  }
  auto c0 = joinNodes_[0]->sources()[0]->outputType();
  std::unordered_map<std::string, int32_t> c0Idx;
  for (int i = 0; i < c0->size(); i++) {
    c0Idx[c0->nameOf(i)] = i;
  }
  auto probeDeferred = [&](const std::string& n) {
    // A probe column that is never a join key -> not seeded, read lazily.
    return lazyPayload_ && c0Idx.count(n) && !probeKeyNames.count(n);
  };

  for (int i = 0; i < c0->size(); i++) {
    if (probeDeferred(c0->nameOf(i))) {
      continue; // deferred (or dead): stays out of the accumulator
    }
    addCol(c0->nameOf(i), fusedTypeWidth(c0->childAt(i)));
  }
  for (int32_t s = 0; s < numSteps_; s++) {
    auto out = joinNodes_[s]->outputType();
    for (int j = 0; j < out->size(); j++) {
      addCol(out->nameOf(j), fusedTypeWidth(out->childAt(j)));
    }
  }
  accWidth_ = (off + 7) & ~7;
  VELOX_CHECK_LE(
      accWidth_, kFusedAccMaxBytes,
      "Fused accumulator exceeds kFusedAccMaxBytes; increase the bound");

  // ---- probeToAcc: seed acc from the fact input row ----
  std::vector<FieldMapping> pmap;
  seedAccOffsets_.clear();
  seedAccWidths_.clear();
  seedProbeCol_.clear();
  for (int i = 0; i < c0->size(); i++) {
    auto name = c0->nameOf(i);
    if (probeDeferred(name)) {
      continue; // not seeded -- read lazily at output
    }
    pmap.push_back(
        {probeFields[i].offset, accOffset[name], probeFields[i].byte_width});
    // OPT 1 needs the same destination, but keyed off the input COLUMN rather
    // than a byte offset within a row. The device array itself is rebuilt each
    // batch (it holds column pointers), so only the offsets are cached here.
    seedAccOffsets_.push_back(accOffset[name]);
    seedAccWidths_.push_back(probeFields[i].byte_width);
    seedProbeCol_.push_back(i);
  }
  numProbeFields_ = static_cast<int32_t>(seedProbeCol_.size());
  probeToAccBuf_ =
      rmm::device_buffer(pmap.data(), pmap.size() * sizeof(FieldMapping), stream);

  // ---- Per-step key fields + build appends ----
  stepKeyFieldBufs_.resize(numSteps_);
  stepBuildToAccBufs_.resize(numSteps_);
  stepBuildColsBufs_.resize(numSteps_);
  std::vector<FusedJoinStep> hostSteps(numSteps_);
  for (int32_t s = 0; s < numSteps_; s++) {
    auto node = joinNodes_[s];
    auto leftType = node->sources()[0]->outputType();
    auto rightType = node->sources()[1]->outputType();
    auto out = node->outputType();

    // Key fields (read from acc). leftKeys[k] joins rightKeys[k], matching the
    // order the build table's key words were packed in.
    std::vector<FusedKeyField> kf;
    for (auto& key : node->leftKeys()) {
      auto name = key->name();
      kf.push_back({accOffset[name], accWidthByName[name]});
    }
    // fusedProbeKernel builds the lookup key in `uint64_t key[kMaxKeyCols]`, a
    // per-thread stack array. More keys than that is an out-of-bounds write, not
    // a graceful failure -- so check here rather than corrupt the stack.
    VELOX_CHECK_LE(
        kf.size(),
        static_cast<size_t>(kMaxKeyCols),
        "Fused join step {} has {} key columns; kMaxKeyCols is {}. "
        "Raise the bound (it is nearly free -- see GpuFusedProbe.cuh).",
        s,
        kf.size(),
        kMaxKeyCols);

    // Build appends: right-source output columns of this join. For a COLUMNAR
    // build (no row store), also record each field's cuDF column base pointer so
    // the probe can fetch it at `data + br*width`.
    const bool colBuild = builds_[s]->columnarBuild;
    std::vector<FieldMapping> bmap;
    std::vector<ColBuildField> colFields;
    cudf::table_view buildTv;
    if (colBuild) {
      VELOX_CHECK_NOT_NULL(builds_[s]->buildTable);
      buildTv = builds_[s]->buildTable->view();
    }
    for (int j = 0; j < out->size(); j++) {
      auto name = out->nameOf(j);
      if (leftType->getChildIdxIfExists(name).has_value()) {
        continue; // left-source column already lives in acc
      }
      int srcCol = rightType->getChildIdx(name);
      if (colBuild) {
        // No row store: width from the column, row-store offset unused.
        auto col = buildTv.column(srcCol);
        int32_t w = static_cast<int32_t>(cudfTypeWidth(col.type().id()));
        const uint8_t* base = static_cast<const uint8_t*>(col.head<uint8_t>()) +
            (int64_t)col.offset() * w;
        bmap.push_back({0, accOffset[name], w});
        colFields.push_back({base, accOffset[name], w});
      } else {
        bmap.push_back(
            {builds_[s]->hostFields[srcCol].offset,
             accOffset[name],
             builds_[s]->hostFields[srcCol].byte_width});
      }
    }

    stepKeyFieldBufs_[s] = rmm::device_buffer(
        kf.data(), kf.size() * sizeof(FusedKeyField), stream);
    stepBuildToAccBufs_[s] = rmm::device_buffer(
        bmap.data(), bmap.size() * sizeof(FieldMapping), stream);

    hostSteps[s].table = builds_[s]->deviceMap;
    hostSteps[s].keyFields =
        static_cast<const FusedKeyField*>(stepKeyFieldBufs_[s].data());
    hostSteps[s].numKeys = static_cast<int32_t>(kf.size());
    // Fingerprint verify reads the build key from the KEY-ONLY store when payload
    // is columnar; otherwise from the full row store.
    hostSteps[s].buildStore = builds_[s]->hasKeyRowStore
        ? builds_[s]->keyRowStore
        : builds_[s]->gpuRowStore;
    hostSteps[s].buildToAcc =
        static_cast<const FieldMapping*>(stepBuildToAccBufs_[s].data());
    hostSteps[s].numBuildFields = static_cast<int32_t>(bmap.size());
    hostSteps[s].columnarBuild = colBuild ? 1 : 0;
    if (colBuild) {
      stepBuildColsBufs_[s] = rmm::device_buffer(
          colFields.data(), colFields.size() * sizeof(ColBuildField), stream);
      hostSteps[s].buildCols =
          static_cast<const ColBuildField*>(stepBuildColsBufs_[s].data());
    }

    // Composite-key packing params, mirrored from the build (the build packed
    // its keys; the probe must pack the same way). keyFields[k] must be in the
    // SAME order the build packed -- both follow rightKeys()/leftKeys() order.
    hostSteps[s].packEnabled = builds_[s]->packEnabled ? 1 : 0;
    if (builds_[s]->packEnabled) {
      VELOX_CHECK_EQ(
          builds_[s]->packMin.size(), kf.size(),
          "packing key-column count mismatch at fused step {}", s);
      // Build key column offsets in the build row store, in the SAME order as
      // keyFields (leftKeys[k] <-> rightKeys[k]). Used by fingerprint-slot
      // verification to re-pack the build key from its row.
      auto rightKeys = node->rightKeys();
      for (size_t k = 0; k < kf.size(); k++) {
        hostSteps[s].packMin[k] = builds_[s]->packMin[k];
        hostSteps[s].packMax[k] = builds_[s]->packMax[k];
        hostSteps[s].packShift[k] = builds_[s]->packShift[k];
        // buildKeyOffset[k] locates key k in whatever store the verify reads:
        // the key-only store (columnar payload) or the full row store. Both list
        // keys in rightKeys()/keyColIndices order, matching packShift[k].
        if (builds_[s]->hasKeyRowStore) {
          hostSteps[s].buildKeyOffset[k] = builds_[s]->keyRowFields[k].offset;
        } else {
          int32_t bcol = static_cast<int32_t>(
              rightType->getChildIdx(rightKeys[k]->name()));
          hostSteps[s].buildKeyOffset[k] = builds_[s]->hostFields[bcol].offset;
        }
      }
      hostSteps[s].packSentinel = builds_[s]->packSentinel;
    }
  }
  // Whether ANY step uses a composite fingerprint slot. Lets the probe launch
  // the fingerprint-free kernel specialization when no step needs it (the common
  // case), so the verify path costs zero registers there.
  hasFingerprint_ = false;
  for (auto& hs : hostSteps) {
    if (hs.table.fingerprintSlot) {
      hasFingerprint_ = true;
      break;
    }
  }
  stepsBuf_ = rmm::device_buffer(
      hostSteps.data(), hostSteps.size() * sizeof(FusedJoinStep), stream);

  // ---- accToOut: acc -> final chain output (outputType column order) ----
  // Split output columns into ACC-sourced (produced by the chain / seeded keys)
  // and DEFERRED (pure probe payload read from the probe at output). outputFields_
  // still covers ALL columns (it sizes the output buffers); colOut/accToOut only
  // the acc-sourced ones; deferred lists the rest.
  std::vector<FieldMapping> omap;
  int32_t doff = 0;
  outputFields_.clear();
  outAccOffsets_.clear();
  accOutIdx_.clear();
  deferredProbeCol_.clear();
  deferredOutIdx_.clear();
  deferredWidth_.clear();
  for (int j = 0; j < outputType_->size(); j++) {
    auto name = outputType_->nameOf(j);
    int32_t w = fusedTypeWidth(outputType_->childAt(j));
    doff = alignUp(doff, w); // fused row-output path writes with reinterpret_cast
    outputFields_.push_back({doff, w});
    if (probeDeferred(name)) {
      deferredProbeCol_.push_back(c0Idx[name]);
      deferredOutIdx_.push_back(j);
      deferredWidth_.push_back(w);
    } else {
      omap.push_back({accOffset[name], doff, w});
      outAccOffsets_.push_back(accOffset[name]); // OPT 2
      accOutIdx_.push_back(j);
    }
    doff += w;
  }
  outputRowWidth_ = (doff + 7) & ~7;
  numOutFields_ = static_cast<int32_t>(accOutIdx_.size());
  numDeferred_ = static_cast<int32_t>(deferredProbeCol_.size());
  accToOutBuf_ =
      rmm::device_buffer(omap.data(), omap.size() * sizeof(FieldMapping), stream);
  outputFieldsBuffer_ = rmm::device_buffer(
      outputFields_.data(), outputFields_.size() * sizeof(FieldDesc), stream);

  outCountBuf_ = rmm::device_buffer(sizeof(int32_t), stream);
  layoutComputed_ = true;
}

RowVectorPtr FusedRowHashJoinProbe::doGetOutput() {
  // ---- Phase 1: harvest the batch launched on the PREVIOUS call ----
  //
  // Its kernel has been running on the GPU while the driver went upstream for
  // the next batch (scan + H2D), so this synchronize should find it already
  // complete. That is the entire point of deferring: see InFlight in the header.
  //
  // Reusing probeRowBuffer_/probeGatherBuffer_/outCountBuf_ for the batch we are
  // about to launch is safe precisely BECAUSE we synchronize here first -- the
  // previous kernel is done with them.
  RowVectorPtr harvested = nullptr;
  if (inflight_.valid) {
    int32_t numMatches = 0;
    cudaMemcpyAsync(
        &numMatches,
        outCountBuf_.data(),
        sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        inflight_.stream.value());
    inflight_.stream.synchronize();

    addRuntimeStat(
        "fusedProbeOutputRows",
        RuntimeCounter(static_cast<int64_t>(numMatches)));

    if (CudfConfig::getInstance().benchmarkSkipOutput) {
      std::vector<VectorPtr> children(outputType_->size());
      for (int i = 0; i < outputType_->size(); i++) {
        children[i] =
            BaseVector::createNullConstant(outputType_->childAt(i), 1, pool());
      }
      harvested = std::make_shared<RowVector>(
          pool(), outputType_, nullptr, 1, std::move(children));
    } else if (numMatches > 0) {
      // Fused probe replaced the whole chain, so its consumer is always a
      // columnar operator.
      harvested = CudfConfig::getInstance().fusedColumnarOutput
          ? makeColumnarOutputDirect(numMatches, inflight_.stream)
          : makeColumnarOutput(numMatches, inflight_.stream);
    }
    inflight_.valid = false;
  }

  // ---- Phase 2: launch the current input, WITHOUT waiting for it ----
  if (!input_) {
    finished_ = noMoreInput_ && !inflight_.valid;
    return harvested;
  }

  // ---- Determine probe input (RowStoreVector Path A or CudfVector Path B) ----
  rmm::cuda_stream_view stream{rmm::cuda_stream_default};
  int32_t probeRows = 0;
  GpuFixedRowStore probeGpuStore;

  auto rowStoreInput = std::dynamic_pointer_cast<RowStoreVector>(input_);
  // OPT 1 needs cuDF columns to read from, so it only applies to a CudfVector
  // input. In practice the fused probe replaces the WHOLE chain, so its input is
  // the columnar source -- but an upstream row-mode operator could hand us a
  // RowStoreVector, and then the row seed is already the right thing.
  const auto& cfg = CudfConfig::getInstance();
  const bool columnarSeed = cfg.fusedColumnarSeed && (rowStoreInput == nullptr);
  const bool columnarOut = cfg.fusedColumnarOutput;
  if (rowStoreInput) {
    stream = rowStoreInput->stream();
    probeRows = rowStoreInput->size();
    if (!initialized_) {
      probeFields_ = rowStoreInput->hostFields();
      probeRowWidth_ = rowStoreInput->rowWidth();
      initialized_ = true;
    }
    if (!fieldsUploaded_) {
      probeFieldsBuffer_ = rmm::device_buffer(
          probeFields_.data(), probeFields_.size() * sizeof(FieldDesc), stream);
      fieldsUploaded_ = true;
    }
    probeGpuStore.row_buffer =
        static_cast<uint8_t*>(rowStoreInput->gpuRowData());
    probeGpuStore.row_width = probeRowWidth_;
    probeGpuStore.num_rows = probeRows;
    probeGpuStore.num_fields = probeFields_.size();
    probeGpuStore.fields =
        static_cast<const FieldDesc*>(probeFieldsBuffer_.data());
  } else {
    auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input_);
    VELOX_CHECK_NOT_NULL(
        cudfInput, "FusedRowHashJoinProbe: input must be RowStoreVector or CudfVector");
    stream = cudfInput->stream();
    auto probeView = cudfInput->getTableView();
    probeRows = probeView.num_rows();
    if (!initialized_) {
      auto [pf, prw] = computeRowLayoutFromTable(probeView);
      probeFields_ = std::move(pf);
      probeRowWidth_ = prw;
      initialized_ = true;
    }
    if (!layoutComputed_) {
      setupLayout(probeFields_, stream);
    }

    if (columnarSeed) {
      // OPT 1: no transpose. Hand the kernel the column base pointers and let
      // each thread read its own tuple's fields directly -- lane i reads col[i],
      // which is coalesced, whereas the row store made lane i read at
      // i*row_width, which is not.
      //
      // Rebuilt every batch: these are raw column POINTERS into the input
      // table, and the input table is a different one each batch.
      // Base pointer of every probe column this batch -- used by the seed here
      // and by the lazy-payload deferred reads at output.
      probeColBase_.assign(probeView.num_columns(), nullptr);
      for (int c = 0; c < probeView.num_columns(); c++) {
        auto col = probeView.column(c);
        VELOX_CHECK(
            col.null_count() == 0,
            "FusedRowHashJoinProbe columnar seed: null probe column not supported");
        probeColBase_[c] = static_cast<const uint8_t*>(col.head<uint8_t>()) +
            (int64_t)col.offset() * probeFields_[c].byte_width;
      }
      std::vector<ColSeedField> seeds(numProbeFields_);
      for (int i = 0; i < numProbeFields_; i++) {
        seeds[i].data = probeColBase_[seedProbeCol_[i]];
        seeds[i].byte_width = seedAccWidths_[i];
        seeds[i].acc_offset = seedAccOffsets_[i];
      }
      colSeedBuf_ = rmm::device_buffer(
          seeds.data(), seeds.size() * sizeof(ColSeedField), stream);
    } else {
      int64_t probeRowBytes = (int64_t)probeRows * probeRowWidth_;
      if (probeRows > probeRowCapacity_) {
        probeRowCapacity_ = probeRows;
        probeRowBuffer_ = rmm::device_buffer(probeRowBytes, stream);
      }
      if (!fieldsUploaded_) {
        probeFieldsBuffer_ = rmm::device_buffer(
            probeFields_.data(), probeFields_.size() * sizeof(FieldDesc), stream);
        fieldsUploaded_ = true;
      }
      transposeToRows(
          probeView, probeFields_, probeRowWidth_,
          static_cast<uint8_t*>(probeRowBuffer_.data()), stream.value());
      probeGpuStore.row_buffer = static_cast<uint8_t*>(probeRowBuffer_.data());
      probeGpuStore.row_width = probeRowWidth_;
      probeGpuStore.num_rows = probeRows;
      probeGpuStore.num_fields = probeFields_.size();
      probeGpuStore.fields =
          static_cast<const FieldDesc*>(probeFieldsBuffer_.data());
    }
  }

  if (!layoutComputed_) {
    setupLayout(probeFields_, stream);
  }

  // ---- Allocate the output, upper-bounded at probeRows ----
  // An inner-join probe tuple survives at most once, so probeRows bounds the
  // output. Both layouts allocate the same TOTAL bytes; they differ only in
  // whether those bytes are one row buffer or N column buffers.
  if (columnarOut) {
    // OPT 2: one column buffer per OUTPUT column (all of them -- acc-sourced and
    // deferred alike write here), reused across batches. Sized at the upper
    // bound and handed to cudf::column at the true (smaller) size later.
    const int nOut = outputType_->size();
    if (probeRows > outColCapacity_ || outColBuffers_.empty()) {
      outColCapacity_ = probeRows;
      outColBuffers_.clear();
      outColBuffers_.reserve(nOut);
      for (int j = 0; j < nOut; j++) {
        outColBuffers_.emplace_back(
            (int64_t)probeRows * outputFields_[j].byte_width, stream);
      }
    }
    // Acc-sourced output columns (read from the accumulator in the kernel).
    std::vector<ColOutField> outs(numOutFields_);
    for (int k = 0; k < numOutFields_; k++) {
      const int j = accOutIdx_[k];
      outs[k].data = static_cast<uint8_t*>(outColBuffers_[j].data());
      outs[k].byte_width = outputFields_[j].byte_width;
      outs[k].acc_offset = outAccOffsets_[k];
    }
    colOutBuf_ = rmm::device_buffer(
        outs.data(), outs.size() * sizeof(ColOutField), stream);
    // OPT 3: deferred (pure probe-payload) output columns, read from the probe
    // column at the survivor's own row index.
    if (numDeferred_ > 0) {
      std::vector<DeferredField> defs(numDeferred_);
      for (int k = 0; k < numDeferred_; k++) {
        defs[k].probeData = probeColBase_[deferredProbeCol_[k]];
        defs[k].outData =
            static_cast<uint8_t*>(outColBuffers_[deferredOutIdx_[k]].data());
        defs[k].byte_width = deferredWidth_[k];
      }
      deferredBuf_ = rmm::device_buffer(
          defs.data(), defs.size() * sizeof(DeferredField), stream);
    }
  } else {
    int64_t neededOutput = (int64_t)probeRows * outputRowWidth_;
    if (probeRows > gatherCapacity_) {
      gatherCapacity_ = probeRows;
      probeGatherBuffer_ = rmm::device_buffer(neededOutput, stream);
    } else if (neededOutput > static_cast<int64_t>(probeGatherBuffer_.size())) {
      probeGatherBuffer_ = rmm::device_buffer(neededOutput, stream);
    }
  }
  cudaMemsetAsync(outCountBuf_.data(), 0, sizeof(int32_t), stream.value());

  // ---- Fused N-way probe: one kernel walks each tuple through all joins ----
  fusedProbe(
      probeGpuStore,
      probeRows,
      static_cast<const FieldMapping*>(probeToAccBuf_.data()),
      numProbeFields_,
      columnarSeed ? static_cast<const ColSeedField*>(colSeedBuf_.data())
                   : nullptr,
      static_cast<const FusedJoinStep*>(stepsBuf_.data()),
      numSteps_,
      accWidth_,
      static_cast<const FieldMapping*>(accToOutBuf_.data()),
      columnarOut ? static_cast<const ColOutField*>(colOutBuf_.data()) : nullptr,
      numOutFields_,
      (columnarOut && numDeferred_ > 0)
          ? static_cast<const DeferredField*>(deferredBuf_.data())
          : nullptr,
      numDeferred_,
      outputRowWidth_,
      columnarOut ? nullptr
                  : static_cast<uint8_t*>(probeGatherBuffer_.data()),
      static_cast<int32_t*>(outCountBuf_.data()),
      hasFingerprint_,
      stream.value());

  // DO NOT synchronize here. Record the batch as in-flight and return the one we
  // harvested at the top of this call. The driver will now go upstream for the
  // next batch (scan + Velox->cuDF + H2D) while this kernel runs, and we collect
  // its outCount on the next doGetOutput() -- by which point it is long done.
  inflight_.valid = true;
  inflight_.probeRows = probeRows;
  inflight_.stream = stream;

  rowStoreInput.reset();
  input_.reset();
  finished_ = false; // a batch is in flight; we are not done until it is drained

  return harvested;
}

RowVectorPtr FusedRowHashJoinProbe::makeColumnarOutput(
    int32_t numMatches,
    rmm::cuda_stream_view stream) {
  const int32_t numCols = outputType_->size();
  VELOX_CHECK_EQ(static_cast<size_t>(numCols), outputFields_.size());

  std::vector<std::unique_ptr<rmm::device_buffer>> colBuffers;
  colBuffers.reserve(numCols);
  std::vector<uint8_t*> colPtrs(numCols);
  for (int i = 0; i < numCols; i++) {
    int64_t bytes =
        static_cast<int64_t>(numMatches) * outputFields_[i].byte_width;
    auto buf = std::make_unique<rmm::device_buffer>(bytes, stream);
    colPtrs[i] = static_cast<uint8_t*>(buf->data());
    colBuffers.push_back(std::move(buf));
  }

  rowsToColumns(
      static_cast<const uint8_t*>(probeGatherBuffer_.data()),
      outputFields_.data(),
      colPtrs.data(),
      numCols,
      numMatches,
      outputRowWidth_,
      stream.value());

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numCols);
  for (int i = 0; i < numCols; i++) {
    auto cudfType = veloxToCudfDataType(outputType_->childAt(i));
    columns.push_back(std::make_unique<cudf::column>(
        cudfType,
        static_cast<cudf::size_type>(numMatches),
        std::move(*colBuffers[i]),
        rmm::device_buffer{},
        0));
  }

  auto table = std::make_unique<cudf::table>(std::move(columns));

  // KEEP the gather buffer. rowsToColumns SCATTERS out of it into fresh column
  // buffers -- unlike RowHashJoinProbe, which hands its buffer to the output
  // RowStoreVector and so must relinquish it. Freeing it here forced a fresh
  // rmm allocation on every single batch, and allocation is the dominant
  // CPU-side cost in this system (see the cudaMemsetAsync/cudaMallocAsync
  // totals in any nsys profile). Reusing it is safe: the next batch's kernel is
  // queued on the SAME stream behind this scatter, so CUDA stream ordering
  // guarantees the read completes before the write begins.

  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      static_cast<vector_size_t>(numMatches),
      std::move(table),
      stream);
}

RowVectorPtr FusedRowHashJoinProbe::makeColumnarOutputDirect(
    int32_t numMatches,
    rmm::cuda_stream_view stream) {
  // OPT 2: the kernel already wrote cuDF-layout columns. There is no
  // rows_to_columns pass here -- that whole kernel is gone.
  //
  // The staging buffers are sized at the upper bound (probeRows), so we copy the
  // live prefix into exactly-sized column buffers rather than handing the
  // oversized ones to cudf::column. That costs one CONTIGUOUS D2D copy of just
  // the surviving rows (a few thousand rows for an aggregation-terminated TPC-H
  // chain) and, crucially, keeps the staging buffers reusable across batches --
  // giving them away would force a fresh rmm allocation every batch, and
  // allocation is the dominant CPU-side cost in this system.
  const int32_t numCols = outputType_->size();
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numCols);
  for (int i = 0; i < numCols; i++) {
    const int64_t bytes =
        static_cast<int64_t>(numMatches) * outputFields_[i].byte_width;
    rmm::device_buffer colData(bytes, stream);
    cudaMemcpyAsync(
        colData.data(),
        outColBuffers_[i].data(),
        bytes,
        cudaMemcpyDeviceToDevice,
        stream.value());
    columns.push_back(std::make_unique<cudf::column>(
        veloxToCudfDataType(outputType_->childAt(i)),
        static_cast<cudf::size_type>(numMatches),
        std::move(colData),
        rmm::device_buffer{},
        0));
  }
  return std::make_shared<CudfVector>(
      pool(),
      outputType_,
      static_cast<vector_size_t>(numMatches),
      std::make_unique<cudf::table>(std::move(columns)),
      stream);
}

void FusedRowHashJoinProbe::doNoMoreInput() {
  Operator::noMoreInput();
}

bool FusedRowHashJoinProbe::isFinished() {
  // Not finished while a batch is still in flight -- it has yet to be harvested.
  return noMoreInput_ && input_ == nullptr && !inflight_.valid;
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
