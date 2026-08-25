/*
 * RowOrderBy.cpp (branch row-sort, 2026-08-24) -- see RowOrderBy.h.
 */
#include "velox/experimental/cudf/exec/HostRowVector.h"
#include "velox/experimental/cudf/exec/RowOrderBy.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/CudfLocalPartition.h"
#include "velox/experimental/cudf/exec/DedicatedStream.h"
#include "velox/experimental/cudf/exec/DeferralPlan.h"
#include "velox/experimental/cudf/exec/DeferralStats.h"
#include "velox/experimental/cudf/exec/GpuRowOps.cuh"
#include "velox/experimental/cudf/exec/RowHashJoin.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include "velox/exec/Driver.h"
#include "velox/exec/Task.h"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <numeric>
#include <thread>

namespace facebook::velox::cudf_velox {

namespace {
int32_t padTo8(int32_t n) {
  return (n + 7) & ~7;
}

// Host-side work in the (single-driver) sort pipeline is embarrassingly
// parallel over rows; the other pipelines have finished by the time the
// blocking sort emits, so the cores are idle. Plain std::threads over
// contiguous row ranges (no shared mutation: disjoint destination bytes).
constexpr int kMaxEmitThreads = 24;
template <typename F>
void parallelRows(int64_t n, F&& body, int64_t kMinPerThread = 1 << 15) {
  int threads = static_cast<int>(std::min<int64_t>(
      std::thread::hardware_concurrency(), kMaxEmitThreads));
  threads = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(threads, n / kMinPerThread)));
  if (threads <= 1) {
    body(0, n, 0);
    return;
  }
  std::vector<std::thread> pool;
  pool.reserve(threads);
  const int64_t per = (n + threads - 1) / threads;
  for (int t = 0; t < threads; t++) {
    const int64_t b = t * per;
    const int64_t e = std::min<int64_t>(n, b + per);
    if (b >= e) {
      break;
    }
    pool.emplace_back([&body, b, e, t] { body(b, e, t); });
  }
  for (auto& th : pool) {
    th.join();
  }
}

// Make `dst` wait for all work enqueued so far on `src`.
void waitOnStream(rmm::cuda_stream_view dst, rmm::cuda_stream_view src) {
  if (dst.value() == src.value()) {
    return;
  }
  cudaEvent_t ev;
  cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
  cudaEventRecord(ev, src.value());
  cudaStreamWaitEvent(dst.value(), ev, 0);
  cudaEventDestroy(ev);
}
} // namespace

bool rowSortConsumesGather(
    const exec::Operator* next,
    const core::PlanNodePtr& planRoot) {
  if (!CudfConfig::getInstance().benchmarkRowSort || next == nullptr ||
      dynamic_cast<const CudfLocalPartition*>(next) == nullptr) {
    return false;
  }
  return orderByBehindGather(planRoot, next->planNodeId()) != nullptr;
}

std::vector<std::string> RowOrderBy::sortKeyNames(
    const core::OrderByNode& node) {
  std::vector<std::string> out;
  for (const auto& k : node.sortingKeys()) {
    out.push_back(k->name());
  }
  return out;
}

RowOrderBy::RowOrderBy(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    std::shared_ptr<const core::OrderByNode> orderByNode)
    : CudfOperatorBase(
          operatorId,
          driverCtx,
          orderByNode->outputType(),
          orderByNode->id(),
          "RowOrderBy",
          nvtx3::rgb{72, 209, 204},
          NvtxMethodFlag::kAll,
          std::nullopt,
          orderByNode),
      orderByNode_(std::move(orderByNode)),
      stream_(acquireDedicatedStream()) {
  keyNames_ = sortKeyNames(*orderByNode_);
  for (size_t i = 0; i < orderByNode_->sortingKeys().size(); ++i) {
    const auto& so = orderByNode_->sortingOrders()[i];
    orders_.push_back(
        so.isAscending() ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    nullOrders_.push_back(
        (so.isNullsFirst() ^ !so.isAscending()) ? cudf::null_order::BEFORE
                                                 : cudf::null_order::AFTER);
  }
}

int32_t RowOrderBy::fieldIndexOf(const std::string& name) const {
  if (!fieldNames_.empty()) {
    for (size_t f = 0; f < fieldNames_.size(); f++) {
      if (fieldNames_[f] == name) {
        return static_cast<int32_t>(f);
      }
    }
    return -1;
  }
  const auto idx = outputType_->getChildIdxIfExists(name);
  return idx.has_value() ? static_cast<int32_t>(idx.value()) : -1;
}

void RowOrderBy::captureLayout(const RowStoreVector& v) {
  fields_ = v.hostFields();
  rowWidth_ = v.rowWidth();
  fieldNames_ = v.fieldNames();
  nullStride_ = v.nullStride();
  nullPad_ = nullStride_ > 0 ? padTo8(nullStride_) : 0;
  rowIdField_ = v.hasProvenance() ? v.rowIdField() : -1;
  hasStrings_ = false;
  for (const auto& f : fields_) {
    if (f.offset >= 0 && f.kind == kFieldString) {
      hasStrings_ = true;
    }
  }
  layoutReady_ = true;
}

void RowOrderBy::doAddInput(RowVectorPtr input) {
  if (input->size() == 0) {
    return;
  }
  if (auto rs = std::dynamic_pointer_cast<RowStoreVector>(input)) {
    VELOX_CHECK(
        cudfInputs_.empty(), "RowOrderBy: mixed RowStoreVector/CudfVector input");
    if (!layoutReady_) {
      captureLayout(*rs);
    } else {
      // Every batch must share ONE layout. A batch-level adaptive pack can
      // only switch layouts mid-query when the chain endpoint reports
      // survivors per batch; a sort endpoint reports once at the end, so
      // its own input never mixes -- this guards the invariant.
      VELOX_CHECK(
          rs->rowWidth() == rowWidth_ && rs->fieldNames() == fieldNames_ &&
              rs->hasProvenance() == (rowIdField_ >= 0),
          "RowOrderBy: input batches with different row layouts "
          "(eager/deferred mix) are not supported");
    }
    totalRows_ += rs->size();
    totalChars_ += rs->charsBytes();
    rowInputs_.push_back(std::move(rs));
    return;
  }
  auto cv = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(
      cv, "RowOrderBy: input must be RowStoreVector or CudfVector");
  VELOX_CHECK(
      rowInputs_.empty(), "RowOrderBy: mixed RowStoreVector/CudfVector input");
  totalRows_ += cv->size();
  cudfInputs_.push_back(std::move(cv));
}

// Row inputs -> one contiguous row store (+ padded null sidecar, string
// heap with rebased offsets, GLOBAL rowids). Leaves the unsorted rows in
// sorted_/sortedNulls_ for sortRows() to permute in place.
void RowOrderBy::concatenateRowInputs() {
  const int64_t rowsBytes = totalRows_ * rowWidth_;
  sorted_ = rmm::device_buffer(rowsBytes, stream_);
  if (hasStrings_) {
    heap_ = rmm::device_buffer(static_cast<size_t>(totalChars_), stream_);
  }
  if (nullPad_ > 0) {
    sortedNulls_ = rmm::device_buffer(totalRows_ * nullPad_, stream_);
    cudaMemsetAsync(
        sortedNulls_.data(), 0, totalRows_ * nullPad_, stream_.value());
  }
  // String-slot offsets for rebasing heap offsets.
  std::vector<int32_t> strOffs;
  for (const auto& f : fields_) {
    if (f.offset >= 0 && f.kind == kFieldString) {
      strOffs.push_back(f.offset);
    }
  }
  rmm::device_buffer strOffsDev;
  if (!strOffs.empty()) {
    strOffsDev = rmm::device_buffer(
        strOffs.data(), strOffs.size() * sizeof(int32_t), stream_);
  }
  int64_t rowsSoFar = 0;
  int64_t heapSoFar = 0;
  int64_t idSpaceSoFar = 0; // global rowid base of the NEXT new store
  uint8_t* rowsBase = static_cast<uint8_t*>(sorted_.data());
  for (auto& v : rowInputs_) {
    const int64_t n = v->size();
    waitOnStream(stream_, v->stream());
    uint8_t* dst = rowsBase + rowsSoFar * rowWidth_;
    cudaMemcpyAsync(
        dst,
        v->gpuRowData(),
        n * rowWidth_,
        cudaMemcpyDeviceToDevice,
        stream_.value());
    if (hasStrings_ && v->charsBytes() > 0) {
      cudaMemcpyAsync(
          static_cast<uint8_t*>(heap_.data()) + heapSoFar,
          v->charsData(),
          v->charsBytes(),
          cudaMemcpyDeviceToDevice,
          stream_.value());
      rebaseStringOffsets(
          dst,
          static_cast<int32_t>(n),
          rowWidth_,
          static_cast<const int32_t*>(strOffsDev.data()),
          static_cast<int32_t>(strOffs.size()),
          heapSoFar,
          stream_.value());
      heapSoFar += v->charsBytes();
    }
    if (nullPad_ > 0) {
      const auto store = v->getGpuRowStore();
      if (store.null_bytes != nullptr && store.null_stride > 0) {
        cudaMemcpy2DAsync(
            static_cast<uint8_t*>(sortedNulls_.data()) + rowsSoFar * nullPad_,
            nullPad_,
            store.null_bytes,
            store.null_stride,
            store.null_stride,
            n,
            cudaMemcpyDeviceToDevice,
            stream_.value());
      }
    }
    if (rowIdField_ >= 0) {
      // Per-batch provenance: make the rowid global = store base + local.
      // The base advances by the STORE's own row count (the pack batch it
      // retains), NOT by this input batch's size: after a join the batch
      // holds only the matches, while its rowids index the whole store.
      auto store = v->provenanceStore();
      VELOX_CHECK_NOT_NULL(store);
      int64_t base;
      if (!stores_.empty() && stores_.back() == store) {
        base = storeBases_.back(); // several batches of one store
      } else {
        base = idSpaceSoFar;
        stores_.push_back(store);
        storeBases_.push_back(base);
        idSpaceSoFar += store->totalRows();
      }
      addInt64Field(
          dst,
          static_cast<int32_t>(n),
          rowWidth_,
          fields_[rowIdField_].offset,
          base,
          stream_.value());
    }
    rowsSoFar += n;
  }
  stream_.synchronize();
  rowInputs_.clear(); // frees the per-batch GPU buffers
}

// Columnar inputs -> one row store via the join's transpose helpers.
void RowOrderBy::transposeCudfInputs() {
  auto tbl = getConcatenatedTable(
      std::exchange(cudfInputs_, {}), outputType_, stream_, get_output_mr());
  VELOX_CHECK_NOT_NULL(tbl);
  const auto view = tbl->view();
  auto [f, w] = rowLayoutFromCudfTable(view);
  fields_ = std::move(f);
  rowWidth_ = w;
  fieldNames_.clear();
  nullStride_ = nullPad_ = 0;
  rowIdField_ = -1;
  hasStrings_ = false;
  for (const auto& fd : fields_) {
    if (fd.kind == kFieldString) {
      hasStrings_ = true;
    }
  }
  layoutReady_ = true;
  sorted_ = rmm::device_buffer(totalRows_ * rowWidth_, stream_);
  heap_ = transposeCudfTableToRows(
      view,
      fields_,
      rowWidth_,
      static_cast<uint8_t*>(sorted_.data()),
      stream_.value());
  stream_.synchronize(); // the table must outlive the transpose
}

// Keys -> cudf columns -> sorted_order -> ONE row gather.
void RowOrderBy::sortRows() {
  VELOX_CHECK_LE(
      totalRows_,
      static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "RowOrderBy: > 2^31 rows");
  const auto n = static_cast<int32_t>(totalRows_);
  GpuFixedRowStore store;
  store.row_buffer = static_cast<uint8_t*>(sorted_.data());
  store.row_width = rowWidth_;
  store.num_rows = n;
  store.num_fields = static_cast<int32_t>(fields_.size());
  store.fields = nullptr;
  store.null_bytes =
      nullPad_ > 0 ? static_cast<const uint8_t*>(sortedNulls_.data()) : nullptr;
  store.null_stride = nullPad_;
  store.chars = hasStrings_ ? static_cast<const uint8_t*>(heap_.data()) : nullptr;
  store.chars_bytes = hasStrings_ ? static_cast<int64_t>(heap_.size()) : 0;

  std::vector<rmm::device_buffer> keyBufs;
  std::vector<rmm::device_buffer> keyMasks;
  std::vector<std::unique_ptr<cudf::column>> keyCols;
  std::vector<cudf::column_view> keyViews;
  for (const auto& name : keyNames_) {
    const int32_t fi = fieldIndexOf(name);
    VELOX_CHECK(
        fi >= 0 && fields_[fi].offset >= 0,
        "RowOrderBy: sort key '{}' is not in the row layout",
        name);
    const auto& fd = fields_[fi];
    const auto type = outputType_->childAt(outputType_->getChildIdx(name));
    // Validity from the input sidecar (bit index = field index).
    rmm::device_buffer mask{};
    cudf::size_type nullCount = 0;
    if (nullPad_ > 0) {
      mask = rmm::device_buffer(
          cudf::bitmask_allocation_size_bytes(n), stream_);
      sidecarToMask(
          static_cast<const uint8_t*>(sortedNulls_.data()),
          nullPad_,
          fi,
          n,
          static_cast<uint32_t*>(mask.data()),
          stream_.value());
      nullCount = cudf::null_count(
          static_cast<const cudf::bitmask_type*>(mask.data()), 0, n, stream_);
    }
    if (fd.kind == kFieldString) {
      const size_t offBytes = (static_cast<size_t>(n) + 1) * sizeof(int64_t);
      if (offBytes > strOffsetsDev_.size()) {
        strOffsetsDev_ = rmm::device_buffer(offBytes, stream_);
      }
      const int64_t totalChars = stringFieldOffsets(
          store.row_buffer,
          n,
          rowWidth_,
          fd.offset,
          static_cast<int64_t*>(strOffsetsDev_.data()),
          stream_.value());
      auto offsetsCol = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT32},
          n + 1,
          cudf::mask_state::UNALLOCATED,
          stream_,
          get_output_mr());
      rmm::device_buffer chars(totalChars, stream_);
      stringFieldToChars(
          store.row_buffer,
          n,
          rowWidth_,
          fd.offset,
          store.chars,
          static_cast<const int64_t*>(strOffsetsDev_.data()),
          offsetsCol->mutable_view().data<int32_t>(),
          static_cast<uint8_t*>(chars.data()),
          stream_.value());
      keyCols.push_back(cudf::make_strings_column(
          n, std::move(offsetsCol), std::move(chars), nullCount, std::move(mask)));
      keyViews.push_back(keyCols.back()->view());
      continue;
    }
    rmm::device_buffer buf(static_cast<size_t>(n) * fd.byte_width, stream_);
    extractKeysFromRows(store, fd.offset, fd.byte_width, buf.data(), stream_.value());
    keyViews.emplace_back(
        veloxToCudfDataType(type),
        n,
        buf.data(),
        nullCount > 0 ? static_cast<const cudf::bitmask_type*>(mask.data())
                      : nullptr,
        nullCount);
    keyBufs.push_back(std::move(buf));
    keyMasks.push_back(std::move(mask));
  }

  cudaEvent_t evSortBegin, evSortEnd, evGatherEnd;
  cudaEventCreate(&evSortBegin);
  cudaEventCreate(&evSortEnd);
  cudaEventCreate(&evGatherEnd);
  cudaEventRecord(evSortBegin, stream_.value());
  const auto tp0 = std::chrono::steady_clock::now();
  auto perm = cudf::sorted_order(
      cudf::table_view(keyViews), orders_, nullOrders_, stream_, get_output_mr());
  cudaEventRecord(evSortEnd, stream_.value());
  const int32_t* permData = perm->view().data<int32_t>();

  // One gather of whole rows by the permutation (slots copied verbatim;
  // the heap is shared, so string offsets stay valid).
  rmm::device_buffer gathered(sorted_.size(), stream_);
  gatherRowsWarp(
      store, permData, n, static_cast<uint8_t*>(gathered.data()), stream_.value());
  if (nullPad_ > 0) {
    GpuFixedRowStore nstore;
    nstore.row_buffer = static_cast<uint8_t*>(sortedNulls_.data());
    nstore.row_width = nullPad_;
    nstore.num_rows = n;
    nstore.num_fields = 0;
    nstore.fields = nullptr;
    rmm::device_buffer gatheredNulls(sortedNulls_.size(), stream_);
    gatherRowsWarp(
        nstore, permData, n, static_cast<uint8_t*>(gatheredNulls.data()),
        stream_.value());
    sortedNulls_ = std::move(gatheredNulls);
  }
  cudaEventRecord(evGatherEnd, stream_.value());
  // While the device sorts and gathers, COALESCE the retained host batches
  // into contiguous per-store columns (the CPU hybrid sort's trick,
  // overlapped here with the GPU sort): the final permutation gather then
  // does one plain indexed read per cell instead of a per-batch scattered
  // extraction.
  const auto tpCoalesce = std::chrono::steady_clock::now();
  coalesceStores();
  addRuntimeStat(
      "rowSortCoalesceNanos",
      RuntimeCounter(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - tpCoalesce)
              .count(),
          RuntimeCounter::Unit::kNanos));
  stream_.synchronize();
  {
    float msSort = 0, msGather = 0;
    cudaEventElapsedTime(&msSort, evSortBegin, evSortEnd);
    cudaEventElapsedTime(&msGather, evSortEnd, evGatherEnd);
    addRuntimeStat(
        "rowSortKeysDevNanos",
        RuntimeCounter(
            static_cast<int64_t>(msSort * 1e6), RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "rowSortGatherDevNanos",
        RuntimeCounter(
            static_cast<int64_t>(msGather * 1e6),
            RuntimeCounter::Unit::kNanos));
    addRuntimeStat(
        "rowSortGatherDevBytes",
        RuntimeCounter(
            static_cast<int64_t>(totalRows_) * rowWidth_,
            RuntimeCounter::Unit::kBytes));
    cudaEventDestroy(evSortBegin);
    cudaEventDestroy(evSortEnd);
    cudaEventDestroy(evGatherEnd);
  }
  sorted_ = std::move(gathered);
  addRuntimeStat(
      "rowSortNanos",
      RuntimeCounter(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - tp0)
              .count(),
          RuntimeCounter::Unit::kNanos));
  addRuntimeStat("rowSortRows", RuntimeCounter(static_cast<int64_t>(n)));
  addRuntimeStat(
      "rowSortRowWidthBytes", RuntimeCounter(static_cast<int64_t>(rowWidth_)));
}

void RowOrderBy::resolveOutputOnce() {
  if (emitHost_ >= 0) {
    return;
  }
  emitHost_ = 0;
  if (auto* driver = operatorCtx_->driver()) {
    const auto ops = driver->operators();
    for (size_t i = 0; i + 1 < ops.size(); i++) {
      if (ops[i] == this) {
        if (dynamic_cast<CudfToVelox*>(ops[i + 1]) != nullptr) {
          emitHost_ = 1; // CPU exit: emit host RowVectors directly
        }
        break;
      }
    }
  }
  outFieldIdx_.assign(outputType_->size(), -1);
  deferredCols_.clear();
  for (size_t i = 0; i < outputType_->size(); i++) {
    const int32_t fi = fieldIndexOf(outputType_->nameOf(i));
    if (fi >= 0 && fields_[fi].offset >= 0) {
      outFieldIdx_[i] = fi;
    } else {
      VELOX_CHECK(
          !stores_.empty(),
          "RowOrderBy: column '{}' absent from the row layout and no "
          "provenance store",
          outputType_->nameOf(i));
      deferredCols_.push_back(static_cast<int32_t>(i));
    }
  }
  addRuntimeStat(
      "rowSortEmitHost", RuntimeCounter(static_cast<int64_t>(emitHost_)));
}

void RowOrderBy::doNoMoreInput() {
  Operator::noMoreInput();
  if (totalRows_ == 0) {
    finished_ = true;
    return;
  }
  if (!rowInputs_.empty()) {
    concatenateRowInputs();
  } else {
    transposeCudfInputs();
  }
  resolveOutputOnce(); // deferred columns are needed by the coalesce
  sortRows();
  // Chain endpoint (batch-level adaptive deferral): a sort never reduces,
  // so EVERY input row survives to the exit and the deferred payload would
  // have to be gathered host-side for all of them -- deferral only pays
  // when an upstream join thinned the spine (S/P = join survival). Unlike
  // the join's CPU-exit rule (S = 0), report the true survivor count.
  DeferralStats::instance().recordSurvived(orderByNode_->id(), totalRows_);
}

// Per-store contiguous columns of every deferred output column, in store
// (= global rowid) order. Each store is extracted once, sequentially, by
// its own thread; the store's batches are released as soon as its columns
// are coalesced (peak host memory ~ +1 store).
void RowOrderBy::coalesceStores() {
  const size_t S = stores_.size();
  const size_t D = deferredCols_.size();
  storeCols_.assign(S, {});
  if (S == 0 || D == 0) {
    return;
  }
  std::vector<int32_t> childIdx(D);
  for (size_t d = 0; d < D; d++) {
    childIdx[d] = static_cast<int32_t>(stores_[0]->rowType()->getChildIdx(
        outputType_->nameOf(deferredCols_[d])));
  }
  parallelRows(
      static_cast<int64_t>(S),
      [&](int64_t b, int64_t e, int /*t*/) {
        std::vector<exec::HybridRowId> rowIds;
        std::vector<const char*> sentinels;
        std::vector<int32_t> seq;
        for (int64_t k = b; k < e; k++) {
          const auto m = static_cast<int32_t>(stores_[k]->totalRows());
          seq.resize(m);
          std::iota(seq.begin(), seq.end(), 0);
          stores_[k]->idsForGlobalRows(seq.data(), m, rowIds);
          auto& cols = storeCols_[k];
          cols.resize(D);
          for (size_t d = 0; d < D; d++) {
            auto vec = BaseVector::create(
                outputType_->childAt(deferredCols_[d]), m, pool());
            stores_[k]->gather(childIdx[d], rowIds, vec, sentinels);
            cols[d] = std::move(vec);
          }
          stores_[k].reset(); // free the retained batches
        }
      },
      /*kMinPerThread=*/1);
}

std::vector<VectorPtr> RowOrderBy::gatherDeferred(
    const std::vector<int64_t>& gids,
    int32_t n) {
  const size_t S = storeCols_.size();
  const size_t D = deferredCols_.size();
  // global id -> (store, local) once for all columns.
  std::vector<int32_t> storeOf(n), localOf(n);
  parallelRows(n, [&](int64_t b, int64_t e, int /*t*/) {
    for (int64_t i = b; i < e; i++) {
      const auto it =
          std::upper_bound(storeBases_.begin(), storeBases_.end(), gids[i]);
      const auto k = static_cast<int32_t>(it - storeBases_.begin() - 1);
      storeOf[i] = k;
      localOf[i] = static_cast<int32_t>(gids[i] - storeBases_[k]);
    }
  });
  std::vector<VectorPtr> out(D);
  for (size_t d = 0; d < D; d++) {
    const auto& type = outputType_->childAt(deferredCols_[d]);
    auto res = BaseVector::create(type, n, pool());
    bool anyNulls = false;
    for (size_t k = 0; k < S; k++) {
      anyNulls = anyNulls || storeCols_[k][d]->mayHaveNulls();
    }
    if (type->kind() == TypeKind::VARCHAR || type->kind() == TypeKind::VARBINARY) {
      // Views from the per-store columns; bytes copied into ONE shared
      // buffer (per-range totals first), set without copying.
      std::vector<const StringView*> src(S);
      for (size_t k = 0; k < S; k++) {
        src[k] = storeCols_[k][d]->template asFlatVector<StringView>()->rawValues();
      }
      auto* fv = res->template asFlatVector<StringView>();
      std::vector<int64_t> rangeBytes(kMaxEmitThreads, 0);
      parallelRows(n, [&](int64_t b, int64_t e, int t) {
        int64_t bytes = 0;
        for (int64_t i = b; i < e; i++) {
          const auto& v = src[storeOf[i]][localOf[i]];
          if (!v.isInline()) {
            bytes += v.size();
          }
        }
        rangeBytes[t] = bytes;
      });
      int64_t total = 0;
      std::vector<int64_t> rangeBase(kMaxEmitThreads, 0);
      for (int t = 0; t < kMaxEmitThreads; t++) {
        rangeBase[t] = total;
        total += rangeBytes[t];
      }
      char* base = nullptr;
      if (total > 0) {
        auto buf = AlignedBuffer::allocate<char>(total, pool());
        base = buf->asMutable<char>();
        fv->setStringBuffers({buf});
      }
      parallelRows(n, [&](int64_t b, int64_t e, int t) {
        int64_t cursor = rangeBase[t];
        for (int64_t i = b; i < e; i++) {
          const auto& v = src[storeOf[i]][localOf[i]];
          if (v.isInline()) {
            fv->setNoCopy(static_cast<vector_size_t>(i), v);
          } else {
            char* dst = base + cursor;
            std::memcpy(dst, v.data(), v.size());
            fv->setNoCopy(static_cast<vector_size_t>(i), StringView(dst, v.size()));
            cursor += v.size();
          }
        }
      });
    } else {
      VELOX_CHECK(
          type->kind() != TypeKind::BOOLEAN,
          "RowOrderBy: BOOLEAN deferred payload not supported");
      const int32_t w = static_cast<int32_t>(type->cppSizeInBytes());
      std::vector<const uint8_t*> src(S);
      for (size_t k = 0; k < S; k++) {
        src[k] = static_cast<const uint8_t*>(storeCols_[k][d]->valuesAsVoid());
      }
      auto* dst = static_cast<uint8_t*>(const_cast<void*>(res->valuesAsVoid()));
      VELOX_CHECK_NOT_NULL(dst);
      parallelRows(n, [&](int64_t b, int64_t e, int /*t*/) {
        for (int64_t i = b; i < e; i++) {
          std::memcpy(
              dst + i * w, src[storeOf[i]] + static_cast<int64_t>(localOf[i]) * w, w);
        }
      });
    }
    if (anyNulls) {
      for (int32_t i = 0; i < n; i++) {
        if (storeCols_[storeOf[i]][d]->isNullAt(localOf[i])) {
          res->setNull(i, true);
        }
      }
    }
    out[d] = std::move(res);
  }
  addRuntimeStat(
      "rowSortDeferredRows", RuntimeCounter(static_cast<int64_t>(n) * D));
  return out;
}

RowVectorPtr RowOrderBy::emitHostChunk(int64_t begin, int32_t n) {
  if (!deferredCols_.empty()) {
    // DEFERRED exit (2026-08-25): uniformly COLUMNAR. Ids are extracted on
    // the device, the deferred payload is gathered on the host, and the
    // GPU-side (key) columns are transposed on the device and brought over
    // as columns. The rows themselves never cross.
    prefetchBegin_ = -1;
    const uint8_t* base =
        static_cast<const uint8_t*>(sorted_.data()) + begin * rowWidth_;
    const uint8_t* nullBase = nullPad_ > 0
        ? static_cast<const uint8_t*>(sortedNulls_.data()) + begin * nullPad_
        : nullptr;
    const int32_t numCols = outputType_->size();
    GpuFixedRowStore chunk;
    chunk.row_buffer = const_cast<uint8_t*>(base);
    chunk.row_width = rowWidth_;
    chunk.num_rows = n;
    chunk.num_fields = static_cast<int32_t>(fields_.size());
    chunk.fields = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(int64_t);
    if (bytes > idsDev_.size()) {
      idsDev_ = rmm::device_buffer(bytes, stream_);
    }
    extractKeysFromRows(
        chunk, fields_[rowIdField_].offset, 8, idsDev_.data(), stream_.value());
    std::vector<int64_t> gids(n);
    cudaMemcpyAsync(
        gids.data(), idsDev_.data(), bytes, cudaMemcpyDeviceToHost,
        stream_.value());
    stream_.synchronize();
    std::vector<VectorPtr> children(numCols);
    const auto tpHG = std::chrono::steady_clock::now();
    auto cols = gatherDeferred(gids, n);
    addRuntimeStat(
        "rowSortHostGatherNanos",
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tpHG)
                .count(),
            RuntimeCounter::Unit::kNanos));
    for (size_t k = 0; k < deferredCols_.size(); k++) {
      children[deferredCols_[k]] = std::move(cols[k]);
    }
    auto gpuCols = transposeChunkColumns(base, nullBase, n);
    std::vector<std::unique_ptr<cudf::column>> subset;
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    std::vector<int32_t> subsetIdx;
    for (int32_t i = 0; i < numCols; i++) {
      if (gpuCols[i] != nullptr) {
        subset.push_back(std::move(gpuCols[i]));
        names.push_back(outputType_->nameOf(i));
        types.push_back(outputType_->childAt(i));
        subsetIdx.push_back(i);
      }
    }
    if (!subset.empty()) {
      auto tbl = std::make_unique<cudf::table>(std::move(subset));
      auto host = with_arrow::toVeloxColumn(
          tbl->view(),
          pool(),
          std::static_pointer_cast<const Type>(
              ROW(std::move(names), std::move(types))),
          stream_,
          get_output_mr());
      for (size_t k = 0; k < subsetIdx.size(); k++) {
        children[subsetIdx[k]] = host->childAt(k);
      }
    }
    addRuntimeStat("hostExitRows", RuntimeCounter(static_cast<int64_t>(n)));
    return std::make_shared<RowVector>(
        pool(), outputType_, nullptr, n, std::move(children));
  }
  // ---- D2H: this chunk was prefetched by the previous call (or now) ----
  const auto rowsBytes = static_cast<size_t>(n) * rowWidth_;
  if (prefetchBegin_ != begin) {
    hostRows_.resize(rowsBytes);
    cudaMemcpyAsync(
        hostRows_.data(),
        static_cast<const uint8_t*>(sorted_.data()) + begin * rowWidth_,
        rowsBytes,
        cudaMemcpyDeviceToHost,
        stream_.value());
  }
  if (nullPad_ > 0) {
    hostNulls_.resize(static_cast<size_t>(n) * nullPad_);
    cudaMemcpyAsync(
        hostNulls_.data(),
        static_cast<const uint8_t*>(sortedNulls_.data()) + begin * nullPad_,
        hostNulls_.size(),
        cudaMemcpyDeviceToHost,
        stream_.value());
  }
  if (hasStrings_ && !hostCharsReady_) {
    hostChars_.resize(heap_.size());
    if (!hostChars_.empty()) {
      cudaMemcpyAsync(
          hostChars_.data(),
          heap_.data(),
          heap_.size(),
          cudaMemcpyDeviceToHost,
          stream_.value());
    }
    hostCharsReady_ = true;
  }
  const auto tpD2H = std::chrono::steady_clock::now();
  stream_.synchronize();
  addRuntimeStat(
      "rowSortD2HWaitNanos",
      RuntimeCounter(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - tpD2H)
              .count(),
          RuntimeCounter::Unit::kNanos));
  addRuntimeStat(
      "rowSortD2HBytes",
      RuntimeCounter(static_cast<int64_t>(rowsBytes), RuntimeCounter::Unit::kBytes));
  std::swap(hostRows_, hostRowsCur_); // hostRowsCur_ = this chunk
  // Prefetch the NEXT chunk into the (now free) other buffer; the next
  // call's synchronize() waits for it.
  const int64_t nextBegin = begin + n;
  if (nextBegin < totalRows_) {
    const auto nextN = static_cast<size_t>(
        std::min<int64_t>(kChunkRows, totalRows_ - nextBegin));
    hostRows_.resize(nextN * rowWidth_);
    cudaMemcpyAsync(
        hostRows_.data(),
        static_cast<const uint8_t*>(sorted_.data()) + nextBegin * rowWidth_,
        nextN * rowWidth_,
        cudaMemcpyDeviceToHost,
        stream_.value());
    prefetchBegin_ = nextBegin;
  } else {
    prefetchBegin_ = -1;
  }

  const uint8_t* rows = hostRowsCur_.data();
  const int32_t numCols = outputType_->size();
  std::vector<VectorPtr> children(numCols); // eager: nothing deferred
  if (CudfConfig::getInstance().benchmarkRowOutputNative) {
    // Native row output: hand this chunk's D2H'd rows to the consumer as-is
    // (with the shared string heap, this chunk's null sidecar and the
    // host-gathered deferred columns). No column extraction; the harness
    // materializes only when results are printed.
    if (hasStrings_ && hostCharsShared_ == nullptr) {
      hostCharsShared_ =
          std::make_shared<const std::vector<uint8_t>>(std::move(hostChars_));
    }
    std::vector<FieldDesc> f(numCols);
    std::vector<int32_t> nullBits(numCols, 0);
    for (int32_t i = 0; i < numCols; i++) {
      const auto fi = outFieldIdx_[i];
      if (fi >= 0) {
        f[i] = fields_[fi];
        nullBits[i] = fi; // the sidecar is indexed by FIELD
      }
    }
    std::vector<uint8_t> rowsOwned = std::move(hostRowsCur_);
    std::vector<uint8_t> nullsOwned;
    if (nullPad_ > 0) {
      nullsOwned = std::move(hostNulls_);
    }
    addRuntimeStat("hostExitRows", RuntimeCounter(static_cast<int64_t>(n)));
    addRuntimeStat(
        "hostExitNativeBytes",
        RuntimeCounter(
            static_cast<int64_t>(rowsBytes), RuntimeCounter::Unit::kBytes));
    return std::make_shared<HostRowVector>(
        pool(),
        outputType_,
        n,
        std::move(rowsOwned),
        rowWidth_,
        std::move(f),
        hostCharsShared_ != nullptr
            ? hostCharsShared_
            : std::make_shared<const std::vector<uint8_t>>(),
        std::move(nullsOwned),
        nullPad_,
        std::move(nullBits));
  }
  const auto tpExtract = std::chrono::steady_clock::now();
  // ---- GPU-gathered columns: TWO parallel passes over row ranges for ALL
  // columns at once (thread teams are per chunk, not per column). Pass 1:
  // fixed-width strided copies + per-range out-of-line byte totals of every
  // string column. Pass 2: string bytes into one shared buffer per column
  // (disjoint ranges) + StringViews set without copying. ----
  struct StrCol {
    int32_t col;
    int32_t fieldOff;
    FlatVector<StringView>* fv;
    char* base = nullptr;
    std::vector<int64_t> rangeBase;
  };
  std::vector<StrCol> strCols;
  struct FixCol {
    int32_t fieldOff;
    int32_t width;
    uint8_t* dst;
  };
  std::vector<FixCol> fixCols;
  for (int32_t i = 0; i < numCols; i++) {
    if (children[i] != nullptr) {
      continue; // deferred
    }
    const auto& fd = fields_[outFieldIdx_[i]];
    auto vec = BaseVector::create(outputType_->childAt(i), n, pool());
    if (fd.kind == kFieldString) {
      strCols.push_back(
          {i, fd.offset, vec->template asFlatVector<StringView>(), nullptr, {}});
    } else {
      auto* dst = static_cast<uint8_t*>(const_cast<void*>(vec->valuesAsVoid()));
      VELOX_CHECK_NOT_NULL(dst);
      fixCols.push_back({fd.offset, fd.byte_width, dst});
    }
    children[i] = std::move(vec);
  }
  const size_t S = strCols.size();
  std::vector<int64_t> rangeBytes(static_cast<size_t>(kMaxEmitThreads) * S, 0);
  const int32_t w = rowWidth_;
  parallelRows(n, [&](int64_t b, int64_t e, int t) {
    for (const auto& fc : fixCols) {
      const int32_t fw = fc.width;
      for (int64_t r = b; r < e; r++) {
        std::memcpy(fc.dst + r * fw, rows + r * w + fc.fieldOff, fw);
      }
    }
    for (size_t k = 0; k < S; k++) {
      int64_t bytes = 0;
      const int32_t off = strCols[k].fieldOff;
      for (int64_t r = b; r < e; r++) {
        uint32_t len;
        std::memcpy(&len, rows + r * w + off, 4);
        if (len > kRowStrInlineMax) {
          bytes += len;
        }
      }
      rangeBytes[static_cast<size_t>(t) * S + k] = bytes;
    }
  });
  for (size_t k = 0; k < S; k++) {
    auto& sc = strCols[k];
    sc.rangeBase.assign(kMaxEmitThreads, 0);
    int64_t total = 0;
    for (int t = 0; t < kMaxEmitThreads; t++) {
      sc.rangeBase[t] = total;
      total += rangeBytes[static_cast<size_t>(t) * S + k];
    }
    if (total > 0) {
      auto buf = AlignedBuffer::allocate<char>(total, pool());
      sc.base = buf->asMutable<char>();
      sc.fv->setStringBuffers({buf});
    }
  }
  if (S > 0) {
    parallelRows(n, [&](int64_t b, int64_t e, int t) {
      for (size_t k = 0; k < S; k++) {
        auto& sc = strCols[k];
        int64_t cursor = sc.rangeBase[t];
        const int32_t off = sc.fieldOff;
        for (int64_t r = b; r < e; r++) {
          const uint8_t* slot = rows + r * w + off;
          uint32_t len;
          std::memcpy(&len, slot, 4);
          if (len <= kRowStrInlineMax) {
            sc.fv->setNoCopy(
                static_cast<vector_size_t>(r),
                StringView(reinterpret_cast<const char*>(slot + 4), len));
          } else {
            uint64_t hoff;
            std::memcpy(&hoff, slot + 8, 8);
            char* dst = sc.base + cursor;
            std::memcpy(dst, hostChars_.data() + hoff, len);
            sc.fv->setNoCopy(static_cast<vector_size_t>(r), StringView(dst, len));
            cursor += len;
          }
        }
      }
    });
  }
  if (nullPad_ > 0) {
    for (int32_t i = 0; i < numCols; i++) {
      const auto fi = outFieldIdx_[i];
      if (fi < 0) {
        continue;
      }
      const uint8_t byteMask = static_cast<uint8_t>(1u << (fi & 7));
      const int32_t byteIdx = fi >> 3;
      for (int32_t r = 0; r < n; r++) {
        if (hostNulls_[static_cast<int64_t>(r) * nullPad_ + byteIdx] & byteMask) {
          children[i]->setNull(r, true);
        }
      }
    }
  }
  addRuntimeStat(
      "rowSortHostExtractNanos",
      RuntimeCounter(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - tpExtract)
              .count(),
          RuntimeCounter::Unit::kNanos));
  addRuntimeStat("hostExitRows", RuntimeCounter(static_cast<int64_t>(n)));
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, n, std::move(children));
}

// Device transpose of one chunk of sorted rows into cudf columns, in
// output order; deferred columns are left nullptr for the caller (uploaded
// for a GPU consumer, host-gathered for a CPU exit). Shared by
// emitColumnarChunk and the deferred path of emitHostChunk.
std::vector<std::unique_ptr<cudf::column>> RowOrderBy::transposeChunkColumns(
    const uint8_t* base,
    const uint8_t* nullBase,
    int32_t n) {
  const int32_t numCols = outputType_->size();
  std::vector<std::unique_ptr<rmm::device_buffer>> colBufs(numCols);
  std::vector<uint8_t*> fixedPtrs;
  std::vector<FieldDesc> fixedFields;
  for (int32_t i = 0; i < numCols; i++) {
    const auto fi = outFieldIdx_[i];
    if (fi < 0 || fields_[fi].kind == kFieldString) {
      continue;
    }
    colBufs[i] = std::make_unique<rmm::device_buffer>(
        static_cast<size_t>(n) * fields_[fi].byte_width, stream_);
    fixedPtrs.push_back(static_cast<uint8_t*>(colBufs[i]->data()));
    fixedFields.push_back(fields_[fi]);
  }
  if (!fixedFields.empty()) {
    rowsToColumns(
        base, fixedFields.data(), fixedPtrs.data(),
        static_cast<int32_t>(fixedFields.size()), n, rowWidth_, stream_.value());
  }

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(numCols);
  for (int32_t i = 0; i < numCols; i++) {
    const auto fi = outFieldIdx_[i];
    if (fi < 0) {
      columns.push_back(nullptr); // deferred: supplied by the caller
      continue;
    }
    const auto& fd = fields_[fi];
    rmm::device_buffer mask{};
    cudf::size_type nullCount = 0;
    if (nullPad_ > 0) {
      mask = rmm::device_buffer(cudf::bitmask_allocation_size_bytes(n), stream_);
      sidecarToMask(
          nullBase, nullPad_, fi, n, static_cast<uint32_t*>(mask.data()),
          stream_.value());
      nullCount = cudf::null_count(
          static_cast<const cudf::bitmask_type*>(mask.data()), 0, n, stream_);
    }
    if (fd.kind == kFieldString) {
      const size_t offBytes = (static_cast<size_t>(n) + 1) * sizeof(int64_t);
      if (offBytes > strOffsetsDev_.size()) {
        strOffsetsDev_ = rmm::device_buffer(offBytes, stream_);
      }
      const int64_t totalChars = stringFieldOffsets(
          base, n, rowWidth_, fd.offset,
          static_cast<int64_t*>(strOffsetsDev_.data()), stream_.value());
      auto offsetsCol = cudf::make_numeric_column(
          cudf::data_type{cudf::type_id::INT32}, n + 1,
          cudf::mask_state::UNALLOCATED, stream_, get_output_mr());
      rmm::device_buffer chars(totalChars, stream_);
      stringFieldToChars(
          base, n, rowWidth_, fd.offset,
          static_cast<const uint8_t*>(heap_.data()),
          static_cast<const int64_t*>(strOffsetsDev_.data()),
          offsetsCol->mutable_view().data<int32_t>(),
          static_cast<uint8_t*>(chars.data()), stream_.value());
      columns.push_back(cudf::make_strings_column(
          n, std::move(offsetsCol), std::move(chars), nullCount, std::move(mask)));
      continue;
    }
    columns.push_back(std::make_unique<cudf::column>(
        veloxToCudfDataType(outputType_->childAt(i)),
        n,
        std::move(*colBufs[i]),
        std::move(mask),
        nullCount));
  }
  return columns;
}

RowVectorPtr RowOrderBy::emitColumnarChunk(int64_t begin, int32_t n) {
  const uint8_t* base =
      static_cast<const uint8_t*>(sorted_.data()) + begin * rowWidth_;
  const uint8_t* nullBase = nullPad_ > 0
      ? static_cast<const uint8_t*>(sortedNulls_.data()) + begin * nullPad_
      : nullptr;
  const int32_t numCols = outputType_->size();
  std::vector<std::unique_ptr<cudf::column>> deferred(numCols);
  if (!deferredCols_.empty()) {
    GpuFixedRowStore chunk;
    chunk.row_buffer = const_cast<uint8_t*>(base);
    chunk.row_width = rowWidth_;
    chunk.num_rows = n;
    chunk.num_fields = static_cast<int32_t>(fields_.size());
    chunk.fields = nullptr;
    const size_t bytes = static_cast<size_t>(n) * sizeof(int64_t);
    if (bytes > idsDev_.size()) {
      idsDev_ = rmm::device_buffer(bytes, stream_);
    }
    extractKeysFromRows(
        chunk, fields_[rowIdField_].offset, 8, idsDev_.data(), stream_.value());
    std::vector<int64_t> gids(n);
    cudaMemcpyAsync(
        gids.data(), idsDev_.data(), bytes, cudaMemcpyDeviceToHost,
        stream_.value());
    stream_.synchronize();
    const auto tpHG = std::chrono::steady_clock::now();
    auto cols = gatherDeferred(gids, n);
    addRuntimeStat(
        "rowSortHostGatherNanos",
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tpHG)
                .count(),
            RuntimeCounter::Unit::kNanos));
    std::vector<std::string> names;
    std::vector<TypePtr> types;
    for (auto outIdx : deferredCols_) {
      names.push_back(outputType_->nameOf(outIdx));
      types.push_back(outputType_->childAt(outIdx));
    }
    auto hostRow = std::make_shared<RowVector>(
        pool(), ROW(std::move(names), std::move(types)), nullptr, n,
        std::move(cols));
    auto tbl = with_arrow::toCudfTable(hostRow, pool(), stream_, get_output_mr());
    auto released = tbl->release();
    for (size_t k = 0; k < deferredCols_.size(); k++) {
      deferred[deferredCols_[k]] = std::move(released[k]);
    }
  }

  // Fixed-width columns: one scatter kernel.
  auto columns = transposeChunkColumns(base, nullBase, n);
  for (int32_t i = 0; i < numCols; i++) {
    if (outFieldIdx_[i] < 0) {
      VELOX_CHECK_NOT_NULL(deferred[i]);
      columns[i] = std::move(deferred[i]);
    }
  }

  auto table = std::make_unique<cudf::table>(std::move(columns));
  addRuntimeStat("rowToColOutputRows", RuntimeCounter(static_cast<int64_t>(n)));
  return std::make_shared<CudfVector>(
      pool(), outputType_, n, std::move(table), stream_);
}

RowVectorPtr RowOrderBy::doGetOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  if (cursor_ >= totalRows_) {
    finished_ = true;
    return nullptr;
  }
  const auto n =
      static_cast<int32_t>(std::min<int64_t>(kChunkRows, totalRows_ - cursor_));
  const int64_t begin = cursor_;
  cursor_ += n;
  auto out = emitHost_ == 1 ? emitHostChunk(begin, n)
                            : emitColumnarChunk(begin, n);
  if (cursor_ >= totalRows_) {
    finished_ = true;
  }
  return out;
}

void RowOrderBy::doClose() {
  Operator::close();
  rowInputs_.clear();
  cudfInputs_.clear();
  stores_.clear();
  storeCols_.clear();
  sorted_ = rmm::device_buffer{};
  sortedNulls_ = rmm::device_buffer{};
  heap_ = rmm::device_buffer{};
  idsDev_ = rmm::device_buffer{};
  strOffsetsDev_ = rmm::device_buffer{};
}

} // namespace facebook::velox::cudf_velox
