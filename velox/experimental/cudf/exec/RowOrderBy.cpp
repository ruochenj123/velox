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
#include <cudf/copying.hpp>
#include <cudf/strings/strings_column_view.hpp>
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
  VELOX_FAIL("RowOrderBy: input must be a RowStoreVector (row pack)");
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
  // Per-batch (row start, heap base) for ONE batched rebase launch after the
  // copies (2026-09-08; was one rebase launch per input batch).
  std::vector<int64_t> batchRowStart;
  std::vector<int64_t> batchHeapBase;
  batchRowStart.reserve(rowInputs_.size() + 1);
  batchHeapBase.reserve(rowInputs_.size());
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
    if (hasStrings_) {
      batchRowStart.push_back(rowsSoFar);
      batchHeapBase.push_back(heapSoFar);
      if (v->charsBytes() > 0) {
        cudaMemcpyAsync(
            static_cast<uint8_t*>(heap_.data()) + heapSoFar,
            v->charsData(),
            v->charsBytes(),
            cudaMemcpyDeviceToDevice,
            stream_.value());
        heapSoFar += v->charsBytes();
      }
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
      if (sortStore_ == nullptr) {
        // ONE coalesced store for the sort: the retained batches are merged
        // in one parallel pass by coalesceFrom() (sortRows' coalesce thread)
        // and gathered by global row id.
        sortStore_ = std::make_unique<BoundaryHostStore>(
            store->rowType(), pool(), /*scattered=*/false);
      }
      const int64_t base = sortStoreRows_;
      for (const auto& b : store->batches()) {
        pendingSortBatches_.push_back(b);
      }
      sortStoreRows_ += store->totalRows();
      idSpaceSoFar += store->totalRows();
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
  if (hasStrings_ && !strOffs.empty() && !batchRowStart.empty()) {
    batchRowStart.push_back(rowsSoFar);
    rmm::device_buffer rowStartDev(
        batchRowStart.data(),
        batchRowStart.size() * sizeof(int64_t),
        stream_);
    rmm::device_buffer heapBaseDev(
        batchHeapBase.data(),
        batchHeapBase.size() * sizeof(int64_t),
        stream_);
    rebaseStringOffsetsBatched(
        rowsBase,
        rowsSoFar,
        rowWidth_,
        static_cast<const int32_t*>(strOffsDev.data()),
        static_cast<int32_t>(strOffs.size()),
        static_cast<const int64_t*>(rowStartDev.data()),
        static_cast<const int64_t*>(heapBaseDev.data()),
        static_cast<int32_t>(batchHeapBase.size()),
        stream_.value());
    stream_.synchronize(); // rowStartDev/heapBaseDev die at scope exit
  }
  stream_.synchronize();
  rowInputs_.clear(); // frees the per-batch GPU buffers
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

  // Deferred payload: coalesce the retained batches on a background thread
  // (HybridContainer::coalesceBatches, as the CPU hybrid sort does in
  // SortBuffer), overlapped with the device sort below; joined before the
  // first host gather.
  idsOnly_ = emitHost_ == 1 && !deferredCols_.empty() && !mixedExit_;
  std::thread coalesceThread;
  std::exception_ptr coalesceError;
  const auto tpCoalesce = std::chrono::steady_clock::now();
  if (sortStore_ != nullptr && !deferredCols_.empty()) {
    coalesceThread = std::thread([&]() {
      try {
        if (!pendingSortBatches_.empty()) {
          sortStore_->coalesceFrom(pendingSortBatches_);
          pendingSortBatches_ = {};
        } else {
          sortStore_->coalesce();
        }
      } catch (...) {
        coalesceError = std::current_exception();
      }
    });
  }
  // Mixed host exit: D2H the string heap now, overlapped with the merge and
  // the device sort (the heap is final after concatenateRowInputs), in
  // parallel slices on their own streams.
  std::thread heapThread;
  std::exception_ptr heapError;
  if (hostExtractPath() && hasStrings_ && !hostCharsReady_) {
    heapThread = std::thread([&]() {
      try {
        const size_t bytes = heap_.size();
        hostChars_.resize(bytes);
        constexpr int kSlices = 4;
        std::vector<std::thread> slices;
        for (int k = 0; k < kSlices; k++) {
          slices.emplace_back([&, k]() {
            const size_t b0 = bytes * k / kSlices;
            const size_t b1 = bytes * (k + 1) / kSlices;
            if (b1 <= b0) {
              return;
            }
            cudaStream_t st;
            cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
            cudaMemcpyAsync(
                hostChars_.data() + b0,
                static_cast<const uint8_t*>(heap_.data()) + b0,
                b1 - b0,
                cudaMemcpyDeviceToHost,
                st);
            cudaStreamSynchronize(st);
            cudaStreamDestroy(st);
          });
        }
        for (auto& t : slices) {
          t.join();
        }
      } catch (...) {
        heapError = std::current_exception();
      }
    });
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

  rmm::device_buffer gathered;
  if (idsOnly_) {
    // Deferred + CPU exit: permute ONLY the __rowid column (8 B/row); the
    // crossing rows themselves are never needed again.
    rmm::device_buffer idsCol(static_cast<size_t>(n) * 8, stream_);
    extractKeysFromRows(
        store, fields_[rowIdField_].offset, 8, idsCol.data(), stream_.value());
    cudf::column_view idsView(
        cudf::data_type{cudf::type_id::INT64}, n, idsCol.data(), nullptr, 0);
    auto permuted = cudf::gather(
        cudf::table_view({idsView}),
        perm->view(),
        cudf::out_of_bounds_policy::DONT_CHECK,
        stream_,
        get_output_mr());
    sortedIds_ = rmm::device_buffer(static_cast<size_t>(n) * 8, stream_);
    cudaMemcpyAsync(
        sortedIds_.data(),
        permuted->get_column(0).view().data<int64_t>(),
        static_cast<size_t>(n) * 8,
        cudaMemcpyDeviceToDevice,
        stream_.value());
    stream_.synchronize(); // permuted's storage is released at scope exit
  } else {
    // One gather of whole rows by the permutation (slots copied verbatim;
    // the heap is shared, so string offsets stay valid).
    gathered = rmm::device_buffer(sorted_.size(), stream_);
    gatherRowsWarp(
        store, permData, n, static_cast<uint8_t*>(gathered.data()), stream_.value());
  }
  if (!idsOnly_ && nullPad_ > 0) {
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
  stream_.synchronize();
  if (heapThread.joinable()) {
    heapThread.join();
    if (heapError) {
      std::rethrow_exception(heapError);
    }
    hostCharsReady_ = true;
  }
  if (coalesceThread.joinable()) {
    coalesceThread.join();
    if (coalesceError) {
      std::rethrow_exception(coalesceError);
    }
  }
  addRuntimeStat(
      "rowSortCoalesceNanos",
      RuntimeCounter(
          std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::steady_clock::now() - tpCoalesce)
              .count(),
          RuntimeCounter::Unit::kNanos));
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
  if (!idsOnly_) {
    sorted_ = std::move(gathered);
  } else {
    sorted_ = rmm::device_buffer{}; // rows not needed: ids only
  }
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
  chunkRows_ = std::max<int32_t>(
      1, outputBatchRows(static_cast<uint64_t>(rowWidth_ + nullPad_)));
  if (sortGatherThreads() > 1) {
    // Parallel gather: bigger chunks amortize the per-chunk fork/join.
    chunkRows_ = std::max<int32_t>(chunkRows_, 1 << 17);
  }
  if (auto* driver = operatorCtx_->driver()) {
    const auto ops = driver->operators();
    for (size_t i = 0; i + 1 < ops.size(); i++) {
      if (ops[i] == this) {
        if (dynamic_cast<CudfToVelox*>(ops[i + 1]) != nullptr) {
          // CPU exit: emit host RowVectors directly -- native rows
          // (--row_output_native), or D2H'd rows extracted on the host
          // (eager and mixed exits), or the deferred ids-only exit.
          emitHost_ = 1;
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
          sortStore_ != nullptr,
          "RowOrderBy: column '{}' absent from the row layout and no "
          "provenance store",
          outputType_->nameOf(i));
      deferredCols_.push_back(static_cast<int32_t>(i));
    }
  }
  // Deferred + CPU exit: ids-only crossing. Every output column -- keys
  // included -- is gathered on the host from the retained batches, so the
  // sorted rows never cross and no device transpose is needed.
  // Sort-after-join (2026-09-10): the retained store is the PROBE's, so
  // only the columns it holds can be gathered there; the others (the eager
  // build side) live in the rows only -> mixed exit: rows cross, deferred
  // columns are gathered by the __rowid carried in each row.
  mixedExit_ = false;
  if (emitHost_ == 1 && !deferredCols_.empty()) {
    const auto& storeType = sortStore_->rowType();
    deferredCols_.clear();
    for (size_t i = 0; i < outputType_->size(); i++) {
      if (storeType->getChildIdxIfExists(outputType_->nameOf(i)).has_value()) {
        deferredCols_.push_back(static_cast<int32_t>(i));
      } else {
        VELOX_CHECK_GE(
            outFieldIdx_[i],
            0,
            "RowOrderBy: column '{}' is neither in the rows nor in the "
            "provenance store",
            outputType_->nameOf(i));
        mixedExit_ = true;
      }
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

// Host gather of the deferred columns at the sorted GLOBAL row ids:
// HybridContainer's coalesced-mode extraction over the single sort store.
// Threads for the deferred exit's host gather: the flag, or (-1) auto =
// min(48, hardware_concurrency) -- the cores are idle during the GPU sort.
int32_t RowOrderBy::sortGatherThreads() {
  const int32_t v = CudfConfig::getInstance().benchmarkSortGatherThreads;
  if (v >= 0) {
    return v;
  }
  const auto hw = static_cast<int32_t>(std::thread::hardware_concurrency());
  return std::max<int32_t>(1, std::min<int32_t>(48, hw));
}

std::vector<VectorPtr> RowOrderBy::gatherDeferred(
    const std::vector<int64_t>& gids,
    int32_t n) {
  VELOX_CHECK_NOT_NULL(sortStore_);
  const size_t D = deferredCols_.size();
  const auto& storeType = sortStore_->rowType();
  std::vector<VectorPtr> out(D);
  auto work = [&](size_t d0,
                  size_t d1,
                  std::vector<const char*>& rowsScratch,
                  std::vector<exec::HybridRowId>& idsScratch) {
    for (size_t d = d0; d < d1; d++) {
      const auto& type = outputType_->childAt(deferredCols_[d]);
      const auto childIdx = static_cast<int32_t>(
          storeType->getChildIdx(outputType_->nameOf(deferredCols_[d])));
      auto res = BaseVector::create(type, n, pool());
      sortStore_->gatherCoalesced(
          childIdx, gids.data(), n, res, rowsScratch, idsScratch);
      out[d] = std::move(res);
    }
  };
  // Columns are independent (each writes its own vector; the store is only
  // read), so the chunk's gather fans out over T threads by column range.
  const size_t T = static_cast<size_t>(std::max<int32_t>(
      1, std::min<int32_t>(sortGatherThreads(), static_cast<int32_t>(D))));
  if (T <= 1) {
    work(0, D, rowsScratch_, idsScratch_);
  } else {
    rowsScratchT_.resize(T);
    idsScratchT_.resize(T);
    std::vector<std::thread> threads;
    threads.reserve(T);
    for (size_t t = 0; t < T; t++) {
      const size_t d0 = D * t / T;
      const size_t d1 = D * (t + 1) / T;
      threads.emplace_back(
          [&, t, d0, d1]() { work(d0, d1, rowsScratchT_[t], idsScratchT_[t]); });
    }
    for (auto& th : threads) {
      th.join();
    }
  }
  addRuntimeStat(
      "rowSortDeferredRows", RuntimeCounter(static_cast<int64_t>(n) * D));
  return out;
}

// CPU exits taken through D2H'd rows (pinned) + parallel host extraction:
// the mixed deferred exit (sort after a join) and the eager exit without
// native output. (The mixed exit's device transpose + arrow alternative was
// removed 2026-09-29: Q46 ps10 11.2 s vs 6.7 s.)
bool RowOrderBy::hostExtractPath() const {
  if (emitHost_ != 1) {
    return false;
  }
  if (mixedExit_) {
    return !deferredCols_.empty();
  }
  return deferredCols_.empty() &&
      !CudfConfig::getInstance().benchmarkRowOutputNative;
}

RowVectorPtr RowOrderBy::emitHostChunk(int64_t begin, int32_t n) {
  if (!deferredCols_.empty() && !mixedExit_) {
    // DEFERRED exit (2026-08-25): uniformly COLUMNAR. Ids are extracted on
    // the device, the deferred payload is gathered on the host, and the
    // GPU-side (key) columns are transposed on the device and brought over
    // as columns. The rows themselves never cross.
    prefetchBegin_ = -1;
    const int32_t numCols = outputType_->size();
    // Ids-only crossing: the sorted __rowid column, D2H'd per chunk.
    VELOX_CHECK(idsOnly_);
    std::vector<int64_t> gids(n);
    cudaMemcpyAsync(
        gids.data(),
        static_cast<const int64_t*>(sortedIds_.data()) + begin,
        static_cast<size_t>(n) * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
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
    addRuntimeStat("hostExitRows", RuntimeCounter(static_cast<int64_t>(n)));
    return std::make_shared<RowVector>(
        pool(), outputType_, nullptr, n, std::move(children));
  }
  // ---- D2H: this chunk was prefetched by the previous call (or now) ----
  const auto rowsBytes = static_cast<size_t>(n) * rowWidth_;
  const auto tpChunkD2H = std::chrono::steady_clock::now();
  if (hostExtractPath()) {
    // Pinned double buffer (see pinnedRows_).
    if (pinnedRows_[0] == nullptr) {
      const size_t cap = static_cast<size_t>(chunkRows_) * rowWidth_;
      for (auto& buf : pinnedRows_) {
        const auto err = cudaHostAlloc(
            reinterpret_cast<void**>(&buf), cap, cudaHostAllocDefault);
        VELOX_CHECK(
            err == cudaSuccess,
            "RowOrderBy: cudaHostAlloc({} B) failed: {}",
            cap,
            cudaGetErrorString(err));
      }
    }
    if (prefetchBegin_ != begin) {
      cudaMemcpyAsync(
          pinnedRows_[pinnedCur_],
          static_cast<const uint8_t*>(sorted_.data()) + begin * rowWidth_,
          rowsBytes,
          cudaMemcpyDeviceToHost,
          stream_.value());
    }
  } else if (prefetchBegin_ != begin) {
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
  const uint8_t* curRows = nullptr;
  const int64_t nextBegin = begin + n;
  if (hostExtractPath()) {
    curRows = pinnedRows_[pinnedCur_];
    if (nextBegin < totalRows_) {
      const auto nextN = static_cast<size_t>(
          std::min<int64_t>(chunkRows_, totalRows_ - nextBegin));
      cudaMemcpyAsync(
          pinnedRows_[1 - pinnedCur_],
          static_cast<const uint8_t*>(sorted_.data()) + nextBegin * rowWidth_,
          nextN * rowWidth_,
          cudaMemcpyDeviceToHost,
          stream_.value());
      prefetchBegin_ = nextBegin;
      pinnedCur_ = 1 - pinnedCur_;
    } else {
      prefetchBegin_ = -1;
    }
    addRuntimeStat(
        "rowSortChunkD2HNanos",
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tpChunkD2H)
                .count(),
            RuntimeCounter::Unit::kNanos));
  } else {
  std::swap(hostRows_, hostRowsCur_); // hostRowsCur_ = this chunk
  curRows = hostRowsCur_.data();
  // Prefetch the NEXT chunk into the (now free) other buffer; the next
  // call's synchronize() waits for it.
  if (nextBegin < totalRows_) {
    const auto nextN = static_cast<size_t>(
        std::min<int64_t>(chunkRows_, totalRows_ - nextBegin));
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
  }

  const int32_t numCols = outputType_->size();
  // Eager / mixed exit: this chunk's D2H'd rows (+ the shared string heap +
  // this chunk's null sidecar) are extracted into Velox columns on the host
  // (hostExtractPath); under --row_output_native an eager chunk is handed
  // over as a HostRowVector instead.
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
    } else {
      f[i] = {-1, 0, kFieldFixed}; // deferred: supplied below
    }
  }
  // Mixed exit (sort after a join): the deferred columns are gathered from
  // the probe's retained store by the __rowid field of each sorted row.
  std::vector<VectorPtr> deferredChildren;
  if (mixedExit_) {
    VELOX_CHECK_GE(rowIdField_, 0);
    const int32_t idOff = fields_[rowIdField_].offset;
    std::vector<int64_t> gids(n);
    for (int32_t r = 0; r < n; r++) {
      memcpy(
          &gids[r],
          curRows + static_cast<int64_t>(r) * rowWidth_ + idOff,
          sizeof(int64_t));
    }
    const auto tpHG = std::chrono::steady_clock::now();
    deferredChildren = gatherDeferred(gids, n);
    addRuntimeStat(
        "rowSortHostGatherNanos",
        RuntimeCounter(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - tpHG)
                .count(),
            RuntimeCounter::Unit::kNanos));
    // Columns present in both the rows and the store are taken from the
    // store like every other deferred column.
    for (auto c : deferredCols_) {
      f[c] = {-1, 0, kFieldFixed};
    }
  }
  if (hostExtractPath()) {
    // 2026-09-28: the eager columns are extracted over the gather threads,
    // (column, row range) tasks, string views pointing into the D2H'd heap
    // (no body copies). Was: HostRowVector::materialize on this thread.
    const auto tpExtract = std::chrono::steady_clock::now();
    auto children = extractHostRowsParallel(
        pool(),
        outputType_,
        n,
        curRows,
        rowWidth_,
        f,
        hostCharsShared_,
        nullPad_ > 0 ? hostNulls_.data() : nullptr,
        nullPad_,
        &nullBits,
        sortGatherThreads());
    for (size_t k = 0; k < deferredCols_.size(); k++) {
      children[deferredCols_[k]] = std::move(deferredChildren[k]);
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
  std::vector<uint8_t> rowsOwned = std::move(hostRowsCur_);
  std::vector<uint8_t> nullsOwned;
  if (nullPad_ > 0) {
    nullsOwned = std::move(hostNulls_);
  }
  addRuntimeStat("hostExitRows", RuntimeCounter(static_cast<int64_t>(n)));
  auto hostRows = std::make_shared<HostRowVector>(
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
  // Native rows (--row_output_native): handed over as-is, no extraction.
  VELOX_CHECK(CudfConfig::getInstance().benchmarkRowOutputNative && !mixedExit_);
  addRuntimeStat(
      "hostExitNativeBytes",
      RuntimeCounter(
          static_cast<int64_t>(rowsBytes), RuntimeCounter::Unit::kBytes));
  return hostRows;
}

// Device transpose of one chunk of sorted rows into cudf columns, in
// output order; deferred columns are left nullptr for the caller (uploaded
// for a GPU consumer, host-gathered for a CPU exit). Shared by
// emitColumnarChunk and the deferred path of emitHostChunk.
std::vector<std::unique_ptr<cudf::column>> RowOrderBy::transposeChunkColumns(
    const uint8_t* base,
    const uint8_t* nullBase,
    int32_t n,
    bool withRowId) {
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
  std::unique_ptr<rmm::device_buffer> rowIdBuf;
  if (withRowId) {
    VELOX_CHECK_GE(rowIdField_, 0);
    rowIdBuf = std::make_unique<rmm::device_buffer>(
        static_cast<size_t>(n) * 8, stream_);
    fixedPtrs.push_back(static_cast<uint8_t*>(rowIdBuf->data()));
    fixedFields.push_back(fields_[rowIdField_]);
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
  if (withRowId) {
    columns.push_back(std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT64},
        n,
        std::move(*rowIdBuf),
        rmm::device_buffer{},
        0));
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
    // Transpose FIRST (round 6): the chunk goes rows -> columns once on the
    // device, with __rowid as one more column; the ids are read from it.
    auto gpuCols = transposeChunkColumns(base, nullBase, n, true);
    auto rowIdCol = std::move(gpuCols.back());
    gpuCols.pop_back();
    std::vector<int64_t> gids(n);
    cudaMemcpyAsync(
        gids.data(), rowIdCol->view().data<int64_t>(),
        static_cast<size_t>(n) * sizeof(int64_t), cudaMemcpyDeviceToHost,
        stream_.value());
    stream_.synchronize();
    gpuColsForChunk_ = std::move(gpuCols);
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
  auto columns = deferredCols_.empty()
      ? transposeChunkColumns(base, nullBase, n)
      : std::move(gpuColsForChunk_);
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
      static_cast<int32_t>(std::min<int64_t>(chunkRows_, totalRows_ - cursor_));
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
  pendingSortBatches_.clear();
  for (auto& buf : pinnedRows_) {
    if (buf != nullptr) {
      cudaFreeHost(buf);
      buf = nullptr;
    }
  }
  sortStore_.reset();
  sortedIds_ = rmm::device_buffer{};
  sorted_ = rmm::device_buffer{};
  sortedNulls_ = rmm::device_buffer{};
  heap_ = rmm::device_buffer{};
  idsDev_ = rmm::device_buffer{};
  strOffsetsDev_ = rmm::device_buffer{};
}

} // namespace facebook::velox::cudf_velox
