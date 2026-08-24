/*
 * RowOrderBy.cpp (branch row-sort, 2026-08-24) -- see RowOrderBy.h.
 */
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

  const auto tp0 = std::chrono::steady_clock::now();
  auto perm = cudf::sorted_order(
      cudf::table_view(keyViews), orders_, nullOrders_, stream_, get_output_mr());
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
  stream_.synchronize();
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
  sortRows();
  resolveOutputOnce();
  // Chain endpoint (batch-level adaptive deferral): a CPU exit reports 0
  // survivors -- with direct host emission the deferred payload never
  // crosses, so deferral always wins there.
  DeferralStats::instance().recordSurvived(
      orderByNode_->id(), emitHost_ == 1 ? 0 : totalRows_);
}

std::vector<VectorPtr> RowOrderBy::gatherDeferred(
    const std::vector<int64_t>& gids,
    int32_t n) {
  std::vector<VectorPtr> out;
  const size_t S = stores_.size();
  std::vector<std::vector<int32_t>> local(S), pos(S);
  for (int32_t i = 0; i < n; i++) {
    const auto it =
        std::upper_bound(storeBases_.begin(), storeBases_.end(), gids[i]);
    const size_t k = static_cast<size_t>(it - storeBases_.begin() - 1);
    local[k].push_back(static_cast<int32_t>(gids[i] - storeBases_[k]));
    pos[k].push_back(i);
  }
  for (auto outIdx : deferredCols_) {
    const auto& name = outputType_->nameOf(outIdx);
    const auto& type = outputType_->childAt(outIdx);
    auto res = BaseVector::create(type, n, pool());
    if (S == 1) {
      stores_[0]->idsForGlobalRows(local[0].data(), n, rowIdsScratch_);
      stores_[0]->gather(
          static_cast<int32_t>(stores_[0]->rowType()->getChildIdx(name)),
          rowIdsScratch_,
          res,
          sentinelScratch_);
    } else {
      for (size_t k = 0; k < S; k++) {
        const auto m = static_cast<int32_t>(local[k].size());
        if (m == 0) {
          continue;
        }
        auto tmp = BaseVector::create(type, m, pool());
        stores_[k]->idsForGlobalRows(local[k].data(), m, rowIdsScratch_);
        stores_[k]->gather(
            static_cast<int32_t>(stores_[k]->rowType()->getChildIdx(name)),
            rowIdsScratch_,
            tmp,
            sentinelScratch_);
        std::vector<BaseVector::CopyRange> ranges(m);
        for (int32_t j = 0; j < m; j++) {
          ranges[j] = {j, pos[k][j], 1};
        }
        res->copyRanges(tmp.get(), ranges);
      }
    }
    out.push_back(std::move(res));
  }
  addRuntimeStat(
      "rowSortDeferredRows",
      RuntimeCounter(static_cast<int64_t>(n) * deferredCols_.size()));
  return out;
}

RowVectorPtr RowOrderBy::emitHostChunk(int64_t begin, int32_t n) {
  const uint8_t* base =
      static_cast<const uint8_t*>(sorted_.data()) + begin * rowWidth_;
  hostRows_.resize(static_cast<size_t>(n) * rowWidth_);
  cudaMemcpyAsync(
      hostRows_.data(),
      base,
      static_cast<size_t>(n) * rowWidth_,
      cudaMemcpyDeviceToHost,
      stream_.value());
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
  stream_.synchronize();

  const uint8_t* rows = hostRows_.data();
  const int32_t numCols = outputType_->size();
  std::vector<VectorPtr> children(numCols);
  if (!deferredCols_.empty()) {
    std::vector<int64_t> gids(n);
    const int32_t idOff = fields_[rowIdField_].offset;
    for (int32_t r = 0; r < n; r++) {
      std::memcpy(&gids[r], rows + static_cast<int64_t>(r) * rowWidth_ + idOff, 8);
    }
    auto cols = gatherDeferred(gids, n);
    for (size_t k = 0; k < deferredCols_.size(); k++) {
      children[deferredCols_[k]] = std::move(cols[k]);
    }
  }
  for (int32_t i = 0; i < numCols; i++) {
    if (children[i] != nullptr) {
      continue;
    }
    const auto fi = outFieldIdx_[i];
    const auto& fd = fields_[fi];
    auto vec = BaseVector::create(outputType_->childAt(i), n, pool());
    if (fd.kind == kFieldString) {
      auto* fv = vec->template asFlatVector<StringView>();
      for (int32_t r = 0; r < n; r++) {
        const uint8_t* slot =
            rows + static_cast<int64_t>(r) * rowWidth_ + fd.offset;
        uint32_t len;
        std::memcpy(&len, slot, 4);
        const char* src;
        if (len <= kRowStrInlineMax) {
          src = reinterpret_cast<const char*>(slot + 4);
        } else {
          uint64_t off;
          std::memcpy(&off, slot + 8, 8);
          src = reinterpret_cast<const char*>(hostChars_.data() + off);
        }
        fv->set(r, StringView(src, len));
      }
    } else {
      auto* dst = static_cast<uint8_t*>(const_cast<void*>(vec->valuesAsVoid()));
      VELOX_CHECK_NOT_NULL(dst);
      const int32_t w = fd.byte_width;
      for (int32_t r = 0; r < n; r++) {
        std::memcpy(
            dst + static_cast<int64_t>(r) * w,
            rows + static_cast<int64_t>(r) * rowWidth_ + fd.offset,
            w);
      }
    }
    if (nullPad_ > 0) {
      const uint8_t byteMask = static_cast<uint8_t>(1u << (fi & 7));
      const int32_t byteIdx = fi >> 3;
      for (int32_t r = 0; r < n; r++) {
        if (hostNulls_[static_cast<int64_t>(r) * nullPad_ + byteIdx] & byteMask) {
          vec->setNull(r, true);
        }
      }
    }
    children[i] = std::move(vec);
  }
  addRuntimeStat("hostExitRows", RuntimeCounter(static_cast<int64_t>(n)));
  return std::make_shared<RowVector>(
      pool(), outputType_, nullptr, n, std::move(children));
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
    auto cols = gatherDeferred(gids, n);
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
      VELOX_CHECK_NOT_NULL(deferred[i]);
      columns.push_back(std::move(deferred[i]));
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
  sorted_ = rmm::device_buffer{};
  sortedNulls_ = rmm::device_buffer{};
  heap_ = rmm::device_buffer{};
  idsDev_ = rmm::device_buffer{};
  strOffsetsDev_ = rmm::device_buffer{};
}

} // namespace facebook::velox::cudf_velox
