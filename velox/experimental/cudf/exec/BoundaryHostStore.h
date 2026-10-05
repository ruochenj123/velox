/*
 * BoundaryHostStore.h
 *
 * Host-side payload retention + gather for the BOUNDARY-HYBRID GPU join
 * (CudfConfig::benchmarkBoundaryHybrid): only join keys cross the device
 * boundary; payload columns stay host-resident in columnar form; the GPU
 * join returns surviving row-id pairs; the host gathers payloads once for
 * the survivors.
 *
 * This is a THIN WRAPPER over exec::HybridContainer (velox/exec/
 * RowContainer.h) in SCATTERED mode — the same machinery the CPU hybrid
 * join uses for host-side batch retention:
 *   - addPayload(): retains the batch, flattens dictionary-encoded children
 *     (so a selective filter's dictionary view does not pin its whole base
 *     vector), and pre-decodes every column into DecodedVector for O(1)
 *     valueAt access;
 *   - the extractPayloadScattered* kernels: gather by (batchId, rowInBatch)
 *     encoded in exec::HybridRowId.
 * We deliberately REUSE that machinery rather than re-implementing retention
 * and gather (per the CPU/GPU hybrid-layout design: one retention mechanism
 * on both sides of the boundary).
 *
 * What the wrapper adds/bridges:
 *   - HybridContainer requires a keys RowContainer (the CPU join stores join
 *     keys + a rowId column there). In boundary-hybrid the GPU owns the keys,
 *     so we construct the container with ZERO key types over a minimal dummy
 *     RowContainer that is never written. Only addPayload + the scattered
 *     extraction paths are exercised; those never touch the keys container.
 *   - HybridContainer's extraction API takes a `rows` array whose only use in
 *     scattered mode is `row == nullptr` => emit NULL (a probe-miss marker in
 *     the CPU outer-join path). Boundary-hybrid gathers only genuine inner-
 *     join survivors, so we pass a reusable array of non-null sentinels.
 *   - Prefix sums over the added batch sizes, mapping a row index in the
 *     CONCATENATION of all batches (== the GPU-side global row id, since the
 *     GPU keys were concatenated in exactly this order) back to
 *     (batchId, rowInBatch).
 */

#pragma once

#include "velox/exec/RowContainer.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/RetainedMerge.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/FlatVector.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace facebook::velox::cudf_velox {

class BoundaryHostStore {
 public:
  /// `rowType` is the retained batches' full row type (all columns, in input
  /// child order); extraction addresses columns by child index in this type.
  /// `scattered` (default, the probe's per-batch store): row ids encode
  /// (batch, row) and extraction reads the batches in place. `false` (the
  /// sort's single store): batches are merged by coalesce() into one
  /// contiguous batch and row ids are global indices (HybridContainer's
  /// coalesced mode, the CPU hybrid sort's own path).
  BoundaryHostStore(
      const RowTypePtr& rowType,
      memory::MemoryPool* pool,
      bool scattered = true,
      bool light = false)
      : light_(light && scattered),
        rowType_(rowType),
        pool_(pool),
        dummyKeys_(std::vector<TypePtr>{BIGINT()}, pool),
        container_(
            /*keyTypes=*/{},
            rowType->children(),
            &dummyKeys_) {
    container_.setScatteredModeEnabled(scattered);
    // Fast scattered gather (2026-09-27; always on since 2026-09-29):
    // forward-cursor id translation + batch-cached, prefetched extraction.
    container_.setFastScatteredExtraction(true);
    container_.setId(0);
    // Register self so the single-container scattered fast path applies.
    std::unordered_map<uint8_t, exec::HybridContainer*> self{{0, &container_}};
    container_.setAllContainers(self);
  }

  // Not copyable/movable: container_ holds a pointer to dummyKeys_.
  BoundaryHostStore(const BoundaryHostStore&) = delete;
  BoundaryHostStore& operator=(const BoundaryHostStore&) = delete;

  /// Retain one batch. Batches must be added in GPU-concatenation order so
  /// that global row ids line up. Returns the batchId assigned.
  uint32_t addBatch(RowVectorPtr batch) {
    const auto batchId = static_cast<uint32_t>(batchStarts_.size());
    // 24-bit batchId in HybridRowId::encodeScattered.
    VELOX_CHECK_LT(
        batchId,
        1u << exec::HybridRowId::kBatchIdBits,
        "BoundaryHostStore: too many retained batches for scattered encoding");
    batchStarts_.push_back(totalRows_);
    totalRows_ += batch->size();
    if (light_) {
      // 2026-09-27 lightweight retention (per-GPU-batch probe stores): no
      // HybridContainer::addPayload -- no flatten, no per-column
      // DecodedVector. A filter's zero-copy dictionary view is kept as is
      // (the store lives for one GPU batch only) and each column is described
      // by raw pointers: values, the view's indices, and the null bitmaps.
      addLightBatch(batch);
      batches_.push_back(std::move(batch));
      return batchId;
    }
    batches_.push_back(batch);
    container_.addPayload(std::move(batch));
    return batchId;
  }

  /// The retained batches, in the order added (a blocking consumer such as
  /// the sort re-adds them to its own single store).
  const std::vector<RowVectorPtr>& batches() const {
    return batches_;
  }

  /// Coalesced mode only: merge all retained batches into one contiguous
  /// batch (HybridContainer::coalesceBatches -- column-at-a-time, sources
  /// released as copied). Safe to run on a background thread as long as
  /// nothing else touches this store meanwhile.
  /// Used only by the single-build-store fallback (a finalizing build driver
  /// without its own retained input); the sort store and the per-driver
  /// build stores are merged by coalesceFrom().
  /// `threads` < 0: --coalesce_threads.
  void coalesce(int32_t threads = -1) {
    // 2026-09-27: optional column-parallel merge.
    container_.setCoalesceThreads(
        threads >= 0 ? threads
                     : CudfConfig::getInstance().benchmarkCoalesceThreads);
    container_.coalesceBatches();
  }
  /// Coalesced mode, EMPTY store: merge `inputs` column-parallel straight
  /// into one flat batch (RetainedMerge.h: fixed-width values through a
  /// filter's dictionary indices, string bodies compacted) and retain only
  /// that batch -- already coalesced, so coalesce() is not needed. Replaces
  /// addBatch() per input + coalesce(): no per-input addPayload flatten
  /// (2026-09-28: a join -> sort chain re-added 63K filtered lineitem
  /// batches one at a time, ~20 s serial in the sort's finish). Used by the
  /// sort store (--coalesce_threads) and, since 2026-09-30, by each build
  /// driver's store on its background thread (--build_coalesce_threads).
  /// `threads` < 0: --coalesce_threads.
  void coalesceFrom(
      const std::vector<RowVectorPtr>& inputs,
      int32_t threads = -1) {
    VELOX_CHECK(!container_.isScatteredModeEnabled());
    VELOX_CHECK(batches_.empty());
    if (inputs.empty()) {
      return;
    }
    auto merged = mergeRetainedBatches(
        rowType_,
        inputs,
        pool_,
        threads >= 0 ? threads
                     : CudfConfig::getInstance().benchmarkCoalesceThreads);
    addBatch(std::move(merged)); // flat children: addPayload is zero-copy
  }
  /// Coalesced mode, after coalesce(): the single merged batch.
  RowVectorPtr coalescedBatch() const {
    return container_.getCoalescedBatch();
  }
  exec::HybridContainer& container() {
    return container_;
  }

  /// int32 global ids (the GPU probe's survivor build ids).
  void gatherCoalesced(
      int32_t childIdx,
      const int32_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      idsScratch[i] = exec::HybridRowId{0, static_cast<uint64_t>(globalRows[i])};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

  /// Coalesced mode only: gather column `childIdx` at GLOBAL row ids.
  void gatherCoalesced(
      int32_t childIdx,
      const int64_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      idsScratch[i] = exec::HybridRowId{0, static_cast<uint64_t>(globalRows[i])};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

  /// Drop all retained batches (per-probe-batch reuse: the probe retains one
  /// GPU batch's host payload, gathers survivors, then clears).
  void clearBatches() {
    container_.clear();
    colRefs_.clear();
    lightKeepAlive_.clear();
    batches_.clear();
    batchStarts_.clear();
    totalRows_ = 0;
  }

  int64_t totalRows() const {
    return totalRows_;
  }

  /// The retained batches' row type (all columns, input child order).
  const RowTypePtr& rowType() const {
    return rowType_;
  }

  size_t numBatches() const {
    return batchStarts_.size();
  }

  /// Map a global row id (index into the concatenation of all added batches,
  /// == the GPU-side row id) to a scattered HybridRowId. O(log #batches).
  exec::HybridRowId idForGlobalRow(int64_t globalRow) const {
    VELOX_DCHECK_GE(globalRow, 0);
    VELOX_DCHECK_LT(globalRow, totalRows_);
    if (!container_.isScatteredModeEnabled()) {
      // Coalesced store (one merged batch, e.g. a coalesced provenance
      // store): ids are global row indices into that batch.
      VELOX_DCHECK(container_.isCoalesced());
      return exec::HybridRowId{0, static_cast<uint64_t>(globalRow)};
    }
    // upper_bound-1: last batch whose start <= globalRow.
    const auto it = std::upper_bound(
        batchStarts_.begin(), batchStarts_.end(), globalRow);
    const auto b = static_cast<uint32_t>(it - batchStarts_.begin() - 1);
    const auto rowInBatch = static_cast<uint32_t>(globalRow - batchStarts_[b]);
    return exec::HybridRowId{
        0, exec::HybridRowId::encodeScattered(b, rowInBatch)};
  }

  /// Bulk form of idForGlobalRow. `globalRows` are GPU-side row ids (e.g. the
  /// build-side survivor ids read back from the GPU probe).
  /// Returns how many ids fell back to the binary search (ids going
  /// backwards; 0 for a probe store, whose survivors arrive in order).
  int32_t idsForGlobalRows(
      const int32_t* globalRows,
      int32_t numRows,
      std::vector<exec::HybridRowId>& out) const {
    out.resize(numRows);
    int32_t fallbacks = 0;
    if (container_.isScatteredModeEnabled() &&
        !batchStarts_.empty()) {
      // 2026-09-27: ids of a probe store arrive in retained-batch order: walk
      // the batch starts with a forward cursor (O(1) amortized); fall back to
      // the binary search whenever an id goes backwards (random build ids).
      const size_t nb = batchStarts_.size();
      size_t b = 0;
      for (int32_t i = 0; i < numRows; ++i) {
        const int64_t g = globalRows[i];
        if (g < batchStarts_[b]) {
          out[i] = idForGlobalRow(g);
          b = static_cast<size_t>(out[i].batchId());
          ++fallbacks;
          continue;
        }
        while (b + 1 < nb && batchStarts_[b + 1] <= g) {
          ++b;
        }
        out[i] = exec::HybridRowId{
            0,
            exec::HybridRowId::encodeScattered(
                static_cast<uint32_t>(b),
                static_cast<uint32_t>(g - batchStarts_[b]))};
      }
      return fallbacks;
    }
    // Coalesced store: ids index the single merged batch directly.
    for (int32_t i = 0; i < numRows; ++i) {
      out[i] = idForGlobalRow(globalRows[i]);
    }
    return 0;
  }

  /// Gather column `childIdx` (index in the construction rowType) of the
  /// retained batches at `ids` into `result` (pre-created with the column's
  /// type; resized by the extraction). Runs HybridContainer's scattered
  /// single-container extraction kernel.
  ///
  /// `rowsScratch` is CALLER-OWNED scratch for the `rows` array
  /// HybridContainer's API requires: in scattered mode rows[i] is consulted
  /// only as a null-row marker (probe-miss in the CPU outer-join path), and
  /// every survivor here is a real row, so it is filled with non-null
  /// sentinels. Caller-owned so that a build-side store SHARED by several
  /// probe drivers stays immutable during gather (thread-safe: extraction
  /// only reads container state).
  void gather(
      int32_t childIdx,
      const std::vector<exec::HybridRowId>& ids,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch) const {
    if (light_) {
      lightGather(childIdx, ids, result);
      return;
    }
    const auto n = static_cast<int32_t>(ids.size());
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < n) {
      rowsScratch.assign(n, &kSentinel);
    }
    // extractColumn is logically const here (pure read in scattered mode,
    // ids included); HybridContainer just does not mark it so. Sharing one
    // ids vector across concurrent column gathers is therefore safe.
    const_cast<exec::HybridContainer&>(container_).extractColumn(
        rowsScratch.data(),
        n,
        /*columnIndex=*/childIdx,
        result,
        const_cast<std::vector<exec::HybridRowId>&>(ids));
  }

 private:
  // ---- lightweight retention (light_) ----
  struct ColRef {
    const void* values{nullptr}; // flat values (bits for BOOLEAN)
    const vector_size_t* indices{nullptr}; // dictionary view's indices
    const uint64_t* wrapNulls{nullptr}; // view's own nulls (by row)
    const uint64_t* baseNulls{nullptr}; // values' nulls (by value index)
    const BaseVector* generic{nullptr}; // non-flat fallback (copy path)
  };

  void addLightBatch(const RowVectorPtr& batch) {
    const auto nc = static_cast<int32_t>(rowType_->size());
    for (int32_t c = 0; c < nc; ++c) {
      auto& child = batch->childAt(c);
      child = BaseVector::loadedVectorShared(child);
      ColRef ref;
      if (child->encoding() == VectorEncoding::Simple::DICTIONARY) {
        auto base = BaseVector::loadedVectorShared(child->valueVector());
        if (base != nullptr &&
            base->encoding() == VectorEncoding::Simple::FLAT) {
          ref.values = base->valuesAsVoid();
          ref.indices = child->wrapInfo()->as<vector_size_t>();
          ref.wrapNulls = child->rawNulls();
          ref.baseNulls = base->rawNulls();
          lightKeepAlive_.push_back(std::move(base));
          colRefs_.push_back(ref);
          continue;
        }
        BaseVector::flattenVector(child);
      }
      if (child->encoding() == VectorEncoding::Simple::FLAT) {
        ref.values = child->valuesAsVoid();
        ref.baseNulls = child->rawNulls();
      } else {
        ref.generic = child.get();
      }
      colRefs_.push_back(ref);
    }
  }

  template <typename T>
  void lightGatherTyped(
      int32_t childIdx,
      const std::vector<exec::HybridRowId>& ids,
      FlatVector<T>* out) const {
    const auto n = static_cast<int32_t>(ids.size());
    const auto nc = static_cast<size_t>(rowType_->size());
    T* dst = out->mutableRawValues();
    constexpr int32_t kDist = 16;
    auto refOf = [&](uint32_t b) -> const ColRef& {
      return colRefs_[b * nc + childIdx];
    };
    uint32_t cur = std::numeric_limits<uint32_t>::max();
    const ColRef* cr = nullptr;
    for (int32_t i = 0; i < n; ++i) {
      if (i + 2 * kDist < n) {
        const auto& f = ids[i + 2 * kDist];
        const auto& fr = refOf(f.batchId());
        if (fr.values != nullptr) {
          const auto r = f.rowInBatch();
          __builtin_prefetch(
              static_cast<const T*>(fr.values) +
                  (fr.indices ? fr.indices[r] : r),
              0,
              1);
        }
      }
      if constexpr (std::is_same_v<T, StringView>) {
        if (i + kDist < n) {
          const auto& f = ids[i + kDist];
          const auto& fr = refOf(f.batchId());
          if (fr.values != nullptr) {
            const auto r = f.rowInBatch();
            const auto& v = static_cast<const StringView*>(
                fr.values)[fr.indices ? fr.indices[r] : r];
            if (!v.isInline()) {
              __builtin_prefetch(v.data(), 0, 1);
            }
          }
        }
      }
      const auto b = ids[i].batchId();
      const auto r = ids[i].rowInBatch();
      if (b != cur) {
        cur = b;
        cr = &refOf(b);
      }
      if (cr->values == nullptr) {
        out->copy(cr->generic, i, r, 1);
        continue;
      }
      const vector_size_t idx = cr->indices ? cr->indices[r] : r;
      if ((cr->wrapNulls && bits::isBitNull(cr->wrapNulls, r)) ||
          (cr->baseNulls && bits::isBitNull(cr->baseNulls, idx))) {
        out->setNull(i, true);
        continue;
      }
      const T v = static_cast<const T*>(cr->values)[idx];
      if constexpr (std::is_same_v<T, StringView>) {
        out->set(i, v); // bodies copied, as in scattered extraction
      } else {
        dst[i] = v;
      }
    }
  }

  void lightGather(
      int32_t childIdx,
      const std::vector<exec::HybridRowId>& ids,
      const VectorPtr& result) const {
    const auto n = static_cast<vector_size_t>(ids.size());
    result->resize(n);
    const auto nc = static_cast<size_t>(rowType_->size());
    switch (result->typeKind()) {
      case TypeKind::BIGINT:
        return lightGatherTyped<int64_t>(childIdx, ids, result->asFlatVector<int64_t>());
      case TypeKind::INTEGER:
        return lightGatherTyped<int32_t>(childIdx, ids, result->asFlatVector<int32_t>());
      case TypeKind::SMALLINT:
        return lightGatherTyped<int16_t>(childIdx, ids, result->asFlatVector<int16_t>());
      case TypeKind::TINYINT:
        return lightGatherTyped<int8_t>(childIdx, ids, result->asFlatVector<int8_t>());
      case TypeKind::DOUBLE:
        return lightGatherTyped<double>(childIdx, ids, result->asFlatVector<double>());
      case TypeKind::REAL:
        return lightGatherTyped<float>(childIdx, ids, result->asFlatVector<float>());
      case TypeKind::HUGEINT:
        return lightGatherTyped<int128_t>(childIdx, ids, result->asFlatVector<int128_t>());
      case TypeKind::TIMESTAMP:
        return lightGatherTyped<Timestamp>(childIdx, ids, result->asFlatVector<Timestamp>());
      case TypeKind::VARCHAR:
      case TypeKind::VARBINARY:
        return lightGatherTyped<StringView>(childIdx, ids, result->asFlatVector<StringView>());
      default: {
        // BOOLEAN (bit-packed) and anything else: generic per-row copy from
        // the retained child.
        for (vector_size_t i = 0; i < n; ++i) {
          const auto b = ids[i].batchId();
          result->copy(
              batches_[b]->childAt(childIdx).get(), i, ids[i].rowInBatch(), 1);
        }
        (void)nc;
      }
    }
  }

  const bool light_;
  std::vector<ColRef> colRefs_; // [batch * numColumns + column]
  std::vector<VectorPtr> lightKeepAlive_; // loaded dictionary bases
  const RowTypePtr rowType_;
  memory::MemoryPool* const pool_;
  // Never-written keys container: HybridContainer's ctor dereferences its
  // column layout (for the rowId-column offset used by getRowIds, which we
  // never call), so it must be a real RowContainer with >= 1 column.
  exec::RowContainer dummyKeys_;
  exec::HybridContainer container_;
  // batchStarts_[b] = global row id of the first row of batch b.
  std::vector<int64_t> batchStarts_;
  std::vector<RowVectorPtr> batches_;
  int64_t totalRows_ = 0;
};


/// The CPU hybrid join's build-side layout, on the GPU boundary: one
/// COALESCED per-driver store per build driver, registered as containers
/// 0..N-1 of one HybridContainer group, so a survivor's build payload is
/// extracted by container id exactly as the CPU probe does (per-container
/// grouped, pre-decoded, prefetched extraction). The only GPU-specific step
/// is mapping the matcher's global build row id to (driver chunk, local row)
/// with a search over the (<=16) chunk starts.
class MultiBoundaryHostStore {
 public:
  explicit MultiBoundaryHostStore(std::vector<std::shared_ptr<BoundaryHostStore>> chunks)
      : chunks_(std::move(chunks)) {
    VELOX_CHECK(!chunks_.empty());
    VELOX_CHECK_LE(chunks_.size(), 255, "too many build drivers for uint8 container ids");
    int64_t start = 0;
    for (size_t k = 0; k < chunks_.size(); ++k) {
      chunkStarts_.push_back(start);
      start += chunks_[k]->totalRows();
      chunks_[k]->container().setId(static_cast<uint8_t>(k));
      all_[static_cast<uint8_t>(k)] = &chunks_[k]->container();
    }
    totalRows_ = start;
    for (auto& c : chunks_) {
      c->container().setAllContainers(all_);
      // As the CPU hybrid join arm (--hybrid_join_reorder_enabled=false):
      // extract in survivor order, no reorder-by-container pass.
      c->container().setReorderEnabled(false);
    }
  }
  int64_t totalRows() const {
    return totalRows_;
  }
  const RowTypePtr& rowType() const {
    return chunks_[0]->rowType();
  }
  static constexpr int kLocalBits = 48;
  static constexpr uint64_t kLocalMask = (1ULL << kLocalBits) - 1;
  /// Gather at BUILD-TIME-ENCODED ids (containerId << 48 | localRow), the
  /// CPU mechanism: no mapping on the extraction path.
  void gatherEncoded(
      int32_t childIdx,
      const uint64_t* encoded,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    for (int32_t i = 0; i < numRows; ++i) {
      const uint64_t v = encoded[i];
      idsScratch[i] = exec::HybridRowId{
          static_cast<uint8_t>(v >> kLocalBits), v & kLocalMask};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(chunks_[0]->container())
        .extractColumn(rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

  /// Gather column `childIdx` at GLOBAL build row ids (the matcher's ids).
  void gather(
      int32_t childIdx,
      const int32_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    idsScratch.resize(numRows);
    const auto nChunks = chunkStarts_.size();
    for (int32_t i = 0; i < numRows; ++i) {
      const int64_t g = globalRows[i];
      size_t k = nChunks - 1;
      if (nChunks > 1) {
        // last chunk whose start <= g
        k = static_cast<size_t>(
            std::upper_bound(chunkStarts_.begin(), chunkStarts_.end(), g) -
            chunkStarts_.begin() - 1);
      }
      idsScratch[i] = exec::HybridRowId{
          static_cast<uint8_t>(k), static_cast<uint64_t>(g - chunkStarts_[k])};
    }
    static const char kSentinel = 0;
    if (static_cast<int32_t>(rowsScratch.size()) < numRows) {
      rowsScratch.assign(numRows, &kSentinel);
    }
    const_cast<exec::HybridContainer&>(chunks_[0]->container())
        .extractColumn(rowsScratch.data(), numRows, childIdx, result, idsScratch);
  }

 private:
  std::vector<std::shared_ptr<BoundaryHostStore>> chunks_;
  std::vector<int64_t> chunkStarts_;
  std::unordered_map<uint8_t, exec::HybridContainer*> all_;
  int64_t totalRows_{0};
};

/// 2026-09-27: DEFERRED wait for the per-driver build-store coalescing. The
/// build publishes its hash table as soon as the GPU side is ready; each build
/// driver's coalesce keeps running as an async task (overlapped with the
/// probe's ingress, transfer and GPU join), and the FIRST probe gather waits
/// for all of them (once, shared by every probe driver) and only then links
/// the chunks into a MultiBoundaryHostStore. Same gather interface as
/// MultiBoundaryHostStore. Before this, the build finalizer joined every
/// coalesce before building the GPU table, so on a short GPU build the whole
/// coalesce showed up in the build's wall time.
class LazyMultiBoundaryHostStore {
 public:
  /// `rows[k]`: rows retained by chunk k (the stores themselves are filled by
  /// the pending merges, so their counts are not readable yet).
  LazyMultiBoundaryHostStore(
      std::vector<std::shared_ptr<BoundaryHostStore>> chunks,
      std::vector<std::shared_future<int64_t>> pending,
      const std::vector<int64_t>& rows)
      : chunks_(std::move(chunks)), pending_(std::move(pending)) {
    VELOX_CHECK(!chunks_.empty());
    VELOX_CHECK_EQ(chunks_.size(), pending_.size());
    VELOX_CHECK_EQ(chunks_.size(), rows.size());
    for (const auto r : rows) {
      totalRows_ += r;
    }
  }

  /// Blocks until every chunk is coalesced; the first caller links the
  /// chunks. Returns the nanoseconds THIS call waited (0 once ready).
  int64_t ensureReady() {
    if (ready_.load(std::memory_order_acquire)) {
      return 0;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::call_once(once_, [this] {
      int64_t sum = 0;
      for (auto& f : pending_) {
        sum += f.get(); // rethrows a failed coalesce
      }
      coalesceNanos_ = sum;
      store_ = std::make_shared<MultiBoundaryHostStore>(chunks_);
      ready_.store(true, std::memory_order_release);
    });
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now() - t0)
        .count();
  }

  /// Summed per-driver coalesce time (valid once ready).
  int64_t coalesceNanos() const {
    return coalesceNanos_;
  }
  /// True exactly once (lets one probe driver report coalesceNanos()).
  bool claimCoalesceReport() {
    return !reported_.exchange(true);
  }

  int64_t totalRows() const {
    return totalRows_;
  }
  const RowTypePtr& rowType() const {
    return chunks_[0]->rowType();
  }

  void gatherEncoded(
      int32_t childIdx,
      const uint64_t* encoded,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    ready()->gatherEncoded(
        childIdx, encoded, numRows, result, rowsScratch, idsScratch);
  }

  void gather(
      int32_t childIdx,
      const int32_t* globalRows,
      int32_t numRows,
      const VectorPtr& result,
      std::vector<const char*>& rowsScratch,
      std::vector<exec::HybridRowId>& idsScratch) const {
    ready()->gather(
        childIdx, globalRows, numRows, result, rowsScratch, idsScratch);
  }

 private:
  const MultiBoundaryHostStore* ready() const {
    const_cast<LazyMultiBoundaryHostStore*>(this)->ensureReady();
    return store_.get();
  }

  std::vector<std::shared_ptr<BoundaryHostStore>> chunks_;
  std::vector<std::shared_future<int64_t>> pending_;
  std::once_flag once_;
  std::atomic<bool> ready_{false};
  std::atomic<bool> reported_{false};
  std::shared_ptr<MultiBoundaryHostStore> store_;
  int64_t coalesceNanos_{0};
  int64_t totalRows_{0};
};

} // namespace facebook::velox::cudf_velox
