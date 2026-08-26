/*
 * RowOrderBy.h (branch row-sort, 2026-08-24)
 *
 * Row-wise GPU sort. Consumes fixed-stride row batches (RowStoreVector from
 * the row pack / row joins, possibly through a gather LocalPartition) or
 * columnar CudfVector batches (transposed on arrival), concatenates them
 * into one row store, sorts on the key fields (extracted to cudf columns,
 * cudf::sorted_order) and applies the permutation with ONE row gather --
 * the row-native counterpart of cudf::sort_by_key's per-column gathers.
 *
 * Spine deferral: with a crossing-set input (keys + __rowid + provenance)
 * only the keys and the rowid are sorted/gathered on the GPU; the payload
 * is materialized from the retained host batches by the permutation, at
 * the exit (host emission when the consumer is CudfToVelox, else uploaded
 * for a columnar consumer). Output is emitted in chunks.
 */
#pragma once

#include "velox/experimental/cudf/exec/BoundaryHostStore.h"
#include "velox/experimental/cudf/exec/CudfOperator.h"
#include "velox/experimental/cudf/exec/GpuFixedRowStore.h"
#include "velox/experimental/cudf/exec/RowStoreVector.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include "velox/core/PlanNode.h"
#include "velox/exec/Operator.h"

#include <cudf/types.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <memory>
#include <string>
#include <vector>

namespace facebook::velox::cudf_velox {

class RowOrderBy : public CudfOperatorBase {
 public:
  RowOrderBy(
      int32_t operatorId,
      exec::DriverCtx* driverCtx,
      std::shared_ptr<const core::OrderByNode> orderByNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }
  exec::BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return exec::BlockingReason::kNotBlocked;
  }
  bool isFinished() override {
    return finished_;
  }

  const std::shared_ptr<const core::OrderByNode>& orderByNode() const {
    return orderByNode_;
  }

  /// Sort-key column names of an OrderByNode (the row pack's crossing set /
  /// null-key guard when it feeds a row sort).
  static std::vector<std::string> sortKeyNames(const core::OrderByNode& node);

 protected:
  void doAddInput(RowVectorPtr input) override;
  RowVectorPtr doGetOutput() override;
  void doNoMoreInput() override;
  void doClose() override;

 private:
  static constexpr int32_t kChunkRows = 1 << 20;

  void captureLayout(const RowStoreVector& v);
  void concatenateRowInputs();
  void transposeCudfInputs();
  void sortRows();
  void resolveOutputOnce();
  int32_t fieldIndexOf(const std::string& name) const;

  RowVectorPtr emitHostChunk(int64_t begin, int32_t n);
  RowVectorPtr emitColumnarChunk(int64_t begin, int32_t n);
  std::vector<std::unique_ptr<cudf::column>> transposeChunkColumns(
      const uint8_t* base,
      const uint8_t* nullBase,
      int32_t n,
      bool withRowId = false);
  // Deferred columnar chunk: transposed GPU-side columns kept between the
  // id read and the splice (transpose-first, round 6).
  std::vector<std::unique_ptr<cudf::column>> gpuColsForChunk_;
  /// Host-gather the deferred output columns for `n` sorted rows whose
  /// GLOBAL rowids are `globalIds` (concatenation order).
  std::vector<VectorPtr> gatherDeferred(
      const std::vector<int64_t>& globalIds,
      int32_t n);

  std::shared_ptr<const core::OrderByNode> orderByNode_;
  std::vector<std::string> keyNames_;
  std::vector<cudf::order> orders_;
  std::vector<cudf::null_order> nullOrders_;
  rmm::cuda_stream_view stream_;

  // ---- inputs ----
  std::vector<std::shared_ptr<RowStoreVector>> rowInputs_;
  std::vector<CudfVectorPtr> cudfInputs_;
  int64_t totalRows_ = 0;
  int64_t totalChars_ = 0;

  // ---- concatenated layout ----
  bool layoutReady_ = false;
  std::vector<FieldDesc> fields_;
  int32_t rowWidth_ = 0;
  std::vector<std::string> fieldNames_; // empty: field i == column i
  int32_t nullStride_ = 0; // input sidecar stride (0: null-free)
  int32_t nullPad_ = 0; // sidecar stride padded to 8 for the row gather
  bool hasStrings_ = false;
  int32_t rowIdField_ = -1;
  std::vector<std::shared_ptr<BoundaryHostStore>> stores_;
  std::vector<int64_t> storeBases_; // global rowid base per store
  // Coalesced per-store columns of the deferred output columns
  // [store][deferred col], built while the device sorts.
  std::vector<std::vector<VectorPtr>> storeCols_;
  void coalesceStores();
  rmm::device_buffer fieldsBuffer_; // FieldDesc[] on device

  // ---- sorted data ----
  rmm::device_buffer sorted_;
  rmm::device_buffer sortedNulls_;
  rmm::device_buffer heap_; // string bytes (shared by all rows)

  // ---- output ----
  int emitHost_ = -1;
  std::vector<int32_t> outFieldIdx_; // per output column, -1 = deferred
  std::vector<int32_t> deferredCols_;
  int64_t cursor_ = 0;
  bool finished_ = false;
  std::vector<uint8_t> hostRows_; // prefetch target (next chunk)
  std::vector<uint8_t> hostRowsCur_; // chunk being extracted
  int64_t prefetchBegin_ = -1;
  std::vector<uint8_t> hostNulls_;
  std::vector<uint8_t> hostChars_;
  std::shared_ptr<const std::vector<uint8_t>> hostCharsShared_; // native output
  bool hostCharsReady_ = false;
  rmm::device_buffer idsDev_;
  rmm::device_buffer strOffsetsDev_; // int64[n+1] scratch
  std::vector<exec::HybridRowId> rowIdsScratch_;
  std::vector<const char*> sentinelScratch_;
};

/// True when `next` is a gather CudfLocalPartition whose plan parent is an
/// OrderByNode that will run as RowOrderBy: rows may be emitted to it.
bool rowSortConsumesGather(
    const exec::Operator* next,
    const core::PlanNodePtr& planRoot);

} // namespace facebook::velox::cudf_velox
