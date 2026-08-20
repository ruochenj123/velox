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

#include <thread>

#include "SortBuffer.h"
#include "velox/exec/MemoryReclaimer.h"
#include "velox/exec/OperatorUtils.h"
#include "velox/exec/Spiller.h"

namespace facebook::velox::exec {

SortBuffer::SortBuffer(
    const RowTypePtr& input,
    const std::vector<column_index_t>& sortColumnIndices,
    const std::vector<CompareFlags>& sortCompareFlags,
    velox::memory::MemoryPool* pool,
    tsan_atomic<bool>* nonReclaimableSection,
    common::PrefixSortConfig prefixSortConfig,
    const common::SpillConfig* spillConfig,
    exec::SpillStats* spillStats,
    bool hybridSortEnabled,
    bool hybridSortScattered,
    uint32_t hybridSortMinPayloadBytes)
    : input_(input),
      sortCompareFlags_(sortCompareFlags),
      pool_(pool),
      nonReclaimableSection_(nonReclaimableSection),
      prefixSortConfig_(prefixSortConfig),
      spillConfig_(spillConfig),
      spillStats_(spillStats),
      sortedRows_(0, memory::StlAllocator<char*>(*pool)),
      hybridSortEnabled_(hybridSortEnabled),
      hybridSortScattered_(hybridSortScattered) {
  VELOX_CHECK_GE(input_->children().size(), sortCompareFlags_.size());
  VELOX_CHECK_GT(sortCompareFlags_.size(), 0);
  VELOX_CHECK_EQ(sortColumnIndices.size(), sortCompareFlags_.size());
  VELOX_CHECK_NOT_NULL(nonReclaimableSection_);

  std::vector<TypePtr> sortedColumnTypes;
  std::vector<TypePtr> nonSortedColumnTypes;
  std::vector<std::string> nonSortedColumnNames;
  std::vector<std::string> sortedSpillColumnNames;
  std::vector<TypePtr> sortedSpillColumnTypes;
  sortedColumnTypes.reserve(sortColumnIndices.size());
  nonSortedColumnTypes.reserve(input->size() - sortColumnIndices.size());
  nonSortedColumnNames.reserve(input->size() - sortColumnIndices.size());
  payloadChannels_.reserve(input->size() - sortColumnIndices.size());
  sortedSpillColumnNames.reserve(input->size());
  sortedSpillColumnTypes.reserve(input->size());
  std::unordered_set<column_index_t> sortedChannelSet;
  // Sorted key columns.
  for (column_index_t i = 0; i < sortColumnIndices.size(); ++i) {
    keyColumnMap_.emplace_back(IdentityProjection(i, sortColumnIndices.at(i)));
    columnMap_.emplace_back(IdentityProjection(i, sortColumnIndices.at(i)));
    sortedColumnTypes.emplace_back(input_->childAt(sortColumnIndices.at(i)));
    sortedSpillColumnTypes.emplace_back(
        input_->childAt(sortColumnIndices.at(i)));
    sortedSpillColumnNames.emplace_back(input->nameOf(sortColumnIndices.at(i)));
    sortedChannelSet.emplace(sortColumnIndices.at(i));
  }
  // Non-sorted key columns.
  for (column_index_t i = 0, nonSortedIndex = sortCompareFlags_.size();
       i < input_->size();
       ++i) {
    if (sortedChannelSet.count(i) != 0) {
      continue;
    }
    columnMap_.emplace_back(nonSortedIndex++, i);
    nonSortedColumnTypes.emplace_back(input_->childAt(i));
    nonSortedColumnNames.emplace_back(input->nameOf(i));
    payloadChannels_.push_back(i);
    sortedSpillColumnTypes.emplace_back(input_->childAt(i));
    sortedSpillColumnNames.emplace_back(input->nameOf(i));
  }

  // Hybrid sort does not support the spill paths yet; gracefully fall back to
  // the standard row-layout path when spilling is configured.
  // hybrid+spill deferred; see transplant plan Phase 5.
  if (spillConfig_ != nullptr) {
    hybridSortEnabled_ = false;
  }
  // Hybrid layout only pays off when there are payload columns to keep
  // columnar.
  hybridSortEnabled_ = hybridSortEnabled_ && !nonSortedColumnTypes.empty();
  // Payload-width eligibility gate (symmetry with the hybrid join gate; the
  // default threshold of 1 byte keeps hybrid on whenever any payload
  // exists). Sort outputs every payload column, so all of them count.
  if (hybridSortEnabled_) {
    int64_t payloadBytes = 0;
    for (const auto& type : nonSortedColumnTypes) {
      payloadBytes += hybridPayloadNominalWidth(type);
    }
    if (payloadBytes < static_cast<int64_t>(hybridSortMinPayloadBytes)) {
      hybridSortEnabled_ = false;
    }
    VLOG(1) << "hybrid sort payload gate: payloadBytes=" << payloadBytes
            << " threshold=" << hybridSortMinPayloadBytes << " hybridSort="
            << (hybridSortEnabled_ ? "enabled" : "disabled");
  }
  if (hybridSortEnabled_) {
    const std::vector<TypePtr> rowIdType = {BIGINT()};
    data_ = std::make_unique<RowContainer>(
        sortedColumnTypes, rowIdType, /*useListRowIndex=*/true, pool_);
    hybridData_ = std::make_unique<HybridContainer>(
        sortedColumnTypes, nonSortedColumnTypes, data_.get());
    // Register the (single) container in the extraction registry.
    std::unordered_map<uint8_t, HybridContainer*> hybridDataChannel;
    hybridDataChannel[0] = hybridData_.get();
    hybridData_->setAllContainers(hybridDataChannel);
    if (hybridSortScattered_) {
      // Must be set before the first addPayload so per-batch decoded
      // payloads are populated for the scattered extraction kernels.
      hybridData_->setScatteredModeEnabled(true);
    }

    payloadTypes_ =
        ROW(std::move(nonSortedColumnNames), std::move(nonSortedColumnTypes));
  } else {
    data_ = std::make_unique<RowContainer>(
        sortedColumnTypes,
        nonSortedColumnTypes,
        /*useListRowIndex=*/true,
        pool_);
  }
  spillerStoreType_ =
      ROW(std::move(sortedSpillColumnNames), std::move(sortedSpillColumnTypes));
}

SortBuffer::~SortBuffer() {
  pool_->release();
}

void SortBuffer::addInput(const VectorPtr& input) {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::SortBuffer::addInput", this);

  VELOX_CHECK(!noMoreInput_);
  ensureInputFits(input);

  const SelectivityVector allRows(input->size());
  std::vector<char*> rows(input->size());
  for (int row = 0; row < input->size(); ++row) {
    rows[row] = data_->newRow();
  }
  const auto* inputRow = input->as<RowVector>();
  if (hybridSortEnabled_) {
    if (hybridSortScattered_) {
      // Scattered ids: driverId (8 bits) | batchId (24) | rowInBatch (32).
      const auto batchId = hybridData_->getNumBatches();
      for (int row = 0; row < input->size(); ++row) {
        uint64_t encodedId = HybridRowId::encodeScattered(batchId, row);
        data_->storeSingleRowId(encodedId, rows[row]);
      }
    } else {
      const auto currentRows = hybridData_->getNumRows();
      for (int row = 0; row < input->size(); ++row) {
        // Store RowId: driverId (8 bits) | globalRowId (56 bits).
        uint64_t encodedId = (static_cast<uint64_t>(0) << 56) |
            (static_cast<uint64_t>(row + currentRows) & ((1ULL << 56) - 1));
        data_->storeSingleRowId(encodedId, rows[row]);
      }
    }
    // Store the sort key columns row-wise.
    for (const auto& columnProjection : keyColumnMap_) {
      DecodedVector decoded(
          *inputRow->childAt(columnProjection.outputChannel), allRows);
      data_->store(
          decoded,
          folly::Range(rows.data(), input->size()),
          columnProjection.inputChannel);
    }
    // Gather payload columns (zero-copy: shares the input's children).
    hybridData_->addPayload(
        wrapColumns(inputRow, payloadChannels_, payloadTypes_, pool_));
  } else {
    for (const auto& columnProjection : columnMap_) {
      DecodedVector decoded(
          *inputRow->childAt(columnProjection.outputChannel), allRows);
      data_->store(
          decoded,
          folly::Range(rows.data(), input->size()),
          columnProjection.inputChannel);
    }
  }
  numInputRows_ += allRows.size();
}

void SortBuffer::noMoreInput() {
  velox::common::testutil::TestValue::adjust(
      "facebook::velox::exec::SortBuffer::noMoreInput", this);
  VELOX_CHECK(!noMoreInput_);
  VELOX_CHECK_NULL(outputSpiller_);

  // It may trigger spill, make sure it's triggered before noMoreInput_ is set.
  ensureSortFits();

  noMoreInput_ = true;

  // No data.
  if (numInputRows_ == 0) {
    return;
  }

  // Coalesced hybrid mode: run the payload merge on a background thread,
  // overlapped with the key sort below. The merge touches only the payload
  // batches; the sort touches only the key container (data_) — fully
  // disjoint. The thread is joined before updateEstimatedOutputRowSize(),
  // which is the first reader of hybridData_ after this point. Scattered
  // mode skips coalescing entirely; extraction reads per-batch vectors.
  std::thread coalesceThread;
  std::exception_ptr coalesceError;
  const bool overlapCoalesce = hybridData_ != nullptr &&
      !hybridSortScattered_ && inputSpiller_ == nullptr;
  if (overlapCoalesce) {
    coalesceThread = std::thread([&]() {
      try {
        hybridData_->coalesceBatches();
      } catch (...) {
        coalesceError = std::current_exception();
      }
    });
  } else if (hybridData_ != nullptr && !hybridSortScattered_) {
    hybridData_->coalesceBatches();
  }

  if (inputSpiller_ == nullptr) {
    try {
      VELOX_CHECK_EQ(numInputRows_, data_->numRows());
      // Sort the pointers to the rows in RowContainer (data_) instead of
      // sorting the rows.
      // TODO: Reuse 'RowContainer::rowPointers_'.
      sortedRows_.resize(numInputRows_);
      RowContainerIterator iter;
      data_->listRows(&iter, numInputRows_, sortedRows_.data());
      PrefixSort::sort(
          data_.get(),
          sortCompareFlags_,
          prefixSortConfig_,
          pool_,
          sortedRows_);
    } catch (...) {
      if (coalesceThread.joinable()) {
        coalesceThread.join();
      }
      throw;
    }
    if (coalesceThread.joinable()) {
      coalesceThread.join();
      if (coalesceError) {
        std::rethrow_exception(coalesceError);
      }
    }
    // Runs after the coalesce join: reads hybridData_ under hybrid mode.
    updateEstimatedOutputRowSize();
  } else {
    // Spill the remaining in-memory state to disk if spilling has been
    // triggered on this sort buffer. This is to simplify query OOM prevention
    // when producing output as we don't support to spill during that stage as
    // for now.
    spill();

    finishSpill();
  }

  // Releases the unused memory reservation after procesing input.
  pool_->release();
}

RowVectorPtr SortBuffer::getOutput(vector_size_t maxOutputRows) {
  SCOPE_EXIT {
    pool_->release();
  };

  VELOX_CHECK(noMoreInput_);

  if (numOutputRows_ == numInputRows_) {
    return nullptr;
  }
  VELOX_CHECK_GT(maxOutputRows, 0);
  VELOX_CHECK_GT(numInputRows_, numOutputRows_);
  const vector_size_t batchSize =
      std::min<uint64_t>(numInputRows_ - numOutputRows_, maxOutputRows);
  ensureOutputFits(batchSize);
  prepareOutput(batchSize);
  if (hasSpilled()) {
    getOutputWithSpill();
  } else {
    getOutputWithoutSpill();
  }
  return std::move(output_);
}

bool SortBuffer::hasSpilled() const {
  if (inputSpiller_ != nullptr) {
    VELOX_CHECK_NULL(outputSpiller_);
    return true;
  }
  return outputSpiller_ != nullptr;
}

void SortBuffer::spill() {
  VELOX_CHECK_NOT_NULL(
      spillConfig_, "spill config is null when SortBuffer spill is called");

  // Check if sort buffer is empty or not, and skip spill if it is empty.
  if (data_->numRows() == 0) {
    return;
  }
  updateEstimatedOutputRowSize();

  if (sortedRows_.empty()) {
    spillInput();
  } else {
    spillOutput();
  }
}

std::optional<uint64_t> SortBuffer::estimateOutputRowSize() const {
  return estimatedOutputRowSize_;
}

void SortBuffer::ensureInputFits(const VectorPtr& input) {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  const int64_t numRows = data_->numRows();
  if (numRows == 0) {
    // 'data_' is empty. Nothing to spill.
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const int64_t flatInputBytes = input->estimateFlatSize();

  // Test-only spill path.
  if (numRows > 0 && testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  const auto currentMemoryUsage = pool_->usedBytes();
  const auto minReservationBytes =
      currentMemoryUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = pool_->availableReservation();
  const int64_t estimatedIncrementalBytes =
      data_->sizeIncrement(input->size(), outOfLineBytes ? flatInputBytes : 0);
  if (availableReservationBytes > minReservationBytes) {
    // If we have enough free rows for input rows and enough variable length
    // free space for the vector's flat size, no need for spilling.
    if (freeRows > input->size() &&
        (outOfLineBytes == 0 || outOfLineFreeBytes >= flatInputBytes)) {
      return;
    }

    // If the current available reservation in memory pool is 2X the
    // estimatedIncrementalBytes, no need to spill.
    if (availableReservationBytes > 2 * estimatedIncrementalBytes) {
      return;
    }
  }

  // Try reserving targetIncrementBytes more in memory pool, if succeed, no
  // need to spill.
  const auto targetIncrementBytes = std::max<int64_t>(
      estimatedIncrementalBytes * 2,
      currentMemoryUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << pool()->name()
               << ", root pool: " << pool()->root()->name()
               << ", used: " << succinctBytes(pool()->usedBytes())
               << ", reservation: " << succinctBytes(pool()->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(pool()->root()->reservedBytes());
}

void SortBuffer::ensureOutputFits(vector_size_t batchSize) {
  VELOX_CHECK_GT(batchSize, 0);
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (!estimatedOutputRowSize_.has_value() || hasSpilled()) {
    return;
  }

  const uint64_t outputBufferSizeToReserve =
      estimatedOutputRowSize_.value() * batchSize * 1.2;
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(outputBufferSizeToReserve)) {
      return;
    }
  }
  LOG(WARNING) << "Failed to reserve "
               << succinctBytes(outputBufferSizeToReserve)
               << " for memory pool " << pool_->name()
               << ", root pool: " << pool_->root()->name()
               << ", used: " << succinctBytes(pool_->usedBytes())
               << ", reservation: " << succinctBytes(pool_->reservedBytes())
               << ", root pool reservation: "
               << succinctBytes(pool_->root()->reservedBytes());
}

void SortBuffer::ensureSortFits() {
  // Check if spilling is enabled or not.
  if (spillConfig_ == nullptr) {
    return;
  }

  // Test-only spill path.
  if (testingTriggerSpill(pool_->name())) {
    spill();
    return;
  }

  if (numInputRows_ == 0 || inputSpiller_ != nullptr) {
    return;
  }

  // The memory for std::vector sorted rows and prefix sort required buffer.
  const auto sortBufferToReserve =
      numInputRows_ * sizeof(char*) +
      PrefixSort::maxRequiredBytes(
          data_.get(), sortCompareFlags_, prefixSortConfig_, pool_);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (pool_->maybeReserve(sortBufferToReserve)) {
      return;
    }
  }

  LOG(WARNING) << fmt::format(
      "Failed to reserve {} for memory pool {}, usage: {}, reservation: {}",
      succinctBytes(sortBufferToReserve),
      pool_->name(),
      succinctBytes(pool_->usedBytes()),
      succinctBytes(pool_->reservedBytes()));
}

void SortBuffer::updateEstimatedOutputRowSize() {
  const auto optionalRowSize = hybridSortEnabled_
      ? hybridData_->estimateRowSize()
      : data_->estimateRowSize();
  if (!optionalRowSize.has_value() || optionalRowSize.value() == 0) {
    return;
  }

  const auto rowSize = optionalRowSize.value();
  if (!estimatedOutputRowSize_.has_value()) {
    estimatedOutputRowSize_ = rowSize;
  } else if (rowSize > estimatedOutputRowSize_.value()) {
    estimatedOutputRowSize_ = rowSize;
  }
}

void SortBuffer::spillInput() {
  if (inputSpiller_ == nullptr) {
    VELOX_CHECK(!noMoreInput_);
    const auto sortingKeys = SpillState::makeSortingKeys(sortCompareFlags_);
    inputSpiller_ = std::make_unique<SortInputSpiller>(
        data_.get(), spillerStoreType_, sortingKeys, spillConfig_, spillStats_);
  }
  inputSpiller_->spill();
  data_->clear();
}

void SortBuffer::spillOutput() {
  if (hasSpilled()) {
    // Already spilled.
    return;
  }
  if (numOutputRows_ == sortedRows_.size()) {
    // All the output has been produced.
    return;
  }

  outputSpiller_ = std::make_unique<SortOutputSpiller>(
      data_.get(), spillerStoreType_, spillConfig_, spillStats_);
  auto spillRows = SpillerBase::SpillRows(
      sortedRows_.begin() + numOutputRows_,
      sortedRows_.end(),
      *memory::spillMemoryPool());
  outputSpiller_->spill(spillRows);
  data_->clear();
  sortedRows_.clear();
  sortedRows_.shrink_to_fit();
  // Finish right after spilling as the output spiller only spills at most
  // once.
  finishSpill();
}

void SortBuffer::prepareOutput(vector_size_t batchSize) {
  if (output_ != nullptr) {
    VectorPtr output = std::move(output_);
    BaseVector::prepareForReuse(output, batchSize);
    output_ = std::static_pointer_cast<RowVector>(output);
  } else {
    output_ = std::static_pointer_cast<RowVector>(
        BaseVector::create(input_, batchSize, pool_));
  }

  if (hasSpilled()) {
    spillSources_.resize(batchSize);
    spillSourceRows_.resize(batchSize);
    prepareOutputWithSpill();
  }

  VELOX_CHECK_GT(output_->size(), 0);
  VELOX_CHECK_LE(output_->size() + numOutputRows_, numInputRows_);
}

void SortBuffer::getOutputWithoutSpill() {
  VELOX_DCHECK_EQ(numInputRows_, sortedRows_.size());
  if (hybridSortEnabled_) {
    // Reused across output batches; getOutput is single-threaded per buffer.
    static thread_local std::vector<HybridRowId> outputRowIds;
    outputRowIds.resize(output_->size());
    hybridData_->getRowIds(
        sortedRows_.data() + numOutputRows_, output_->size(), outputRowIds);
    for (const auto& columnProjection : columnMap_) {
      hybridData_->extractColumn(
          sortedRows_.data() + numOutputRows_,
          output_->size(),
          columnProjection.inputChannel,
          output_->childAt(columnProjection.outputChannel),
          outputRowIds);
    }
  } else {
    for (const auto& columnProjection : columnMap_) {
      data_->extractColumn(
          sortedRows_.data() + numOutputRows_,
          output_->size(),
          columnProjection.inputChannel,
          output_->childAt(columnProjection.outputChannel));
    }
  }
  numOutputRows_ += output_->size();
}

void SortBuffer::getOutputWithSpill() {
  VELOX_CHECK_NOT_NULL(spillMerger_);
  VELOX_DCHECK_EQ(sortedRows_.size(), 0);

  int32_t outputRow = 0;
  int32_t outputSize = 0;
  bool isEndOfBatch = false;
  while (outputRow + outputSize < output_->size()) {
    SpillMergeStream* stream = spillMerger_->next();
    VELOX_CHECK_NOT_NULL(stream);

    spillSources_[outputSize] = &stream->current();
    spillSourceRows_[outputSize] = stream->currentIndex(&isEndOfBatch);
    ++outputSize;
    if (FOLLY_UNLIKELY(isEndOfBatch)) {
      // The stream is at end of input batch. Need to copy out the rows before
      // fetching next batch in 'pop'.
      gatherCopy(
          output_.get(),
          outputRow,
          outputSize,
          spillSources_,
          spillSourceRows_,
          columnMap_);
      outputRow += outputSize;
      outputSize = 0;
    }
    // Advance the stream.
    stream->pop();
  }
  VELOX_CHECK_EQ(outputRow + outputSize, output_->size());

  if (FOLLY_LIKELY(outputSize != 0)) {
    gatherCopy(
        output_.get(),
        outputRow,
        outputSize,
        spillSources_,
        spillSourceRows_,
        columnMap_);
  }

  numOutputRows_ += output_->size();
}

void SortBuffer::finishSpill() {
  VELOX_CHECK_NULL(spillMerger_);
  VELOX_CHECK(spillPartitionSet_.empty());
  VELOX_CHECK_EQ(
      !!(outputSpiller_ != nullptr) + !!(inputSpiller_ != nullptr),
      1,
      "inputSpiller_ {}, outputSpiller_ {}",
      inputSpiller_ == nullptr ? "set" : "null",
      outputSpiller_ == nullptr ? "set" : "null");
  if (inputSpiller_ != nullptr) {
    VELOX_CHECK(!inputSpiller_->finalized());
    inputSpiller_->finishSpill(spillPartitionSet_);
  } else {
    VELOX_CHECK(!outputSpiller_->finalized());
    outputSpiller_->finishSpill(spillPartitionSet_);
  }
  VELOX_CHECK_EQ(spillPartitionSet_.size(), 1);
}

void SortBuffer::prepareOutputWithSpill() {
  VELOX_CHECK(hasSpilled());
  if (spillMerger_ != nullptr) {
    VELOX_CHECK(spillPartitionSet_.empty());
    return;
  }

  VELOX_CHECK_EQ(spillPartitionSet_.size(), 1);
  spillMerger_ = spillPartitionSet_.begin()->second->createOrderedReader(
      *spillConfig_, pool(), spillStats_);
  spillPartitionSet_.clear();
}
} // namespace facebook::velox::exec
