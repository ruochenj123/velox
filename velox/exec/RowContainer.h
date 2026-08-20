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
#pragma once

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/common/memory/MemoryAllocator.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/ContainerRowSerde.h"
#include "velox/exec/Spill.h"
#include "velox/vector/FlatVector.h"
#include "velox/vector/VectorTypeUtils.h"

namespace facebook::velox::exec {
namespace test {
class RowContainerTestHelper;
}

class Aggregate;

class Accumulator {
 public:
  Accumulator(
      bool isFixedSize,
      int32_t fixedSize,
      bool usesExternalMemory,
      int32_t alignment,
      TypePtr spillType,
      std::function<void(folly::Range<char**> groups, VectorPtr& result)>
          spillExtractFunction,
      std::function<void(folly::Range<char**> groups)> destroyFunction);

  explicit Accumulator(Aggregate* aggregate, TypePtr spillType);

  bool isFixedSize() const;

  int32_t fixedWidthSize() const;

  bool usesExternalMemory() const;

  int32_t alignment() const;

  const TypePtr& spillType() const;

  void extractForSpill(folly::Range<char**> groups, VectorPtr& result) const;

  void destroy(folly::Range<char**> groups);

 private:
  const bool isFixedSize_;
  const int32_t fixedSize_;
  const bool usesExternalMemory_;
  const int32_t alignment_;
  const TypePtr spillType_;
  std::function<void(folly::Range<char**>, VectorPtr&)> spillExtractFunction_;
  std::function<void(folly::Range<char**> groups)> destroyFunction_;
};

using normalized_key_t = uint64_t;

struct RowContainerIterator {
  int32_t allocationIndex = 0;
  int32_t rowOffset = 0;
  /// Number of unvisited entries that are prefixed by an uint64_t for
  /// normalized key. Set in listRows() on first call.
  int64_t normalizedKeysLeft = 0;
  int normalizedKeySize = 0;

  /// Ordinal position of 'currentRow' in RowContainer.
  int32_t rowNumber{0};
  char* rowBegin{nullptr};
  /// First byte after the end of the range containing 'currentRow'.
  char* endOfRun{nullptr};
  /// Cursor of the list row operation.
  int32_t listRowCursor{0};

  /// Returns the current row, skipping a possible normalized key below the
  /// first byte of row.
  inline char* currentRow() const {
    return (rowBegin && normalizedKeysLeft) ? rowBegin + normalizedKeySize
                                            : rowBegin;
  }

  void reset() {
    *this = {};
  }

  std::string toString() const;
};

/// Container with a 8-bit partition number field for each row in a
/// RowContainer. The partition number bytes correspond 1:1 to rows. Used only
/// for parallel hash join build.
class RowPartitions {
 public:
  /// Initializes this to hold up to 'numRows'.
  RowPartitions(int32_t numRows, memory::MemoryPool& pool);

  /// Appends 'partitions' to the end of 'this'. Throws if adding more than the
  /// capacity given at construction.
  void appendPartitions(folly::Range<const uint8_t*> partitions);

  auto& allocation() const {
    return allocation_;
  }

  int32_t size() const {
    return size_;
  }

  void reset() {
    size_ = 0;
  }

 private:
  const int32_t capacity_;

  // Number of partition numbers added.
  int32_t size_{0};

  // Partition numbers. 1 byte each.
  memory::Allocation allocation_;
};

/// Packed representation of offset, null byte offset and null mask for
/// a column inside a RowContainer.
class RowColumn {
 public:
  /// Used as null offset for a non-null column.
  static constexpr int32_t kNotNullOffset = -1;

  RowColumn(int32_t offset, int32_t nullOffset)
      : packedOffsets_(PackOffsets(offset, nullOffset)) {}

  int32_t offset() const {
    return packedOffsets_ >> 32;
  }

  int32_t nullByte() const {
    return static_cast<uint32_t>(packedOffsets_) >> 8;
  }

  uint8_t nullMask() const {
    return packedOffsets_ & 0xff;
  }

  /// The null bits and the initialized bits for accumulators start at the
  /// beginning of the first byte following the null bits for the keys.  This
  /// guarantees that they always appear on the same byte for any given
  /// accumulator (since 2 evenly divides 8).
  int32_t initializedByte() const {
    return nullByte();
  }

  /// The initialized bit for an accumulator is guaranteed to appear on the same
  /// byte immediately following the null bit for that accumulator.
  int32_t initializedMask() const {
    return nullMask() << 1;
  }

  /// Aggregated stats of a column in the 'RowContainer'.
  class Stats {
   public:
    Stats() = default;

    void addCellSize(int32_t bytes) {
      if (UNLIKELY(nonNullCount_ == 0)) {
        minBytes_ = bytes;
        maxBytes_ = bytes;
      } else {
        minBytes_ = std::min(minBytes_, bytes);
        maxBytes_ = std::max(maxBytes_, bytes);
      }
      sumBytes_ += bytes;
      ++nonNullCount_;
    }

    void addNullCell() {
      ++nullCount_;
    }

    void removeOrUpdateCellStats(int32_t bytes, bool wasNull, bool setToNull);

    int32_t maxBytes() const {
      return maxBytes_;
    }

    int32_t minBytes() const {
      return minBytes_;
    }

    uint64_t sumBytes() const {
      return sumBytes_;
    }

    uint64_t avgBytes() const {
      if (nonNullCount_ == 0) {
        return 0;
      }
      return sumBytes_ / nonNullCount_;
    }

    uint32_t nonNullCount() const {
      return nonNullCount_;
    }

    uint32_t nullCount() const {
      return nullCount_;
    }

    uint32_t numCells() const {
      return nullCount_ + nonNullCount_;
    }

    void invalidateMinMaxColumnStats() {
      minMaxStatsValid_ = false;
    }

    bool minMaxColumnStatsValid() const {
      return minMaxStatsValid_;
    }

    /// Merges multiple aggregated stats of the same column into a single one.
    static Stats merge(const std::vector<Stats>& statsList);

   private:
    // Aggregated stats for non-null rows of the column.
    int32_t minBytes_{0};
    int32_t maxBytes_{0};
    bool minMaxStatsValid_{true};
    uint64_t sumBytes_{0};

    uint32_t nonNullCount_{0};
    uint32_t nullCount_{0};
  };

 private:
  static uint64_t PackOffsets(int32_t offset, int32_t nullOffset) {
    if (nullOffset == kNotNullOffset) {
      // If the column is not nullable, The low word is 0, meaning
      // that a null check will AND 0 to the 0th byte of the row,
      // which is always false and always safe to do.
      return static_cast<uint64_t>(offset) << 32;
    }
    return (1UL << (nullOffset & 7)) | ((nullOffset & ~7UL) << 5) |
        static_cast<uint64_t>(offset) << 32;
  }

  const uint64_t packedOffsets_;
};

/// Collection of rows for aggregation, hash join, order by.
class RowContainer {
 public:
  static constexpr uint64_t kUnlimited = std::numeric_limits<uint64_t>::max();
  /// The number of flags (bits) per accumulator, one for null and one for
  /// initialized.
  static constexpr size_t kNumAccumulatorFlags = 2;
  using Eraser = std::function<void(folly::Range<char**> rows)>;

  /// 'keyTypes' gives the type of row and use 'allocator' for bulk
  /// allocation.
  RowContainer(const std::vector<TypePtr>& keyTypes, memory::MemoryPool* pool)
      : RowContainer(keyTypes, std::vector<TypePtr>{}, pool) {}

  RowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes,
      memory::MemoryPool* pool)
      : RowContainer(
            keyTypes,
            dependentTypes,
            /*useListRowIndex=*/false,
            pool) {}

  /// If 'useListRowIndex' is true, the container maintains an internal array of
  /// row pointers so that listRowsFast() can return rows without scanning
  /// underlying allocations or checking free/probe flags. It is intended to be
  /// used in SortBuffer and SortInputSpiller to improve performance.
  RowContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& dependentTypes,
      bool useListRowIndex,
      memory::MemoryPool* pool)
      : RowContainer(
            keyTypes,
            true, // nullableKeys
            std::vector<Accumulator>{},
            dependentTypes,
            false, // hasNext
            false, // isJoinBuild
            false, // hasProbedFlag
            false, // hasCountFlag
            false, // hasNormalizedKey
            useListRowIndex,
            pool) {}

  ~RowContainer();

  static int32_t combineAlignments(int32_t a, int32_t b);

  /// 'keyTypes' gives the type of the key of each row. For a group by,
  /// order by or right outer join build side these may be
  /// nullable. 'nullableKeys' specifies if these have a null flag.
  /// 'aggregates' is a vector of Aggregate for a group by payload,
  /// empty otherwise. 'DependentTypes' gives the types of non-key
  /// columns for a hash join build side or an order by. 'hasNext' is
  /// true for a hash join build side where keys can be
  /// non-unique. 'isJoinBuild' is true for hash join build sides. This
  /// implies that hashing of keys ignores null keys even if these were
  /// allowed. 'hasProbedFlag' indicates that an extra bit is reserved
  /// for a probed state of a full or right outer
  /// join. 'hasNormalizedKey' specifies that an extra word is left
  /// below each row for a normalized key that collapses all parts
  /// into one word for faster comparison. The bulk allocation is done
  /// from 'allocator'. ContainerRowSerde is used for serializing complex
  /// type values into the container.
  RowContainer(
      const std::vector<TypePtr>& keyTypes,
      bool nullableKeys,
      const std::vector<Accumulator>& accumulators,
      const std::vector<TypePtr>& dependentTypes,
      bool hasNext,
      bool isJoinBuild,
      bool hasProbedFlag,
      bool hasCountFlag,
      bool hasNormalizedKey,
      bool useListRowIndex,
      memory::MemoryPool* pool);

  /// Allocates a new row and initializes possible aggregates to null.
  char* newRow();

  uint32_t rowSize(const char* row) const {
    return fixedRowSize_ +
        (rowSizeOffset_
             ? *reinterpret_cast<const uint32_t*>(row + rowSizeOffset_)
             : 0);
  }

  /// Sets all fields, aggregates, keys and dependents to null. Used when making
  /// a row with uninitialized keys for aggregates with no-op partial
  /// aggregation.
  void setAllNull(char* row);

  /// The row size excluding any out-of-line stored variable length values.
  int32_t fixedRowSize() const {
    return fixedRowSize_;
  }

  /// Adds 'rows' to the free rows list and frees any associated variable length
  /// data.
  void eraseRows(folly::Range<char**> rows);

  /// Copies elements of 'rows' where the char* points to a row inside 'this' to
  /// 'result' and returns the number copied. 'result' should have space for
  /// 'rows.size()'.
  int32_t findRows(folly::Range<char**> rows, char** result) const;

  void incrementRowSize(char* row, uint64_t bytes) {
    uint32_t* ptr = reinterpret_cast<uint32_t*>(row + rowSizeOffset_);
    uint64_t size = *ptr + bytes;
    *ptr = std::min<uint64_t>(size, std::numeric_limits<uint32_t>::max());
  }

  /// Initialize row. 'reuse' specifies whether the 'row' is reused or not. If
  /// it is reused, it will free memory associated with the row elsewhere (such
  /// as in HashStringAllocator).
  /// Note: Fields of the row are not zero-initialized. If the row contains
  /// variable-width fields, the caller must populate these fields by calling
  /// 'store' or initialize them to zero by calling 'initializeFields'.
  char* initializeRow(char* row, bool reuse);

  /// Zero out all the fields of the 'row'.
  void initializeFields(char* row) {
    ::memset(row, 0, fixedRowSize_);
  }

  // Store a single row id into the row at the reserved offset for hybrid design
  void storeSingleRowId(uint64_t& value, char* row) {
    *reinterpret_cast<int64_t*>(row + rowIdOffset_) = value;
  }

  // Get the stored row id from the row at the reserved offset for hybrid design
  uint64_t getSingleRowId(char* row) const {
    return *reinterpret_cast<int64_t*>(row + rowIdOffset_);
  }

  /// Stores the 'index'th value in 'decoded' into 'row' at 'columnIndex'.
  void store(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      char* row,
      int32_t columnIndex);

  /// Stores the first 'rows.size' values from the 'decoded' vector into the
  /// 'columnIndex' column of 'rows'.
  void store(
      const DecodedVector& decoded,
      folly::Range<char**> rows,
      int32_t columnIndex);

  HashStringAllocator& stringAllocator() {
    return *stringAllocator_;
  }

  /// Returns the number of used rows in 'this'. This is the number of rows a
  /// RowContainerIterator would access.
  int64_t numRows() const {
    return numRows_;
  }

  /// Copy key and dependent columns into a flat VARBINARY vector. All columns
  /// of a row are copied into a single buffer. The format of that buffer is an
  /// implementation detail. The data can be loaded back into the RowContainer
  /// using 'storeSerializedRow'.
  ///
  /// Used for spilling as it is more efficient than converting from row to
  /// columnar format.
  void extractSerializedRows(folly::Range<char**> rows, const VectorPtr& result)
      const;

  /// Copies serialized row produced by 'extractSerializedRow' into the
  /// container.
  void storeSerializedRow(
      const FlatVector<StringView>& vector,
      vector_size_t index,
      char* row);

  /// Copies the values at 'col' into 'result' (starting at 'resultOffset')
  /// for the 'numRows' rows pointed to by 'rows'. If a 'row' is null, sets
  /// corresponding row in 'result' to null.
  /// @param columnHasNulls indicates whether the 'col' column contains null
  /// values. If 'columnHasNulls' is false, a null-free optimization will be
  /// applied. It is the caller's responsibility to ensure this flag is set
  /// correctly.
  static void extractColumn(
      const char* const* rows,
      int32_t numRows,
      RowColumn col,
      bool columnHasNulls,
      vector_size_t resultOffset,
      const VectorPtr& result);

  /// Copies the values at 'col' into 'result' for the 'numRows' rows pointed to
  /// by 'rows'. If an entry in 'rows' is null, sets corresponding row in
  /// 'result' to null.
  /// @param columnHasNulls indicates whether the 'col' column contains null
  /// values. If 'columnHasNulls' is false, a null-free optimization will be
  /// applied. It is the caller's responsibility to ensure this flag is set
  /// correctly.
  static void extractColumn(
      const char* const* rows,
      int32_t numRows,
      RowColumn col,
      bool columnHasNulls,
      const VectorPtr& result) {
    extractColumn(rows, numRows, col, columnHasNulls, 0, result);
  }

  /// Copies the values from the array pointed to by 'rows' at 'col' into
  /// 'result' (starting at 'resultOffset') for the rows at positions in
  /// the 'rowNumbers' array. If a 'row' is null, sets corresponding row in
  /// 'result' to null. The positions in 'rowNumbers' array can repeat and also
  /// appear out of order. If rowNumbers has a negative value, then the
  /// corresponding row in 'result' is set to null.
  /// @param columnHasNulls indicates whether the 'col' column contains null
  /// values. If 'columnHasNulls' is false, a null-free optimization will be
  /// applied. It is the caller's responsibility to ensure this flag is set
  /// correctly.
  static void extractColumn(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      RowColumn col,
      bool columnHasNulls,
      vector_size_t resultOffset,
      const VectorPtr& result);

  /// Sets in result all locations with null values in col for rows (for numRows
  /// number of rows).
  static void extractNulls(
      const char* const* rows,
      int32_t numRows,
      RowColumn col,
      const BufferPtr& result);

  /// Copies the values at 'columnIndex' into 'result' for the 'numRows' rows
  /// pointed to by 'rows'. If an entry in 'rows' is null, sets corresponding
  /// row in 'result' to null.
  void extractColumn(
      const char* const* rows,
      int32_t numRows,
      int32_t columnIndex,
      const VectorPtr& result) const {
    extractColumn(
        rows,
        numRows,
        columnAt(columnIndex),
        columnHasNulls(columnIndex),
        result);
  }

  /// Copies the values at 'columnIndex' into 'result' (starting at
  /// 'resultOffset') for the 'numRows' rows pointed to by 'rows'. If an
  /// entry in 'rows' is null, sets corresponding row in 'result' to null.
  void extractColumn(
      const char* const* rows,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result) const {
    extractColumn(
        rows,
        numRows,
        columnAt(columnIndex),
        columnHasNulls(columnIndex),
        resultOffset,
        result);
  }

  /// Copies the values at 'columnIndex' at positions in the 'rowNumbers' array
  /// for the rows pointed to by 'rows'. The values are copied into the 'result'
  /// vector at the offset pointed by 'resultOffset'. If an entry in 'rows'
  /// is null, sets corresponding row in 'result' to null. The positions in
  /// 'rowNumbers' array can repeat and also appear out of order. If rowNumbers
  /// has a negative value, then the corresponding row in 'result' is set to
  /// null.
  void extractColumn(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t columnIndex,
      const vector_size_t resultOffset,
      const VectorPtr& result) const {
    extractColumn(
        rows,
        rowNumbers,
        columnAt(columnIndex),
        columnHasNulls(columnIndex),
        resultOffset,
        result);
  }

  /// Sets in result all locations with null values in columnIndex for rows.
  void extractNulls(
      const char* const* rows,
      int32_t numRows,
      int32_t columnIndex,
      const BufferPtr& result) const {
    extractNulls(rows, numRows, columnAt(columnIndex), result);
  }

  /// Copies the 'probed' flags for the specified rows into 'result'.
  /// The 'result' is expected to be flat vector of type boolean.
  /// For rows with null keys, sets null in 'result' if 'setNullForNullKeysRow'
  /// is true and false otherwise. For rows with 'false' probed flag, sets null
  /// in 'result' if 'setNullForNonProbedRow' is true and false otherwise. This
  /// is used for null aware and regular right semi project join types.
  void extractProbedFlags(
      const char* const* rows,
      int32_t numRows,
      bool setNullForNullKeysRow,
      bool setNullForNonProbedRow,
      const VectorPtr& result) const;

  static inline int32_t nullByte(int32_t nullOffset) {
    return nullOffset / 8;
  }

  static inline uint8_t nullMask(int32_t nullOffset) {
    return 1 << (nullOffset & 7);
  }

  /// Only accumulators have initialized flags. accumulatorFlagsOffset is the
  /// offset at which the flags for an accumulator begin. Currently this is the
  /// null flag, followed by the initialized flag. So it's equivalent to the
  /// nullOffset.
  ///
  /// It's guaranteed that the flags for an accumulator appear in the same byte.
  static inline int32_t initializedByte(int32_t accumulatorFlagsOffset) {
    return nullByte(accumulatorFlagsOffset);
  }

  /// accumulatorFlagsOffset is the offset at which the flags for an accumulator
  /// begin.
  static inline int32_t initializedMask(int32_t accumulatorFlagsOffset) {
    return nullMask(accumulatorFlagsOffset) << 1;
  }

  /// No tsan because probed flags may have been set by a different thread.
  /// There is a barrier but tsan does not know this.
  enum class ProbeType { kAll, kProbed, kNotProbed };

  template <ProbeType probeType>
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
  __attribute__((__no_sanitize__("thread")))
#endif
#endif
  int32_t listRows(
      RowContainerIterator* iter,
      int32_t maxRows,
      uint64_t maxBytes,
      char** rows) const {
    int32_t count = 0;
    uint64_t totalBytes = 0;
    auto numAllocations = rows_.numRanges();
    if (iter->allocationIndex == 0 && iter->rowOffset == 0) {
      iter->normalizedKeysLeft = numRowsWithNormalizedKey_;
      iter->normalizedKeySize = originalNormalizedKeySize_;
    }
    int32_t rowSize = fixedRowSize_ +
        (iter->normalizedKeysLeft > 0 ? originalNormalizedKeySize_ : 0);
    for (auto i = iter->allocationIndex; i < numAllocations; ++i) {
      auto range = rows_.rangeAt(i);
      auto* data =
          range.data() + memory::alignmentPadding(range.data(), alignment_);
      auto limit = range.size() -
          (reinterpret_cast<uintptr_t>(data) -
           reinterpret_cast<uintptr_t>(range.data()));
      auto row = iter->rowOffset;
      while (row + rowSize <= limit) {
        rows[count++] = data + row +
            (iter->normalizedKeysLeft > 0 ? originalNormalizedKeySize_ : 0);
        VELOX_DCHECK_EQ(
            reinterpret_cast<uintptr_t>(rows[count - 1]) % alignment_, 0);
        row += rowSize;
        auto newTotalBytes = totalBytes + rowSize;
        if (--iter->normalizedKeysLeft == 0) {
          rowSize -= originalNormalizedKeySize_;
        }
        if (bits::isBitSet(rows[count - 1], freeFlagOffset_)) {
          --count;
          continue;
        }
        if constexpr (probeType == ProbeType::kNotProbed) {
          if (bits::isBitSet(rows[count - 1], probedFlagOffset_)) {
            --count;
            continue;
          }
        }
        if constexpr (probeType == ProbeType::kProbed) {
          if (not(bits::isBitSet(rows[count - 1], probedFlagOffset_))) {
            --count;
            continue;
          }
        }
        totalBytes = newTotalBytes;
        if (rowSizeOffset_) {
          totalBytes += variableRowSize(rows[count - 1]);
        }
        if (count == maxRows || totalBytes > maxBytes) {
          iter->rowOffset = row;
          iter->allocationIndex = i;
          return count;
        }
      }
      iter->rowOffset = 0;
    }
    iter->allocationIndex = std::numeric_limits<int32_t>::max();
    return count;
  }

  /// Fast path for `listRows` that returns `rowPointers_` directly. Used by
  /// `SortBuffer` and `SortInputSpiller`, so it skips checking the free and
  /// probe flags.
  int32_t listRowsFast(RowContainerIterator* iter, int32_t maxRows, char** rows)
      const {
    int32_t count = 0;
    while (count < maxRows && iter->listRowCursor < rowPointers_.size()) {
      char* row = rowPointers_[iter->listRowCursor];
      rows[count++] = row;
      ++iter->listRowCursor;
    }
    return count;
  }

  /// Extracts up to 'maxRows' rows starting at the position of 'iter'. A
  /// default constructed or reset iter starts at the beginning. Returns the
  /// number of rows written to 'rows'. Returns 0 when at end. Stops after the
  /// total size of returned rows exceeds maxBytes.
  int32_t listRows(
      RowContainerIterator* iter,
      int32_t maxRows,
      uint64_t maxBytes,
      char** rows) const {
    return listRows<ProbeType::kAll>(iter, maxRows, maxBytes, rows);
  }

  int32_t listRows(RowContainerIterator* iter, int32_t maxRows, char** rows)
      const {
    if (useListRowIndex_) {
      return listRowsFast(iter, maxRows, rows);
    }
    return listRows<ProbeType::kAll>(iter, maxRows, kUnlimited, rows);
  }

  /// Sets 'probed' flag for the specified rows. Used by the right and
  /// full join to mark build-side rows that matches join
  /// condition. 'rows' may contain duplicate entries for the cases
  /// where single probe row matched multiple build rows. In case of
  /// the full join, 'rows' may include null entries that correspond
  /// to probe rows with no match. No tsan because any thread can set
  /// this without synchronization. There is a barrier between setting
  /// and reading.
#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
  __attribute__((__no_sanitize__("thread")))
#endif
#endif
  void setProbedFlag(char** rows, int32_t numRows);

  /// Compares the value at 'column' in 'row' with the value at 'index' in
  /// 'decoded'. Returns 0 for equal, < 0 for 'row' < 'decoded', > 0 otherwise.
  /// 'mayHaveNulls' specifies if nulls need to be checked. This is a fast path
  /// for compare().
  template <bool mayHaveNulls = true>
  int32_t compare(
      const char* row,
      RowColumn column,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags = CompareFlags()) const;

  /// Compares the value at 'columnIndex' between 'left' and 'right'. Returns
  /// 0 for equal, < 0 for left < right, > 0 otherwise.
  int32_t compare(
      const char* left,
      const char* right,
      int32_t columnIndex,
      CompareFlags flags = CompareFlags()) const;

  /// Compares the value between 'left' at 'leftIndex' and 'right' and
  /// 'rightIndex'. Returns 0 for equal, < 0 for left < right, > 0 otherwise.
  /// Both columns should have the same type.
  int32_t compare(
      const char* left,
      const char* right,
      int leftColumnIndex,
      int rightColumnIndex,
      CompareFlags flags = CompareFlags()) const;

  /// Allows get/set of the normalized key. If normalized keys are used, they
  /// are stored in the word immediately below the hash table row.
  static inline normalized_key_t& normalizedKey(char* group) {
    return reinterpret_cast<normalized_key_t*>(group)[-1];
  }

  void disableNormalizedKeys() {
    normalizedKeySize_ = 0;
  }

  RowColumn columnAt(int32_t index) const {
    return rowColumns_[index];
  }

  /// Returns the size of a string or complex types value stored in the
  /// specified row and column.
  int32_t variableSizeAt(const char* row, column_index_t column) const;

  /// Returns the per row size of a fixed size column.
  int32_t fixedSizeAt(column_index_t column) const;

  /// Bit offset of the probed flag for a full or right outer join  payload.
  /// 0 if not applicable.
  int32_t probedFlagOffset() const {
    return probedFlagOffset_;
  }

  /// Byte offset of the per-row count for counting joins. 0 if not applicable.
  int32_t countOffset() const {
    return countOffset_;
  }

  /// Returns the count stored at the given row. Used for counting joins.
  int32_t count(const char* row) const {
    VELOX_DCHECK_NE(countOffset_, 0);
    return countRef(const_cast<char*>(row));
  }

  /// Increments the count at the given row. Used during hash table build for
  /// counting joins.
  void incrementCount(char* row) const {
    VELOX_DCHECK_NE(countOffset_, 0);
    ++countRef(row);
  }

  /// Decrements the count at the given row. Used during hash table probe for
  /// counting joins.
  void decrementCount(char* row) const {
    VELOX_DCHECK_NE(countOffset_, 0);
    --countRef(row);
  }

  /// Adds 'n' to the count at the given row. Used during hash table merge
  /// to combine counts from multiple build-side tables.
  void addCount(char* row, int32_t n) const {
    VELOX_DCHECK_NE(countOffset_, 0);
    countRef(row) += n;
  }

  /// Returns the offset of a uint32_t row size or 0 if the row has no variable
  /// width fields or accumulators.
  int32_t rowSizeOffset() const {
    return rowSizeOffset_;
  }

  /// For a hash join table with possible non-unique entries, the offset of the
  /// pointer to the next row with the same key. 0 if keys are guaranteed
  /// unique, e.g. for a group by or semijoin build.
  int32_t nextOffset() const {
    return nextOffset_;
  }

  /// Creates a next-row-vector if it doesn't exist. Appends the row address to
  /// the next-row-vector, and store the address of the next-row-vector in the
  /// 'nextOffset_' slot for all duplicate rows.
  void appendNextRow(char* current, char* nextRow);

  /// Hashes the values of 'columnIndex' for 'rows'.  If 'mix' is true, mixes
  /// the hash with the existing value in 'result'.
  void hash(
      int32_t columnIndex,
      folly::Range<char**> rows,
      bool mix,
      uint64_t* result) const;

  uint64_t allocatedBytes() const {
    return rows_.allocatedBytes() + stringAllocator_->retainedSize();
  }

  /// Returns the number of fixed size rows that can be allocated without
  /// growing the container and the number of unused bytes of reserved storage
  /// for variable length data.
  std::pair<uint64_t, uint64_t> freeSpace() const {
    return std::make_pair<uint64_t, uint64_t>(
        rows_.freeBytes() / fixedRowSize_ + numFreeRows_,
        stringAllocator_->freeSpace());
  }

  /// Returns the average size of rows in bytes stored in this container.
  std::optional<int64_t> estimateRowSize() const;

  /// Returns a cap on extra memory that may be needed when adding 'numRows'
  /// and variableLengthBytes of out-of-line variable length data.
  int64_t sizeIncrement(vector_size_t numRows, int64_t variableLengthBytes)
      const;

  /// Resets the state to be as after construction. Frees memory for payload.
  void clear();

  int32_t compareRows(
      const char* left,
      const char* right,
      const std::vector<CompareFlags>& flags = {}) const {
    VELOX_DCHECK(flags.empty() || flags.size() == keyTypes_.size());
    for (auto i = 0; i < keyTypes_.size(); ++i) {
      auto result =
          compare(left, right, i, flags.empty() ? CompareFlags() : flags[i]);
      if (result) {
        return result;
      }
    }
    return 0;
  }

  const std::vector<char*, StlAllocator<char*>>& testingRowPointers() const {
    return rowPointers_;
  }

  memory::MemoryPool* pool() const {
    return stringAllocator_->pool();
  }

  /// Returns the types of all non-aggregate columns of 'this', keys first.
  const auto& columnTypes() const {
    return types_;
  }

  /// Returns the aggregated column stats of the column with given
  /// 'columnIndex'. nullopt will be returned if the column stats was previous
  /// invalidated. Any row erase operations will invalidate column stats.
  std::optional<RowColumn::Stats> columnStats(int32_t columnIndex) const;

  uint32_t columnNullCount(int32_t columnIndex) const {
    return rowColumnsStats_[columnIndex].nullCount();
  }

  const auto& keyTypes() const {
    return keyTypes_;
  }

  /// Returns true if specified column has nulls, false otherwise.
  inline bool columnHasNulls(int32_t columnIndex) const {
    return columnStats(columnIndex)->numCells() > 0 &&
        columnStats(columnIndex)->nullCount() > 0;
  }

  const std::vector<Accumulator>& accumulators() const {
    return accumulators_;
  }

  const HashStringAllocator& stringAllocator() const {
    return *stringAllocator_;
  }

  static inline bool
  isNullAt(const char* row, int32_t nullByte, uint8_t nullMask) {
    return (row[nullByte] & nullMask) != 0;
  }

  static inline bool isNullAt(const char* row, const RowColumn& rowColumn) {
    return (row[rowColumn.nullByte()] & rowColumn.nullMask()) != 0;
  }

  /// Returns true if the value at rowColumn in row is NaN.
  template <
      typename T,
      std::enable_if_t<std::is_floating_point_v<T>, int32_t> = 0>
  static inline bool isNanAt(const char* row, const RowColumn& rowColumn) {
    if (isNullAt(row, rowColumn.nullByte(), rowColumn.nullMask())) {
      return false;
    }
    return std::isnan(valueAt<T>(row, rowColumn.offset()));
  }

  /// Creates a container to store a partition number for each row in this row
  /// container. This is used by parallel join build which is responsible for
  /// filling this. This function also marks this row container as immutable
  /// after this call, we expect the user only call this once.
  std::unique_ptr<RowPartitions> createRowPartitions(memory::MemoryPool& pool);

  /// Retrieves rows from 'iterator' whose partition equals 'partition'. Writes
  /// up to 'maxRows' pointers to the rows in 'result'. 'rowPartitions' contains
  /// the partition number of each row in this container. The function returns
  /// the number of rows retrieved, 0 when no more rows are found. 'iterator' is
  /// expected to be in initial state on first call.
  int32_t listPartitionRows(
      RowContainerIterator& iterator,
      uint8_t partition,
      int32_t maxRows,
      const RowPartitions& rowPartitions,
      char** result) const;

  /// Advances 'iterator' by 'numRows'. The current row after skip is
  /// in iter.currentRow(). This is null if past end. Public for testing.
  void skip(RowContainerIterator& iterator, int32_t numRows) const;

  bool testingMutable() const {
    return mutable_;
  }

  /// Returns a summary of the container: key types, dependent types, number of
  /// accumulators and number of rows.
  std::string toString() const;

  /// Returns a string representation of the specified row in the same format as
  /// BaseVector::toString(index).
  std::string toString(const char* row) const;

 private:
  // Offset of the pointer to the next free row on a free row.
  static constexpr int32_t kNextFreeOffset = 0;

  template <typename T>
  static inline T valueAt(const char* group, int32_t offset) {
    return *reinterpret_cast<const T*>(group + offset);
  }

  template <typename T>
  static inline T& valueAt(char* group, int32_t offset) {
    return *reinterpret_cast<T*>(group + offset);
  }

  // Copies a string or complex type value from the specified row and column
  // into provided buffer. Stored the size of the data in the first 4 bytes of
  // the buffer. If the value is null, writes zero into the first 4 bytes of
  // destination and returns.
  // @return The number of bytes written to 'destination' including the 4 bytes
  // of the size.
  int32_t extractVariableSizeAt(
      const char* row,
      column_index_t column,
      char* output) const;

  // Copies a string or complex type value from 'data' into the specified row
  // and column. Expects first 4 bytes in 'data' to contain the size of the
  // string or complex value.
  // @return The number of bytes read from 'data': 4 bytes for size + that many
  // bytes.
  int32_t
  storeVariableSizeAt(const char* data, char* row, column_index_t column);

  template <TypeKind Kind>
  static void extractColumnTyped(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      bool columnHasNulls,
      int32_t resultOffset,
      const VectorPtr& result) {
    if (rowNumbers.size() > 0) {
      extractColumnTypedInternal<true, Kind>(
          rows,
          rowNumbers,
          rowNumbers.size(),
          column,
          columnHasNulls,
          resultOffset,
          result);
    } else {
      extractColumnTypedInternal<false, Kind>(
          rows,
          rowNumbers,
          numRows,
          column,
          columnHasNulls,
          resultOffset,
          result);
    }
  }

  template <bool useRowNumbers, TypeKind Kind>
  static void extractColumnTypedInternal(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      bool columnHasNulls,
      int32_t resultOffset,
      const VectorPtr& result) {
    // Resize the result vector before all copies.
    result->resize(numRows + resultOffset);

    if constexpr (
        Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
        Kind == TypeKind::MAP) {
      extractComplexType<useRowNumbers>(
          rows, rowNumbers, numRows, column, resultOffset, result);
      return;
    }
    using T = typename KindToFlatVector<Kind>::HashRowType;
    auto* flatResult = result->as<FlatVector<T>>();
    auto nullMask = column.nullMask();
    auto offset = column.offset();
    if (!nullMask || !columnHasNulls) {
      extractValuesNoNulls<useRowNumbers, T>(
          rows, rowNumbers, numRows, offset, resultOffset, flatResult);
    } else {
      extractValuesWithNulls<useRowNumbers, T>(
          rows,
          rowNumbers,
          numRows,
          offset,
          column.nullByte(),
          nullMask,
          resultOffset,
          flatResult);
    }
  }

  /// Removes or updates the column stats of a given row by updating each column
  /// stats.
  /// @param row - Points to the row to be removed or updated.
  /// @param setToNull - If true, the row stats is set to a null row,
  /// otherwise, the stats is erased from the columns stats.
  void removeOrUpdateRowColumnStats(const char* row, bool setToNull);

  char*& nextFree(char* row) const {
    return *reinterpret_cast<char**>(row + kNextFreeOffset);
  }

  uint32_t& variableRowSize(char* row) const {
    VELOX_DCHECK(rowSizeOffset_);
    return *reinterpret_cast<uint32_t*>(row + rowSizeOffset_);
  }

  template <TypeKind Kind>
  inline void storeWithNulls(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      bool isKey,
      char* row,
      int32_t offset,
      int32_t nullByte,
      uint8_t nullMask,
      int32_t columnIndex) {
    using T = typename TypeTraits<Kind>::NativeType;
    if (decoded.isNullAt(rowIndex)) {
      row[nullByte] |= nullMask;
      // Do not leave an uninitialized value in the case of a
      // null. This is an error with valgrind/asan.
      *reinterpret_cast<T*>(row + offset) = T();
      return;
    }
    if constexpr (std::is_same_v<T, StringView>) {
      RowSizeTracker tracker(row[rowSizeOffset_], *stringAllocator_);
      stringAllocator_->copyMultipart(
          decoded.valueAt<T>(rowIndex), row, offset);
    } else {
      *reinterpret_cast<T*>(row + offset) = decoded.valueAt<T>(rowIndex);
    }
  }

  template <TypeKind Kind>
  inline void storeNoNulls(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      bool isKey,
      char* row,
      int32_t offset) {
    using T = typename TypeTraits<Kind>::NativeType;
    if constexpr (std::is_same_v<T, StringView>) {
      RowSizeTracker tracker(row[rowSizeOffset_], *stringAllocator_);
      stringAllocator_->copyMultipart(
          decoded.valueAt<T>(rowIndex), row, offset);
    } else {
      *reinterpret_cast<T*>(row + offset) = decoded.valueAt<T>(rowIndex);
    }
  }

  template <TypeKind Kind>
  inline void storeWithNullsBatch(
      const DecodedVector& decoded,
      folly::Range<char**> rows,
      bool isKey,
      int32_t offset,
      int32_t nullByte,
      uint8_t nullMask,
      int32_t column) {
    for (int32_t i = 0; i < rows.size(); ++i) {
      storeWithNulls<Kind>(
          decoded, i, isKey, rows[i], offset, nullByte, nullMask, column);
      updateColumnStats(decoded, i, rows[i], column);
    }
  }

  template <TypeKind Kind>
  inline void storeNoNullsBatch(
      const DecodedVector& decoded,
      folly::Range<char**> rows,
      bool isKey,
      int32_t offset,
      int32_t column) {
    for (int32_t i = 0; i < rows.size(); ++i) {
      storeNoNulls<Kind>(decoded, i, isKey, rows[i], offset);
      updateColumnStats(decoded, i, rows[i], column);
    }
  }

  template <bool useRowNumbers, typename T>
  static void extractValuesWithNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t offset,
      int32_t nullByte,
      uint8_t nullMask,
      int32_t resultOffset,
      FlatVector<T>* result) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr& nullBuffer = result->mutableNulls(maxRows, true);
    auto nulls = nullBuffer->asMutable<uint64_t>();
    BufferPtr valuesBuffer = result->mutableValues();
    [[maybe_unused]] auto values = valuesBuffer->asMutableRange<T>();
    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }
      auto resultIndex = resultOffset + i;
      if (row == nullptr || isNullAt(row, nullByte, nullMask)) {
        bits::setNull(nulls, resultIndex, true);
      } else {
        bits::setNull(nulls, resultIndex, false);
        if constexpr (std::is_same_v<T, StringView>) {
          extractString(valueAt<StringView>(row, offset), result, resultIndex);
        } else {
          values[resultIndex] = valueAt<T>(row, offset);
        }
      }
    }
  }

  template <bool useRowNumbers, typename T>
  static void extractValuesNoNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t offset,
      int32_t resultOffset,
      FlatVector<T>* result) {
    [[maybe_unused]] auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());
    BufferPtr valuesBuffer = result->mutableValues();
    [[maybe_unused]] auto values = valuesBuffer->asMutableRange<T>();
    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }
      auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
      } else {
        result->setNull(resultIndex, false);
        if constexpr (std::is_same_v<T, StringView>) {
          extractString(valueAt<StringView>(row, offset), result, resultIndex);
        } else {
          values[resultIndex] = valueAt<T>(row, offset);
        }
      }
    }
  }

  static HashStringAllocator::InputStream prepareRead(
      const char* row,
      int32_t offset);

  template <bool typeProvidesCustomComparison, TypeKind Kind>
  void hashTyped(
      const Type* type,
      RowColumn column,
      bool nullable,
      folly::Range<char**> rows,
      bool mix,
      uint64_t* result) const;

  template <bool mayHaveNulls, TypeKind Kind>
  inline int compare(
      const char* row,
      RowColumn column,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags) const {
    if (decoded.base()->typeUsesCustomComparison()) {
      return compare<true, Kind, mayHaveNulls>(
          row, column, decoded, index, flags);
    } else {
      return compare<false, Kind, mayHaveNulls>(
          row, column, decoded, index, flags);
    }
  }

  template <
      bool typeProvidesCustomComparison,
      TypeKind Kind,
      bool mayHaveNulls,
      std::enable_if_t<
          Kind != TypeKind::OPAQUE && Kind != TypeKind::UNKNOWN,
          int32_t> = 0>
  inline int compare(
      const char* row,
      RowColumn column,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags) const {
    using T = typename KindToFlatVector<Kind>::HashRowType;

    if constexpr (mayHaveNulls) {
      bool rowIsNull = isNullAt(row, column.nullByte(), column.nullMask());
      bool indexIsNull = decoded.isNullAt(index);
      if (rowIsNull) {
        return indexIsNull ? 0 : flags.nullsFirst ? -1 : 1;
      }
      if (indexIsNull) {
        return flags.nullsFirst ? 1 : -1;
      }
    }

    if constexpr (
        Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
        Kind == TypeKind::MAP) {
      return compareComplexType(row, column.offset(), decoded, index, flags);
    } else if constexpr (is_string_kind(Kind)) {
      auto result = compareStringAsc(
          valueAt<StringView>(row, column.offset()), decoded, index);
      return flags.ascending ? result : result * -1;
    } else {
      auto left = valueAt<T>(row, column.offset());
      auto right = decoded.valueAt<T>(index);

      int result;
      if constexpr (typeProvidesCustomComparison) {
        result =
            SimpleVector<T>::template comparePrimitiveAscWithCustomComparison<
                Kind>(decoded.base()->type().get(), left, right);
      } else {
        result = SimpleVector<T>::comparePrimitiveAsc(left, right);
      }

      return flags.ascending ? result : result * -1;
    }
  }

  template <
      bool typeProvidesCustomComparison,
      TypeKind Kind,
      bool mayHaveNulls,
      std::enable_if_t<Kind == TypeKind::UNKNOWN, int32_t> = 0>
  inline int compare(
      const char* row,
      RowColumn column,
      const DecodedVector& /*decoded*/,
      vector_size_t /*index*/,
      CompareFlags flags) const {
    const bool rowIsNull = isNullAt(row, column.nullByte(), column.nullMask());
    return rowIsNull ? 0 : flags.nullsFirst ? 1 : -1;
  }

  template <
      bool typeProvidesCustomComparison,
      TypeKind Kind,
      bool mayHaveNulls,
      std::enable_if_t<Kind == TypeKind::OPAQUE, int32_t> = 0>
  inline int compare(
      const char* /*row*/,
      RowColumn /*column*/,
      const DecodedVector& /*decoded*/,
      vector_size_t /*index*/,
      CompareFlags /*flags*/) const {
    VELOX_UNSUPPORTED("Comparing Opaque types is not supported.");
  }

  template <
      bool typeProvidesCustomComparison,
      TypeKind Kind,
      std::enable_if_t<Kind != TypeKind::OPAQUE, int32_t> = 0>
  inline int compare(
      const char* left,
      const char* right,
      const Type* type,
      RowColumn leftColumn,
      RowColumn rightColumn,
      CompareFlags flags) const {
    using T = typename KindToFlatVector<Kind>::HashRowType;
    bool leftIsNull =
        isNullAt(left, leftColumn.nullByte(), leftColumn.nullMask());
    bool rightIsNull =
        isNullAt(right, rightColumn.nullByte(), rightColumn.nullMask());
    if (leftIsNull) {
      return rightIsNull ? 0 : flags.nullsFirst ? -1 : 1;
    }
    if (rightIsNull) {
      return flags.nullsFirst ? 1 : -1;
    }

    auto leftOffset = leftColumn.offset();
    auto rightOffset = rightColumn.offset();
    if constexpr (
        Kind == TypeKind::ROW || Kind == TypeKind::ARRAY ||
        Kind == TypeKind::MAP) {
      return compareComplexType(
          left, right, type, leftOffset, rightOffset, flags);
    } else if constexpr (is_string_kind(Kind)) {
      auto leftValue = valueAt<StringView>(left, leftOffset);
      auto rightValue = valueAt<StringView>(right, rightOffset);
      auto result = compareStringAsc(leftValue, rightValue);
      return flags.ascending ? result : result * -1;
    } else {
      auto leftValue = valueAt<T>(left, leftOffset);
      auto rightValue = valueAt<T>(right, rightOffset);

      int result;
      if constexpr (typeProvidesCustomComparison) {
        result =
            SimpleVector<T>::template comparePrimitiveAscWithCustomComparison<
                Kind>(type, leftValue, rightValue);
      } else {
        result = SimpleVector<T>::comparePrimitiveAsc(leftValue, rightValue);
      }

      return flags.ascending ? result : result * -1;
    }
  }

  template <
      bool typeProvidesCustomComparison,
      TypeKind Kind,
      std::enable_if_t<Kind == TypeKind::OPAQUE, int32_t> = 0>
  inline int compare(
      const char* /*left*/,
      const char* /*right*/,
      const Type* /*type*/,
      RowColumn /*leftColumn*/,
      RowColumn /*rightColumn*/,
      CompareFlags /*flags*/) const {
    VELOX_UNSUPPORTED("Comparing Opaque types is not supported.");
  }

  template <bool typeProvidesCustomComparison, TypeKind Kind>
  inline int compare(
      const char* left,
      const char* right,
      const Type* type,
      RowColumn column,
      CompareFlags flags) const {
    return compare<typeProvidesCustomComparison, Kind>(
        left, right, type, column, column, flags);
  }

  void storeComplexType(
      const DecodedVector& decoded,
      vector_size_t index,
      bool isKey,
      char* row,
      int32_t offset,
      int32_t nullByte = 0,
      uint8_t nullMask = 0,
      int32_t column = 0);

  template <bool useRowNumbers>
  static void extractComplexType(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      RowColumn column,
      int32_t resultOffset,
      const VectorPtr& result) {
    auto nullByte = column.nullByte();
    auto nullMask = column.nullMask();
    auto offset = column.offset();

    VELOX_DCHECK_LE(numRows + resultOffset, result->size());
    for (int i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }
      auto resultIndex = resultOffset + i;
      if (!row || isNullAt(row, nullByte, nullMask)) {
        result->setNull(resultIndex, true);
      } else {
        auto stream = prepareRead(row, offset);
        ContainerRowSerde::deserialize(stream, resultIndex, result.get());
      }
    }
  }

  static void extractString(
      StringView value,
      FlatVector<StringView>* values,
      vector_size_t index);

  static int32_t compareStringAsc(
      StringView left,
      const DecodedVector& decoded,
      vector_size_t index);

  static int32_t compareStringAsc(StringView left, StringView right);

  int32_t compareComplexType(
      const char* row,
      int32_t offset,
      const DecodedVector& decoded,
      vector_size_t index,
      CompareFlags flags = CompareFlags()) const;

  int32_t compareComplexType(
      const char* left,
      const char* right,
      const Type* type,
      int32_t offset,
      CompareFlags flags) const;

  int32_t compareComplexType(
      const char* left,
      const char* right,
      const Type* type,
      int32_t leftOffset,
      int32_t rightOffset,
      CompareFlags flags = CompareFlags()) const;

  // Free variable-width fields at column `column_index` associated with the
  // 'rows'. `FieldType` is the type of data representation of the fields in
  // row, and can be one of StringView(represents VARCHAR) and
  // std::string_view(represents ARRAY, MAP or ROW).
  template <typename FieldType>
  void freeVariableWidthFieldsAtColumn(
      size_t column_index,
      folly::Range<char**> rows) {
    static_assert(
        std::is_same_v<FieldType, StringView> ||
        std::is_same_v<FieldType, std::string_view>);

    const auto column = columnAt(column_index);
    for (auto row : rows) {
      if (isNullAt(row, column.nullByte(), column.nullMask())) {
        continue;
      }

      auto& view = valueAt<FieldType>(row, column.offset());
      if constexpr (std::is_same_v<FieldType, StringView>) {
        if (view.isInline()) {
          continue;
        }
      } else {
        if (view.empty()) {
          continue;
        }
      }
      stringAllocator_->free(HashStringAllocator::headerOf(view.data()));
    }
  }

  // Free any variable-width fields associated with the 'rows' and zero out
  // complex-typed field in 'rows'.
  void freeVariableWidthFields(folly::Range<char**> rows);

  // Free any aggregates associated with the 'rows'.
  void freeAggregates(folly::Range<char**> rows);

  void freeRowsExtraMemory(folly::Range<char**> rows);

  inline void updateColumnStats(
      const DecodedVector& decoded,
      vector_size_t rowIndex,
      char* row,
      int32_t columnIndex);

  // Updates column stats for serialized row.
  inline void updateColumnStats(char* row, int32_t columnIndex);

  // Min/max column stats do not support row erasures. This
  // method is called whenever a row is erased.
  void invalidateMinMaxColumnStats() {
    for (auto columnStats : rowColumnsStats_) {
      columnStats.invalidateMinMaxColumnStats();
    }
  }

  int32_t& countRef(char* row) const {
    return *reinterpret_cast<int32_t*>(row + countOffset_);
  }

  const std::vector<TypePtr> keyTypes_;
  const bool nullableKeys_;
  const bool isJoinBuild_;
  // True if normalized keys are enabled in initial state.
  const bool hasNormalizedKeys_;
  // True if use 'listRowsFast'.
  const bool useListRowIndex_;
  const std::unique_ptr<HashStringAllocator> stringAllocator_;

  // Indicates if we can add new row to this row container. It is set to false
  // after user calls 'getRowPartitions()' to create 'rowPartitions' object for
  // parallel join build.
  bool mutable_{true};

  std::vector<Accumulator> accumulators_;

  bool usesExternalMemory_ = false;
  // Types of non-aggregate columns. Keys first. Corresponds pairwise
  // to 'typeKinds_' and 'rowColumns_'.
  std::vector<TypePtr> types_;
  std::vector<TypeKind> typeKinds_;
  int32_t nextOffset_ = 0;
  // Indicates if this row container has rows with duplicate keys. This only
  // applies if 'nextOffset_' is set.
  tsan_atomic<bool> hasDuplicateRows_{false};
  // Bit position of null bit in the row. 0 if no null flag. Order is keys,
  // accumulators, dependent.
  std::vector<int32_t> nullOffsets_;
  // Position of field or accumulator. Corresponds 1:1 to 'nullOffset_'.
  std::vector<int32_t> offsets_;
  // Position of row ID field, used in hybrid design.
  int32_t rowIdOffset_{0};
  // Offset and null indicator offset of non-aggregate fields as a single word.
  // Corresponds pairwise to 'types_'.
  std::vector<RowColumn> rowColumns_;
  // Aggregated column stats(e.g. min/max size) for non-aggregate
  // fields. Index aligns with 'rowColumns_'.
  std::vector<RowColumn::Stats> rowColumnsStats_;
  // Bit offset of the probed flag for a full or right outer join  payload. 0 if
  // not applicable.
  int32_t probedFlagOffset_ = 0;

  // Byte offset of the per-row count for counting joins. 0 if not applicable.
  int32_t countOffset_ = 0;

  // Bit position of free bit.
  int32_t freeFlagOffset_ = 0;
  int32_t rowSizeOffset_ = 0;

  int32_t fixedRowSize_;
  // How many bytes do the flags (null, probed, free) occupy.
  int32_t flagBytes_;
  // The count of entries that have an extra normalized_key_t before the
  // start.
  int64_t numRowsWithNormalizedKey_ = 0;
  // This is the original normalized key size regardless of whether
  // disableNormalizedKeys() is called or not.
  int originalNormalizedKeySize_;
  // Extra bytes to reserve before  each added row for a normalized key. Set to
  // 0 after deciding not to use normalized keys.
  int normalizedKeySize_;
  uint64_t numRows_ = 0;
  // Head of linked list of free rows.
  char* firstFreeRow_ = nullptr;
  uint64_t numFreeRows_ = 0;

  memory::AllocationPool rows_;
  std::vector<char*, StlAllocator<char*>> rowPointers_;

  int alignment_ = 1;

  friend class test::RowContainerTestHelper;
  friend class HybridContainer;
};

template <>
inline int128_t RowContainer::valueAt<int128_t>(
    const char* group,
    int32_t offset) {
  return HugeInt::deserialize(group + offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::ROW>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    int32_t columnIndex) {
  storeComplexType(
      decoded, rowIndex, isKey, row, offset, nullByte, nullMask, columnIndex);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::ROW>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  storeComplexType(decoded, rowIndex, isKey, row, offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::ARRAY>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    int32_t columnIndex) {
  storeComplexType(
      decoded, rowIndex, isKey, row, offset, nullByte, nullMask, columnIndex);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::ARRAY>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  storeComplexType(decoded, rowIndex, isKey, row, offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::MAP>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    int32_t columnIndex) {
  storeComplexType(
      decoded, rowIndex, isKey, row, offset, nullByte, nullMask, columnIndex);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::MAP>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool isKey,
    char* row,
    int32_t offset) {
  storeComplexType(decoded, rowIndex, isKey, row, offset);
}

template <>
inline void RowContainer::storeWithNulls<TypeKind::HUGEINT>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool /*isKey*/,
    char* row,
    int32_t offset,
    int32_t nullByte,
    uint8_t nullMask,
    int32_t columnIndex) {
  if (decoded.isNullAt(rowIndex)) {
    row[nullByte] |= nullMask;
    memset(row + offset, 0, sizeof(int128_t));
    return;
  }
  HugeInt::serialize(decoded.valueAt<int128_t>(rowIndex), row + offset);
}

template <>
inline void RowContainer::storeNoNulls<TypeKind::HUGEINT>(
    const DecodedVector& decoded,
    vector_size_t rowIndex,
    bool /*isKey*/,
    char* row,
    int32_t offset) {
  HugeInt::serialize(decoded.valueAt<int128_t>(rowIndex), row + offset);
}

template <>
inline void RowContainer::extractColumnTyped<TypeKind::OPAQUE>(
    const char* const* /*rows*/,
    folly::Range<const vector_size_t*> /*rowNumbers*/,
    int32_t /*numRows*/,
    RowColumn /*column*/,
    bool /*columnHasNulls*/,
    int32_t /*resultOffset*/,
    const VectorPtr& /*result*/) {
  VELOX_UNSUPPORTED("RowContainer doesn't support values of type OPAQUE");
}

inline void RowContainer::extractColumn(
    const char* const* rows,
    int32_t numRows,
    RowColumn column,
    bool columnHasNulls,
    int32_t resultOffset,
    const VectorPtr& result) {
  VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractColumnTyped,
      result->typeKind(),
      rows,
      {},
      numRows,
      column,
      columnHasNulls,
      resultOffset,
      result);
}

inline void RowContainer::extractColumn(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    RowColumn column,
    bool columnHasNulls,
    int32_t resultOffset,
    const VectorPtr& result) {
  VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractColumnTyped,
      result->typeKind(),
      rows,
      rowNumbers,
      rowNumbers.size(),
      column,
      columnHasNulls,
      resultOffset,
      result);
}

inline void RowContainer::extractNulls(
    const char* const* rows,
    int32_t numRows,
    RowColumn column,
    const BufferPtr& result) {
  VELOX_DCHECK(result->size() >= bits::nbytes(numRows));
  auto* rawResult = result->asMutable<uint64_t>();
  bits::fillBits(rawResult, 0, numRows, false);

  auto nullMask = column.nullMask();
  if (!nullMask) {
    return;
  }

  auto nullByte = column.nullByte();
  for (int32_t i = 0; i < numRows; ++i) {
    const char* row = rows[i];
    if (row == nullptr || isNullAt(row, nullByte, nullMask)) {
      bits::setBit(rawResult, i, true);
    }
  }
}

template <bool mayHaveNulls>
inline int RowContainer::compare(
    const char* row,
    RowColumn column,
    const DecodedVector& decoded,
    vector_size_t index,
    CompareFlags flags) const {
  return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
      compare,
      mayHaveNulls,
      decoded.base()->typeKind(),
      row,
      column,
      decoded,
      index,
      flags);
}

inline int RowContainer::compare(
    const char* left,
    const char* right,
    int columnIndex,
    CompareFlags flags) const {
  auto type = types_[columnIndex].get();
  if (type->providesCustomComparison()) {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        true,
        type->kind(),
        left,
        right,
        type,
        columnAt(columnIndex),
        flags);
  } else {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        false,
        type->kind(),
        left,
        right,
        type,
        columnAt(columnIndex),
        flags);
  }
}

inline int RowContainer::compare(
    const char* left,
    const char* right,
    int leftColumnIndex,
    int rightColumnIndex,
    CompareFlags flags) const {
  auto leftType = types_[leftColumnIndex].get();
  auto rightType = types_[rightColumnIndex].get();
  VELOX_CHECK(leftType->equivalent(*rightType));

  if (leftType->providesCustomComparison()) {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        true,
        leftType->kind(),
        left,
        right,
        leftType,
        columnAt(leftColumnIndex),
        columnAt(rightColumnIndex),
        flags);
  } else {
    return VELOX_DYNAMIC_TEMPLATE_TYPE_DISPATCH_ALL(
        compare,
        false,
        leftType->kind(),
        left,
        right,
        leftType,
        columnAt(leftColumnIndex),
        columnAt(rightColumnIndex),
        flags);
  }
}

/// A comparator of rows stored in the RowContainer compatible with
/// std::priority_queue. Uses specified columns and sorting orders for
/// comparison.
class RowComparator {
 public:
  RowComparator(
      const RowTypePtr& rowType,
      const std::vector<core::FieldAccessTypedExprPtr>& sortingKeys,
      const std::vector<core::SortOrder>& sortingOrders,
      RowContainer* rowContainer);

  /// Returns true if lhs < rhs, false otherwise.
  bool operator()(const char* lhs, const char* rhs);

  /// Returns 0 for equal, < 0 for lhs < rhs, > 0 otherwise.
  int compare(const char* lhs, const char* rhs);

  /// Returns true if decodedVectors[index] < other, false otherwise.
  bool operator()(
      const std::vector<DecodedVector>& decodedVectors,
      vector_size_t index,
      const char* other);

  /// Returns 0 for equal, < 0 for decodedVectors[index] < other,
  /// > 0 otherwise.
  int32_t compare(
      const std::vector<DecodedVector>& decodedVectors,
      vector_size_t index,
      const char* other);

 private:
  std::vector<std::pair<column_index_t, core::SortOrder>> keyInfo_;
  RowContainer* rowContainer_;
};

/// Hybrid container

/// Row identifier for hybrid join mode.
/// In coalesced mode: rowId_ is a global row index (0 to totalRows-1).
/// In scattered mode: rowId_ encodes (batchId << 32 | rowInBatch).
struct HybridRowId {
  uint8_t containerId_;
  uint64_t rowId_;

  // Constants for scattered mode encoding
  static constexpr int kBatchIdBits = 24;
  static constexpr int kRowInBatchBits = 32;
  static constexpr uint64_t kRowInBatchMask = (1ULL << kRowInBatchBits) - 1;
  static constexpr uint64_t kBatchIdMask = ((1ULL << kBatchIdBits) - 1)
      << kRowInBatchBits;

  // Encode batchId and rowInBatch for scattered mode
  static uint64_t encodeScattered(uint32_t batchId, uint32_t rowInBatch) {
    return (static_cast<uint64_t>(batchId) << kRowInBatchBits) | rowInBatch;
  }

  // Decode batchId from rowId_ (scattered mode)
  uint32_t batchId() const {
    return static_cast<uint32_t>((rowId_ & kBatchIdMask) >> kRowInBatchBits);
  }

  // Decode rowInBatch from rowId_ (scattered mode)
  uint32_t rowInBatch() const {
    return static_cast<uint32_t>(rowId_ & kRowInBatchMask);
  }

  // For coalesced mode, rowId_ is the global row index
  uint64_t globalRowId() const {
    return rowId_;
  }
};

// Forward declarations for late materialization

// Phase-2 defensive helper: late-m payload storage can retain dictionary/lazy
// encoded children (peer-driver containers that were never coalesced, probe
// outputs stored as-is). Extraction paths require FLAT. Flatten in place on
// demand; no-op when already flat. TODO(reorg): flatten at STORE time instead.
inline void ensureFlatChild(RowVector* batch, int32_t columnIndex) {
  auto& child = batch->childAt(columnIndex);
  if (child->encoding() != VectorEncoding::Simple::FLAT) {
    BaseVector::flattenVector(child);
  }
}

class HybridContainer;

// Releaser that keeps a coalesced payload string buffer alive for the
// lifetime of a BufferView acquired into an extraction result vector. The
// view's reported size is the number of body bytes the result batch actually
// references, so byte-based flow control (e.g. LocalExchangeQueue weights
// computed from BaseVector::retainedSize()) sees the amortized per-batch
// footprint instead of the full shared buffer capacity.
struct HybridPayloadBufferReleaser {
  BufferPtr buffer;
  void addRef() const {}
  void release() const {}
};

class HybridContainer {
 public:
  HybridContainer(
      const std::vector<TypePtr>& keyTypes,
      const std::vector<TypePtr>& payloadTypes,
      RowContainer* rows);
  ~HybridContainer();

  std::optional<int64_t> estimateRowSize() const;
  int32_t fixedSizeAt(column_index_t column) const;
  int32_t estimateVariableSizeAt(const char* row, column_index_t column) const;
  void addPayload(RowVectorPtr input);
  void clear();
  std::vector<TypePtr> columnTypes() const;

  void extractNulls(
      const char* const* rows,
      int32_t numRows,
      int32_t columnIndex,
      const BufferPtr& result,
      std::vector<HybridRowId>& outputRowIds);

  void extractPayload(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds,
      bool exactSize);
  // The function to get the stored RowIds from RowContainer to materialize
  // payload columns Separating it from materialization logic because in many
  // cases the same set of RowIds will be used to materialize multiple columns.
  void getRowIds(
      const char* const* rows,
      int32_t numRows,
      std::vector<HybridRowId>& outputRowIds) {
    getRowIdsInternal<false>(rows, {}, numRows, outputRowIds);
  }
  void getRowIds(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      std::vector<HybridRowId>& outputRowIds) {
    getRowIdsInternal<true>(rows, rowNumbers, rowNumbers.size(), outputRowIds);
  }
  void extractColumn(
      const char* const* rows,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds,
      bool exactSize = false) {
    // keys
    if (isKey(columnIndex)) {
      keys_->extractColumn(rows, numRows, columnIndex, resultOffset, result);
    } else {
      // payloads
      // getRowIds should be called out of extracting projection columns
      VELOX_CHECK_EQ(
          numRows,
          outputRowIds.size(),
          "Number of rowIds is not equal to number of rows.");
      VELOX_CHECK_GT(payloadTypes_.size(), 0, "No payload columns stored.");
      extractPayload(
          rows,
          {},
          numRows,
          columnIndex - numKeys_,
          resultOffset,
          result,
          outputRowIds,
          exactSize);
    }
  };

  void extractColumn(
      const char* const* rows,
      int32_t numRows,
      int32_t columnIndex,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds,
      bool exactSize = false) {
    extractColumn(
        rows, numRows, columnIndex, 0, result, outputRowIds, exactSize);
  };

  void extractColumn(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds,
      bool exactSize = false) {
    // keys
    if (isKey(columnIndex)) {
      // exactSize was deprecated
      keys_->extractColumn(rows, rowNumbers, columnIndex, resultOffset, result);
    } else {
      // payloads
      // getRowIds should be called out of extracting projection columns
      VELOX_CHECK_EQ(
          rowNumbers.size(),
          outputRowIds.size(),
          "Number of rowIds is not equal to number of rows.");
      VELOX_CHECK_GT(payloadTypes_.size(), 0, "No payload columns stored.");
      extractPayload(
          rows,
          rowNumbers,
          rowNumbers.size(),
          columnIndex - numKeys_,
          resultOffset,
          result,
          outputRowIds,
          exactSize);
    }
  }
  bool isKey(int32_t columnIndex) const {
    return columnIndex < numKeys_;
  }

  // Returns true if payload column 'payloadColumn' may contain nulls in ANY
  // registered container. 'isNullable_' is per-container state, updated only
  // by this container's own addPayload() calls; after a multi-driver join
  // build merge, extraction runs through one container but pulls rows from
  // all registered containers. Dispatching the null/no-null extraction fast
  // path on the local flag alone silently drops null flags for rows owned by
  // sibling containers (a null VARCHAR extracts as "", making NULL keys
  // compare equal downstream). Multi-container paths must use this instead.
  bool anyContainerNullable(int32_t payloadColumn) const {
    if (isNullable_[payloadColumn]) {
      return true;
    }
    for (const auto& entry : allContainers_) {
      if (entry.second->isNullable_[payloadColumn]) {
        return true;
      }
    }
    return false;
  }

  int32_t numKeys() const {
    return numKeys_;
  }

  RowContainer* getKeys() const {
    return keys_;
  }

  uint32_t getNumRows() const {
    return totalRows_;
  }

  uint32_t getNumBatches() const {
    return totalBatches_;
  }

  void setId(uint8_t id) {
    id_ = id;
  }

  uint8_t getId() {
    return id_;
  }

  void setAllContainers(
      std::unordered_map<uint8_t, HybridContainer*>& hybridDataChannel) {
    allContainers_ = hybridDataChannel;
    maxContainerId_ = 0;
    for (const auto& [cid, _] : allContainers_) {
      maxContainerId_ = std::max<uint8_t>(maxContainerId_, cid);
    }
  }

  uint8_t getNumContainers() const {
    return allContainers_.size();
  }

  // Fast path check for single container - avoids sorting overhead.
  // In single container case, all rows come from the same container.
  bool isSingleContainer() const {
    return allContainers_.size() == 1;
  }

  /// Get all containers map (keyed by containerId/driverId).
  /// Used for looking up the correct container when extracting upstream refs.
  const std::unordered_map<uint8_t, HybridContainer*>& getAllContainers()
      const {
    return allContainers_;
  }

  // Controls whether to reorder rows by containerId during extraction.
  // Can be disabled for testing to get deterministic output order.
  void setReorderEnabled(bool enabled) {
    reorderEnabled_ = enabled;
  }

  bool isReorderEnabled() const {
    return reorderEnabled_;
  }

  // Controls whether scattered (non-coalesced) mode is used for payloads.
  // In scattered mode, batches are kept separate and row IDs encode (batchId,
  // rowInBatch).
  void setScatteredModeEnabled(bool enabled) {
    scatteredModeEnabled_ = enabled;
  }

  bool isScatteredModeEnabled() const {
    return scatteredModeEnabled_;
  }

  // Returns whether sorting should be used for extraction.
  // Sorting is used when: reorder is enabled AND there are multiple
  // containers AND payload is coalesced. In scattered mode locality is
  // per batch, not per container, so container-major reorder costs an
  // O(n) permutation per output batch and buys nothing.
  bool shouldUseSorting() const {
    return reorderEnabled_ && !isSingleContainer() && !scatteredModeEnabled_;
  }

  /// Get hybridRowId from a row pointer
  uint64_t getHybridRowId(char* row) const {
    return keys_->valueAt<uint64_t>(row, rowIdColumnOffset_);
  }

  /// Check if payload batches have been coalesced
  bool isCoalesced() const {
    return owningInputs_.size() <= 1;
  }

  /// Get the coalesced batch (for payload extraction)
  RowVectorPtr getCoalescedBatch() const {
    VELOX_CHECK(isCoalesced(), "Must call coalesceBatches() first");
    if (owningInputs_.empty()) {
      return nullptr;
    }
    return owningInputs_[0];
  }

  /// Get payload types
  const std::vector<TypePtr>& payloadTypes() const {
    return payloadTypes_;
  }

  // Reorder rows and rowIds by containerId to improve locality for extraction.
  // Returns reordered rows, rowIds, and optionally rowNumbers when provided.
  struct SortedRows {
    std::vector<const char*> rows;
    std::vector<vector_size_t> rowNumbers;
    std::vector<HybridRowId> rowIds;
  };

  SortedRows sortByContainerId(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      const std::vector<HybridRowId>& outputRowIds) {
    const int size = outputRowIds.size();
    SortedRows out;
    out.rows.resize(size);
    out.rowIds.resize(size);
    if (!rowNumbers.empty()) {
      out.rowNumbers.resize(size);
    }

    // Build permutation by counting sort on containerId.
    std::vector<int32_t> count(maxContainerId_ + 1, 0);
    for (const auto& item : outputRowIds) {
      ++count[item.containerId_];
    }
    for (int i = 1; i <= maxContainerId_; ++i) {
      count[i] += count[i - 1];
    }
    std::vector<int32_t> perm(size);
    for (int i = size - 1; i >= 0; --i) {
      auto cid = outputRowIds[i].containerId_;
      auto pos = --count[cid];
      perm[pos] = i;
    }

    // Apply permutation.
    for (int outIdx = 0; outIdx < size; ++outIdx) {
      const int srcIdx = perm[outIdx];
      out.rowIds[outIdx] = outputRowIds[srcIdx];
      if (!out.rowNumbers.empty()) {
        out.rowNumbers[outIdx] = rowNumbers[srcIdx];
      }
      out.rows[outIdx] = rows[srcIdx];
    }

    return out;
  }

  // Coalesce all payload batches into a single batch to improve locality.
  void coalesceBatches() {
    // Skip if no payload columns defined.
    if (payloadTypes_.empty()) {
      return;
    }

    auto* pool = keys_->pool();
    const auto numPayloadCols = payloadTypes_.size();

    // Handle empty container: create an empty batch to maintain single-batch
    // invariant. This ensures getSingleContainerData() works even when no data
    // was added.
    if (owningInputs_.empty()) {
      std::vector<VectorPtr> emptyChildren;
      emptyChildren.reserve(numPayloadCols);
      std::vector<std::string> payloadNames;
      payloadNames.reserve(numPayloadCols);
      for (int32_t col = 0; col < numPayloadCols; ++col) {
        payloadNames.push_back(fmt::format("c{}", col));
        emptyChildren.push_back(BaseVector::create(payloadTypes_[col], 0, pool));
      }
      owningInputs_.push_back(std::make_shared<RowVector>(
          pool,
          ROW(std::move(payloadNames), std::vector<TypePtr>(payloadTypes_)),
          BufferPtr(nullptr),
          0,
          std::move(emptyChildren)));
      totalBatches_ = 1;
      return;
    }

    const auto totalRows = totalRows_;
    const auto numBatches = owningInputs_.size();

    std::vector<VectorPtr> newChildren;
    newChildren.reserve(numPayloadCols);
    std::vector<std::string> payloadNames;
    payloadNames.reserve(numPayloadCols);
    for (int32_t col = 0; col < numPayloadCols; ++col) {
      payloadNames.push_back(fmt::format("c{}", col));
    }
    // Merge column-at-a-time and release each source batch's column as soon
    // as it is copied: merged columns accumulate at the same rate source
    // columns are freed, so the transient peak stays near 1x payload plus
    // one in-flight column per worker. (Pre-allocating the whole merged
    // table before copying — the original scheme — holds sources and copies
    // simultaneously, ~2x payload transient; measured as the dominant
    // hybrid-sort finish cost.)
    // Columns are independent, and the owning operator thread is the only
    // one touching this container, so the per-column merge fans out over a
    // small thread pool: each worker claims a column, builds its merged
    // vector, and resets only childAt(col) of each batch — disjoint slots,
    // no shared mutable state. Full-batch release happens after the join.
    // Merge column-at-a-time and release each source batch's column slot as
    // soon as it is copied: merged columns accumulate at the same rate
    // source columns are freed, so the transient peak stays near 1x payload
    // plus the single column being built. (Pre-allocating the whole merged
    // table before copying holds sources and copies simultaneously — ~2x
    // payload transient, measured as the dominant hybrid-sort finish cost.
    // Parallelizing the per-column merges was evaluated and reverted: the
    // copy work moved off the operator thread but end-to-end wall time did
    // not improve, while concurrent in-flight columns pushed the transient
    // peak back toward 2x unless byte-capped — complexity without benefit.)
    for (int32_t col = 0; col < numPayloadCols; ++col) {
      // VARCHAR/VARBINARY: compact out-of-line string bodies into a single
      // buffer instead of merging via BaseVector::copy. copy()'s string path
      // shares the source string buffers, which would leave the merged child
      // holding one BufferPtr per input batch (thousands); the extraction
      // kernels then acquire those shared buffers into every output batch,
      // and O(batches) BufferPtr transfers per extraction call dominate.
      // With one compacted buffer, extraction acquires exactly one buffer
      // per call and copies only 16-byte views. The body copy here is a
      // one-time sequential pass over build rows (cheaper than the per
      // output-row random-offset copies it replaces).
      const auto kind = payloadTypes_[col]->kind();
      if (kind == TypeKind::VARCHAR || kind == TypeKind::VARBINARY) {
        // Pass 1: flatten sources and size the compacted body buffer.
        uint64_t bodyBytes = 0;
        for (auto& batch : owningInputs_) {
          ensureFlatChild(batch.get(), col);
          auto* src =
              batch->childAt(col)->asUnchecked<FlatVector<StringView>>();
          const auto* views = src->rawValues();
          const auto* nulls = src->rawNulls();
          const auto batchSize = batch->size();
          for (vector_size_t i = 0; i < batchSize; ++i) {
            if ((nulls == nullptr || !bits::isBitNull(nulls, i)) &&
                !views[i].isInline()) {
              bodyBytes += views[i].size();
            }
          }
        }
        auto merged = BaseVector::create(payloadTypes_[col], totalRows, pool);
        auto* mergedFlat = merged->asUnchecked<FlatVector<StringView>>();
        BufferPtr bodyBuffer;
        char* bodyData = nullptr;
        if (bodyBytes > 0) {
          bodyBuffer = AlignedBuffer::allocate<char>(bodyBytes, pool);
          bodyBuffer->setSize(bodyBytes);
          bodyData = bodyBuffer->asMutable<char>();
          mergedFlat->setStringBuffers({bodyBuffer});
        }
        // Pass 2: copy views (and bodies for out-of-line strings), then
        // release each source batch's column slot.
        auto* outViews = mergedFlat->mutableRawValues();
        uint64_t bodyOffset = 0;
        vector_size_t offset = 0;
        for (auto& batch : owningInputs_) {
          auto* src =
              batch->childAt(col)->asUnchecked<FlatVector<StringView>>();
          const auto* views = src->rawValues();
          const auto* nulls = src->rawNulls();
          const auto batchSize = batch->size();
          for (vector_size_t i = 0; i < batchSize; ++i) {
            const auto out = offset + i;
            if (nulls != nullptr && bits::isBitNull(nulls, i)) {
              mergedFlat->setNull(out, true);
              continue;
            }
            const auto& v = views[i];
            if (v.isInline()) {
              outViews[out] = v;
            } else {
              memcpy(bodyData + bodyOffset, v.data(), v.size());
              outViews[out] = StringView(bodyData + bodyOffset, v.size());
              bodyOffset += v.size();
            }
          }
          batch->childAt(col).reset();
          offset += batchSize;
        }
        newChildren.push_back(std::move(merged));
        continue;
      }
      newChildren.push_back(
          BaseVector::create(payloadTypes_[col], totalRows, pool));
      auto& merged = newChildren.back();
      vector_size_t offset = 0;
      for (auto& batch : owningInputs_) {
        const auto batchSize = batch->size();
        merged->copy(batch->childAt(col).get(), offset, 0, batchSize);
        batch->childAt(col).reset();
        offset += batchSize;
      }
    }
    for (auto& batch : owningInputs_) {
      batch.reset();
    }

    // Rebuild owningInputs_ with a single RowVector.
    owningInputs_.clear();
    owningInputs_.push_back(std::make_shared<RowVector>(
        pool,
        ROW(std::move(payloadNames), std::vector<TypePtr>(payloadTypes_)),
        BufferPtr(nullptr),
        totalRows,
        std::move(newChildren)));

    totalBatches_ = 1;
  }

 private:
  // Get the single container's coalesced data (only valid when
  // isSingleContainer()). Validates that the single container is actually this
  // container.
  RowVector* getSingleContainerData() const {
    VELOX_DCHECK_EQ(allContainers_.size(), 1);
    VELOX_DCHECK_EQ(owningInputs_.size(), 1);
    VELOX_DCHECK_NOT_NULL(owningInputs_[0]);
    auto it = allContainers_.begin();
    // Validate that the single container is self
    VELOX_DCHECK_EQ(it->first, id_, "Single container ID mismatch with self ID");
    VELOX_DCHECK(it->second == this, "Single container is not self");
    return owningInputs_[0].get();
  }

  // Retain 'source''s (compacted) string buffers on behalf of 'result'
  // without copying bodies: one BufferView per source buffer, sized as the
  // bytes this extraction call actually references (honest amortized
  // accounting; see HybridPayloadBufferReleaser). All-inline batches need no
  // buffer at all.
  static void addSharedStringBufferViews(
      FlatVector<StringView>* result,
      const FlatVector<StringView>* source,
      uint64_t referencedBytes) {
    if (referencedBytes == 0) {
      return;
    }
    bool first = true;
    for (const auto& buffer : source->stringBuffers()) {
      if (first) {
        result->addStringBuffer(
            BufferView<HybridPayloadBufferReleaser>::create(
                buffer->as<uint8_t>(),
                std::min<uint64_t>(referencedBytes, buffer->size()),
                HybridPayloadBufferReleaser{buffer}));
        first = false;
      } else {
        // Defensive: coalesceBatches() compaction leaves at most one string
        // buffer; acquire any unexpected extras outright.
        result->addStringBuffer(buffer);
      }
    }
  }

  template <TypeKind Kind>
  void extractPayloadTyped(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds,
      bool exactSize) {
    if (rowNumbers.size() > 0) {
      extractPayloadTypedInternal<Kind, true>(
          rows,
          rowNumbers,
          rowNumbers.size(),
          columnIndex,
          resultOffset,
          result,
          outputRowIds,
          exactSize);
    } else {
      extractPayloadTypedInternal<Kind, false>(
          rows,
          rowNumbers,
          numRows,
          columnIndex,
          resultOffset,
          result,
          outputRowIds,
          exactSize);
    }
  }

  template <TypeKind Kind, bool useRowNumbers>
  void extractPayloadTypedInternal(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds,
      bool exactSize) {
    result->resize(numRows + resultOffset);
    VELOX_CHECK_EQ(numRows, outputRowIds.size());
    VELOX_CHECK(Kind != TypeKind::ROW, "Currently do not support ROW");
    if (Kind == TypeKind::ARRAY || Kind == TypeKind::MAP) {
      extractPayloadComplex(
          numRows, columnIndex, resultOffset, result, outputRowIds);
      return;
    }
    VELOX_CHECK(Kind != TypeKind::ROW && Kind != TypeKind::MAP);
    using T = typename KindToFlatVector<Kind>::HashRowType;
    auto flatResult = result->as<FlatVector<T>>();

    // Scattered mode: payloads kept in separate batches
    if (scatteredModeEnabled_) {
      if (isSingleContainer()) {
        if (isNullable_[columnIndex]) {
          extractPayloadScatteredWithNulls<T, useRowNumbers>(
              rows, rowNumbers, numRows, columnIndex, resultOffset,
              flatResult, outputRowIds);
        } else {
          extractPayloadScatteredNoNulls<T, useRowNumbers>(
              rows, rowNumbers, numRows, columnIndex, resultOffset,
              flatResult, outputRowIds);
        }
      } else {
        // Multi-container scattered mode. Nullability must consider all
        // registered containers, not just this one (see
        // anyContainerNullable()).
        if (anyContainerNullable(columnIndex)) {
          extractPayloadScatteredWithNullsMulti<T, useRowNumbers>(
              rows, rowNumbers, numRows, columnIndex, resultOffset,
              flatResult, outputRowIds);
        } else {
          extractPayloadScatteredNoNullsMulti<T, useRowNumbers>(
              rows, rowNumbers, numRows, columnIndex, resultOffset,
              flatResult, outputRowIds);
        }
      }
      return;
    }

    // Fast path for single container (spilling, sort) - avoids map lookups
    if (isSingleContainer()) {
      if (isNullable_[columnIndex]) {
        extractPayloadWithNullsSingleContainer<T, useRowNumbers>(
            rows,
            rowNumbers,
            numRows,
            columnIndex,
            resultOffset,
            flatResult,
            outputRowIds);
      } else {
        extractPayloadNoNullsSingleContainer<T, useRowNumbers>(
            rows,
            rowNumbers,
            numRows,
            columnIndex,
            resultOffset,
            flatResult,
            outputRowIds);
      }
      return;
    }

    // Multi-container path (hash join after table merge). Nullability must
    // consider all registered containers, not just this one (see
    // anyContainerNullable()).
    if (anyContainerNullable(columnIndex)) {
      extractPayloadWithNulls<T, useRowNumbers>(
          rows,
          rowNumbers,
          numRows,
          columnIndex,
          resultOffset,
          flatResult,
          outputRowIds,
          exactSize);
    } else {
      extractPayloadNoNulls<T, useRowNumbers>(
          rows,
          rowNumbers,
          numRows,
          columnIndex,
          resultOffset,
          flatResult,
          outputRowIds,
          exactSize);
    }
  }

  void extractPayloadComplex(
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      const VectorPtr& result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    // Cache per-container child pointers; coalesced payload lives in
    // owningInputs_[0] for each container.
    std::vector<BaseVector*> sources(maxContainerId_ + 1, nullptr);
    for (const auto& entry : allContainers_) {
      // Skip containers with no data (e.g., drivers that received no input)
      if (entry.second->owningInputs_.empty()) {
        continue;
      }
      auto* child = entry.second->owningInputs_[0]->childAt(columnIndex).get();
      VELOX_CHECK_NOT_NULL(child);
      sources[entry.first] = child;
    }

    auto* rowIdPtr = outputRowIds.data();
    for (int i = 0; i < numRows; ++i) {
      const auto& rec = rowIdPtr[i];
      auto* source = sources[rec.containerId_];
      VELOX_DCHECK_NOT_NULL(source);
      auto resultIndex = resultOffset + i;
      if (source->isNullAt(rec.rowId_)) {
        result->setNull(resultIndex, true);
        continue;
      }
      result->setNull(resultIndex, false);
      result->copy(source, resultIndex, rec.rowId_, 1);
    }
  }

  // ========== Single-container fast path implementations ==========
  // These avoid map lookups and container ID checks in hot loops.
  // Uses 4-way unrolled prefetch for best performance.

  template <typename T, bool useRowNumbers>
  void extractPayloadWithNullsSingleContainer(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr& nullBuffer = result->mutableNulls(maxRows);
    auto nulls = nullBuffer->asMutable<uint64_t>();
    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    // Single container - direct access without map lookup
    ensureFlatChild(getSingleContainerData(), columnIndex);
    auto* flatChild = getSingleContainerData()
                          ->childAt(columnIndex)
                          ->template as<FlatVector<T>>();
    VELOX_CHECK_NOT_NULL(flatChild);
    const T* rawValues = flatChild->rawValues();
    const uint64_t* rawNulls = flatChild->rawNulls();
    // Strings are extracted as raw 16-byte views; out-of-line bodies stay
    // in the compacted source buffer, retained via a BufferView added after
    // the loops (sized by the bytes this call references).
    [[maybe_unused]] uint64_t stringRefBytes = 0;

    constexpr vector_size_t kPrefetchDist = 128;

    int32_t i = 0;

    // ---- Main loop: process 4 rows per iteration ----
    for (; i + 3 < numRows; i += 4) {
      // ---- Prefetch next 4 records at distance ----
      const int32_t p = i + kPrefetchDist;
      if (FOLLY_LIKELY(p + 3 < numRows)) {
        __builtin_prefetch(rawValues + rowIdPtr[p].rowId_, 0, 1);
        __builtin_prefetch(rawValues + rowIdPtr[p + 1].rowId_, 0, 1);
        __builtin_prefetch(rawValues + rowIdPtr[p + 2].rowId_, 0, 1);
        __builtin_prefetch(rawValues + rowIdPtr[p + 3].rowId_, 0, 1);
      }

      // ---- Process 4 rows ----
      for (int32_t u = 0; u < 4; ++u) {
        const int32_t idx = i + u;

        const char* row;
        if constexpr (useRowNumbers) {
          auto rowNumber = rowNumbers[idx];
          row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
        } else {
          row = rows[idx];
        }

        const auto resultIndex = resultOffset + idx;
        if (row == nullptr) {
          bits::setNull(nulls, resultIndex, true);
          continue;
        }

        const auto rid = rowIdPtr[idx].rowId_;
        if (rawNulls != nullptr && bits::isBitNull(rawNulls, rid)) {
          bits::setNull(nulls, resultIndex, true);
          continue;
        }

        bits::setNull(nulls, resultIndex, false);
        if constexpr (std::is_same_v<T, StringView>) {
          const auto& v = rawValues[rid];
          if (!v.isInline()) {
            stringRefBytes += v.size();
          }
          values[resultIndex] = v;
        } else {
          values[resultIndex] = rawValues[rid];
        }
      }
    }

    // ---- Tail loop ----
    for (; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        bits::setNull(nulls, resultIndex, true);
        continue;
      }

      const auto rid = rowIdPtr[i].rowId_;
      if (rawNulls != nullptr && bits::isBitNull(rawNulls, rid)) {
        bits::setNull(nulls, resultIndex, true);
        continue;
      }

      bits::setNull(nulls, resultIndex, false);
      if constexpr (std::is_same_v<T, StringView>) {
        const auto& v = rawValues[rid];
        if (!v.isInline()) {
          stringRefBytes += v.size();
        }
        values[resultIndex] = v;
      } else {
        values[resultIndex] = rawValues[rid];
      }
    }

    if constexpr (std::is_same_v<T, StringView>) {
      addSharedStringBufferViews(result, flatChild, stringRefBytes);
    }
  }

  template <typename T, bool useRowNumbers>
  void extractPayloadNoNullsSingleContainer(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    // Single container - direct access without map lookup
    ensureFlatChild(getSingleContainerData(), columnIndex);
    auto* flatChild = getSingleContainerData()
                          ->childAt(columnIndex)
                          ->template as<FlatVector<T>>();
    VELOX_CHECK_NOT_NULL(flatChild);
    const T* rawValues = flatChild->rawValues();
    // Strings: see extractPayloadWithNullsSingleContainer.
    [[maybe_unused]] uint64_t stringRefBytes = 0;

    constexpr vector_size_t kPrefetchDist = 128;

    int32_t i = 0;

    // ---- Main loop: process 4 rows per iteration ----
    for (; i + 3 < numRows; i += 4) {
      // ---- Prefetch next 4 records at distance ----
      const int32_t p = i + kPrefetchDist;
      if (FOLLY_LIKELY(p + 3 < numRows)) {
        __builtin_prefetch(rawValues + rowIdPtr[p].rowId_, 0, 1);
        __builtin_prefetch(rawValues + rowIdPtr[p + 1].rowId_, 0, 1);
        __builtin_prefetch(rawValues + rowIdPtr[p + 2].rowId_, 0, 1);
        __builtin_prefetch(rawValues + rowIdPtr[p + 3].rowId_, 0, 1);
      }

      // ---- Process 4 rows ----
      for (int32_t u = 0; u < 4; ++u) {
        const int32_t idx = i + u;

        const char* row;
        if constexpr (useRowNumbers) {
          auto rowNumber = rowNumbers[idx];
          row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
        } else {
          row = rows[idx];
        }

        const auto resultIndex = resultOffset + idx;
        if (row == nullptr) {
          result->setNull(resultIndex, true);
          continue;
        }

        result->setNull(resultIndex, false);
        const auto rid = rowIdPtr[idx].rowId_;
        if constexpr (std::is_same_v<T, StringView>) {
          const auto& v = rawValues[rid];
          if (!v.isInline()) {
            stringRefBytes += v.size();
          }
          values[resultIndex] = v;
        } else {
          values[resultIndex] = rawValues[rid];
        }
      }
    }

    // ---- Tail loop ----
    for (; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
        continue;
      }

      result->setNull(resultIndex, false);
      const auto rid = rowIdPtr[i].rowId_;
      if constexpr (std::is_same_v<T, StringView>) {
        const auto& v = rawValues[rid];
        if (!v.isInline()) {
          stringRefBytes += v.size();
        }
        values[resultIndex] = v;
      } else {
        values[resultIndex] = rawValues[rid];
      }
    }

    if constexpr (std::is_same_v<T, StringView>) {
      addSharedStringBufferViews(result, flatChild, stringRefBytes);
    }
  }

  // ========== Scattered mode extraction (non-coalesced batches) ==========
  // Optimization: 4-way unrolled prefetch for better cache utilization.
  // Cache data pointers per batch and prefetch future row data locations.

  template <typename T, bool useRowNumbers>
  void extractPayloadScatteredNoNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    // Simple loop using DecodedVector - no prefetch complexity
    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
        continue;
      }

      result->setNull(resultIndex, false);
      const auto& rid = rowIdPtr[i];
      auto batchIdx = rid.batchId();
      auto rowInBatch = rid.rowInBatch();

      T value = decodedPayloads_[batchIdx][columnIndex]->valueAt<T>(rowInBatch);
      // Strings keep the per-row body copy here: scattered sources are one
      // small buffer per input batch, and per-batch buffer acquisition is
      // O(contributing batches) per extraction call, which loses badly when
      // batches are numerous. Only the coalesced paths share buffers.
      if constexpr (std::is_same_v<T, StringView>) {
        result->set(resultIndex, value);
      } else {
        values[resultIndex] = value;
      }
    }
  }

  template <typename T, bool useRowNumbers>
  void extractPayloadScatteredWithNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    // Simple loop using DecodedVector - no prefetch complexity
    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
        continue;
      }

      const auto& rid = rowIdPtr[i];
      auto batchIdx = rid.batchId();
      auto rowInBatch = rid.rowInBatch();

      // Check for null in the payload
      if (decodedPayloads_[batchIdx][columnIndex]->isNullAt(rowInBatch)) {
        result->setNull(resultIndex, true);
        continue;
      }

      result->setNull(resultIndex, false);
      T value = decodedPayloads_[batchIdx][columnIndex]->valueAt<T>(rowInBatch);
      // Strings keep the per-row body copy here: scattered sources are one
      // small buffer per input batch, and per-batch buffer acquisition is
      // O(contributing batches) per extraction call, which loses badly when
      // batches are numerous. Only the coalesced paths share buffers.
      if constexpr (std::is_same_v<T, StringView>) {
        result->set(resultIndex, value);
      } else {
        values[resultIndex] = value;
      }
    }
  }

  // ========== Multi-container scattered mode extraction ==========
  // Each container has its own decodedPayloads_, indexed by
  // [batchIdx][columnIndex]. Uses containerId_ to find the right container,
  // then batchId()/rowInBatch() to locate data.

  template <typename T, bool useRowNumbers>
  void extractPayloadScatteredNoNullsMulti(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    // Flat containerId -> decodedPayloads table; rows interleave containers
    // under multi-driver probes, so a per-switch hash lookup runs ~per row.
    std::array<const std::vector<std::vector<std::unique_ptr<DecodedVector>>>*,
               256> payloadsById{};
    for (const auto& [id, container] : allContainers_) {
      payloadsById[id] = &container->decodedPayloads_;
    }
    uint8_t currentContainerId = UINT8_MAX;
    const std::vector<std::vector<std::unique_ptr<DecodedVector>>>*
        currentDecodedPayloads = nullptr;

    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
        continue;
      }

      result->setNull(resultIndex, false);
      const auto& rid = rowIdPtr[i];

      // Switch container if needed
      if (rid.containerId_ != currentContainerId) {
        currentContainerId = rid.containerId_;
        currentDecodedPayloads = payloadsById[currentContainerId];
        VELOX_CHECK(currentDecodedPayloads != nullptr,
                    "Container {} not found", currentContainerId);
      }

      auto batchIdx = rid.batchId();
      auto rowInBatch = rid.rowInBatch();

      VELOX_DCHECK_LT(batchIdx, currentDecodedPayloads->size());
      VELOX_DCHECK_LT(columnIndex, (*currentDecodedPayloads)[batchIdx].size());

      T value =
          (*currentDecodedPayloads)[batchIdx][columnIndex]->valueAt<T>(
              rowInBatch);
      // Strings: per-row body copy (see extractPayloadScatteredNoNulls).
      if constexpr (std::is_same_v<T, StringView>) {
        result->set(resultIndex, value);
      } else {
        values[resultIndex] = value;
      }
    }
  }

  template <typename T, bool useRowNumbers>
  void extractPayloadScatteredWithNullsMulti(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    // Flat containerId -> decodedPayloads table (see NoNullsMulti above).
    std::array<const std::vector<std::vector<std::unique_ptr<DecodedVector>>>*,
               256> payloadsById{};
    for (const auto& [id, container] : allContainers_) {
      payloadsById[id] = &container->decodedPayloads_;
    }
    uint8_t currentContainerId = UINT8_MAX;
    const std::vector<std::vector<std::unique_ptr<DecodedVector>>>*
        currentDecodedPayloads = nullptr;

    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
        continue;
      }

      const auto& rid = rowIdPtr[i];

      // Switch container if needed
      if (rid.containerId_ != currentContainerId) {
        currentContainerId = rid.containerId_;
        currentDecodedPayloads = payloadsById[currentContainerId];
        VELOX_CHECK(currentDecodedPayloads != nullptr,
                    "Container {} not found", currentContainerId);
      }

      auto batchIdx = rid.batchId();
      auto rowInBatch = rid.rowInBatch();

      VELOX_DCHECK_LT(batchIdx, currentDecodedPayloads->size());
      VELOX_DCHECK_LT(columnIndex, (*currentDecodedPayloads)[batchIdx].size());

      // Check for null in the payload
      if ((*currentDecodedPayloads)[batchIdx][columnIndex]->isNullAt(
              rowInBatch)) {
        result->setNull(resultIndex, true);
        continue;
      }

      result->setNull(resultIndex, false);
      T value =
          (*currentDecodedPayloads)[batchIdx][columnIndex]->valueAt<T>(
              rowInBatch);
      // Strings: per-row body copy (see extractPayloadScatteredNoNulls).
      if constexpr (std::is_same_v<T, StringView>) {
        result->set(resultIndex, value);
      } else {
        values[resultIndex] = value;
      }
    }
  }

  // ========== End scattered mode extraction implementations ==========
  // ========== End single-container fast path implementations ==========

  template <typename T, bool useRowNumbers>
  void extractPayloadWithNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds,
      bool /*exactSize*/) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());

    BufferPtr& nullBuffer = result->mutableNulls(maxRows);
    auto nulls = nullBuffer->asMutable<uint64_t>();
    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();
    auto* rowIdPtr = outputRowIds.data();

    constexpr vector_size_t kPrefetchDist = 128;

    std::vector<const T*> rawValuesByContainer(maxContainerId_ + 1, nullptr);
    std::vector<const uint64_t*> rawNullsByContainer(
        maxContainerId_ + 1, nullptr);
    // For strings: per-container sources and referenced-byte counters; a
    // BufferView per contributing container is added after the loops.
    [[maybe_unused]] std::vector<const FlatVector<T>*> sourceByContainer(
        maxContainerId_ + 1, nullptr);
    [[maybe_unused]] std::vector<uint64_t> stringRefBytesByContainer(
        maxContainerId_ + 1, 0);
    for (const auto& entry : allContainers_) {
      // Skip containers with no data (e.g., drivers that received no input)
      if (entry.second->owningInputs_.empty()) {
        continue;
      }
      ensureFlatChild(entry.second->owningInputs_[0].get(), columnIndex);
      auto* flatChild = entry.second->owningInputs_[0]
                            ->childAt(columnIndex)
                            ->template as<FlatVector<T>>();
      VELOX_CHECK_NOT_NULL(flatChild);
      rawValuesByContainer[entry.first] = flatChild->rawValues();
      rawNullsByContainer[entry.first] = flatChild->rawNulls();
      sourceByContainer[entry.first] = flatChild;
    }

    int32_t curCid = -1;
    const T* curRaw = nullptr;
    const uint64_t* curNulls = nullptr;

    int32_t pfCid = -1;
    const T* pfRaw = nullptr;

    if (FOLLY_LIKELY(numRows > 0)) {
      curCid = rowIdPtr[0].containerId_;
      curRaw = rawValuesByContainer[curCid];
      curNulls = rawNullsByContainer[curCid];
      VELOX_DCHECK_NOT_NULL(curRaw);
      if (kPrefetchDist < numRows) {
        pfCid = rowIdPtr[kPrefetchDist].containerId_;
        pfRaw = rawValuesByContainer[pfCid];
        VELOX_DCHECK_NOT_NULL(pfRaw);
      }
    }

    int32_t i = 0;

    for (; i + 3 < numRows; i += 4) {
      const int32_t p = i + kPrefetchDist;
      if (FOLLY_LIKELY(p + 3 < numRows)) {
        const auto& r0 = rowIdPtr[p];
        if (FOLLY_UNLIKELY(r0.containerId_ != pfCid)) {
          pfCid = r0.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r0.rowId_, 0, 1);

        const auto& r1 = rowIdPtr[p + 1];
        if (FOLLY_UNLIKELY(r1.containerId_ != pfCid)) {
          pfCid = r1.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r1.rowId_, 0, 1);

        const auto& r2 = rowIdPtr[p + 2];
        if (FOLLY_UNLIKELY(r2.containerId_ != pfCid)) {
          pfCid = r2.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r2.rowId_, 0, 1);

        const auto& r3 = rowIdPtr[p + 3];
        if (FOLLY_UNLIKELY(r3.containerId_ != pfCid)) {
          pfCid = r3.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r3.rowId_, 0, 1);
      }

      for (int32_t u = 0; u < 4; ++u) {
        const int32_t idx = i + u;

        const char* row;
        if constexpr (useRowNumbers) {
          auto rowNumber = rowNumbers[idx];
          row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
        } else {
          row = rows[idx];
        }

        const auto resultIndex = resultOffset + idx;
        if (row == nullptr) {
          bits::setNull(nulls, resultIndex, true);
          continue;
        }

        const auto& rowIdRec = rowIdPtr[idx];
        if (FOLLY_UNLIKELY(rowIdRec.containerId_ != curCid)) {
          curCid = rowIdRec.containerId_;
          curRaw = rawValuesByContainer[curCid];
          curNulls = rawNullsByContainer[curCid];
          VELOX_DCHECK_NOT_NULL(curRaw);
        }

        const auto rid = rowIdRec.rowId_;
        if (curNulls != nullptr && bits::isBitNull(curNulls, rid)) {
          bits::setNull(nulls, resultIndex, true);
          continue;
        }

        bits::setNull(nulls, resultIndex, false);
        if constexpr (std::is_same_v<T, StringView>) {
          const auto& v = curRaw[rid];
          if (!v.isInline()) {
            stringRefBytesByContainer[curCid] += v.size();
          }
          values[resultIndex] = v;
        } else {
          values[resultIndex] = curRaw[rid];
        }
      }
    }

    for (; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        bits::setNull(nulls, resultIndex, true);
        continue;
      }

      const auto& rowIdRec = rowIdPtr[i];
      if (FOLLY_UNLIKELY(rowIdRec.containerId_ != curCid)) {
        curCid = rowIdRec.containerId_;
        curRaw = rawValuesByContainer[curCid];
        curNulls = rawNullsByContainer[curCid];
        VELOX_DCHECK_NOT_NULL(curRaw);
      }

      const auto rid = rowIdRec.rowId_;
      if (curNulls != nullptr && bits::isBitNull(curNulls, rid)) {
        bits::setNull(nulls, resultIndex, true);
        continue;
      }

      bits::setNull(nulls, resultIndex, false);
      if constexpr (std::is_same_v<T, StringView>) {
        const auto& v = curRaw[rid];
        if (!v.isInline()) {
          stringRefBytesByContainer[curCid] += v.size();
        }
        values[resultIndex] = v;
      } else {
        values[resultIndex] = curRaw[rid];
      }
    }

    if constexpr (std::is_same_v<T, StringView>) {
      for (int32_t cid = 0; cid <= maxContainerId_; ++cid) {
        if (stringRefBytesByContainer[cid] > 0) {
          addSharedStringBufferViews(
              result, sourceByContainer[cid], stringRefBytesByContainer[cid]);
        }
      }
    }
  }

  template <typename T, bool useRowNumbers>
  void extractPayloadNoNulls(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      int32_t numRows,
      int32_t columnIndex,
      int32_t resultOffset,
      FlatVector<T>* result,
      std::vector<HybridRowId>& outputRowIds,
      bool /*exactSize*/) {
    auto maxRows = numRows + resultOffset;
    VELOX_DCHECK_LE(maxRows, result->size());
    VELOX_DCHECK_LT(columnIndex, payloadTypes_.size());

    BufferPtr valuesBuffer = result->mutableValues(maxRows);
    auto values = valuesBuffer->asMutableRange<T>();

    auto* rowIdPtr = outputRowIds.data();

    constexpr vector_size_t kPrefetchDist = 128;

    std::vector<const T*> rawValuesByContainer(maxContainerId_ + 1, nullptr);
    // For strings: per-container sources and referenced-byte counters; a
    // BufferView per contributing container is added after the loops.
    [[maybe_unused]] std::vector<const FlatVector<T>*> sourceByContainer(
        maxContainerId_ + 1, nullptr);
    [[maybe_unused]] std::vector<uint64_t> stringRefBytesByContainer(
        maxContainerId_ + 1, 0);
    for (const auto& entry : allContainers_) {
      // Skip containers with no data (e.g., drivers that received no input)
      if (entry.second->owningInputs_.empty()) {
        continue;
      }
      ensureFlatChild(entry.second->owningInputs_[0].get(), columnIndex);
      auto* flatChild = entry.second->owningInputs_[0]
                            ->childAt(columnIndex)
                            ->template as<FlatVector<T>>();
      VELOX_CHECK_NOT_NULL(flatChild);
      rawValuesByContainer[entry.first] = flatChild->rawValues();
      sourceByContainer[entry.first] = flatChild;
    }
    // cached for load
    int32_t curCid = -1;
    const T* curRaw = nullptr;

    // cached for prefetch
    int32_t pfCid = -1;
    const T* pfRaw = nullptr;

    if (FOLLY_LIKELY(numRows > 0)) {
      curCid = rowIdPtr[0].containerId_;
      curRaw = rawValuesByContainer[curCid];
      VELOX_DCHECK_NOT_NULL(curRaw);
      if (kPrefetchDist < numRows) {
        pfCid = rowIdPtr[kPrefetchDist].containerId_;
        pfRaw = rawValuesByContainer[pfCid];
        VELOX_DCHECK_NOT_NULL(pfRaw);
      }
    }

    int32_t i = 0;

    // ---- Main loop: process 4 rows per iteration ----
    for (; i + 3 < numRows; i += 4) {
      // ---- Prefetch next 4 records at distance ----
      const int32_t p = i + kPrefetchDist;
      // Correct bound for prefetching p..p+3:
      if (FOLLY_LIKELY(p + 3 < numRows)) {
        // r0
        const auto& r0 = rowIdPtr[p];
        if (FOLLY_UNLIKELY(r0.containerId_ != pfCid)) {
          pfCid = r0.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r0.rowId_, 0, 1);

        // r1
        const auto& r1 = rowIdPtr[p + 1];
        if (FOLLY_UNLIKELY(r1.containerId_ != pfCid)) {
          pfCid = r1.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r1.rowId_, 0, 1);

        // r2
        const auto& r2 = rowIdPtr[p + 2];
        if (FOLLY_UNLIKELY(r2.containerId_ != pfCid)) {
          pfCid = r2.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r2.rowId_, 0, 1);

        // r3
        const auto& r3 = rowIdPtr[p + 3];
        if (FOLLY_UNLIKELY(r3.containerId_ != pfCid)) {
          pfCid = r3.containerId_;
          pfRaw = rawValuesByContainer[pfCid];
          VELOX_DCHECK_NOT_NULL(pfRaw);
        }
        __builtin_prefetch(pfRaw + r3.rowId_, 0, 1);
      }

      // ---- Consume 4 rows ----
      for (int32_t u = 0; u < 4; ++u) {
        const int32_t idx = i + u;

        const char* row;
        if constexpr (useRowNumbers) {
          auto rowNumber = rowNumbers[idx];
          row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
        } else {
          row = rows[idx];
        }

        const auto resultIndex = resultOffset + idx;
        if (row == nullptr) {
          result->setNull(resultIndex, true);
          continue;
        }

        result->setNull(resultIndex, false);

        const auto& rowIdRec = rowIdPtr[idx];

        // Refresh cached container pointer only when containerId changes.
        if (FOLLY_UNLIKELY(rowIdRec.containerId_ != curCid)) {
          curCid = rowIdRec.containerId_;
          curRaw = rawValuesByContainer[curCid];
          VELOX_DCHECK_NOT_NULL(curRaw);
        }

        const auto rid = rowIdRec.rowId_;
        if constexpr (std::is_same_v<T, StringView>) {
          const auto& v = curRaw[rid];
          if (!v.isInline()) {
            stringRefBytesByContainer[curCid] += v.size();
          }
          values[resultIndex] = v;
        } else {
          values[resultIndex] = curRaw[rid];
        }
      }
    }

    // ---- Tail loop ----
    for (; i < numRows; ++i) {
      const char* row;
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }

      const auto resultIndex = resultOffset + i;
      if (row == nullptr) {
        result->setNull(resultIndex, true);
        continue;
      }

      result->setNull(resultIndex, false);

      const auto& rowIdRec = rowIdPtr[i];
      if (FOLLY_UNLIKELY(rowIdRec.containerId_ != curCid)) {
        curCid = rowIdRec.containerId_;
        curRaw = rawValuesByContainer[curCid];
        VELOX_DCHECK_NOT_NULL(curRaw);
      }

      const auto rid = rowIdRec.rowId_;
      if constexpr (std::is_same_v<T, StringView>) {
        const auto& v = curRaw[rid];
        if (!v.isInline()) {
          stringRefBytesByContainer[curCid] += v.size();
        }
        values[resultIndex] = v;
      } else {
        values[resultIndex] = curRaw[rid];
      }
    }

    if constexpr (std::is_same_v<T, StringView>) {
      for (int32_t cid = 0; cid <= maxContainerId_; ++cid) {
        if (stringRefBytesByContainer[cid] > 0) {
          addSharedStringBufferViews(
              result, sourceByContainer[cid], stringRefBytesByContainer[cid]);
        }
      }
    }
  }

  template <bool useRowNumbers>
  void getRowIdsInternal(
      const char* const* rows,
      folly::Range<const vector_size_t*> rowNumbers,
      uint32_t numRows,
      std::vector<HybridRowId>& outputRowIds) {
    VELOX_CHECK_EQ(numRows, outputRowIds.size());
    // NOTE: For N-way late-m, payloadTypes can be empty (join only stores
    // keys). We still need to extract rowIds for upstream traversal.
    // The rowIdColumnOffset_ is always valid since it's the uint64_t column
    // appended after the key columns.
    constexpr int32_t kPrefetchDist = 16;
    for (int32_t i = 0; i < numRows; ++i) {
      const char* row;
      if (i + kPrefetchDist < numRows) {
        const char* pfRow;
        if constexpr (useRowNumbers) {
          auto pfNumber = rowNumbers[i + kPrefetchDist];
          pfRow = pfNumber >= 0 ? rows[pfNumber] : nullptr;
        } else {
          pfRow = rows[i + kPrefetchDist];
        }
        if (pfRow != nullptr) {
          __builtin_prefetch(pfRow + rowIdColumnOffset_, 0, 1);
        }
      }
      if constexpr (useRowNumbers) {
        auto rowNumber = rowNumbers[i];
        row = rowNumber >= 0 ? rows[rowNumber] : nullptr;
      } else {
        row = rows[i];
      }
      if (row == nullptr) {
        outputRowIds[i] = {0, 0};
      } else {
        auto encodedId = keys_->valueAt<uint64_t>(row, rowIdColumnOffset_);
        uint8_t driverId = encodedId >> 56;
        uint64_t rowId = encodedId & ((1ULL << 56) - 1);
        outputRowIds[i] = {driverId, rowId};
      }
    }
  }

  int32_t rowIdColumnOffset_;
  std::vector<RowVectorPtr> owningInputs_;
  const std::vector<TypePtr> keyTypes_;
  const std::vector<TypePtr> payloadTypes_;
  std::vector<TypePtr> types_;
  std::vector<uint64_t> payloadFlatBytesSum_;

  const int numKeys_;
  RowContainer* keys_;

  // null bitmap for each column
  std::vector<char> isNullable_;

  uint64_t totalRows_{0};
  uint32_t totalBatches_{0};

  uint8_t id_{0};
  std::unordered_map<uint8_t, HybridContainer*> allContainers_;
  uint8_t maxContainerId_{0};

  // Controls whether to reorder rows by containerId during extraction.
  // Default true for better cache locality. Can be disabled for testing.
  bool reorderEnabled_{true};

  // Controls whether scattered (non-coalesced) mode is used.
  // In scattered mode, payload batches are kept separate and row IDs
  // encode (batchId, rowInBatch) instead of global row index.
  bool scatteredModeEnabled_{false};

  // Decoded payload vectors for scattered mode extraction.
  // Outer vector: per batch (same index as owningInputs_)
  // Inner vector: per payload column
  // Using DecodedVector allows efficient access to any encoding (flat,
  // dictionary, lazy).
  std::vector<std::vector<std::unique_ptr<DecodedVector>>> decodedPayloads_;
};

template <>
inline void HybridContainer::extractPayloadTyped<TypeKind::OPAQUE>(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    int32_t columnIndex,
    int32_t resultOffset,
    const VectorPtr& result,
    std::vector<HybridRowId>& outputRowIds,
    bool exactSize) {
  VELOX_UNSUPPORTED("HybridContainer doesn't support OPAQUE payload types.");
}

inline void HybridContainer::extractPayload(
    const char* const* rows,
    folly::Range<const vector_size_t*> rowNumbers,
    int32_t numRows,
    int32_t columnIndex,
    int32_t resultOffset,
    const VectorPtr& result,
    std::vector<HybridRowId>& outputRowIds,
    bool exactSize) {
  // Ablation knob (benchmark only, env-gated so no QueryConfig churn):
  // HYBRID_NAIVE_EXTRACT=1 replaces the optimized kernels with the textbook
  // gather described in Velox's "Why Sort is row-based" post -- for each
  // output row, locate the owning input vector, locate the source row, copy
  // that one cell -- with no pre-decoded vectors, no prefetching, no
  // unrolling, and per-cell virtual dispatch. Used to reproduce the
  // published negative result and to attribute our recovery to each
  // mechanism (naive -> +decode/prefetch -> +coalescing).
  static const bool kNaiveExtract =
      (::getenv("HYBRID_NAIVE_EXTRACT") != nullptr);
  if (kNaiveExtract && !owningInputs_.empty()) {
    result->resize(numRows + resultOffset);
    for (int32_t i = 0; i < numRows; ++i) {
      const auto& rid = outputRowIds[i];
      auto* container = this;
      if (!allContainers_.empty()) {
        auto it = allContainers_.find(rid.containerId_);
        if (it != allContainers_.end()) {
          container = it->second;
        }
      }
      const auto& batches = container->owningInputs_;
      int32_t batchIdx;
      int32_t rowInBatch;
      if (container->scatteredModeEnabled_) {
        batchIdx = static_cast<int32_t>(rid.batchId());
        rowInBatch = static_cast<int32_t>(rid.rowInBatch());
      } else {
        batchIdx = 0;
        rowInBatch = static_cast<int32_t>(rid.rowId_);
      }
      if (batchIdx >= static_cast<int32_t>(batches.size())) {
        continue;
      }
      // One cell at a time, through the generic vector copy path.
      result->copy(
          batches[batchIdx]->childAt(columnIndex).get(),
          i + resultOffset,
          rowInBatch,
          1);
    }
    return;
  }
  VELOX_DYNAMIC_TYPE_DISPATCH_ALL(
      extractPayloadTyped,
      result->typeKind(),
      rows,
      rowNumbers,
      numRows,
      columnIndex,
      resultOffset,
      result,
      outputRowIds,
      exactSize);
}

inline void HybridContainer::extractNulls(
    const char* const* rows,
    int32_t numRows,
    int32_t columnIndex,
    const BufferPtr& result,
    std::vector<HybridRowId>& outputRowIds) {
  if (isKey(columnIndex)) {
    keys_->extractNulls(rows, numRows, columnIndex, result);
  } else {
    auto payloadColumnIndex = columnIndex - numKeys_;
    VELOX_DCHECK(result->size() >= bits::nbytes(numRows));
    auto* rawResult = result->asMutable<uint64_t>();
    bits::fillBits(rawResult, 0, numRows, false);
    // NOTE: must index nullability by the payload column index (isNullable_
    // is sized to payloadTypes_) and consider all registered containers.
    if (!anyContainerNullable(payloadColumnIndex)) {
      return;
    }
    std::vector<const uint64_t*> rawNullsByContainer(
        maxContainerId_ + 1, nullptr);
    for (const auto& entry : allContainers_) {
      // Skip containers with no data (e.g., drivers that received no input)
      if (entry.second->owningInputs_.empty()) {
        continue;
      }
      auto* child =
          entry.second->owningInputs_[0]->childAt(payloadColumnIndex).get();
      rawNullsByContainer[entry.first] = child->rawNulls();
    }
    auto* rowIdPtr = outputRowIds.data();
    for (int32_t i = 0; i < numRows; ++i) {
      const char* row = rows[i];
      if (row == nullptr) {
        bits::setBit(rawResult, i, true);
        continue;
      }
      const auto& rec = rowIdPtr[i];
      const auto* nulls = rawNullsByContainer[rec.containerId_];
      if (nulls != nullptr && bits::isBitNull(nulls, rec.rowId_)) {
        bits::setBit(rawResult, i, true);
      }
    }
  }
}

} // namespace facebook::velox::exec
