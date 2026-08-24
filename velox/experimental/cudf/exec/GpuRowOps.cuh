/*
 * GpuRowOps.cuh
 *
 * GPU kernel declarations for fixed-stride row store operations:
 *   - extractKeysFromRows:  extract a fixed-width column from row buffer
 *   - gatherRowsWarp:       warp-per-row gather (coalesced writes)
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "GpuFixedRowStore.h"

/// Extract a single fixed-width column from the row buffer into a contiguous
/// device array. The key must be a fixed-width type (e.g. INT64).
void extractKeysFromRows(
    const GpuFixedRowStore& store,
    int32_t key_offset,
    int32_t key_width,
    void* d_out,
    cudaStream_t stream);

/// Gather rows by index from a source GpuFixedRowStore.
/// Uses warp-collaborative copy (32 threads per row) for coalesced writes.
///
/// @param store         Source row store on device.
/// @param d_gather_map  Device array of int32_t row indices [num_output].
/// @param num_output    Number of rows to gather.
/// @param d_out_buffer  Pre-allocated output buffer [num_output * store.row_width].
/// @param stream        CUDA stream.
void gatherRowsWarp(
    const GpuFixedRowStore& store,
    const int32_t* d_gather_map,
    int32_t num_output,
    uint8_t* d_out_buffer,
    cudaStream_t stream);

/// Fused gather + concatenate: gather rows from two source row stores and
/// concatenate them into a single wider output row. Output layout is:
///   [probe_row (probeStore.row_width bytes) | build_row (buildStore.row_width bytes)]
/// padded to output_row_width (8-byte aligned).
///
/// @param probeStore     Source row store for probe (left) side.
/// @param d_probe_map    Device array of probe row indices [num_output].
/// @param buildStore     Source row store for build (right) side.
/// @param d_build_map    Device array of build row indices [num_output].
/// @param num_output     Number of output rows.
/// @param output_row_width  Total output row width (must be >= probe + build widths, 8-byte aligned).
/// @param d_out_buffer   Pre-allocated output buffer [num_output * output_row_width].
/// @param stream         CUDA stream.
void gatherAndConcatRows(
    const GpuFixedRowStore& probeStore,
    const int32_t* d_probe_map,
    const GpuFixedRowStore& buildStore,
    const int32_t* d_build_map,
    int32_t num_output,
    int32_t output_row_width,
    uint8_t* d_out_buffer,
    cudaStream_t stream);

/// Selective gather + concatenate: gather only the specified fields from two
/// source row stores into output rows. Each output row contains only the
/// columns listed in the FieldMapping arrays, laid out at the specified
/// destination offsets. Column order in the output follows outputType.
///
/// @param probeStore         Source row store for probe (left) side.
/// @param d_probe_map        Device array of probe row indices [num_output].
/// @param d_probe_mappings   Device array of FieldMapping for probe columns.
/// @param num_probe_mappings Number of probe fields to gather.
/// @param buildStore         Source row store for build (right) side.
/// @param d_build_map        Device array of build row indices [num_output].
/// @param d_build_mappings   Device array of FieldMapping for build columns.
/// @param num_build_mappings Number of build fields to gather.
/// @param num_output         Number of output rows.
/// @param output_row_width   Total output row width (8-byte aligned).
/// @param d_out_buffer       Pre-allocated output buffer [num_output * output_row_width].
/// @param stream             CUDA stream.
/// Null-sidecar arguments for the selective gather (2026-08-17 null
/// support). All pointers optional: out_null_bytes == nullptr disables the
/// null path entirely; per-side src pointers may independently be null for
/// null-free stores. *_src_field / *_dst_field are device arrays parallel to
/// the FieldMapping arrays, giving each mapping's source and destination
/// FIELD INDEX (sidecar bits are addressed by field index, not byte offset).
struct NullGatherArgs {
  const uint8_t* probe_null_bytes = nullptr;
  int32_t probe_null_stride = 0;
  const uint8_t* build_null_bytes = nullptr;
  int32_t build_null_stride = 0;
  const int32_t* probe_src_field = nullptr;
  const int32_t* probe_dst_field = nullptr;
  const int32_t* build_src_field = nullptr;
  const int32_t* build_dst_field = nullptr;
  uint8_t* out_null_bytes = nullptr;
  int32_t out_null_stride = 0;
};

/// Sidecar -> Arrow validity mask for one column (mask bit set = valid).
void sidecarToMask(
    const uint8_t* d_null_bytes,
    int32_t null_stride,
    int32_t field_idx,
    int32_t num_rows,
    uint32_t* d_mask_words,
    cudaStream_t stream);

void selectiveGatherAndConcat(
    const GpuFixedRowStore& probeStore,
    const int32_t* d_probe_map,
    const FieldMapping* d_probe_mappings,
    int32_t num_probe_mappings,
    const GpuFixedRowStore& buildStore,
    const int32_t* d_build_map,
    const FieldMapping* d_build_mappings,
    int32_t num_build_mappings,
    int32_t num_output,
    int32_t output_row_width,
    uint8_t* d_out_buffer,
    cudaStream_t stream,
    const int32_t* d_probe_src_field = nullptr,
    const int32_t* d_probe_dst_field = nullptr,
    const int32_t* d_build_src_field = nullptr,
    const int32_t* d_build_dst_field = nullptr,
    uint8_t* d_out_null_bytes = nullptr,
    int32_t out_null_stride = 0);

/// Transpose columnar data into a fixed-stride row buffer on GPU.
void columnsToRows(
    const uint8_t* const* d_col_ptrs,
    const FieldDesc* field_descs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    uint8_t* d_row_buffer,
    cudaStream_t stream);

/// Transpose a fixed-stride row buffer back into columnar device buffers
/// (reverse of columnsToRows). Each d_col_ptrs[i] must point to a
/// pre-allocated device buffer of at least num_rows * field_descs[i].byte_width
/// bytes. Used at the last row-mode join to hand columnar data to downstream
/// cudf operators (aggregation, orderBy, ...).
void rowsToColumns(
    const uint8_t* d_row_buffer,
    const FieldDesc* field_descs,
    uint8_t* const* d_col_ptrs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    cudaStream_t stream);

// ============================================================================
// Out-of-line strings (2026-08-21) -- see GpuFixedRowStore.h for the slot
// format. All functions are stream-ordered; the two *Layout/*Offsets helpers
// synchronize the stream to return the heap total.
// ============================================================================

/// One string field of a gathered output row and which source heap it came
/// from (0 = probe, 1 = build).
struct StringGatherField {
  int32_t dst_offset;
  int32_t heap;
};

/// Add delta to every out-of-line offset of the listed string fields.
void rebaseStringOffsets(
    uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    const int32_t* d_str_field_offsets,
    int32_t num_str_fields,
    int64_t delta,
    cudaStream_t stream);

/// cudf strings column (offsets + chars) -> 16B slots in a row buffer; the
/// column's chars are assumed copied into the combined heap such that
/// slot offset = absolute column offset + chars_delta.
void stringsToSlots(
    const void* d_offsets,
    bool offsets_are_int64,
    const uint8_t* d_chars,
    int64_t chars_delta,
    int32_t num_rows,
    uint8_t* d_rows,
    int32_t row_width,
    int32_t field_offset,
    cudaStream_t stream);

/// Per-row heap base for compaction (d_row_base has num_rows + 1 entries);
/// returns the total heap bytes needed. Synchronizes.
int64_t stringHeapLayout(
    const uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    const int32_t* d_str_field_offsets,
    int32_t num_str_fields,
    int64_t* d_row_base,
    cudaStream_t stream);

/// Copy out-of-line bytes of the listed fields into d_out_chars and rewrite
/// the slots' offsets (run after the fixed gather copied slots verbatim).
void compactStrings(
    uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    const StringGatherField* d_fields,
    int32_t num_fields,
    const uint8_t* d_heap0,
    const uint8_t* d_heap1,
    const int64_t* d_row_base,
    uint8_t* d_out_chars,
    cudaStream_t stream);

/// One string field -> exclusive-scanned int64 offsets (num_rows + 1); returns
/// total chars. Synchronizes.
int64_t stringFieldOffsets(
    const uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    int64_t* d_offsets64,
    cudaStream_t stream);

/// Materialize one string field: writes int32 offsets (num_rows + 1) and the
/// chars, from inline slots or the heap.
void stringFieldToChars(
    const uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    const uint8_t* d_heap,
    const int64_t* d_offsets64,
    int32_t* d_offsets32,
    uint8_t* d_out_chars,
    cudaStream_t stream);
