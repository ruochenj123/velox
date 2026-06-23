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
    cudaStream_t stream);

/// Transpose columnar data into a fixed-stride row buffer on GPU.
void columnsToRows(
    const uint8_t* const* d_col_ptrs,
    const FieldDesc* field_descs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    uint8_t* d_row_buffer,
    cudaStream_t stream);

