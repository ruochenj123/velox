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

/// Transpose columnar data into a fixed-stride row buffer on GPU.
void columnsToRows(
    const uint8_t* const* d_col_ptrs,
    const FieldDesc* field_descs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    uint8_t* d_row_buffer,
    cudaStream_t stream);

