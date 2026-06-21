/*
 * GpuRowOps.cu
 *
 * CUDA kernels for fixed-stride row store operations (no-strings variant).
 * - extract_keys_kernel:         extract key column from row buffer
 * - gather_rows_vectorized:      vectorized row gather (uint4 loads/stores)
 * - columns_to_rows_kernel:      transpose columnar data to row layout
 *
 * Optimized: vectorized memory access, pre-allocated device descriptors,
 *            thread-per-row for short rows, uint4 bulk copies.
 */

#include "GpuRowOps.cuh"
#include <cuda_runtime.h>

// ============================================================================
// Kernel: extract a fixed-width key column from row buffer
// ============================================================================

__global__ void extract_keys_kernel(
    const uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    int32_t key_offset,
    int32_t key_width,
    uint8_t* __restrict__ d_out,
    int32_t num_rows) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_rows) return;

  const uint8_t* src = row_buffer + (int64_t)idx * row_width + key_offset;
  uint8_t* dst = d_out + (int64_t)idx * key_width;

  // Use wide copy for common key widths
  if (key_width == 8) {
    *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src);
  } else if (key_width == 4) {
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
  } else {
    for (int b = 0; b < key_width; b++) {
      dst[b] = src[b];
    }
  }
}

void extractKeysFromRows(
    const GpuFixedRowStore& store,
    int32_t key_offset,
    int32_t key_width,
    void* d_out,
    cudaStream_t stream) {
  if (store.num_rows == 0) return;
  int block = 256;
  int grid = (store.num_rows + block - 1) / block;
  extract_keys_kernel<<<grid, block, 0, stream>>>(
      store.row_buffer,
      store.row_width,
      key_offset,
      key_width,
      static_cast<uint8_t*>(d_out),
      store.num_rows);
}

// ============================================================================
// Kernel: vectorized row gather — thread-per-row with uint4/uint2/uint32 bulk copy
// Uses one thread per output row and copies using the widest possible type.
// Row widths are guaranteed 8-byte aligned (from computeRowLayout).
// ============================================================================

__global__ void gather_rows_vectorized_kernel(
    const int32_t* __restrict__ gather_map,
    const uint8_t* __restrict__ src_row_buffer,
    uint8_t* __restrict__ dst_row_buffer,
    int32_t row_width,
    int32_t num_output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_output) return;

  int32_t src_row = gather_map[idx];
  const uint8_t* src = src_row_buffer + (int64_t)src_row * row_width;
  uint8_t* dst = dst_row_buffer + (int64_t)idx * row_width;

  // row_width is guaranteed 8-byte aligned, so use uint2 (8 bytes) as max width.
  // Cannot use uint4 (16B) because row_width may not be a multiple of 16,
  // causing odd-indexed rows to start at non-16-byte-aligned addresses.
  int offset = 0;
  int remaining = row_width;

  // uint2 = 8 bytes (= uint64_t)
  const int n8 = remaining >> 3;  // / 8
  for (int i = 0; i < n8; i++) {
    reinterpret_cast<uint2*>(dst + offset)[0] =
        reinterpret_cast<const uint2*>(src + offset)[0];
    offset += 8;
  }
  remaining -= n8 << 3;

  // uint32 = 4 bytes
  if (remaining >= 4) {
    *reinterpret_cast<uint32_t*>(dst + offset) =
        *reinterpret_cast<const uint32_t*>(src + offset);
    offset += 4;
    remaining -= 4;
  }

  // Remaining bytes (0-3)
  for (int i = 0; i < remaining; i++) {
    dst[offset + i] = src[offset + i];
  }
}

void gatherRowsWarp(
    const GpuFixedRowStore& store,
    const int32_t* d_gather_map,
    int32_t num_output,
    uint8_t* d_out_buffer,
    cudaStream_t stream) {
  if (num_output == 0) return;

  int block = 256;
  int grid = (num_output + block - 1) / block;
  gather_rows_vectorized_kernel<<<grid, block, 0, stream>>>(
      d_gather_map,
      store.row_buffer,
      d_out_buffer,
      store.row_width,
      num_output);
}

// ============================================================================
// Kernel: transpose columnar cudf::table_view into row layout on GPU
// ============================================================================

/// Per-column descriptor for the transpose kernel.
struct ColDesc {
  const uint8_t* data;    // pointer to column data on device
  int32_t offset;         // byte offset within a row
  int32_t byte_width;     // column element width
};

__global__ void columns_to_rows_kernel(
    const ColDesc* __restrict__ cols,
    int32_t num_cols,
    uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    int32_t num_rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  uint8_t* dst_row = row_buffer + (int64_t)row * row_width;
  for (int c = 0; c < num_cols; c++) {
    const uint8_t* src = cols[c].data + (int64_t)row * cols[c].byte_width;
    uint8_t* dst = dst_row + cols[c].offset;
    int w = cols[c].byte_width;
    // Source (cudf columns) is always naturally aligned.
    // Destination alignment depends on field offset within the row.
    // computeRowLayout packs sequentially, so fields may not be naturally aligned.
    if (w == 8 && ((reinterpret_cast<uintptr_t>(dst) & 7) == 0)) {
      *reinterpret_cast<uint64_t*>(dst) =
          *reinterpret_cast<const uint64_t*>(src);
    } else if (w == 4 && ((reinterpret_cast<uintptr_t>(dst) & 3) == 0)) {
      *reinterpret_cast<uint32_t*>(dst) =
          *reinterpret_cast<const uint32_t*>(src);
    } else if (w == 2 && ((reinterpret_cast<uintptr_t>(dst) & 1) == 0)) {
      *reinterpret_cast<uint16_t*>(dst) =
          *reinterpret_cast<const uint16_t*>(src);
    } else {
      for (int b = 0; b < w; b++) {
        dst[b] = src[b];
      }
    }
  }
}

// Thread-local host buffer for ColDesc (avoids heap allocation + race condition)
static thread_local ColDesc tl_host_col_desc[64];

/// Transpose a cudf table_view (columnar GPU data) into a row buffer on GPU.
/// Uses cudaMallocAsync (stream-ordered O(1) pool allocation) to avoid the
/// overhead of synchronous cudaMalloc while remaining thread-safe.
void columnsToRows(
    const uint8_t* const* d_col_ptrs,
    const FieldDesc* field_descs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  if (num_rows == 0 || num_cols == 0) return;

  // Build ColDesc array on thread-local stack (no heap alloc, no race)
  for (int i = 0; i < num_cols; i++) {
    tl_host_col_desc[i].data = d_col_ptrs[i];
    tl_host_col_desc[i].offset = field_descs[i].offset;
    tl_host_col_desc[i].byte_width = field_descs[i].byte_width;
  }

  // Stream-ordered allocation: O(1) from driver pool, no sync
  ColDesc* d_cols;
  cudaMallocAsync(&d_cols, num_cols * sizeof(ColDesc), stream);
  cudaMemcpyAsync(d_cols, tl_host_col_desc, num_cols * sizeof(ColDesc),
      cudaMemcpyHostToDevice, stream);

  int block = 256;
  int grid = (num_rows + block - 1) / block;
  columns_to_rows_kernel<<<grid, block, 0, stream>>>(
      d_cols, num_cols, d_row_buffer, row_width, num_rows);

  // Stream-ordered free: O(1), returns to pool
  cudaFreeAsync(d_cols, stream);
}

