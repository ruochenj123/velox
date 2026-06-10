/*
 * GpuRowOps.cu
 *
 * CUDA kernels for fixed-stride row store operations (no-strings variant).
 * - extract_keys_kernel:         extract key column from row buffer
 * - gather_rows_warp_kernel:     warp-collaborative row gather
 * - columns_to_rows_kernel:      transpose columnar data to row layout
 */

#include "GpuRowOps.cuh"

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

  // Byte-wise copy to handle potentially misaligned row offsets
  for (int b = 0; b < key_width; b++) {
    dst[b] = src[b];
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
// Kernel: warp-collaborative row gather (fixed-stride, no strings)
// ============================================================================

__global__ void gather_rows_warp_kernel(
    const int32_t* __restrict__ gather_map,
    const uint8_t* __restrict__ src_row_buffer,
    uint8_t* __restrict__ dst_row_buffer,
    int32_t row_width,
    int32_t num_output) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  if (warp_id >= num_output) return;

  int32_t src_row = gather_map[warp_id];
  const uint8_t* src = src_row_buffer + (int64_t)src_row * row_width;
  uint8_t* dst = dst_row_buffer + (int64_t)warp_id * row_width;

  for (int i = lane; i < row_width; i += 32) {
    dst[i] = src[i];
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
  int warps_per_block = block / 32;
  int grid = (num_output + warps_per_block - 1) / warps_per_block;
  gather_rows_warp_kernel<<<grid, block, 0, stream>>>(
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
    // Use byte-wise copy to avoid alignment issues with packed row layout
    int w = cols[c].byte_width;
    for (int b = 0; b < w; b++) {
      dst[b] = src[b];
    }
  }
}

/// Transpose a cudf table_view (columnar GPU data) into a row buffer on GPU.
/// Caller must provide:
///   - d_row_buffer: pre-allocated [num_rows * row_width] bytes
///   - field_descs: host-side FieldDesc array describing the row layout
/// The column order in table_view must match field_descs order.
void columnsToRows(
    const uint8_t* const* d_col_ptrs,
    const FieldDesc* field_descs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    uint8_t* d_row_buffer,
    cudaStream_t stream) {
  if (num_rows == 0 || num_cols == 0) return;

  // Build ColDesc array on host, upload to device
  ColDesc* h_cols = new ColDesc[num_cols];
  for (int i = 0; i < num_cols; i++) {
    h_cols[i].data = d_col_ptrs[i];
    h_cols[i].offset = field_descs[i].offset;
    h_cols[i].byte_width = field_descs[i].byte_width;
  }

  ColDesc* d_cols;
  cudaMalloc(&d_cols, num_cols * sizeof(ColDesc));
  cudaMemcpyAsync(d_cols, h_cols, num_cols * sizeof(ColDesc),
      cudaMemcpyHostToDevice, stream);
  delete[] h_cols;

  int block = 256;
  int grid = (num_rows + block - 1) / block;
  columns_to_rows_kernel<<<grid, block, 0, stream>>>(
      d_cols, num_cols, d_row_buffer, row_width, num_rows);

  // Free temporary — schedule on stream so it happens after kernel
  // We use cudaFreeAsync if available, otherwise sync + free
  cudaStreamSynchronize(stream);
  cudaFree(d_cols);
}
