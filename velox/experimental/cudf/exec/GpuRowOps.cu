/*
 * GpuRowOps.cu
 *
 * CUDA kernels for fixed-stride row store operations (fixed-width + 16B string slots).
 * - extract_keys_kernel:         extract key column from row buffer
 * - gather_rows_vectorized:      vectorized row gather (uint4 loads/stores)
 * - columns_to_rows_kernel:      transpose columnar data to row layout
 *
 * Optimized: vectorized memory access, pre-allocated device descriptors,
 *            thread-per-row for short rows, uint4 bulk copies.
 */

#include "GpuRowOps.cuh"
#include <cub/device/device_scan.cuh>
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

  // Use wide copy for common key widths. The row pack is TIGHT (no field
  // alignment — see CudfFromVelox's row-mode layout), so a key can sit at a
  // misaligned offset (e.g. an 8-byte computed key after a 4-byte column);
  // guard the wide loads exactly like copy_field does. dst is always
  // naturally aligned (idx * key_width).
  if (key_width == 8) {
    if ((reinterpret_cast<uintptr_t>(src) & 7) == 0) {
      *reinterpret_cast<uint64_t*>(dst) =
          *reinterpret_cast<const uint64_t*>(src);
    } else if ((reinterpret_cast<uintptr_t>(src) & 3) == 0) {
      const uint32_t lo = *reinterpret_cast<const uint32_t*>(src);
      const uint32_t hi = *reinterpret_cast<const uint32_t*>(src + 4);
      *reinterpret_cast<uint64_t*>(dst) =
          static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    } else {
      for (int b = 0; b < 8; b++) {
        dst[b] = src[b];
      }
    }
  } else if (key_width == 4) {
    if ((reinterpret_cast<uintptr_t>(src) & 3) == 0) {
      *reinterpret_cast<uint32_t*>(dst) =
          *reinterpret_cast<const uint32_t*>(src);
    } else {
      for (int b = 0; b < 4; b++) {
        dst[b] = src[b];
      }
    }
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
// Kernel: fused gather + concatenate from two row stores into one output row
// ============================================================================

/// Each thread handles one output row. It copies the probe portion first,
/// then the build portion, into a contiguous output row of width
/// (probe_row_width + build_row_width), padded to 8-byte alignment.
__global__ void gather_concat_rows_kernel(
    const int32_t* __restrict__ probe_map,
    const uint8_t* __restrict__ probe_src,
    int32_t probe_row_width,
    const int32_t* __restrict__ build_map,
    const uint8_t* __restrict__ build_src,
    int32_t build_row_width,
    uint8_t* __restrict__ dst,
    int32_t output_row_width,
    int32_t num_output) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_output) return;

  uint8_t* out_row = dst + (int64_t)idx * output_row_width;

  // Copy probe portion
  {
    int32_t src_row = probe_map[idx];
    const uint8_t* src = probe_src + (int64_t)src_row * probe_row_width;
    int offset = 0;
    int n8 = probe_row_width >> 3;
    for (int i = 0; i < n8; i++) {
      reinterpret_cast<uint2*>(out_row + offset)[0] =
          reinterpret_cast<const uint2*>(src + offset)[0];
      offset += 8;
    }
    int remaining = probe_row_width - (n8 << 3);
    if (remaining >= 4) {
      *reinterpret_cast<uint32_t*>(out_row + offset) =
          *reinterpret_cast<const uint32_t*>(src + offset);
      offset += 4;
      remaining -= 4;
    }
    for (int i = 0; i < remaining; i++) {
      out_row[offset + i] = src[offset + i];
    }
  }

  // Copy build portion (offset by probe_row_width in output)
  {
    int32_t src_row = build_map[idx];
    const uint8_t* src = build_src + (int64_t)src_row * build_row_width;
    uint8_t* out_build = out_row + probe_row_width;
    int offset = 0;
    int n8 = build_row_width >> 3;
    for (int i = 0; i < n8; i++) {
      reinterpret_cast<uint2*>(out_build + offset)[0] =
          reinterpret_cast<const uint2*>(src + offset)[0];
      offset += 8;
    }
    int remaining = build_row_width - (n8 << 3);
    if (remaining >= 4) {
      *reinterpret_cast<uint32_t*>(out_build + offset) =
          *reinterpret_cast<const uint32_t*>(src + offset);
      offset += 4;
      remaining -= 4;
    }
    for (int i = 0; i < remaining; i++) {
      out_build[offset + i] = src[offset + i];
    }
  }
}

void gatherAndConcatRows(
    const GpuFixedRowStore& probeStore,
    const int32_t* d_probe_map,
    const GpuFixedRowStore& buildStore,
    const int32_t* d_build_map,
    int32_t num_output,
    int32_t output_row_width,
    uint8_t* d_out_buffer,
    cudaStream_t stream) {
  if (num_output == 0) return;

  int block = 256;
  int grid = (num_output + block - 1) / block;
  gather_concat_rows_kernel<<<grid, block, 0, stream>>>(
      d_probe_map,
      probeStore.row_buffer,
      probeStore.row_width,
      d_build_map,
      buildStore.row_buffer,
      buildStore.row_width,
      d_out_buffer,
      output_row_width,
      num_output);
}

// ============================================================================
// Device helper: copy a single field with alignment-aware loads/stores
// ============================================================================

__device__ __forceinline__ void copy_field(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int32_t width) {
  if (width == 16) {
    if (((reinterpret_cast<uintptr_t>(src) |
          reinterpret_cast<uintptr_t>(dst)) & 15) == 0) {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      return;
    }
    if (((reinterpret_cast<uintptr_t>(src) |
          reinterpret_cast<uintptr_t>(dst)) & 7) == 0) {
      reinterpret_cast<uint64_t*>(dst)[0] =
          reinterpret_cast<const uint64_t*>(src)[0];
      reinterpret_cast<uint64_t*>(dst)[1] =
          reinterpret_cast<const uint64_t*>(src)[1];
      return;
    }
  }
  if (width == 8) {
    if (((reinterpret_cast<uintptr_t>(src) |
          reinterpret_cast<uintptr_t>(dst)) & 7) == 0) {
      *reinterpret_cast<uint64_t*>(dst) =
          *reinterpret_cast<const uint64_t*>(src);
      return;
    }
  }
  if (width == 4) {
    if (((reinterpret_cast<uintptr_t>(src) |
          reinterpret_cast<uintptr_t>(dst)) & 3) == 0) {
      *reinterpret_cast<uint32_t*>(dst) =
          *reinterpret_cast<const uint32_t*>(src);
      return;
    }
  }
  if (width == 2) {
    if (((reinterpret_cast<uintptr_t>(src) |
          reinterpret_cast<uintptr_t>(dst)) & 1) == 0) {
      *reinterpret_cast<uint16_t*>(dst) =
          *reinterpret_cast<const uint16_t*>(src);
      return;
    }
  }
  for (int b = 0; b < width; b++) {
    dst[b] = src[b];
  }
}

// ============================================================================
// Kernel: selective gather + concatenate from two row stores
// Only gathers the fields specified by FieldMapping arrays.
// ============================================================================

__global__ void selective_gather_concat_kernel(
    const int32_t* __restrict__ probe_map,
    const uint8_t* __restrict__ probe_src,
    int32_t probe_row_width,
    const FieldMapping* __restrict__ probe_mappings,
    int32_t num_probe_mappings,
    const int32_t* __restrict__ build_map,
    const uint8_t* __restrict__ build_src,
    int32_t build_row_width,
    const FieldMapping* __restrict__ build_mappings,
    int32_t num_build_mappings,
    uint8_t* __restrict__ dst,
    int32_t output_row_width,
    int32_t num_output,
    // ---- null sidecar (2026-08-17): all nullable-path params optional ----
    NullGatherArgs nulls) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_output) return;

  uint8_t* out_row = dst + (int64_t)idx * output_row_width;
  uint8_t* out_nb = nulls.out_null_bytes == nullptr
      ? nullptr
      : nulls.out_null_bytes + (int64_t)idx * nulls.out_null_stride;

  // Gather selected probe fields
  {
    int32_t src_row_idx = probe_map[idx];
    const uint8_t* src_row =
        probe_src + (int64_t)src_row_idx * probe_row_width;
    for (int f = 0; f < num_probe_mappings; f++) {
      copy_field(
          src_row + probe_mappings[f].src_offset,
          out_row + probe_mappings[f].dst_offset,
          probe_mappings[f].byte_width);
    }
    if (out_nb != nullptr && nulls.probe_null_bytes != nullptr) {
      const uint8_t* src_nb =
          nulls.probe_null_bytes + (int64_t)src_row_idx * nulls.probe_null_stride;
      for (int f = 0; f < num_probe_mappings; f++) {
        const int32_t sf = nulls.probe_src_field[f];
        if (src_nb[sf >> 3] & (1u << (sf & 7))) {
          const int32_t df = nulls.probe_dst_field[f];
          out_nb[df >> 3] |= (1u << (df & 7));
        }
      }
    }
  }

  // Gather selected build fields
  {
    int32_t src_row_idx = build_map[idx];
    const uint8_t* src_row =
        build_src + (int64_t)src_row_idx * build_row_width;
    for (int f = 0; f < num_build_mappings; f++) {
      copy_field(
          src_row + build_mappings[f].src_offset,
          out_row + build_mappings[f].dst_offset,
          build_mappings[f].byte_width);
    }
    if (out_nb != nullptr && nulls.build_null_bytes != nullptr) {
      const uint8_t* src_nb =
          nulls.build_null_bytes + (int64_t)src_row_idx * nulls.build_null_stride;
      for (int f = 0; f < num_build_mappings; f++) {
        const int32_t sf = nulls.build_src_field[f];
        if (src_nb[sf >> 3] & (1u << (sf & 7))) {
          const int32_t df = nulls.build_dst_field[f];
          out_nb[df >> 3] |= (1u << (df & 7));
        }
      }
    }
  }
}

// Sidecar -> Arrow-style validity mask for one output column: mask bit SET =
// VALID = sidecar bit CLEAR. Word-per-thread; trailing bits of the last word
// are left set (cudf ignores bits past num_rows).
__global__ void sidecar_to_mask_kernel(
    const uint8_t* __restrict__ null_bytes,
    int32_t null_stride,
    int32_t field_idx,
    int32_t num_rows,
    uint32_t* __restrict__ mask_words) {
  int w = blockIdx.x * blockDim.x + threadIdx.x;
  int num_words = (num_rows + 31) / 32;
  if (w >= num_words) return;
  uint32_t out = 0xffffffffu;
  const int base = w * 32;
  const int n = min(32, num_rows - base);
  for (int i = 0; i < n; i++) {
    const uint8_t nb = null_bytes[(int64_t)(base + i) * null_stride +
                                  (field_idx >> 3)];
    if (nb & (1u << (field_idx & 7))) {
      out &= ~(1u << i);
    }
  }
  mask_words[w] = out;
}

void sidecarToMask(
    const uint8_t* d_null_bytes,
    int32_t null_stride,
    int32_t field_idx,
    int32_t num_rows,
    uint32_t* d_mask_words,
    cudaStream_t stream) {
  if (num_rows == 0) return;
  int num_words = (num_rows + 31) / 32;
  int block = 256;
  int grid = (num_words + block - 1) / block;
  sidecar_to_mask_kernel<<<grid, block, 0, stream>>>(
      d_null_bytes, null_stride, field_idx, num_rows, d_mask_words);
}

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
    const int32_t* d_probe_src_field,
    const int32_t* d_probe_dst_field,
    const int32_t* d_build_src_field,
    const int32_t* d_build_dst_field,
    uint8_t* d_out_null_bytes,
    int32_t out_null_stride) {
  if (num_output == 0) return;

  int block = 256;
  int grid = (num_output + block - 1) / block;
  NullGatherArgs nulls{};
  nulls.probe_null_bytes = probeStore.null_bytes;
  nulls.probe_null_stride = probeStore.null_stride;
  nulls.build_null_bytes = buildStore.null_bytes;
  nulls.build_null_stride = buildStore.null_stride;
  nulls.probe_src_field = d_probe_src_field;
  nulls.probe_dst_field = d_probe_dst_field;
  nulls.build_src_field = d_build_src_field;
  nulls.build_dst_field = d_build_dst_field;
  nulls.out_null_bytes = d_out_null_bytes;
  nulls.out_null_stride = out_null_stride;
  selective_gather_concat_kernel<<<grid, block, 0, stream>>>(
      d_probe_map,
      probeStore.row_buffer,
      probeStore.row_width,
      d_probe_mappings,
      num_probe_mappings,
      d_build_map,
      buildStore.row_buffer,
      buildStore.row_width,
      d_build_mappings,
      num_build_mappings,
      d_out_buffer,
      output_row_width,
      num_output,
      nulls);
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

// ============================================================================
// Kernel: transpose fixed-stride row buffer back into columnar layout on GPU
// (reverse of columns_to_rows_kernel). Used at the last row-mode join to
// produce cudf columns consumable by downstream columnar operators.
// ============================================================================

__global__ void rows_to_columns_kernel(
    const uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    const ColDesc* __restrict__ cols,
    int32_t num_cols,
    int32_t num_rows) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  const uint8_t* src_row = row_buffer + (int64_t)row * row_width;
  for (int c = 0; c < num_cols; c++) {
    const uint8_t* src = src_row + cols[c].offset;
    // cols[c].data is the destination column base (naturally aligned).
    uint8_t* dst = const_cast<uint8_t*>(cols[c].data) +
        (int64_t)row * cols[c].byte_width;
    int w = cols[c].byte_width;
    // Source alignment within the row is not guaranteed; destination is.
    if (w == 8 && ((reinterpret_cast<uintptr_t>(src) & 7) == 0)) {
      *reinterpret_cast<uint64_t*>(dst) =
          *reinterpret_cast<const uint64_t*>(src);
    } else if (w == 4 && ((reinterpret_cast<uintptr_t>(src) & 3) == 0)) {
      *reinterpret_cast<uint32_t*>(dst) =
          *reinterpret_cast<const uint32_t*>(src);
    } else if (w == 2 && ((reinterpret_cast<uintptr_t>(src) & 1) == 0)) {
      *reinterpret_cast<uint16_t*>(dst) =
          *reinterpret_cast<const uint16_t*>(src);
    } else {
      for (int b = 0; b < w; b++) {
        dst[b] = src[b];
      }
    }
  }
}

/// Transpose a fixed-stride row buffer into columnar device buffers.
/// d_col_ptrs[i] must point to a pre-allocated device buffer of at least
/// num_rows * field_descs[i].byte_width bytes.
void rowsToColumns(
    const uint8_t* d_row_buffer,
    const FieldDesc* field_descs,
    uint8_t* const* d_col_ptrs,
    int32_t num_cols,
    int32_t num_rows,
    int32_t row_width,
    cudaStream_t stream) {
  if (num_rows == 0 || num_cols == 0) return;

  // Build ColDesc array on thread-local stack (no heap alloc, no race)
  for (int i = 0; i < num_cols; i++) {
    tl_host_col_desc[i].data = d_col_ptrs[i];
    tl_host_col_desc[i].offset = field_descs[i].offset;
    tl_host_col_desc[i].byte_width = field_descs[i].byte_width;
  }

  ColDesc* d_cols;
  cudaMallocAsync(&d_cols, num_cols * sizeof(ColDesc), stream);
  cudaMemcpyAsync(d_cols, tl_host_col_desc, num_cols * sizeof(ColDesc),
      cudaMemcpyHostToDevice, stream);

  int block = 256;
  int grid = (num_rows + block - 1) / block;
  rows_to_columns_kernel<<<grid, block, 0, stream>>>(
      d_row_buffer, row_width, d_cols, num_cols, num_rows);

  cudaFreeAsync(d_cols, stream);
}


// ============================================================================
// Out-of-line strings (2026-08-21). Slot format: GpuFixedRowStore.h.
// ============================================================================

__device__ __forceinline__ uint32_t slot_len(const uint8_t* slot) {
  uint32_t v;
  memcpy(&v, slot, 4);
  return v;
}
__device__ __forceinline__ uint64_t slot_off(const uint8_t* slot) {
  uint64_t v;
  memcpy(&v, slot + 8, 8);
  return v;
}
__device__ __forceinline__ void slot_set_off(uint8_t* slot, uint64_t off) {
  memcpy(slot + 8, &off, 8);
}

// Add `delta` to every out-of-line offset of the given string fields, for
// rows [0, num_rows) of `rows`. Used when heaps are appended (concat/build).
__global__ void rebase_string_offsets_kernel(
    uint8_t* __restrict__ rows,
    int32_t num_rows,
    int32_t row_width,
    const int32_t* __restrict__ str_field_offsets,
    int32_t num_str_fields,
    int64_t delta) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  uint8_t* r = rows + (int64_t)row * row_width;
  for (int f = 0; f < num_str_fields; f++) {
    uint8_t* slot = r + str_field_offsets[f];
    if (slot_len(slot) > kRowStrInlineMax) {
      slot_set_off(slot, slot_off(slot) + (uint64_t)delta);
    }
  }
}

void rebaseStringOffsets(
    uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    const int32_t* d_str_field_offsets,
    int32_t num_str_fields,
    int64_t delta,
    cudaStream_t stream) {
  if (num_rows == 0 || num_str_fields == 0 || delta == 0) return;
  int block = 256;
  int grid = (num_rows + block - 1) / block;
  rebase_string_offsets_kernel<<<grid, block, 0, stream>>>(
      d_rows, num_rows, row_width, d_str_field_offsets, num_str_fields, delta);
}

// cudf strings column -> slots. offsets may be int32 or int64 (cudf >= 24.x
// large strings). chars_delta = where this column's chars were copied in the
// combined heap MINUS the column's first absolute offset.
template <typename OffsetT>
__global__ void strings_to_slots_kernel(
    const OffsetT* __restrict__ offsets,
    const uint8_t* __restrict__ chars,
    int64_t chars_delta,
    int32_t num_rows,
    uint8_t* __restrict__ rows,
    int32_t row_width,
    int32_t field_offset) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  const int64_t b = (int64_t)offsets[row];
  const int64_t e = (int64_t)offsets[row + 1];
  const uint32_t len = (uint32_t)(e - b);
  uint8_t* slot = rows + (int64_t)row * row_width + field_offset;
  uint8_t tmp[kRowStrSlotBytes];
  memcpy(tmp, &len, 4);
  if (len <= kRowStrInlineMax) {
    for (uint32_t i = 0; i < kRowStrInlineMax; i++) {
      tmp[4 + i] = i < len ? chars[b + i] : 0;
    }
  } else {
    for (int i = 0; i < 4; i++) {
      tmp[4 + i] = chars[b + i];
    }
    const uint64_t off = (uint64_t)(b + chars_delta);
    memcpy(tmp + 8, &off, 8);
  }
  memcpy(slot, tmp, kRowStrSlotBytes);
}

void stringsToSlots(
    const void* d_offsets,
    bool offsets_are_int64,
    const uint8_t* d_chars,
    int64_t chars_delta,
    int32_t num_rows,
    uint8_t* d_rows,
    int32_t row_width,
    int32_t field_offset,
    cudaStream_t stream) {
  if (num_rows == 0) return;
  int block = 256;
  int grid = (num_rows + block - 1) / block;
  if (offsets_are_int64) {
    strings_to_slots_kernel<int64_t><<<grid, block, 0, stream>>>(
        static_cast<const int64_t*>(d_offsets), d_chars, chars_delta,
        num_rows, d_rows, row_width, field_offset);
  } else {
    strings_to_slots_kernel<int32_t><<<grid, block, 0, stream>>>(
        static_cast<const int32_t*>(d_offsets), d_chars, chars_delta,
        num_rows, d_rows, row_width, field_offset);
  }
}

// Per-row out-of-line byte count over a set of string fields (sizes the
// compaction heap). Output has num_rows + 1 entries; the extra is 0 so an
// exclusive scan yields the total in the last slot.
__global__ void string_out_of_line_bytes_kernel(
    const uint8_t* __restrict__ rows,
    int32_t num_rows,
    int32_t row_width,
    const int32_t* __restrict__ str_field_offsets,
    int32_t num_str_fields,
    int64_t* __restrict__ out_bytes) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > num_rows) return;
  if (row == num_rows) {
    out_bytes[row] = 0;
    return;
  }
  const uint8_t* r = rows + (int64_t)row * row_width;
  int64_t sum = 0;
  for (int f = 0; f < num_str_fields; f++) {
    const uint32_t len = slot_len(r + str_field_offsets[f]);
    if (len > kRowStrInlineMax) sum += len;
  }
  out_bytes[row] = sum;
}

// Exclusive scan in place over num_rows + 1 int64 entries; returns the total
// (synchronizes the stream).
static int64_t exclusive_scan_total(
    int64_t* d_vals, int32_t n_plus_1, cudaStream_t stream) {
  void* tmp = nullptr;
  size_t tmpBytes = 0;
  cub::DeviceScan::ExclusiveSum(tmp, tmpBytes, d_vals, d_vals, n_plus_1, stream);
  cudaMallocAsync(&tmp, tmpBytes, stream);
  cub::DeviceScan::ExclusiveSum(tmp, tmpBytes, d_vals, d_vals, n_plus_1, stream);
  cudaFreeAsync(tmp, stream);
  int64_t total = 0;
  cudaMemcpyAsync(&total, d_vals + (n_plus_1 - 1), sizeof(int64_t),
      cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return total;
}

// Slots of one string field -> cudf strings column pieces: lengths (num_rows
// + 1, exclusive-scanned by the host helper into int32 offsets) then chars.
__global__ void string_field_lengths_kernel(
    const uint8_t* __restrict__ rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    int64_t* __restrict__ out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > num_rows) return;
  out[row] = row == num_rows
      ? 0
      : (int64_t)slot_len(rows + (int64_t)row * row_width + field_offset);
}

__global__ void string_field_to_chars_kernel(
    const uint8_t* __restrict__ rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    const uint8_t* __restrict__ heap,
    const int64_t* __restrict__ offsets64,
    int32_t* __restrict__ offsets32,
    uint8_t* __restrict__ out_chars) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row > num_rows) return;
  offsets32[row] = (int32_t)offsets64[row];
  if (row == num_rows) return;
  const uint8_t* slot = rows + (int64_t)row * row_width + field_offset;
  const uint32_t len = slot_len(slot);
  const uint8_t* src = len <= kRowStrInlineMax ? slot + 4 : heap + slot_off(slot);
  uint8_t* dst = out_chars + offsets64[row];
  for (uint32_t i = 0; i < len; i++) dst[i] = src[i];
}

int64_t stringHeapLayout(
    const uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    const int32_t* d_str_field_offsets,
    int32_t num_str_fields,
    int64_t* d_row_base,
    cudaStream_t stream) {
  if (num_str_fields == 0) return 0;
  int block = 256;
  int grid = (num_rows + 1 + block - 1) / block;
  string_out_of_line_bytes_kernel<<<grid, block, 0, stream>>>(
      d_rows, num_rows, row_width, d_str_field_offsets, num_str_fields,
      d_row_base);
  return exclusive_scan_total(d_row_base, num_rows + 1, stream);
}

// Compaction: for each output row, copy the out-of-line bytes of every
// string field from its source heap (per field: probe or build) into the
// fresh heap at row_base[row], rewriting the slot offset. Slots were copied
// verbatim by the fixed gather; inline slots need nothing.
__global__ void compact_strings_kernel(
    uint8_t* __restrict__ rows,
    int32_t num_rows,
    int32_t row_width,
    const StringGatherField* __restrict__ fields,
    int32_t num_fields,
    const uint8_t* __restrict__ heap0,
    const uint8_t* __restrict__ heap1,
    const int64_t* __restrict__ row_base,
    uint8_t* __restrict__ out_chars) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  uint8_t* r = rows + (int64_t)row * row_width;
  int64_t cursor = row_base[row];
  for (int f = 0; f < num_fields; f++) {
    uint8_t* slot = r + fields[f].dst_offset;
    const uint32_t len = slot_len(slot);
    if (len <= kRowStrInlineMax) continue;
    const uint8_t* src =
        (fields[f].heap == 0 ? heap0 : heap1) + slot_off(slot);
    uint8_t* dst = out_chars + cursor;
    uint32_t i = 0;
    if ((((uintptr_t)src | (uintptr_t)dst) & 7) == 0) {
      for (; i + 8 <= len; i += 8) {
        *reinterpret_cast<uint64_t*>(dst + i) =
            *reinterpret_cast<const uint64_t*>(src + i);
      }
    }
    for (; i < len; i++) dst[i] = src[i];
    slot_set_off(slot, (uint64_t)cursor);
    cursor += len;
  }
}

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
    cudaStream_t stream) {
  if (num_rows == 0 || num_fields == 0) return;
  int block = 256;
  int grid = (num_rows + block - 1) / block;
  compact_strings_kernel<<<grid, block, 0, stream>>>(
      d_rows, num_rows, row_width, d_fields, num_fields, d_heap0, d_heap1,
      d_row_base, d_out_chars);
}

int64_t stringFieldOffsets(
    const uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    int64_t* d_offsets64,
    cudaStream_t stream) {
  int block = 256;
  int grid = (num_rows + 1 + block - 1) / block;
  string_field_lengths_kernel<<<grid, block, 0, stream>>>(
      d_rows, num_rows, row_width, field_offset, d_offsets64);
  return exclusive_scan_total(d_offsets64, num_rows + 1, stream);
}

void stringFieldToChars(
    const uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    const uint8_t* d_heap,
    const int64_t* d_offsets64,
    int32_t* d_offsets32,
    uint8_t* d_out_chars,
    cudaStream_t stream) {
  int block = 256;
  int grid = (num_rows + 1 + block - 1) / block;
  string_field_to_chars_kernel<<<grid, block, 0, stream>>>(
      d_rows, num_rows, row_width, field_offset, d_heap, d_offsets64,
      d_offsets32, d_out_chars);
}

// ============================================================================
// Row-wise sort support (2026-08-24, branch row-sort)
// ============================================================================
__global__ void add_int64_field_kernel(
    uint8_t* __restrict__ rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    int64_t delta) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  uint8_t* p = rows + (int64_t)row * row_width + field_offset;
  int64_t v;
  memcpy(&v, p, 8);
  v += delta;
  memcpy(p, &v, 8);
}

void addInt64Field(
    uint8_t* d_rows,
    int32_t num_rows,
    int32_t row_width,
    int32_t field_offset,
    int64_t delta,
    cudaStream_t stream) {
  if (num_rows == 0 || delta == 0) return;
  int block = 256;
  int grid = (num_rows + block - 1) / block;
  add_int64_field_kernel<<<grid, block, 0, stream>>>(
      d_rows, num_rows, row_width, field_offset, delta);
}
