/*
 * GpuRowHashTable.cu
 *
 * Device implementation of the row-native N:M chained hash table
 * (see GpuRowHashTable.cuh for design and semantics).
 *
 * Hash/verify formulas are adapted from GpuFusedProbe.cu (mix64/fnv chain,
 * fingerprint-then-verify) but the table itself is a multimap: head slots +
 * per-build-row next chains, probed with cudf-style count + scan + refill.
 */

#include "GpuRowHashTable.cuh"

#include <cub/device/device_scan.cuh>

#include <cuda_runtime.h>

#include <algorithm>

namespace facebook::velox::cudf_velox {

namespace {

// ---- read a fixed-width value from `p`, zero-extended to uint64 ----
// (same semantics as GpuFusedProbe.cu::loadKeyWord, but alignment-guarded:
// the full-row CPU pack is TIGHT, so a key can sit at a misaligned offset —
// e.g. an 8-byte computed key after a 4-byte column. The guard is a no-op
// branch for aligned layouts.)
__device__ __forceinline__ uint64_t loadKeyWord(
    const uint8_t* p,
    int32_t width) {
  if (width == 8) {
    if ((reinterpret_cast<uintptr_t>(p) & 7) == 0) {
      return *reinterpret_cast<const uint64_t*>(p);
    }
    if ((reinterpret_cast<uintptr_t>(p) & 3) == 0) {
      const uint32_t lo = *reinterpret_cast<const uint32_t*>(p);
      const uint32_t hi = *reinterpret_cast<const uint32_t*>(p + 4);
      return static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    uint64_t v = 0;
    for (int b = 7; b >= 0; b--) {
      v = (v << 8) | p[b];
    }
    return v;
  }
  // width == 4 (INT32 keys)
  if ((reinterpret_cast<uintptr_t>(p) & 3) == 0) {
    return static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(p));
  }
  uint32_t v = 0;
  for (int b = 3; b >= 0; b--) {
    v = (v << 8) | p[b];
  }
  return static_cast<uint64_t>(v);
}

// ---- 64-bit mix (splitmix64 finalizer; identical to GpuFusedProbe.cu) ----
__device__ __forceinline__ uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

// ---- full 64-bit hash of a composite key ----
// Low 32 bits index the slot, high 32 bits are the fingerprint (the same
// low/high split cuco-style fingerprint slots use in the fused path).
__device__ __forceinline__ uint64_t hashKey64(
    const uint64_t* words,
    int32_t numKeys) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int w = 0; w < numKeys; w++) {
    h = mix64(h ^ words[w]);
  }
  return h;
}

// ---- read the key row (contiguous fields) into registers ----
__device__ __forceinline__ void loadKeyRow(
    const uint8_t* row,
    const RowKeyLayout& kl,
    uint64_t* out) {
  for (int k = 0; k < kl.numKeys; k++) {
    out[k] = loadKeyWord(row + kl.offset[k], kl.width[k]);
  }
}

// ---- read the EFFECTIVE key: raw key words, or one range-packed word ----
// With packing enabled, folds the composite key into out[0] and reports the
// effective word count as 1. Returns false iff any component falls outside
// the build side's [min, min+span] range — possible for PROBE keys only
// (build values are in range by construction), in which case the key cannot
// match any build row. Unsigned compare after subtraction: rel > span also
// catches v < min via wraparound.
__device__ __forceinline__ bool loadEffectiveKey(
    const uint8_t* row,
    const RowKeyLayout& kl,
    const RowKeyPack& pk,
    uint64_t* out,
    int32_t& numWords) {
  if (!pk.enabled) {
    loadKeyRow(row, kl, out);
    numWords = kl.numKeys;
    return true;
  }
  uint64_t packed = 0;
  for (int k = 0; k < kl.numKeys; k++) {
    const uint64_t v = loadKeyWord(row + kl.offset[k], kl.width[k]);
    const uint64_t rel = v - pk.mins[k];
    if (rel > pk.spans[k]) {
      return false;
    }
    packed |= rel << pk.shifts[k];
  }
  out[0] = packed;
  numWords = 1;
  return true;
}

// ---- verify the real key against a build key row (fingerprint verify) ----
// Unpacked arm only: packed verifies compare the stored packedKeys word (one
// dense 8-byte load) in the probe kernel, or are skipped outright when the
// packed key fits the 32-bit fingerprint (pack.exactFp).
__device__ __forceinline__ bool keyEqualsBuildRow(
    const uint64_t* probeKey,
    const uint8_t* buildRow,
    const RowKeyLayout& kl) {
  for (int k = 0; k < kl.numKeys; k++) {
    if (loadKeyWord(buildRow + kl.offset[k], kl.width[k]) != probeKey[k]) {
      return false;
    }
  }
  return true;
}

// ---- chain fingerprint for an effective key ----
// exactFp: the packed key itself (<= 32 bits) — fp equality IS key equality.
// Otherwise: high 32 bits of the 64-bit hash (low 32 pick the slot).
__device__ __forceinline__ uint32_t chainFingerprint(
    const RowKeyPack& pk,
    const uint64_t* key,
    uint64_t h) {
  return pk.enabled && pk.exactFp ? static_cast<uint32_t>(key[0])
                                  : static_cast<uint32_t>(h >> 32);
}

// ============================================================================
// Build kernel: one thread per build row. Lock-free chained insert:
//   old = atomicExch(&heads[slot], i);  nextFp[i] = fp<<32 | old
// The nextFp word is this row's own cell — no other thread writes it — and the
// probe kernels run strictly after (stream ordering), so no ordering hazard:
// by the time any probe walks the chain, every nextFp cell is published.
// ============================================================================
__global__ void insertRowsChained(
    const uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    RowKeyLayout keys,
    RowKeyPack pack,
    int32_t num_rows,
    int32_t* __restrict__ heads,
    uint64_t* __restrict__ nextFp,
    uint64_t* __restrict__ packedKeys, // present iff pack && !exactFp
    int32_t capacity) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) {
    return;
  }
  const uint8_t* row = row_buffer + (int64_t)i * row_width;
  uint64_t key[kRowTableMaxKeys];
  int32_t numWords;
  loadEffectiveKey(row, keys, pack, key, numWords); // build: always in range
  const uint64_t h = hashKey64(key, numWords);
  const int32_t slot =
      static_cast<int32_t>(static_cast<uint32_t>(h) & (capacity - 1));
  const uint32_t fp = chainFingerprint(pack, key, h);
  if (packedKeys != nullptr) {
    packedKeys[i] = key[0];
  }
  const int32_t old = atomicExch(heads + slot, i);
  nextFp[i] = (static_cast<uint64_t>(fp) << 32) |
      static_cast<uint64_t>(static_cast<uint32_t>(old));
}

// ============================================================================
// Probe kernel (count / fill share the walk; FILL is compile-time).
// One thread per probe row:
//   - read the probe KEY ROW ONCE (contiguous fields within one row),
//   - walk the chain from heads[slot],
//   - per entry: ONE 8-byte nextFp load gives fingerprint + next pointer;
//     on fp match, verify the real key against the build key row (row-native
//     compare — the build row bytes it touches are exactly the key row),
//   - count matches (pass 1) or write pairs at scanned offsets (pass 2).
// ============================================================================
template <bool FILL>
__global__ void probeChainedKernel(
    RowNativeHashTable t,
    const uint8_t* __restrict__ probe_buffer,
    int32_t probe_row_width,
    RowKeyLayout probeKeys,
    int32_t num_probe_rows,
    int32_t* __restrict__ counts, // pass 1 out
    const int32_t* __restrict__ offsets, // pass 2 in
    int32_t* __restrict__ leftIdx, // pass 2 out (probe row ids)
    int32_t* __restrict__ rightIdx) { // pass 2 out (build row ids)
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_probe_rows) {
    return;
  }
  const uint8_t* prow = probe_buffer + (int64_t)i * probe_row_width;
  uint64_t key[kRowTableMaxKeys];
  int32_t numWords;
  if (!loadEffectiveKey(prow, probeKeys, t.pack, key, numWords)) {
    // A key component is outside the build side's value range: no build row
    // can match. Skip the walk entirely (a packed-domain hash would alias).
    if constexpr (!FILL) {
      counts[i] = 0;
    }
    return;
  }
  const uint64_t h = hashKey64(key, numWords);
  const int32_t slot =
      static_cast<int32_t>(static_cast<uint32_t>(h) & (t.capacity - 1));
  const uint32_t fp = chainFingerprint(t.pack, key, h);

  int32_t r = __ldg(t.heads + slot);
  int32_t n = 0;
  int32_t out = FILL ? offsets[i] : 0;
  while (r >= 0) {
    const uint64_t nf = __ldg(t.nextFp + r);
    if (static_cast<uint32_t>(nf >> 32) == fp) {
      // Verify ladder: exactFp -> fp match IS key equality (no load);
      // packed -> one dense 8-byte load from packedKeys;
      // unpacked -> field-by-field compare against the build key row.
      bool match;
      if (t.pack.enabled) {
        match = t.pack.exactFp || __ldg(t.packedKeys + r) == key[0];
      } else {
        const uint8_t* brow =
            t.buildStore.row_buffer + (int64_t)r * t.buildStore.row_width;
        match = keyEqualsBuildRow(key, brow, t.keys);
      }
      if (match) {
        if constexpr (FILL) {
          leftIdx[out] = i;
          rightIdx[out] = r;
          out++;
        } else {
          n++;
        }
      }
    }
    const uint32_t nxt = static_cast<uint32_t>(nf);
    r = (nxt == RowNativeHashTable::kChainEnd) ? RowNativeHashTable::kEmpty
                                               : static_cast<int32_t>(nxt);
  }
  if constexpr (!FILL) {
    counts[i] = n;
  }
}

// total = (n == 0) ? 0 : offsets[n-1] + counts[n-1]
__global__ void computeTotalKernel(
    const int32_t* __restrict__ counts,
    const int32_t* __restrict__ offsets,
    int32_t n,
    int32_t* __restrict__ total) {
  *total = (n == 0) ? 0 : offsets[n - 1] + counts[n - 1];
}

// Per-key min/max over the build store's key words (unsigned). Grid-stride;
// each thread accumulates locally and lands ONE atomicMin/Max pair per key,
// so atomic traffic is O(threads * keys), independent of numRows.
__global__ void keyMinMaxKernel(
    const uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    RowKeyLayout keys,
    int32_t num_rows,
    uint64_t* __restrict__ minmax) { // [numKeys mins][numKeys maxs]
  uint64_t lmin[kRowTableMaxKeys];
  uint64_t lmax[kRowTableMaxKeys];
  for (int k = 0; k < keys.numKeys; k++) {
    lmin[k] = ~0ULL;
    lmax[k] = 0;
  }
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
       i += stride) {
    const uint8_t* row = row_buffer + (int64_t)i * row_width;
    for (int k = 0; k < keys.numKeys; k++) {
      const uint64_t v = loadKeyWord(row + keys.offset[k], keys.width[k]);
      lmin[k] = v < lmin[k] ? v : lmin[k];
      lmax[k] = v > lmax[k] ? v : lmax[k];
    }
  }
  for (int k = 0; k < keys.numKeys; k++) {
    atomicMin(
        reinterpret_cast<unsigned long long*>(minmax + k),
        static_cast<unsigned long long>(lmin[k]));
    atomicMax(
        reinterpret_cast<unsigned long long*>(minmax + keys.numKeys + k),
        static_cast<unsigned long long>(lmax[k]));
  }
}

} // namespace

// ============================================================================
// Host entry points
// ============================================================================

int32_t rowTableCapacity(int32_t numRows) {
  // Load factor <= 0.5 on head slots: capacity >= 2 * numRows, power of two.
  int64_t target = (int64_t)numRows * 2 + 1;
  int32_t cap = 2;
  while (cap < target) {
    cap <<= 1;
  }
  return cap;
}

void buildRowNativeHashTable(const RowNativeHashTable& t, cudaStream_t stream) {
  // 0xFF-fill == every head is kEmpty (-1).
  cudaMemsetAsync(
      t.heads, 0xFF, static_cast<size_t>(t.capacity) * sizeof(int32_t), stream);
  if (t.numRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (t.numRows + block - 1) / block;
  insertRowsChained<<<grid, block, 0, stream>>>(
      t.buildStore.row_buffer,
      t.buildStore.row_width,
      t.keys,
      t.pack,
      t.numRows,
      t.heads,
      t.nextFp,
      const_cast<uint64_t*>(t.packedKeys),
      t.capacity);
}

void rowTableComputeKeyMinMax(
    const GpuFixedRowStore& store,
    const RowKeyLayout& keys,
    uint64_t* d_minmax,
    cudaStream_t stream) {
  if (store.num_rows == 0) {
    return;
  }
  const int block = 256;
  const int grid = std::min(
      1024, (store.num_rows + block - 1) / block);
  keyMinMaxKernel<<<grid, block, 0, stream>>>(
      store.row_buffer, store.row_width, keys, store.num_rows, d_minmax);
}

size_t rowTableScanTempBytes(int32_t numRows) {
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr,
      bytes,
      static_cast<const int32_t*>(nullptr),
      static_cast<int32_t*>(nullptr),
      numRows);
  return bytes;
}

void rowTableProbeCount(
    const RowNativeHashTable& t,
    const GpuFixedRowStore& probe,
    const RowKeyLayout& probeKeys,
    int32_t numProbeRows,
    int32_t* d_counts,
    cudaStream_t stream) {
  if (numProbeRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numProbeRows + block - 1) / block;
  probeChainedKernel<false><<<grid, block, 0, stream>>>(
      t,
      probe.row_buffer,
      probe.row_width,
      probeKeys,
      numProbeRows,
      d_counts,
      nullptr,
      nullptr,
      nullptr);
}

void rowTableScanOffsets(
    const int32_t* d_counts,
    int32_t* d_offsets,
    int32_t numRows,
    void* d_temp,
    size_t tempBytes,
    int32_t* d_total,
    cudaStream_t stream) {
  if (numRows > 0) {
    cub::DeviceScan::ExclusiveSum(
        d_temp, tempBytes, d_counts, d_offsets, numRows, stream);
  }
  computeTotalKernel<<<1, 1, 0, stream>>>(
      d_counts, d_offsets, numRows, d_total);
}

void rowTableProbeFill(
    const RowNativeHashTable& t,
    const GpuFixedRowStore& probe,
    const RowKeyLayout& probeKeys,
    int32_t numProbeRows,
    const int32_t* d_offsets,
    int32_t* d_leftIdx,
    int32_t* d_rightIdx,
    cudaStream_t stream) {
  if (numProbeRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numProbeRows + block - 1) / block;
  probeChainedKernel<true><<<grid, block, 0, stream>>>(
      t,
      probe.row_buffer,
      probe.row_width,
      probeKeys,
      numProbeRows,
      nullptr,
      d_offsets,
      d_leftIdx,
      d_rightIdx);
}

} // namespace facebook::velox::cudf_velox
