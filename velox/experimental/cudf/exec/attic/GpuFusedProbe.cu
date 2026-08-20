/*
 * GpuFusedProbe.cu
 *
 * Device implementation of the fused multi-way row-wise probe:
 *   - open-addressing (linear-probe) hash table build
 *   - fused N-way probe kernel (one thread per probe tuple, resident acc)
 *
 * See GpuFusedProbe.cuh for the design and restrictions.
 */

#include "GpuFusedProbe.cuh"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <type_traits>

namespace facebook::velox::cudf_velox {

namespace {

// ---- field copy (matches GpuRowOps copy_field semantics) ----
__device__ __forceinline__ void
copyField(const uint8_t* src, uint8_t* dst, int32_t width) {
  if (width == 8) {
    *reinterpret_cast<uint64_t*>(dst) = *reinterpret_cast<const uint64_t*>(src);
  } else if (width == 4) {
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<const uint32_t*>(src);
  } else {
    for (int b = 0; b < width; b++) {
      dst[b] = src[b];
    }
  }
}

// ---- field copy from read-only global (build rows / slots) via __ldg ----
// Routes the load through the read-only data cache, which is a good fit for the
// random, thread-divergent access pattern of appending build-side fields.
__device__ __forceinline__ void
copyFieldLdg(const uint8_t* src, uint8_t* dst, int32_t width) {
  if (width == 8) {
    *reinterpret_cast<uint64_t*>(dst) =
        __ldg(reinterpret_cast<const uint64_t*>(src));
  } else if (width == 4) {
    *reinterpret_cast<uint32_t*>(dst) =
        __ldg(reinterpret_cast<const uint32_t*>(src));
  } else {
    for (int b = 0; b < width; b++) {
      dst[b] = __ldg(src + b);
    }
  }
}

// ---- read a fixed-width value from `p`, zero-extended to uint64 ----
__device__ __forceinline__ uint64_t loadKeyWord(const uint8_t* p, int32_t width) {
  if (width == 8) {
    return *reinterpret_cast<const uint64_t*>(p);
  }
  // width == 4 (INT32 keys)
  return static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(p));
}

// ---- 64-bit mix (splitmix64 finalizer) ----
__device__ __host__ __forceinline__ uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

// ---- hash of a composite key (numKeyWords words) ----
__device__ __forceinline__ uint32_t
hashKey(const uint64_t* words, int32_t numKeyWords, int32_t mask) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for (int w = 0; w < numKeyWords; w++) {
    h = mix64(h ^ words[w]);
  }
  return static_cast<uint32_t>(h) & static_cast<uint32_t>(mask);
}

__device__ __forceinline__ bool keysEqual(
    const uint64_t* a,
    const uint64_t* b,
    int32_t numKeyWords) {
  for (int w = 0; w < numKeyWords; w++) {
    if (a[w] != b[w]) {
      return false;
    }
  }
  return true;
}

// ---- device-side lookup: returns build row index or -1 ----
// AoS slots: each slot is (numKeyWords + 1) consecutive uint64 words
// [key_0..key_{n-1}, val]. Linear probing keeps subsequent probes within the
// same cache line, so only the initial (hash) access is a cold miss.
__device__ __forceinline__ int32_t
tableFind(const GpuHashTable& t, const uint64_t* key) {
  const int32_t mask = t.capacity - 1;
  int32_t slot = static_cast<int32_t>(hashKey(key, t.numKeyWords, mask));

  if (t.combinedKeyBits > 0) {
    // Combined slots: ONE 64-bit word = [key in low combinedKeyBits | index].
    // A single 8-byte read-only load per probe (vs the 16-byte ulonglong2).
    const uint64_t keyMask =
        (t.combinedKeyBits >= 64) ? ~0ULL : ((1ULL << t.combinedKeyBits) - 1);
    const uint64_t k0 = key[0];
    for (;;) {
      uint64_t w = __ldg(t.slots + slot);
      if (w == GpuHashTable::kEmptyWord) {
        return -1;
      }
      if ((w & keyMask) == k0) {
        return static_cast<int32_t>(w >> t.combinedKeyBits);
      }
      slot = (slot + 1) & mask;
    }
  }

  if (t.numKeyWords == 1) {
    // Fast path: 16-byte slot read as a single 128-bit read-only load.
    const ulonglong2* base = reinterpret_cast<const ulonglong2*>(t.slots);
    const uint64_t k0 = key[0];
    for (;;) {
      ulonglong2 kv = __ldg(base + slot); // {key, val} in one transaction
      if (kv.y == GpuHashTable::kEmptyWord) {
        return -1;
      }
      if (kv.x == k0) {
        return static_cast<int32_t>(static_cast<uint32_t>(kv.y));
      }
      slot = (slot + 1) & mask;
    }
  }

  // General path: composite key.
  const int32_t slotWords = t.numKeyWords + 1;
  for (;;) {
    const uint64_t* s = t.slots + (int64_t)slot * slotWords;
    uint64_t v = __ldg(s + t.numKeyWords);
    if (v == GpuHashTable::kEmptyWord) {
      return -1;
    }
    bool eq = true;
    for (int w = 0; w < t.numKeyWords; w++) {
      if (__ldg(s + w) != key[w]) {
        eq = false;
        break;
      }
    }
    if (eq) {
      return static_cast<int32_t>(static_cast<uint32_t>(v));
    }
    slot = (slot + 1) & mask;
  }
}

// ============================================================================
// Build kernels
// ============================================================================

__global__ void initVals(uint64_t* slots, int32_t capacity, int32_t slotWords) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < capacity) {
    // Only the val word must be sentinelled; key words are never read until
    // after the (stream-ordered) insert kernel has published them.
    slots[(int64_t)i * slotWords + (slotWords - 1)] = GpuHashTable::kEmptyWord;
  }
}

// Insert one row per thread. Build keys are UNIQUE (N:1 join precondition),
// so any occupied slot we encounter belongs to a different key and we simply
// probe onward. We claim an empty slot by CAS-ing its val word from
// kEmptyWord to our row index, then write the key words. insertRows never
// reads other slots' key words (only CASes val words), so the claim-then-write
// ordering is race-free; the probe kernel runs strictly after this kernel
// (stream ordering) and therefore observes fully published keys.
__global__ void insertRows(
    const uint64_t* __restrict__ key_words,
    int32_t num_rows,
    uint64_t* __restrict__ slots,
    int32_t capacity,
    int32_t numKeyWords) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;

  const uint64_t* myKey = key_words + (int64_t)i * numKeyWords;
  const int32_t slotWords = numKeyWords + 1;
  const int32_t mask = capacity - 1;
  const unsigned long long myVal =
      static_cast<unsigned long long>(static_cast<uint32_t>(i));
  int32_t slot = static_cast<int32_t>(hashKey(myKey, numKeyWords, mask));
  for (;;) {
    unsigned long long* valPtr = reinterpret_cast<unsigned long long*>(
        slots + (int64_t)slot * slotWords + numKeyWords);
    unsigned long long prev = atomicCAS(
        valPtr,
        static_cast<unsigned long long>(GpuHashTable::kEmptyWord),
        myVal);
    if (prev == static_cast<unsigned long long>(GpuHashTable::kEmptyWord)) {
      // claimed: publish key words
      uint64_t* dst = slots + (int64_t)slot * slotWords;
      for (int w = 0; w < numKeyWords; w++) {
        dst[w] = myKey[w];
      }
      return;
    }
    slot = (slot + 1) & mask;
  }
}

// Packed-key build: read numKeyCols columns, pack into ONE uint64 word
// (val_j - mins[j]) << shifts[j], and insert into a 2-word slot table. Build
// values are in range by construction (mins/shifts derived from them).
__global__ void insertRowsFromColumnsPacked(
    const ColSeedField* __restrict__ keyCols,
    int32_t numKeyCols,
    const int64_t* __restrict__ mins,
    const int32_t* __restrict__ shifts,
    int32_t num_rows,
    uint64_t* __restrict__ slots,
    int32_t capacity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;

  uint64_t packed = 0;
  for (int k = 0; k < numKeyCols; k++) {
    const ColSeedField& c = keyCols[k];
    int64_t v = static_cast<int64_t>(loadKeyWord(
        c.data + (int64_t)i * c.byte_width, c.byte_width));
    packed |= static_cast<uint64_t>(v - mins[k]) << shifts[k];
  }
  uint64_t myKey[1] = {packed};
  const int32_t slotWords = 2; // [key, val]
  const int32_t mask = capacity - 1;
  const unsigned long long myVal =
      static_cast<unsigned long long>(static_cast<uint32_t>(i));
  int32_t slot = static_cast<int32_t>(hashKey(myKey, 1, mask));
  for (;;) {
    unsigned long long* valPtr = reinterpret_cast<unsigned long long*>(
        slots + (int64_t)slot * slotWords + 1);
    unsigned long long prev = atomicCAS(
        valPtr,
        static_cast<unsigned long long>(GpuHashTable::kEmptyWord),
        myVal);
    if (prev == static_cast<unsigned long long>(GpuHashTable::kEmptyWord)) {
      slots[(int64_t)slot * slotWords] = packed;
      return;
    }
    slot = (slot + 1) & mask;
  }
}

// Combined-slot packed build (mirrors cuco's 8-byte (key,index) slot): pack the
// K key columns into one key word, then store [key | (row_index << keyBits)] in
// ONE slot word via a single atomicCAS (no separate key store). Halves the table
// and the scattered writes -- the build's dominant cost.
__global__ void insertRowsFromColumnsCombined(
    const ColSeedField* __restrict__ keyCols,
    int32_t numKeyCols,
    const int64_t* __restrict__ mins,
    const int32_t* __restrict__ shifts,
    int32_t keyBits,
    int32_t num_rows,
    uint64_t* __restrict__ slots,
    int32_t capacity,
    int32_t noCas) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;

  uint64_t packed = 0;
  for (int k = 0; k < numKeyCols; k++) {
    const ColSeedField& c = keyCols[k];
    int64_t v = static_cast<int64_t>(loadKeyWord(
        c.data + (int64_t)i * c.byte_width, c.byte_width));
    packed |= static_cast<uint64_t>(v - mins[k]) << shifts[k];
  }
  const uint64_t combined =
      packed | (static_cast<uint64_t>(static_cast<uint32_t>(i)) << keyBits);
  uint64_t myKey[1] = {packed};
  const int32_t mask = capacity - 1;
  int32_t slot = static_cast<int32_t>(hashKey(myKey, 1, mask));
  if (noCas) {
    // ABLATION (VELOX_CUDF_NO_CAS): same key read + hash + slot, but ONE plain
    // (non-atomic) random write instead of the atomicCAS+probe loop. Isolates
    // the atomic/probe cost from the random-write memory latency. NOT correct
    // (collisions overwrite) -- for timing only.
    slots[slot] = combined;
    return;
  }
  for (;;) {
    unsigned long long* wptr =
        reinterpret_cast<unsigned long long*>(slots + slot);
    unsigned long long prev = atomicCAS(
        wptr,
        static_cast<unsigned long long>(GpuHashTable::kEmptyWord),
        static_cast<unsigned long long>(combined));
    if (prev == static_cast<unsigned long long>(GpuHashTable::kEmptyWord)) {
      return; // key+index written atomically
    }
    slot = (slot + 1) & mask;
  }
}

// Fingerprint build (mirrors cuco's pair<hash32,index32> composite slot): pack
// the K key columns into one EXACT word, hash it, and store [fp32 | index<<32]
// in ONE 8-byte slot via a single atomicCAS. The exact key does not fit
// alongside the index in 63 bits, so we store a 32-bit fingerprint instead; the
// probe verifies the real key on a fingerprint match. slot and fp come from the
// SAME 64-bit hash (low bits -> slot, high bits -> fp), matching hashKey().
__global__ void insertRowsFromColumnsHashed(
    const ColSeedField* __restrict__ keyCols,
    int32_t numKeyCols,
    const int64_t* __restrict__ mins,
    const int32_t* __restrict__ shifts,
    int32_t fpBits,
    int32_t num_rows,
    uint64_t* __restrict__ slots,
    int32_t capacity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;

  uint64_t packed = 0;
  for (int k = 0; k < numKeyCols; k++) {
    const ColSeedField& c = keyCols[k];
    int64_t v = static_cast<int64_t>(loadKeyWord(
        c.data + (int64_t)i * c.byte_width, c.byte_width));
    packed |= static_cast<uint64_t>(v - mins[k]) << shifts[k];
  }
  const uint64_t h = mix64(0xcbf29ce484222325ULL ^ packed);
  const int32_t mask = capacity - 1;
  int32_t slot = static_cast<int32_t>(static_cast<uint32_t>(h) & mask);
  const uint64_t fp = static_cast<uint64_t>(static_cast<uint32_t>(h >> 32));
  const uint64_t combined = fp | (static_cast<uint64_t>(static_cast<uint32_t>(i))
                                  << fpBits);
  for (;;) {
    unsigned long long* wptr =
        reinterpret_cast<unsigned long long*>(slots + slot);
    unsigned long long prev = atomicCAS(
        wptr,
        static_cast<unsigned long long>(GpuHashTable::kEmptyWord),
        static_cast<unsigned long long>(combined));
    if (prev == static_cast<unsigned long long>(GpuHashTable::kEmptyWord)) {
      return;
    }
    slot = (slot + 1) & mask;
  }
}

// Fused build reading keys from the COLUMNAR build input: lane i reads col[i],
// so the key loads are COALESCED, unlike the row store where lane i reads at
// i*row_width (strided). The insert itself is identical.
__global__ void insertRowsFromColumns(
    const ColSeedField* __restrict__ keyCols, // [numKeyWords] {data, width}
    int32_t numKeyWords,
    int32_t num_rows,
    uint64_t* __restrict__ slots,
    int32_t capacity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;

  uint64_t myKey[kMaxKeyCols];
  for (int k = 0; k < numKeyWords; k++) {
    const ColSeedField& c = keyCols[k];
    myKey[k] = loadKeyWord(c.data + (int64_t)i * c.byte_width, c.byte_width);
  }
  const int32_t slotWords = numKeyWords + 1;
  const int32_t mask = capacity - 1;
  const unsigned long long myVal =
      static_cast<unsigned long long>(static_cast<uint32_t>(i));
  int32_t slot = static_cast<int32_t>(hashKey(myKey, numKeyWords, mask));
  for (;;) {
    unsigned long long* valPtr = reinterpret_cast<unsigned long long*>(
        slots + (int64_t)slot * slotWords + numKeyWords);
    unsigned long long prev = atomicCAS(
        valPtr,
        static_cast<unsigned long long>(GpuHashTable::kEmptyWord),
        myVal);
    if (prev == static_cast<unsigned long long>(GpuHashTable::kEmptyWord)) {
      uint64_t* dst = slots + (int64_t)slot * slotWords;
      for (int k = 0; k < numKeyWords; k++) {
        dst[k] = myKey[k];
      }
      return;
    }
    slot = (slot + 1) & mask;
  }
}

// Fused build: read keys straight from the row store, hash, and insert -- no
// separate key-word array, no separate packing pass. One thread per build row.
// (See insertRows for the claim-then-publish ordering rationale.)
__global__ void insertRowsFromStore(
    const uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    const int32_t* __restrict__ keyOffsets,
    const int32_t* __restrict__ keyWidths,
    int32_t numKeyWords,
    int32_t num_rows,
    uint64_t* __restrict__ slots,
    int32_t capacity) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;

  const uint8_t* row = row_buffer + (int64_t)i * row_width;
  uint64_t myKey[kMaxKeyCols];
  for (int k = 0; k < numKeyWords; k++) {
    myKey[k] = loadKeyWord(row + keyOffsets[k], keyWidths[k]);
  }
  const int32_t slotWords = numKeyWords + 1;
  const int32_t mask = capacity - 1;
  const unsigned long long myVal =
      static_cast<unsigned long long>(static_cast<uint32_t>(i));
  int32_t slot = static_cast<int32_t>(hashKey(myKey, numKeyWords, mask));
  for (;;) {
    unsigned long long* valPtr = reinterpret_cast<unsigned long long*>(
        slots + (int64_t)slot * slotWords + numKeyWords);
    unsigned long long prev = atomicCAS(
        valPtr,
        static_cast<unsigned long long>(GpuHashTable::kEmptyWord),
        myVal);
    if (prev == static_cast<unsigned long long>(GpuHashTable::kEmptyWord)) {
      uint64_t* dst = slots + (int64_t)slot * slotWords;
      for (int k = 0; k < numKeyWords; k++) {
        dst[k] = myKey[k];
      }
      return;
    }
    slot = (slot + 1) & mask;
  }
}

// Pack composite keys out of a row store into contiguous key words.
__global__ void buildKeyWordsKernel(
    const uint8_t* __restrict__ row_buffer,
    int32_t row_width,
    const int32_t* __restrict__ keyOffsets,
    const int32_t* __restrict__ keyWidths,
    int32_t numKeys,
    int32_t num_rows,
    uint64_t* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_rows) return;
  const uint8_t* row = row_buffer + (int64_t)i * row_width;
  uint64_t* dst = out + (int64_t)i * numKeys;
  for (int k = 0; k < numKeys; k++) {
    dst[k] = loadKeyWord(row + keyOffsets[k], keyWidths[k]);
  }
}

// ============================================================================
// Fused probe kernel
// ============================================================================

// COLUMNAR_SEED / COLUMNAR_OUT are compile-time so the seed and output paths
// cost nothing in the other configuration -- no runtime branch, no divergence,
// and the unused pointers fold away.
template <bool COLUMNAR_SEED, bool COLUMNAR_OUT, bool HAS_FP>
__global__ void fusedProbeKernel(
    GpuFixedRowStore probe,
    int32_t numRows,
    const FieldMapping* __restrict__ probeToAcc,
    int32_t numProbeFields,
    const ColSeedField* __restrict__ colSeed,
    const FusedJoinStep* __restrict__ steps,
    int32_t numSteps,
    int32_t accWidth,
    const FieldMapping* __restrict__ accToOut,
    const ColOutField* __restrict__ colOut,
    int32_t numOutFields,
    const DeferredField* __restrict__ deferred,
    int32_t numDeferred,
    int32_t outRowWidth,
    uint8_t* __restrict__ out,
    int32_t* __restrict__ outCount) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // NOTE: do NOT early-return before the warp ballot below. The output uses a
  // warp-aggregated atomic that must be driven by a full-warp __ballot_sync
  // (mask 0xffffffff) so every lane participates and agrees on the survivor
  // set. Threads that are out of range or miss the join simply set
  // survived=false and still take part in the ballot. (Using __activemask()
  // here instead is unreliable under independent thread scheduling on sm_70+:
  // lanes can observe divergent masks, compute inconsistent leader/rank/total,
  // and two survivors can then reserve the same output slot — dropping rows
  // nondeterministically, e.g. only on a cold first iteration.)
  uint8_t acc[kFusedAccMaxBytes];
  bool survived = false;

  if (i < numRows) {
    survived = true;

    // Seed the accumulator from the probe (fact) tuple.
    if constexpr (COLUMNAR_SEED) {
      // OPT 1: read straight from cuDF columns. Lane i reads col[i], so the
      // warp's reads are ADJACENT -> fully coalesced, at any row width. This
      // also means no columns_to_rows pass exists at all: the transpose it
      // replaces was the single largest GPU cost in this operator (3.09ms of
      // 6.53ms on TPC-H Q8 SF10).
      for (int f = 0; f < numProbeFields; f++) {
        const ColSeedField& c = colSeed[f];
        copyField(
            c.data + (int64_t)i * c.byte_width,
            acc + c.acc_offset,
            c.byte_width);
      }
    } else {
      // Row store: lane i reads at i*row_width, so lanes are row_width apart
      // and the warp's reads smear across cache lines.
      const uint8_t* prow = probe.row_buffer + (int64_t)i * probe.row_width;
      for (int f = 0; f < numProbeFields; f++) {
        copyField(
            prow + probeToAcc[f].src_offset,
            acc + probeToAcc[f].dst_offset,
            probeToAcc[f].byte_width);
      }
    }

    // Walk the join chain, tuple stays resident in acc.
    for (int j = 0; j < numSteps; j++) {
      const FusedJoinStep& s = steps[j];

      // Build the lookup key from acc (key columns may have been produced by
      // earlier steps — the row-native win: a local read, not a gather).
      uint64_t key[kMaxKeyCols];
      if (s.packEnabled) {
        // Pack the K key columns into ONE word (single-word fast-path lookup).
        uint64_t packed = 0;
        bool inRange = true;
        for (int k = 0; k < s.numKeys; k++) {
          int64_t v = static_cast<int64_t>(
              loadKeyWord(acc + s.keyFields[k].acc_offset, s.keyFields[k].width));
          if (v < s.packMin[k] || v > s.packMax[k]) {
            inRange = false; // out of build range -> cannot match
            break;
          }
          packed |= static_cast<uint64_t>(v - s.packMin[k]) << s.packShift[k];
        }
        key[0] = inRange ? packed : s.packSentinel;
      } else {
        for (int k = 0; k < s.numKeys; k++) {
          key[k] =
              loadKeyWord(acc + s.keyFields[k].acc_offset, s.keyFields[k].width);
        }
      }

      int32_t br;
      if constexpr (HAS_FP) {
       if (s.table.fingerprintSlot) {
        // Composite/wide key: slots store [fp32 | index], so a slot hit is only
        // a FINGERPRINT match -- verify the real key by re-packing it from the
        // build row (already cache-hot; we read that row for payload next).
        const uint64_t pk = key[0];
        if (pk == s.packSentinel) {
          br = -1; // probe key out of build range
        } else {
          const uint64_t h = mix64(0xcbf29ce484222325ULL ^ pk);
          const int32_t mask = s.table.capacity - 1;
          const int32_t fpBits = s.table.combinedKeyBits;
          const uint64_t fpMask =
              (fpBits >= 64) ? ~0ULL : ((1ULL << fpBits) - 1);
          const uint64_t fp =
              static_cast<uint64_t>(static_cast<uint32_t>(h >> 32));
          int32_t slot = static_cast<int32_t>(static_cast<uint32_t>(h) & mask);
          br = -1;
          for (;;) {
            uint64_t w = __ldg(s.table.slots + slot);
            if (w == GpuHashTable::kEmptyWord) {
              break; // miss
            }
            if ((w & fpMask) == fp) {
              const int32_t idx = static_cast<int32_t>(w >> fpBits);
              const uint8_t* bkrow = s.buildStore.row_buffer +
                  (int64_t)idx * s.buildStore.row_width;
              uint64_t bpk = 0;
              for (int k = 0; k < s.numKeys; k++) {
                int64_t bv = static_cast<int64_t>(loadKeyWord(
                    bkrow + s.buildKeyOffset[k], s.keyFields[k].width));
                bpk |=
                    static_cast<uint64_t>(bv - s.packMin[k]) << s.packShift[k];
              }
              if (bpk == pk) {
                br = idx; // verified real key match
                break;
              }
            }
            slot = (slot + 1) & mask;
          }
        }
       } else {
         br = tableFind(s.table, key); // fingerprint kernel, non-fp step
       }
      } else {
        br = tableFind(s.table, key); // no fingerprint step in this plan
      }
      if (br < 0) {
        survived = false; // inner-join miss -> tuple dropped
        break;
      }

      // Append selected build fields into acc.
      if (s.columnarBuild) {
        // Columnar fetch: read each build field straight from its cuDF column at
        // the matched build row -- no pre-transposed row store.
        for (int f = 0; f < s.numBuildFields; f++) {
          const ColBuildField& c = s.buildCols[f];
          copyFieldLdg(
              c.data + (int64_t)br * c.byte_width,
              acc + c.acc_offset,
              c.byte_width);
        }
      } else {
        const uint8_t* brow =
            s.buildStore.row_buffer + (int64_t)br * s.buildStore.row_width;
        for (int f = 0; f < s.numBuildFields; f++) {
          copyFieldLdg(
              brow + s.buildToAcc[f].src_offset,
              acc + s.buildToAcc[f].dst_offset,
              s.buildToAcc[f].byte_width);
        }
      }
    }
  }

  // Survivor: compact-append the output row. Only a small fraction of probe
  // rows survive an N-way inner join, so use a warp-aggregated atomic: one
  // atomicAdd per warp reserves a contiguous run for all surviving lanes,
  // dramatically cutting global-counter contention versus one atomic per row.
  // The full-warp ballot makes the survivor mask consistent across all lanes.
  const unsigned survivors = __ballot_sync(0xffffffffu, survived);
  if (survived) {
    const int lane = threadIdx.x & 31;
    const int leader = __ffs(survivors) - 1;
    const int rank = __popc(survivors & ((1u << lane) - 1));
    const int total = __popc(survivors);
    int32_t base;
    if (lane == leader) {
      base = atomicAdd(outCount, total);
    }
    base = __shfl_sync(survivors, base, leader);
    int32_t pos = base + rank;

    if constexpr (COLUMNAR_OUT) {
      // OPT 2: scatter straight into cuDF columns -- no row buffer, and no
      // rows_to_columns pass afterwards.
      //
      // This is coalesced FOR FREE. The warp-aggregated atomic above hands
      // surviving lanes CONSECUTIVE slots (pos = base + rank, rank dense
      // 0..total-1), so for a given column the warp writes adjacent addresses.
      // The row-major write below does the opposite: each lane writes a whole
      // row, so lanes are outRowWidth apart -- strided.
      for (int f = 0; f < numOutFields; f++) {
        const ColOutField& c = colOut[f];
        copyField(
            acc + c.acc_offset,
            c.data + (int64_t)pos * c.byte_width,
            c.byte_width);
      }
      // OPT 3 (lazy payload): output columns that were NEVER seeded into the
      // accumulator (pure probe payload, not a join key) are read now, straight
      // from the probe column at THIS thread's own probe row `i`. Only survivors
      // reach here, so this touches ~selectivity x fewer rows than seeding every
      // probe row would. Reads are read-only global -> route through __ldg.
      for (int f = 0; f < numDeferred; f++) {
        const DeferredField& d = deferred[f];
        copyFieldLdg(
            d.probeData + (int64_t)i * d.byte_width,
            d.outData + (int64_t)pos * d.byte_width,
            d.byte_width);
      }
    } else {
      uint8_t* orow = out + (int64_t)pos * outRowWidth;
      for (int f = 0; f < numOutFields; f++) {
        copyField(
            acc + accToOut[f].src_offset,
            orow + accToOut[f].dst_offset,
            accToOut[f].byte_width);
      }
    }
  }
}

} // namespace

// ============================================================================
// Host entry points
// ============================================================================

int32_t fusedTableCapacity(int32_t numRows) {
  // Load factor <= 0.5: capacity >= 2 * numRows, rounded up to power of two.
  int64_t target = (int64_t)numRows * 2 + 1;
  int32_t cap = 1;
  while (cap < target) {
    cap <<= 1;
  }
  if (cap < 2) {
    cap = 2;
  }
  return cap;
}

void buildHashTable(
    const uint64_t* d_key_words,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    int32_t numKeyWords,
    cudaStream_t stream) {
  const int32_t slotWords = numKeyWords + 1;
  int block = 256;
  int gridInit = (capacity + block - 1) / block;
  initVals<<<gridInit, block, 0, stream>>>(d_slots, capacity, slotWords);
  if (numRows == 0) {
    return;
  }
  int gridIns = (numRows + block - 1) / block;
  insertRows<<<gridIns, block, 0, stream>>>(
      d_key_words, numRows, d_slots, capacity, numKeyWords);
}

void buildHashTableFromColumnsPacked(
    const ColSeedField* d_keyCols,
    int32_t numKeyCols,
    const int64_t* d_mins,
    const int32_t* d_shifts,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream) {
  // VELOX_CUDF_BUILD_PROFILE: split the packed build into memset vs insert. Only
  // times the big builds (numRows > 1M) to keep noise out of the log.
  const bool prof =
      (std::getenv("VELOX_CUDF_BUILD_PROFILE") != nullptr) && numRows > 1000000;
  cudaEvent_t e0, e1, e2;
  if (prof) {
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e0, stream);
  }
  cudaMemsetAsync(
      d_slots, 0xFF, static_cast<size_t>(capacity) * 2 * sizeof(uint64_t),
      stream);
  if (prof) {
    cudaEventRecord(e1, stream);
  }
  if (numRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numRows + block - 1) / block;
  insertRowsFromColumnsPacked<<<grid, block, 0, stream>>>(
      d_keyCols, numKeyCols, d_mins, d_shifts, numRows, d_slots, capacity);
  if (prof) {
    cudaEventRecord(e2, stream);
    cudaEventSynchronize(e2);
    float tMemset = 0, tInsert = 0;
    cudaEventElapsedTime(&tMemset, e0, e1);
    cudaEventElapsedTime(&tInsert, e1, e2);
    fprintf(
        stderr,
        "[PACKED_BUILD] nRows=%d cap=%d tableMB=%.0f | memset=%.2f insert=%.2f "
        "ms  (LF=%.2f)\n",
        numRows,
        capacity,
        (double)capacity * 2 * sizeof(uint64_t) / (1024 * 1024),
        tMemset,
        tInsert,
        (double)numRows / capacity);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
  }
}

void buildHashTableFromColumnsCombined(
    const ColSeedField* d_keyCols,
    int32_t numKeyCols,
    const int64_t* d_mins,
    const int32_t* d_shifts,
    int32_t keyBits,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream) {
  const bool prof =
      (std::getenv("VELOX_CUDF_BUILD_PROFILE") != nullptr) && numRows > 1000000;
  cudaEvent_t e0, e1, e2;
  if (prof) {
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e0, stream);
  }
  // ONE word per slot.
  cudaMemsetAsync(
      d_slots, 0xFF, static_cast<size_t>(capacity) * sizeof(uint64_t), stream);
  if (prof) {
    cudaEventRecord(e1, stream);
  }
  if (numRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numRows + block - 1) / block;
  const int32_t noCas = (std::getenv("VELOX_CUDF_NO_CAS") != nullptr) ? 1 : 0;
  insertRowsFromColumnsCombined<<<grid, block, 0, stream>>>(
      d_keyCols, numKeyCols, d_mins, d_shifts, keyBits, numRows, d_slots,
      capacity, noCas);
  if (prof) {
    cudaEventRecord(e2, stream);
    cudaEventSynchronize(e2);
    float tMemset = 0, tInsert = 0;
    cudaEventElapsedTime(&tMemset, e0, e1);
    cudaEventElapsedTime(&tInsert, e1, e2);
    fprintf(
        stderr,
        "[COMBINED_BUILD] nRows=%d cap=%d tableMB=%.0f | memset=%.2f "
        "insert=%.2f ms  (LF=%.2f)\n",
        numRows,
        capacity,
        (double)capacity * sizeof(uint64_t) / (1024 * 1024),
        tMemset,
        tInsert,
        (double)numRows / capacity);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
  }
}

void buildHashTableFromColumnsHashed(
    const ColSeedField* d_keyCols,
    int32_t numKeyCols,
    const int64_t* d_mins,
    const int32_t* d_shifts,
    int32_t fpBits,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream) {
  const bool prof =
      (std::getenv("VELOX_CUDF_BUILD_PROFILE") != nullptr) && numRows > 1000000;
  cudaEvent_t e0, e1, e2;
  if (prof) {
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaEventCreate(&e2);
    cudaEventRecord(e0, stream);
  }
  cudaMemsetAsync(
      d_slots, 0xFF, static_cast<size_t>(capacity) * sizeof(uint64_t), stream);
  if (prof) {
    cudaEventRecord(e1, stream);
  }
  if (numRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numRows + block - 1) / block;
  insertRowsFromColumnsHashed<<<grid, block, 0, stream>>>(
      d_keyCols, numKeyCols, d_mins, d_shifts, fpBits, numRows, d_slots,
      capacity);
  if (prof) {
    cudaEventRecord(e2, stream);
    cudaEventSynchronize(e2);
    float tMemset = 0, tInsert = 0;
    cudaEventElapsedTime(&tMemset, e0, e1);
    cudaEventElapsedTime(&tInsert, e1, e2);
    fprintf(
        stderr,
        "[HASHED_BUILD] nRows=%d cap=%d tableMB=%.0f | memset=%.2f insert=%.2f "
        "ms  (LF=%.2f)\n",
        numRows,
        capacity,
        (double)capacity * sizeof(uint64_t) / (1024 * 1024),
        tMemset,
        tInsert,
        (double)numRows / capacity);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
  }
}

void buildHashTableFromColumns(
    const ColSeedField* d_keyCols,
    int32_t numKeyWords,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream) {
  const int32_t slotWords = numKeyWords + 1;
  cudaMemsetAsync(
      d_slots,
      0xFF,
      static_cast<size_t>(capacity) * slotWords * sizeof(uint64_t),
      stream);
  if (numRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numRows + block - 1) / block;
  insertRowsFromColumns<<<grid, block, 0, stream>>>(
      d_keyCols, numKeyWords, numRows, d_slots, capacity);
}

void buildHashTableFromStore(
    const GpuFixedRowStore& store,
    const int32_t* d_keyOffsets,
    const int32_t* d_keyWidths,
    int32_t numKeyWords,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream) {
  const int32_t slotWords = numKeyWords + 1;
  // Sentinel the val words. kEmptyWord is all-0xFF, so a single memset of the
  // WHOLE buffer to 0xFF marks every slot empty: val words become the sentinel;
  // key words become garbage that is never read for an empty slot (tableFind
  // checks the val word first). Replaces the initVals kernel with a
  // bandwidth-bound memset.
  cudaMemsetAsync(
      d_slots,
      0xFF,
      static_cast<size_t>(capacity) * slotWords * sizeof(uint64_t),
      stream);
  if (store.num_rows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (store.num_rows + block - 1) / block;
  insertRowsFromStore<<<grid, block, 0, stream>>>(
      store.row_buffer,
      store.row_width,
      d_keyOffsets,
      d_keyWidths,
      numKeyWords,
      store.num_rows,
      d_slots,
      capacity);
}

void buildKeyWords(
    const GpuFixedRowStore& store,
    const int32_t* d_keyOffsets,
    const int32_t* d_keyWidths,
    int32_t numKeys,
    uint64_t* d_out,
    cudaStream_t stream) {
  if (store.num_rows == 0) {
    return;
  }
  int block = 256;
  int grid = (store.num_rows + block - 1) / block;
  buildKeyWordsKernel<<<grid, block, 0, stream>>>(
      store.row_buffer,
      store.row_width,
      d_keyOffsets,
      d_keyWidths,
      numKeys,
      store.num_rows,
      d_out);
}

void fusedProbe(
    const GpuFixedRowStore& probe,
    int32_t numRows,
    const FieldMapping* d_probeToAcc,
    int32_t numProbeFields,
    const ColSeedField* d_colSeed,
    const FusedJoinStep* d_steps,
    int32_t numSteps,
    int32_t accWidth,
    const FieldMapping* d_accToOut,
    const ColOutField* d_colOut,
    int32_t numOutFields,
    const DeferredField* d_deferred,
    int32_t numDeferred,
    int32_t outRowWidth,
    uint8_t* d_out,
    int32_t* d_outCount,
    bool hasFingerprint,
    cudaStream_t stream) {
  if (numRows == 0) {
    return;
  }
  const int block = 256;
  const int grid = (numRows + block - 1) / block;

  // Dispatch (seed layout) x (output layout) x (has-fingerprint). All three are
  // compile-time in the kernel, so a plan with NO fingerprint step compiles the
  // fingerprint verify path OUT entirely -- no extra registers, no occupancy
  // hit. Only plans that actually use a composite fingerprint slot pay for it.
  auto launch = [&](auto seedTag, auto outTag, auto fpTag) {
    constexpr bool kSeed = decltype(seedTag)::value;
    constexpr bool kOut = decltype(outTag)::value;
    constexpr bool kFp = decltype(fpTag)::value;
    fusedProbeKernel<kSeed, kOut, kFp><<<grid, block, 0, stream>>>(
        probe,
        numRows,
        d_probeToAcc,
        numProbeFields,
        d_colSeed,
        d_steps,
        numSteps,
        accWidth,
        d_accToOut,
        d_colOut,
        numOutFields,
        d_deferred,
        numDeferred,
        outRowWidth,
        d_out,
        d_outCount);
  };
  using T = std::true_type;
  using F = std::false_type;
  const bool colSeed = (d_colSeed != nullptr);
  const bool colOut = (d_colOut != nullptr);
  auto dispatchFp = [&](auto seedTag, auto outTag) {
    if (hasFingerprint) {
      launch(seedTag, outTag, T{});
    } else {
      launch(seedTag, outTag, F{});
    }
  };
  if (colSeed && colOut) {
    dispatchFp(T{}, T{});
  } else if (colSeed) {
    dispatchFp(T{}, F{});
  } else if (colOut) {
    dispatchFp(F{}, T{});
  } else {
    dispatchFp(F{}, F{});
  }
}

} // namespace facebook::velox::cudf_velox
