/*
 * GpuFusedProbe.cuh
 *
 * Fused multi-way row-wise hash join probe.
 *
 * Instead of running an N-way inner-join chain as N separate probe operators
 * (each doing extract-keys + hash-probe + full intermediate row materialization
 * to global memory), the fused probe assigns ONE GPU thread to ONE probe
 * (fact) tuple and walks that tuple through all N build hash tables, keeping
 * the growing tuple resident in a small per-thread accumulator. Downstream join
 * keys that are *produced* by earlier joins (e.g. TPC-H Q5's
 * (s_nationkey,o_custkey) key for the customer join) are read directly from the
 * accumulator instead of being re-gathered from a materialized intermediate.
 * This is the row-native advantage: N dependent global gathers of intermediate
 * keys become N local reads.
 *
 * Restrictions (prototype):
 *   - Inner joins only.
 *   - Every fused build side must have UNIQUE keys (N:1 semantics), so each
 *     probe tuple matches at most one build row per step. Many-to-many steps
 *     cannot be fused inline and must terminate the fused region.
 *   - Fixed-width key columns (INT32/INT64), up to kMaxKeyCols per join.
 *   - Accumulator bounded by kFusedAccMaxBytes.
 *
 * This header has NO Velox/folly dependencies so it can be included from both
 * host (.cpp) and device (.cu) translation units.
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "GpuFixedRowStore.h" // FieldDesc, FieldMapping, GpuFixedRowStore

namespace facebook::velox::cudf_velox {

// Compile-time bounds.
//
// The accumulator is a per-thread stack array (`uint8_t acc[kFusedAccMaxBytes]`
// in fusedProbeKernel), so its size MUST be a compile-time constant -- CUDA has
// no variable-length arrays in device code. `accWidth_` is only the high-water
// mark used to check the plan fits; it does not size anything.
//
// Oversizing is close to free. The array lives in LOCAL memory, which the
// hardware stores INTERLEAVED across threads (word w of thread t at
// base + w*4*numThreads + t*4). Consequences:
//   - untouched words live in their own cache lines and are never fetched, so
//     unused capacity costs no bandwidth and no L1/L2;
//   - local memory consumes neither registers nor shared memory, so it does not
//     cap occupancy (measured: register count is identical at 40/64/128/256 B);
//   - the only cost is device memory reserved for local storage
//     (frame_size x max_resident_threads ~ tens of MB on an H100).
// Hence 128 rather than 64: TPC-H Q9's chain needs ~60-68 B and Q8 ~60 B, and
// paying for headroom we don't use is cheaper than a VELOX_CHECK failure.
static constexpr int kFusedAccMaxBytes = 128;

// Max key columns per fused join step. Bounds `uint64_t key[kMaxKeyCols]` in
// fusedProbeKernel -- exceeding it is an out-of-bounds STACK WRITE, so
// setupLayout VELOX_CHECKs against it. TPC-H's widest fused key is 2 columns
// (Q9's partsupp on (ps_partkey, ps_suppkey); Q5's customer on
// (c_nationkey, c_custkey)); 8 gives headroom on the same free-capacity logic
// and lets composite-key microbenchmarks sweep arity across all slot regimes.
static constexpr int kMaxKeyCols = 8;

static constexpr int kMaxFusedSteps = 8;

// ============================================================================
// Device-probeable open-addressing hash table (linear probing).
// Maps a fixed-width composite key -> build-row index. Built once per build
// side; probed inline (device function) from the fused kernel.
//
// AoS ("array of structs") slot layout for coalesced, single-transaction
// probes: each slot is `numKeyWords + 1` consecutive uint64 words:
//     [ key_0, key_1, ..., key_{n-1}, val ]
// where `val` holds the build-row index in its low 32 bits, or the sentinel
// `kEmptyWord` (all-ones) when the slot is empty. Row indices are in
// [0, 2^31), so a valid val can never collide with the sentinel.
//
// For the common single-key case (numKeyWords == 1) a slot is exactly 16 bytes
// and is read with ONE aligned 128-bit (ulonglong2) load — key and val arrive
// in a single memory transaction, versus the previous split-array layout that
// needed two independent uncoalesced loads (key array + val array).
// ============================================================================

struct GpuHashTable {
  uint64_t* slots;     // [capacity * (numKeyWords + 1)]  (2 words per slot if
                       //  combinedKeyBits==0; ONE word per slot otherwise)
  int32_t capacity;    // power of two
  int32_t numKeyWords; // number of key columns (1..kMaxKeyCols)

  // COMBINED slots (single-word, key+index packed together) -- this mirrors
  // cudf::distinct_hash_join's cuco layout (an 8-byte (key,index) slot written
  // with ONE atomic). When > 0, each slot is ONE uint64: the low
  // `combinedKeyBits` bits hold the (already-packed) key, the high bits hold the
  // build row index. Halves the table and the scattered writes vs the 2-word
  // [key,val] layout -- the dominant cost of the build (latency-bound random
  // atomics). Only used when combinedKeyBits + indexBits <= 63 (so a valid word
  // can never equal the all-ones sentinel). Requires a packed single-word key.
  int32_t combinedKeyBits{0};

  // FINGERPRINT slots (cuco's composite-key design): when set, the low
  // `combinedKeyBits` bits hold a 32-bit HASH of the key (not the exact key),
  // and the high bits hold the build row index -- an 8-byte slot even for a
  // composite/wide key that will not pack exactly into 63 bits. A fingerprint
  // match is NOT a guaranteed key match, so the probe MUST verify the real key
  // (re-read from the build row store, which it already touches for payload).
  // This is exactly what cudf::distinct_hash_join stores: pair<hash32,index32>.
  int32_t fingerprintSlot{0};

  // Empty-slot sentinel stored in the val word. All-ones can never equal a
  // valid (zero-extended, < 2^31) row index.
  static constexpr uint64_t kEmptyWord = ~0ULL;
};

// Describes how to read one key column out of the per-thread accumulator to
// form a lookup key that matches how the build table was constructed.
struct FusedKeyField {
  int32_t acc_offset; // byte offset of the key column within the accumulator
  int32_t width;      // 4 or 8
};

// ---- Columnar seed / columnar output descriptors (OPT 1 and OPT 2) ----
//
// The probe input and the chain output are each touched EXACTLY ONCE, in
// thread-id order. That is the coalesced case, and a row layout is the wrong
// tool for it:
//
//   row  : thread i touches base + i*row_width -> lanes are row_width apart,
//          so a warp's accesses smear across cache lines. Measured: a 72-byte
//          row transposes at ~242 GB/s (8% of peak) versus ~1250 GB/s for a
//          32-byte row. It degrades SUPERLINEARLY with width.
//   col  : thread i touches col[i] -> lanes are adjacent. Perfectly coalesced,
//          at any width.
//
// So we read the probe straight from cuDF columns (no columns_to_rows pass at
// all) and write survivors straight into cuDF columns (no rows_to_columns pass).
// The output write is coalesced for free because the warp-aggregated atomic
// hands surviving lanes CONSECUTIVE output slots (pos = base + rank).
//
// The BUILD payload keeps its row store: it is fetched randomly (by hash
// result), where one contiguous row beats N independent gathers. That is the
// one place the row layout genuinely earns its keep.
struct ColSeedField {
  const uint8_t* data; // cuDF column base pointer (fixed-width, non-nullable)
  int32_t byte_width;
  int32_t acc_offset; // where it lands in the accumulator
};

struct ColOutField {
  uint8_t* data; // output column base pointer
  int32_t byte_width;
  int32_t acc_offset; // where it comes from in the accumulator
};

// OPT 3 (lazy payload): a probe-payload output column that is NOT seeded into the
// accumulator. A surviving thread reads it directly from the probe column at its
// own probe row index and writes it to the output column -- survivors only, which
// is 1/selectivity fewer reads than seeding every probe row.
struct DeferredField {
  const uint8_t* probeData; // probe column base pointer
  uint8_t* outData;         // output column base pointer
  int32_t byte_width;
};

// Columnar build-payload fetch: on a match, read the build field straight from
// its cuDF column at the matched build row `br` (data + br*width), instead of
// from a pre-transposed row store. Skips the build-side columns->rows transpose,
// which is a fixed cost over ALL build rows; worth it when each build row is
// fetched < ~1x (large build, few matches) -- see fetches/build_rows analysis.
struct ColBuildField {
  const uint8_t* data; // build cuDF column base pointer (fixed-width)
  int32_t acc_offset;  // where it lands in the accumulator
  int32_t byte_width;
};

// One fused join step (one edge of the join chain).
struct FusedJoinStep {
  GpuHashTable table;              // build-side hash table (key -> row index)
  const FusedKeyField* keyFields;  // [numKeys] where to read the key from acc
  int32_t numKeys;

  // ---- Composite-key PACKING (single-word fast path) ----
  // When packEnabled, the numKeys key columns are packed into ONE uint64 word:
  //   packed = OR_j ( (val_j - packMin[j]) << packShift[j] )
  // and table.numKeyWords == 1, so tableFind uses the fast 128-bit-load path.
  // cudf's distinct/hash join, by contrast, hashes and compares the key
  // COLUMN BY COLUMN (numKeys separate arrays touched per probe) -- so packing is
  // where the row-native design pulls ahead as the key widens.
  //
  // A probe value outside [packMin[j], packMax[j]] can never match a build row,
  // so its packed key is forced to packSentinel (a value no build key produces),
  // which misses. (Assumes non-negative fixed-width keys whose packed width
  // fits in <= 63 bits; the build falls back to the multi-word path otherwise.)
  int32_t packEnabled{0};
  int64_t packMin[kMaxKeyCols];
  int64_t packMax[kMaxKeyCols];
  int32_t packShift[kMaxKeyCols];
  uint64_t packSentinel;
  // Byte offset of each build key column within the build row store. Used only
  // by fingerprint-slot verification: on a fingerprint match, re-pack the build
  // key from buildStore[index] and compare to the probe key. Populated in the
  // SAME key order as keyFields/packMin/packShift.
  int32_t buildKeyOffset[kMaxKeyCols];
  GpuFixedRowStore buildStore;     // build payload rows (to append fields from)
  const FieldMapping* buildToAcc;  // [numBuildFields] build field -> acc offset
  int32_t numBuildFields;

  // COLUMNAR build fetch (skips the row-store transpose). When set, append build
  // fields from `buildCols[f].data + br*width` instead of buildStore. Only used
  // for non-fingerprint builds (fingerprint verify still needs the row store).
  int32_t columnarBuild{0};
  const ColBuildField* buildCols;  // [numBuildFields] when columnarBuild
};

// ============================================================================
// Host entry points
// ============================================================================

/// Allocate + build a device hash table from `numRows` composite keys.
/// `d_key_words` is a device array of [numRows * numKeyWords] uint64, where
/// word (r*numKeyWords + c) is key column c of row r, zero-extended to 64 bits.
/// `d_slots` must be pre-allocated with `capacity * (numKeyWords + 1)` uint64
/// words (capacity must be a power of two, load factor <= 0.5 recommended).
/// The slot val words are pre-initialized to GpuHashTable::kEmptyWord here.
void buildHashTable(
    const uint64_t* d_key_words,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    int32_t numKeyWords,
    cudaStream_t stream);

/// Compute the recommended power-of-two capacity for `numRows` (load <= 0.5).
int32_t fusedTableCapacity(int32_t numRows);

/// Build the device hash table DIRECTLY from a fixed-stride row store: sentinel
/// the slots (memset) and insert every row, computing its composite key inline
/// from the store. Replaces buildKeyWords + buildHashTable (no intermediate
/// key-word array, no separate packing pass). `d_slots` must be pre-allocated
/// with `capacity * (numKeyWords + 1)` uint64 words.
void buildHashTableFromStore(
    const GpuFixedRowStore& store,
    const int32_t* d_keyOffsets,
    const int32_t* d_keyWidths,
    int32_t numKeyWords,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream);

/// Build the device hash table reading composite keys from the COLUMNAR build
/// input (one ColSeedField per key column: base pointer + width). Lane i reads
/// col[i], so key loads are coalesced (vs the row store's strided i*row_width).
void buildHashTableFromColumns(
    const ColSeedField* d_keyCols,
    int32_t numKeyWords,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream);

/// Build a SINGLE-WORD hash table from `numKeyCols` columns packed into one
/// uint64 per row: OR_j ((val_j - mins[j]) << shifts[j]). Build values are in
/// range by construction, so no sentinel handling here. `d_mins`/`d_shifts` are
/// device arrays of length numKeyCols. Slots are 2 words each ([packedKey, val]).
void buildHashTableFromColumnsPacked(
    const ColSeedField* d_keyCols,
    int32_t numKeyCols,
    const int64_t* d_mins,
    const int32_t* d_shifts,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream);

/// Build a COMBINED single-word table: each slot holds [key | (index<<keyBits)]
/// in one uint64 (mirrors cuco's 8-byte (key,index) slot). Halves the table vs
/// the [key,val] layout and lets each insert write key+index with a single
/// atomicCAS. Requires keyBits + indexBits <= 63.
void buildHashTableFromColumnsCombined(
    const ColSeedField* d_keyCols,
    int32_t numKeyCols,
    const int64_t* d_mins,
    const int32_t* d_shifts,
    int32_t keyBits,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream);

/// Build a FINGERPRINT single-word table for a composite/wide key that will not
/// pack exactly into 63 bits: each slot is [fp32 | (index << fpBits)] where fp32
/// is a 32-bit hash of the (exactly-packed) key. Mirrors cuco's pair<hash,index>
/// 8-byte slot. `fpBits` is the fingerprint field width (== 32). The probe must
/// verify the real key on a fingerprint match.
void buildHashTableFromColumnsHashed(
    const ColSeedField* d_keyCols,
    int32_t numKeyCols,
    const int64_t* d_mins,
    const int32_t* d_shifts,
    int32_t fpBits,
    int32_t numRows,
    uint64_t* d_slots,
    int32_t capacity,
    cudaStream_t stream);

/// Pack composite keys out of a fixed-stride row store into the [numRows *
/// numKeyWords] uint64 word array expected by buildHashTable(). Word
/// (r*numKeyWords + c) is key column c of row r (at byte offset keyOffsets[c],
/// width keyWidths[c]) zero-extended to 64 bits. keyOffsets/keyWidths are
/// device arrays of length numKeys (== numKeyWords).
void buildKeyWords(
    const GpuFixedRowStore& store,
    const int32_t* d_keyOffsets,
    const int32_t* d_keyWidths,
    int32_t numKeys,
    uint64_t* d_out,
    cudaStream_t stream);

/// Launch the fused N-way probe.
/// One thread per probe row seeds an accumulator from the probe row (via
/// `d_probeToAcc`), walks all `numSteps` steps (each: read key from acc ->
/// probe table -> on hit append build fields into acc, on miss drop the row),
/// and on survival appends the compacted output row (via `d_accToOut`).
///
/// `d_steps` is a device array of `numSteps` FusedJoinStep (pointers inside
/// each step must also be device pointers). `d_outCount` is a device int32
/// initialized to 0; on return it holds the number of surviving rows.
///
/// SEED (pick one):
///   d_colSeed != nullptr  -> read the probe straight from cuDF columns (OPT 1;
///                            `probe` is then unused and numRows must be given)
///   d_colSeed == nullptr  -> read the probe from the `probe` row store
/// OUTPUT (pick one):
///   d_colOut != nullptr   -> scatter survivors straight into cuDF columns
///                            (OPT 2; d_out/outRowWidth then unused)
///   d_colOut == nullptr   -> append survivors to the `d_out` row buffer
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
    cudaStream_t stream);

} // namespace facebook::velox::cudf_velox
