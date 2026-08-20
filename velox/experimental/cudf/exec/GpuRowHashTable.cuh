/*
 * GpuRowHashTable.cuh
 *
 * ROW-NATIVE N:M GPU hash table for a SINGLE hash join: a classic chained
 * (separate-chaining) multimap built over the build side's fixed-stride
 * KEY ROW STORE, probed with keys read straight out of the probe side's
 * row store. It replaces cudf::hash_join as the join MATCHER: match-finding
 * itself reads row-format keys — no key extraction to columnar buffers on
 * either side.
 *
 * This de-chains the fused-probe row-table design (GpuFusedProbe.cuh) into a
 * general single-join matcher:
 *   - the fused GpuHashTable is single-match (open addressing, one build row
 *     per key -> N:1 only) and entangled with chain walking / accumulators;
 *   - this table is a true multimap: slot array of head indices + a per-build-
 *     row next[] chain, so duplicate build keys yield ALL (probe,build) pairs
 *     (N:M), mirroring cudf::hash_join::inner_join's count+retrieve contract.
 *
 * Reused fused-era ideas (adapted, not shared code):
 *   - composite-key hash: mix64(fnv-chained) over zero-extended key words
 *     (identical formula to GpuFusedProbe.cu's hashKey);
 *   - fingerprint filtering: each chain entry carries a 32-bit hash
 *     fingerprint packed next to its next-pointer in ONE uint64
 *     ([fp32 | next32] — the 8-byte packed-slot idea), so a chain step costs
 *     a single 8-byte load and false key compares are ~2^-32;
 *   - device-side key verify against the row store (the "fingerprint verify"
 *     pattern): on fp match, compare the real key field-by-field from the
 *     build key row via FieldDesc-style offsets/widths.
 *
 * Layout / semantics:
 *   - heads[capacity]: int32 build-row index or -1 (empty). capacity is the
 *     first power of two >= 2 * buildRows (load factor <= 0.5 on slots).
 *   - nextFp[buildRows]: low 32 bits = next build row in this slot's chain
 *     (0xFFFFFFFF terminates), high 32 bits = key fingerprint.
 *   - insert: hash key row -> slot; old = atomicExch(&heads[slot], row);
 *     nextFp[row] = fp<<32 | old.  Lock-free, one atomic per build row.
 *   - probe (two-pass, mirroring cudf's count+retrieve):
 *     pass 1: per probe row, read the probe KEY ROW ONCE, walk the chain,
 *             verify keys, count matches; exclusive scan (cub) over counts;
 *     pass 2: re-walk, write (probeIdx, buildIdx) pairs at scanned offsets.
 *   - fixed-width keys (4/8 bytes), 1..kRowTableMaxKeys composite columns.
 *   - null keys: not representable — the row stores carry no null masks
 *     (parity with the existing RowHashJoin path, whose extracted key
 *     column_views are built with a null null-mask and probed with
 *     null_equality::UNEQUAL; the bench data has non-null keys).
 *
 * This header has NO Velox/folly/rmm dependencies so it can be included from
 * both host (.cpp) and device (.cu) translation units. All device memory is
 * allocated by the caller.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

#include "GpuFixedRowStore.h" // FieldDesc, GpuFixedRowStore

namespace facebook::velox::cudf_velox {

// Max composite key columns. Bounds the fixed arrays in RowKeyLayout and the
// per-thread key-word array in the probe kernels.
static constexpr int kRowTableMaxKeys = 8;

/// Where the key columns live inside a fixed-stride row store.
/// offset/width are in bytes; width is 4 or 8 (fixed-width integer keys).
/// Probe and build layouts may differ in offsets but must agree in widths
/// and column order.
struct RowKeyLayout {
  int32_t numKeys{0};
  int32_t offset[kRowTableMaxKeys];
  int32_t width[kRowTableMaxKeys];
};

/// Range packing of a composite key into ONE uint64 word (Eiger-style /
/// GpuFusedProbe-style): packed = OR_k ((v_k - mins[k]) << shifts[k]).
/// mins/spans are the BUILD side's per-key value range (raw key words
/// interpreted as unsigned, exactly as loadKeyWord zero-extends them), so
/// build keys pack losslessly by construction; a probe key with any
/// component outside [mins[k], mins[k]+spans[k]] cannot match any build row
/// and is rejected before hashing. Within range the mapping is injective,
/// so packed equality <=> component-wise key equality: hash, fingerprint,
/// and verify all operate on the single packed word.
/// Engaged only when sum_k bits(spans[k]) <= 64 (decided at build time).
/// Layout-independent: the same pack applies to build and probe key layouts
/// (mins/shifts are per key INDEX, offsets come from each side's layout).
struct RowKeyPack {
  int32_t enabled{0};
  /// Total packed bit-width <= 32: the chain fingerprint IS the packed key,
  /// so a fingerprint match is an exact match and the per-match key verify
  /// is skipped entirely (no build-side access during the walk).
  int32_t exactFp{0};
  uint64_t mins[kRowTableMaxKeys];
  uint64_t spans[kRowTableMaxKeys]; // max - min (inclusive span), unsigned
  int32_t shifts[kRowTableMaxKeys]; // bit position of key k in the word
};

/// POD device handle (memcpy-safe, passed by value to kernels).
struct RowNativeHashTable {
  int32_t* heads{nullptr};   // [capacity] head build-row index, kEmpty if none
  uint64_t* nextFp{nullptr}; // [numRows] (fp32 << 32) | next32 (kChainEnd ends)
  int32_t capacity{0};       // power of two, >= 2 * numRows
  int32_t numRows{0};        // build rows
  GpuFixedRowStore buildStore{}; // build key rows (keys-only or full row store)
  RowKeyLayout keys{};       // key columns within buildStore rows
  RowKeyPack pack{};         // optional composite-key range packing
  /// [numRows] packed key per build row, present iff pack.enabled (and not
  /// needed when pack.exactFp). Verify then compares ONE dense 8-byte load
  /// against the probe's packed word instead of K strided loads from the
  /// build row — the stored-packed-key access pattern Eiger relies on.
  const uint64_t* packedKeys{nullptr};

  static constexpr int32_t kEmpty = -1;
  static constexpr uint32_t kChainEnd = 0xFFFFFFFFu;
};

/// First power of two >= 2 * numRows (>= 2).
int32_t rowTableCapacity(int32_t numRows);

/// Per-key min/max over the build store's key words (unsigned, as loaded).
/// d_minmax layout: [numKeys mins][numKeys maxs], each uint64. Caller must
/// pre-fill mins with 0xFF bytes (UINT64_MAX) and maxs with 0x00. Used to
/// derive RowKeyPack at build time; one pass over numRows * numKeys words.
void rowTableComputeKeyMinMax(
    const GpuFixedRowStore& store,
    const RowKeyLayout& keys,
    uint64_t* d_minmax,
    cudaStream_t stream);

/// BUILD: sentinel heads (memset 0xFF == kEmpty) and insert every build row.
/// One kernel launch pair; t.heads/t.nextFp must be pre-allocated
/// ([capacity] int32 / [numRows] uint64).
void buildRowNativeHashTable(const RowNativeHashTable& t, cudaStream_t stream);

/// cub temp-storage bytes needed by rowTableScanOffsets for numRows counts.
size_t rowTableScanTempBytes(int32_t numRows);

/// PROBE pass 1: d_counts[i] = number of build rows matching probe row i.
void rowTableProbeCount(
    const RowNativeHashTable& t,
    const GpuFixedRowStore& probe,
    const RowKeyLayout& probeKeys,
    int32_t numProbeRows,
    int32_t* d_counts,
    cudaStream_t stream);

/// Exclusive scan d_counts -> d_offsets ([numRows] each) and write the total
/// match count into d_total (device int32).
void rowTableScanOffsets(
    const int32_t* d_counts,
    int32_t* d_offsets,
    int32_t numRows,
    void* d_temp,
    size_t tempBytes,
    int32_t* d_total,
    cudaStream_t stream);

/// PROBE pass 2: refill (probeIdx, buildIdx) pairs into d_leftIdx/d_rightIdx
/// at the scanned offsets. Arrays must hold `total` elements.
void rowTableProbeFill(
    const RowNativeHashTable& t,
    const GpuFixedRowStore& probe,
    const RowKeyLayout& probeKeys,
    int32_t numProbeRows,
    const int32_t* d_offsets,
    int32_t* d_leftIdx,
    int32_t* d_rightIdx,
    cudaStream_t stream);

} // namespace facebook::velox::cudf_velox
