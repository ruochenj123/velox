# Review Guide — Round 2 (payload-width gate + zero-copy string extraction)

Scope: only the changes made after the Area-1 review. Everything else on
`hybrid-merge` was reviewed already (see REVIEW-GUIDE.md).

Motivation (measured, exp/.../results/prof_extract): the hybrid join's
getOutput lost to baseline for two separate reasons — a fixed ~5.4 ns/row
row-id decode tax paid even when NO build payload column is projected
(Q17: the entire 2.04x penalty), and, for VARCHAR-heavy payloads, string
BODIES re-copied from random offsets of huge coalesced buffers (Q10:
9-13 ns per column-row). Fix 1 removes the first by not enabling hybrid at
all; Fix 2 removes the second by never copying bodies during extraction.

---

## FIX 1 — plan-time payload-width gate

The paper's sentence ("hybrid enabled only when the projected payload size
exceeds a configurable byte threshold") implemented literally.

### Files
- `velox/core/QueryConfig.h:1053` `hybrid_join_min_payload_bytes` (default
  **24**), `:1066` `hybrid_sort_min_payload_bytes` (default **1**).
- `velox/exec/OperatorUtils.h:341` `hybridPayloadNominalWidth(type)`:
  exact `cppSizeInBytes()` for fixed-width; **32** for VARCHAR/VARBINARY
  (16B view + 16B nominal body); 16 for other variable-width.
- `velox/exec/HashBuild.cpp:139-192` — the gate (read it as one block).
- `velox/exec/SortBuffer.cpp:99-113` + `OrderBy.cpp:74` — sort sibling.
- `velox/benchmarks/QueryBenchmarkBase.cpp:129-141,320-323` — gflags.

### What the gate counts (the subtle part)
Only dependents the probe ACTUALLY READS:
- read-by-output: name present in `joinNode_->outputType()` — mirrors how
  HashProbe builds `tableOutputProjections_`;
- read-by-filter: name referenced by `joinNode_->filter()` AND absent from
  the probe type — mirrors `HashProbe::initializeFilter()`'s probe-first
  resolution.
A dependent that is stored but never read contributes **0** (this is
exactly Q17's join[5], and why it now gates off).

### Review questions
1. Is the read-set derivation faithful to HashProbe? If HashProbe resolves
   a column differently (e.g. duplicate names across sides, or a filter
   field that exists on both sides), the gate could over- or under-count.
   The probe-first rule at :172-177 is the one to scrutinise.
2. The filter walk (`collectFields`, :150-163) is a recursive
   `dynamic_cast` over `ITypedExpr` — are there expression node types that
   reference fields without a `FieldAccessTypedExpr` (lambdas, subfield
   access)? Under-counting there disables hybrid unnecessarily (safe
   direction, but worth knowing).
3. Threshold semantics: gate is `readPayloadBytes < threshold -> disable`.
   With VARCHAR=32 nominal, ANY single string payload passes the default
   24. That is why q12 is not gated (see "known wart" below).
4. Sort side: width is over all non-sorted columns because sort outputs
   all of them. Default raised 1 -> **8** (user review, 2026-08-12): the
   break-even implied by the design is that hybrid swaps sum(payload
   widths) bytes/row in the container for ONE 8-byte row reference, so
   the container only shrinks above 8B. Confirmed on the sort sweep
   (OrderBy CPU, hybrid/baseline): 24B 1.24x, 16B 1.07x, 8B 1.03x (wash,
   still enabled), 0B already self-disabled. Below 8B the reference costs
   more than the payload it replaces. Note the join threshold (24) has a
   different rationale (per-output-row decode tax + fan-out) and may be
   worth re-deriving after the PROCREPS=4 rerun now that Fix 2 removed
   the extraction penalty.
5. `stats_.wlock()->addRuntimeStat(...)` at :188-190 is called in the ctor
   — confirm that is safe/allowed at operator construction time (it is
   used elsewhere in ctors, but worth a look).

### Known wart (flagged, not fixed)
q12's single read dependent is `o_orderpriority`, a VARCHAR -> nominal 32
>= 24, so hybrid stays ON. Harmless today (q12 measures 1.00x after Fix 2)
but nominal-32 overstates short strings. Options if we want it gated:
raise the threshold above 32, or estimate VARCHAR width from stats.

---

## FIX 2 — zero-copy string extraction

Three pieces; the second and third exist because the naive version failed
in measurable, non-obvious ways. Review them together.

### (a) Extraction kernels copy views only
`velox/exec/RowContainer.h` — the `extractPayload{WithNulls,NoNulls}
{SingleContainer,}` family (~2700-2900 and ~3300-3560). For
VARCHAR/VARBINARY the kernels now write the raw 16-byte `StringView`
(`values[i] = v`) instead of `FlatVector<StringView>::set()`, which would
copy the body. Inline strings (<=12B) were always by-value — unchanged.
Call sites of the buffer helper: :2789, :2896, :3356, :3550.

### (b) Coalesce-time body compaction
`RowContainer.h` inside `coalesceBatches()` (~2355-2430). Why it exists:
`BaseVector::copy`'s string path SHARES source buffers, so the merged
child would hold one BufferPtr per input batch (~6K buffers for q10@SF100)
and each extraction call would then have to acquire thousands of buffers —
measured **q10 getOutput 55.5s, 15.9x baseline**. Compaction copies bodies
once, sequentially, into a single exact-size buffer per string column
(pass 1 sizes it, pass 2 copies views+bodies). Review questions: the
inline/out-of-line split at :2378 and :2411; the body-offset arithmetic at
the `StringView(bodyData + bodyOffset, v.size())` construction; and that
this cost lands inside the *overlapped* coalesce (so it is off the
critical path — verify it is not accidentally on the operator thread).

### (c) Honest-sized BufferViews instead of raw acquire
`RowContainer.h:1989` `HybridPayloadBufferReleaser`, `:2472`
`addSharedStringBufferViews()`. Why it exists: acquiring the whole
compacted buffer made every output batch report a `retainedSize()` of the
entire buffer (probe output reported **43-56 TB**), and `LocalPartition`
enqueues by `retainedSize()`, so its MB-scale queue limits admitted ~1
batch and serialized the pipeline — measured **q21 0.54x, q16 0.64x**.
Now each extraction call accumulates the bytes it actually references and
adds ONE `BufferView` sized to that, whose releaser holds the real
`BufferPtr` (lifetime identical to acquiring; accounting honest).
Review questions:
1. **Lifetime**: the view's data pointer is `buffer->as<uint8_t>()` and the
   releaser holds `BufferPtr buffer` — confirm the releaser really keeps
   it alive (addRef/release are no-ops by design; the member shared_ptr is
   the owner) and that nothing resets the container's coalesced base while
   output batches are in flight.
2. **Sizing** — RESOLVED during review (2026-08-12), kept here as the
   argument: the attached view can be SMALLER than the region actually
   addressed by the StringViews it accompanies (`referencedBytes` counts
   bytes referenced, not a contiguous extent). Verified safe because
   (i) `BufferView` sets `capacity_ == size_` (Buffer.h:291 via the ctor,
   :819), so `FlatVector<StringView>::getBufferWithSpace`'s reuse test
   `size() + n <= capacity()` is always false — a downstream writer can
   never append into our view and will allocate its own buffer;
   (ii) `Buffer::isMutable()` is `!isView() && unique()`, false for any
   view; (iii) no code path validates StringView pointers against
   stringBuffer extents (grep: no range checks in the vector layer).
   Consumers dereference through the views themselves, so the declared
   size only feeds `retainedSize()` accounting — its intended purpose.
3. The `first`/else branch at :2481-2492 assumes compaction leaves at most
   one buffer; the else-branch acquires extras outright. Is that
   reachable, and is it correct if so?
4. Scattered kernels intentionally KEEP body copies (per-batch acquire is
   O(contributing batches) per call and loses when batches are numerous) —
   a documented asymmetry between the coalesced and scattered paths.

---

## Gate results this round (for reviewing against claims)
- byte-identical smokes: q31 p4 and p12 (both with VARCHAR payloads),
  q40 k2p4 — baseline vs coalesced vs scattered, all identical.
- gate behaviour: q17 all joins off (16/0/0 B), q9 big joins on (40B),
  TPC-DS winners on (128-328B); row counts match baseline.
- phases (hybrid/baseline getOutput CPU): q10 d1 1.37 -> **0.60**,
  d16 1.70 -> **0.94**; q17 -> ~1.0; q9 ~1.2 (2-rep noise).
- suites: TPC-H 22q floor 0.93 -> **0.96** (all remaining losses in the
  0.96-0.98 noise band); TPC-DS exhibit aggregate -1.3% -> **-6.7%**,
  q81 +9.0% -> -7.6%.
- caveat: all PROCREPS=2. Figure-grade numbers need the PROCREPS=4 rerun.
NULL-payload finding -> review notes
## Area 2 finding (2026-08-17, from fig-1b Q30 gate): NULL payloads silently corrupted
Real Q30@SF100 join inputs: build payload with 400 NULLs of 65,340 rows.
col vs rowpack(+row_table) gate: identical row counts (65,340) but DIFFERENT
content checksums (de3815027b562277 vs b2c94cc8db3343e7). Zero NULLs elsewhere.
Strong indication the row-store pack/gather drops or fabricates null payload
values while the cudf path preserves them. Expected behavior per the design's
own rule: HARD FAIL on unsupported batches (CudfConversion.cpp ~630), not
silent divergence. Reproducer: exp/2026-08-17-fig1b-colsweep/q30real.sh with
the pre-2026-08-17 build_wr.parquet (regenerate without COALESCE).

## Null support for the row path (implemented overnight 2026-08-17/18)

### Design: sidecar null bitmap, bit SET = null
One byte-array per store, `null_bytes[num_rows * null_stride]`,
`null_stride = ceil(num_fields / 8)`, addressed by FIELD INDEX:
`null_bytes[row * stride + (f >> 3)] & (1 << (f & 7))`. Bit set = NULL, so a
zeroed (or absent) sidecar means all-valid, and every legacy construction
site is null-free by default. The sidecar rides in the TAIL of the pinned
pack slab and the RowStoreVector's device buffer (rows region, then null
region) so the existing single H2D carries both. NOTE the polarity is the
INVERSE of Velox/Arrow validity (their bit set = valid); conversions happen
at exactly two places: pack-in (bits::isBitNull) and mask-out
(sidecar_to_mask_kernel).

### Diff map (review anchors)
- GpuFixedRowStore.h — `null_bytes`/`null_stride` appended with NSDMI
  defaults (legacy partial initialization stays safe).
- RowStoreVector.h — `setNullSidecar()/nullStride()`; getGpuRowStore() wires
  the tail pointer.
- CudfConversion.cpp tryPinnedPack — nullable FLAT children now packable in
  row mode (boundary keys-only mode and col-major mode keep the refusal);
  slab sized rows+nulls; sidecar filled in a second pass from rawNulls;
  emitted vector marked via setNullSidecar.
- RowHashJoin.h — BuildData gains nullBuffer/nullStride; probe gains four
  per-mapping field-index device buffers + outputNullBuffer_/stride.
- RowHashJoin.cpp — build concat concatenates sidecars (zero-fill for
  null-free batches; strides must agree); output-layout pass records each
  mapping's src/dst FIELD index; gather allocates+zeroes the output sidecar
  when either input has one; makeColumnarOutput converts sidecar ->
  per-column Arrow masks (sidecar_to_mask_kernel) with cudf::null_count
  (one sync per nullable output batch, nullable path only).
- GpuRowOps.cuh/.cu — NullGatherArgs (all-optional, NSDMI); gather kernel
  copies null bits per mapping (thread owns its output row: plain OR, no
  atomics); sidecarToMask kernel (word-per-thread, mask bit = !sidecar bit).

### Scope and residual limits
- Null KEYS remain unsupported (matcher reads key bytes raw). Callers must
  drop or reject them; inner-join semantics permit dropping. Not enforced
  device-side — review question: enforce at pack time?
- The GPU-transpose ingest path (cudf table -> row store, used when inputs
  are NOT RowStoreVectors) has NO sidecar; the Level-1 hard-fail guard
  (checkTransposeInputNullFree, 3 call sites) makes it loud instead of
  silently corrupting.
- RowStoreVector OUTPUT chains (emitRowStore across joins) do not propagate
  the output sidecar yet — single-join columnar output only.
- Boundary-hybrid payloads were always null-safe (host HybridContainer);
  boundary KEY pack keeps its hard fail.

### Validation (job nullval, 3 phases)
A. null-free gate col==rowpack (no regression);
B. 400-real-NULL gate col==rowpack (end-to-end sidecar);
C. transpose path on NULLs -> guard fires.

### Validation VERDICT (job nullval 13660925, 2026-08-18)
A null-free: PASS (9622e9f9210add97 both arms -- bit-identical to pre-change).
B 400-real-NULLs: PASS (de3815027b562277 both arms -- rowpack now reproduces
  the checksum the columnar reference produced when the original corruption
  was discovered; end-to-end null path proven on real TPC-DS data).
C transpose-path guard: FIRED (exit 134, "cannot represent" in stderr).

## Addition (2026-08-19): composite-key range packing in the row-native matcher

Motivation: multi-key joins should pack K key columns into ONE uint64 word
(Eiger does this; GpuFusedProbe already had it) so hash/fingerprint/verify
touch one word instead of K. User-requested during Area 2 review.

### Design
`RowKeyPack` (GpuRowHashTable.cuh): packed = OR_k ((v_k - mins[k]) << shifts[k]).
- mins/spans = BUILD side per-key min/max over raw key words interpreted
  UNSIGNED (exactly as loadKeyWord zero-extends; signed data with mixed
  signs just yields a huge span and packing auto-disables — correct, not
  wrong). Injective within range => packed equality <=> key equality.
- Probe keys out of range cannot match: rejected BEFORE hashing
  (`rel = v - min; rel > span` catches v < min via unsigned wraparound).
- Engages iff sum of per-key bit-widths <= 64 AND numKeys >= 2.
- Layout-independent: same pack applies to build and probe RowKeyLayouts.

### Diff map
- GpuRowHashTable.cuh: RowKeyPack POD; RowNativeHashTable.pack field;
  rowTableComputeKeyMinMax() decl.
- GpuRowHashTable.cu: loadEffectiveKey() (fold or passthrough, range check);
  keyEqualsBuildRow now compares effective keys (packs the build row on the
  fly — no stored packed keys); insertRowsChained/probeChainedKernel use
  effective keys; keyMinMaxKernel (grid-stride, thread-local accumulate,
  one atomicMin/Max pair per thread per key).
- RowHashJoin.cpp buildRowTable lambda: min/max pass + 2K-word D2H + sync
  (ONE extra sync per join, at build finalize), bit-width decision,
  shift-64 UB guard (zero-span keys), runtime stat rowTablePackedKeyBits.
- CudfConfig.h: benchmarkRowTablePackKeys{true} (appended at end, stale-
  object layout rule).
- Bench: --row_table_pack_keys (default true).

### Review hotspots
- Unsigned min/span/wraparound reasoning in loadEffectiveKey.
- Fingerprint domain changes when packed (hash over 1 word) — build and
  probe must agree, which they do because both go through loadEffectiveKey
  with the same t.pack; a mismatch would show as zero matches, caught by
  the gate.
- Null keys: same unsupported status as the unpacked table (row stores
  carry no key null masks); packing does not change that contract.

### Validation (job packgate)
SF1, 3 arms (col reference / row_table pack off / pack on) x 4 key configs
(k=1 no-op, k=2, k=3, k=2 + build_key_mod=1000 N:M): rows+checksum must
match col for BOTH row_table arms; multi-key pack arms must log
rowTablePackedKeyBits (engagement proof). Then SF30 join_keys=3 timed
preview (pack on/off/col, 24 drivers).

### v2 (same day): stored packed keys + exact fingerprint
v1 measured PERF-NEUTRAL at SF30/k3 (130.2 vs 132.7 ms/driver): our table
stores row indices, not keys, so the packed verify still loaded all K
components from the build row to fold them — packing only saved hash ALU.
(Eiger's win comes from comparing a STORED packed key.) v2 fixes the access
pattern:
- pack enabled, bits in (32, 64]: build writes packedKeys[numRows] (dense
  uint64 array, +8B/build row); verify = ONE coalesced load
  `packedKeys[r] == key[0]` instead of K strided build-row loads.
- bits <= 32 (pack.exactFp): the chain fingerprint IS the packed key —
  fp match is exact match, verify disappears; no packedKeys array either.
- keyEqualsBuildRow reverted to the plain unpacked compare (packed arms
  never call it).
Diff: RowKeyPack.exactFp + RowNativeHashTable.packedKeys (cuh);
chainFingerprint() + verify ladder in probe, insert writes packedKeys (cu);
BuildData.rowTablePackedKeys buffer + exactFp decision (RowHashJoin.h/.cpp).
Gate adds k2xfp (mod=60000 -> 2x16-bit keys = 32 bits, exercises exactFp).

### v2 VERDICT (job packgate 13741962, 2026-08-19)
Correctness: ALL FIVE configs PASS bit-exact vs cudf reference (k1, k2=43b
stored-key, k3=60b stored-key, k2nm=40b N:M, k2xfp=32b exactFp verify-free).
Performance at SF30/k3/24dr: rt_pack 126.2 vs rt_nopack 125.6 ms/driver —
NEUTRAL even with stored keys / exact fingerprints. Interpretation: in a
CHAINED table the 32-bit fingerprint already reduces key verification to
~once per true match, so range packing has nothing left to save at
realistic key cardinalities — packing's win in Eiger-style open addressing
comes from per-slot key compares that this design never does. Paper-ready
claim: fingerprinted chaining subsumes composite-key packing for the
matcher; the multi-key advantage over cudf comes from row-native key access
(col probe 21.5 -> rt 5.3 ms/driver at k=3, and col transfer grows with K
while the packed row crosses once). Keep --row_table_pack_keys for the
sweep's packed-vs-unpacked line; feature stays default-ON (harmless, never
slower, engages only when ranges fit).

## Post-review fixup queue (apply AFTER the user finishes the Area 2 pass)
- Rename buildNullBuffer2/buildNullStride2 -> buildNullBuffer/buildNullStride
  (RowHashJoin.cpp L410-411 + uses). The "2" suffix is residue from the
  null-session scoping fix; no un-suffixed originals remain. Semantics-free;
  incremental rebuild, no gate required.
