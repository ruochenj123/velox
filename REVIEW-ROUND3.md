# Review Round 3: mode-consolidation refactor (2026-08-20)

Scope agreed with the user after the Area 2 review: remove the fused path,
merge the boundary/main probe duplication, relax the boundary null refusal,
close the null-key gap, apply queued renames. Whole-query deferral (design
doc + implementation) starts AFTER this round is reviewed.

Baseline: checkpoint commit fc4bffe49 ("Checkpoint: null sidecar,
boundary-hybrid store, row-native table + key packing").

## 1. Fused path removal
- `GpuFusedProbe.cu/.cuh` -> `exec/attic/` (with README; operator-side code
  recoverable from the checkpoint commit).
- RowHashJoin.cpp: deleted `buildFusedMap` (+ pack locals), `wantFusedMap`/
  `skipHashJoin` decision, the columnar-build-fetch experiment
  (VELOX_CUDF_COL_BUILD), the key-row-store support, and the whole
  `FusedRowHashJoinProbe` implementation (~40KB total).
- RowHashJoin.h: `FusedRowHashJoinProbe` class + BuildData fields
  (deviceMap/mapSlots/pack*/columnarBuild/buildTable/keyRow*) removed.
  BuildData is only touched by TUs in the incremental rebuild set, so field
  removal is ABI-safe there.
- ToCudf.cpp: fused pre-pass removed; `isAnyOf<FusedRowHashJoinProbe,
  CudfBatchConcat>` lists reduced to CudfBatchConcat (still needed by the
  rebatch path); markProbeFused/isProbeFused definitions removed.
- CudfConfig.h: fused DATA members kept as deprecated stubs (append-only
  ABI rule for stale objects); methods removed; docs replaced with
  deprecation notes. `benchmarkDistinctHashJoin` kept (used by the columnar
  CudfHashJoin, not fused).
- Stale `GpuFusedProbe.cu.o` deleted from the prebuilt archive (`ar d`).
- REVIEW HOTSPOT: the BuildData handoff block was accidentally over-cut and
  restored from the checkpoint (gpuRowStore/rowBuffer/fieldsBuffer/hashJoin/
  keyBuffers/numRows/rowWidth/hostFields assignments + stream.synchronize()).
  Worth one look: RowHashJoin.cpp, "Push to bridge" block.

## 2. boundary/main probe merge (runMatcher)
New private helper `RowHashJoinProbe::runMatcher(probeStore, keyDescs,
probeRows, stream, leftIndices, rightIndices)` — the shared MATCHER front
half (row-native table vs extract+cudf fork, including matcherWallNanos
timing). The two former copies differed ONLY in key addressing:
- regular path: keyDescs[k] = probeFields_[leftKeyIndices_[k]]
- boundary path: keyDescs[k] = input's hostFields()[k] (keys-only layout)
Both call sites now build a keyDescs vector and call the helper. Output
halves remain separate by design (GPU gather+transpose vs id D2H + host
gather) — that is the next natural seam if we ever want one doGetOutput.

## 3. Boundary null relaxation (CudfConversion.cpp)
Old: boundary hard-failed if ANY child was nullable/non-flat. New:
- payload channels under boundary are UNCONSTRAINED here (skipped entirely):
  BoundaryHostStore/HybridContainer::addPayload owns their loading,
  flattening, and null handling.
- key channels keep the strict requirements, with the hard fail now scoped
  to keys.
This unlocks nullable-payload workloads (TPC-DS) for the boundary arm;
gate config B_bnd covers exactly that (400 payload NULLs through the
boundary path, checked against the columnar reference).

## 4. Null-KEY guard (CudfConversion.cpp)
Key names of the adjacent row-join op are now captured for EVERY row pack
(not just boundary); channels resolved lazily on first pack. A key column
with actual NULLs (countNulls > 0) VELOX_FAILs with a clear message —
closing the silent-corruption gap (matcher hashes raw key bytes; NULL must
never match). A mayHaveNulls-flagged but actually clean key column is
accepted directly (does NOT force the per-batch columnar fallback).
NOT gate-covered: no dataset with null keys in the suite; the guard is
fail-loud by construction.

## 5. Renames
buildNullBuffer2/buildNullStride2 -> buildNullBuffer/buildNullStride.
File reorg kept light-touch per discussion: attic/ + updated header
comments; GpuRowOps.cu / GpuRowHashTable.cu split retained.

## Validation (job refgate)
One binary, three sections:
1. pack/matcher matrix: k1/k2/k3/k2nm/k2xfp x {col, row_table pack-off,
   row_table pack-on}, bit-exact vs col + pack-engagement stat.
2. boundary arms: col vs boundary+row_table vs boundary+cudf (SF1 k1).
3. null phases: A/B col==rowpack (null-free / 400 NULLs), A_bnd/B_bnd
   col==boundary (B_bnd exercises the relaxation), C transpose guard fires.

## Gate round 1 (job 13767525): boundary FAIL -> bug found & fixed
Pack matrix (5/5) and null phases (A/B/C) passed. ALL boundary configs
SEGV'd in tryPinnedPack. Root cause: the null relaxation made `keepAlive`
SPARSE (payload channels skipped), but the boundary host-retention block
indexes it positionally as `keepAlive[b * numCols + c]` -- out-of-bounds
VectorPtr reads. This dependency was implicit (dense push_back happened to
line up with positional indexing). Fix: `keepAlive` is now an explicitly
positional array; skipped payload channels still get LOADED into their slot
(host extraction needs loaded children) but carry no flat/null constraints.
Note: boundary was NEVER covered by a gate after the 08-17 null rework --
the new bnd/A_bnd/B_bnd configs exist precisely to close that hole, and
they caught this on first run.

Also learned: boundary + GPU re-batching (gpu_batch > batch) has never been
supported -- CudfBatchConcat sits between CudfFromVelox and the join, so the
keys-only resolution (which asks the NEXT operator for join keys) never
engages. Pre-existing design constraint, now documented in the gate script;
the whole-query deferral design should decide whether to lift it.
