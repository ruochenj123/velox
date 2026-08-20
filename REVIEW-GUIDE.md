# Review Guide — hybrid-merge branch

Two review areas, ordered by how the data flows. All paths relative to repo
root. Line numbers current as of this writing.

---

## Area 1 — Overlapped coalesceBatches (CPU hybrid, Sort + HashJoin)

### The design in one paragraph

A hybrid operator retains payload batches columnar and, in coalesced mode,
merges them into one contiguous batch before output extraction
(`HybridContainer::coalesceBatches`). The merge is (a) restructured
column-at-a-time with progressive source release so its transient memory is
~1x payload + one column instead of ~2x, and (b) moved onto a background
thread overlapped with the operator's other finalization work — the key sort
in SortBuffer, the hash-table build in HashBuild — since the merge touches
only payload batches while sort/table-build touch only the key container.
Joined before the first reader of the merged data.

### Files and anchors

**velox/exec/RowContainer.h — the merge itself**
- `coalesceBatches()` @ **2279**: column-at-a-time loop; per-batch
  `childAt(col).reset()` release; the comment block records why the
  parallel-workers variant was evaluated and reverted (measured: moved CPU
  off the operator ledger but never improved wall time; uncapped workers
  re-created the 2x transient).
- `addPayload()` decl @ 1994, body in RowContainer.cpp (~1340-1400):
  **dictionary-flatten on add** @ cpp:1366 — copies survivors instead of
  retaining dictionary views that pin pre-filter scan batches (found via
  TPC-H Q12: 19K ~49-row views pinning 3.6GB). Flat children pass through
  zero-copy; constants stay constant.
- `anyContainerNullable()` @ **2114**, used @ 2456/2495: the NULL-corruption
  fix — extraction after a multi-driver merge must dispatch its
  WithNulls/NoNulls fast paths on the OR of ALL registered containers'
  nullability flags, not the local one (local-only dispatch silently turned
  sibling drivers' NULL varchars into "" — found via TPC-DS Q64; Bolt has
  the same latent bug).

**velox/exec/SortBuffer.cpp — sort-side overlap**
- ctor @ 45: `hybridSortScattered_` (own QueryConfig key
  `hybrid_sort_scattered_enabled`, default false — scattered sort measured
  0.7x, kept only as a flag).
- `addInput` hybrid fork @ 107-160: scattered vs coalesced rowId encoding
  (`HybridRowId::encodeScattered(batchId, row)` @ 148 vs global row index).
- **`noMoreInput` overlap @ 207-260**: spawn `coalesceThread` @ 212 (only
  when hybrid + coalesced + no spiller), PrefixSort runs on the operator
  thread, exception-safe join (try/catch around the sort joins the thread
  before rethrow — a dangling std::thread dtor would terminate), then
  `updateEstimatedOutputRowSize()` moved AFTER the join because it is the
  first hybridData_ reader.
- Review question: confirm nothing between spawn and join reads
  `hybridData_` (PrefixSort touches only `data_`, the keys container).

**velox/exec/HashBuild.{h,cpp} — join-side overlap (multi-driver)**
- HashBuild.h: `hybridCoalesceThread_`/`hybridCoalesceError_` members +
  `joinHybridCoalesceThread()` (public — called cross-operator).
- `noMoreInputInternal` @ **895-905**: every driver launches its own
  container's merge on a thread. Non-last drivers park afterwards, so their
  merges overlap the last driver's remaining input; the last driver's merge
  overlaps `prepareJoinTable`.
- `finishHashBuild`: cache-hit early join @ 945; **the main join point @
  1045-1048** — after `prepareJoinTable` returns, join own thread then every
  peer's (`otherBuilds`), before the probe handoff. Peers are alive because
  the promises are realized at scope exit, after this point.
- `joinHybridCoalesceThread()` @ 1530 (rethrows the captured exception);
  `close()` @ 1544 joins without rethrow (defensive teardown path).
- Review questions: (a) the not-last-driver path returns with a live thread
  — verify every route to probe-visible data passes through the last
  driver's peer-join loop; (b) abort/cancel while parked → close() join.

### Related but separate (same files, earlier work)
`HashBuild::addInput` hybrid store fork (~610-668), `HashProbe`
`extractColumns` hybrid branch + `initializeResultIter` hybrid sizing (the
velox-only OOB fix), `HashTable` hybridMode ctor/registry — the base
transplant; review if time allows, but they predate this design and have
byte-identical-output gates across TPC-H/TPC-DS behind them.

---

## Area 2 — Boundary-hybrid transfer + row-native single-operator join (GPU)

### The design in one paragraph

For a GPU-executed join, only the join-key columns cross the device
boundary, packed as fixed-stride rows into pinned memory; payload columns
never leave the host and are retained columnar with (batchId, rowInBatch)
identity — by reusing the CPU `exec::HybridContainer` (Area 1's machinery)
as the host-side store. The GPU matches keys either via cudf::hash_join
(columnar cuco, after key extraction) or — the new matcher — a chained
N:M row-native hash table probed directly against key rows, with no
extraction step. Survivor (probe, build) id pairs are copied back through
pinned staging; the host gathers payloads for survivors only, producing a
CPU RowVector (no output D2H).

### Files and anchors (data-flow order)

**velox/experimental/cudf/CudfConfig.h** — `benchmarkBoundaryHybrid` @ ~257,
`benchmarkRowTable` @ ~296 (both appended at struct end: partial-rebuild ABI
safety for stale TUs).

**velox/experimental/cudf/exec/CudfConversion.{h,cpp}** — the boundary pack:
- key-name resolution from the downstream join op @ cpp:444-473;
- keys-only natural-aligned pinned pack @ cpp:~512-562 (alignment matters:
  `extract_keys_kernel` does wide loads — see the misaligned-key guard);
- hard-fail (not silent fallback) for unsupported batches @ ~630;
- host-batch retention attach to the emitted RowStoreVector.

**velox/experimental/cudf/exec/RowStoreVector.h** @ 94-145 — boundary
payload attachment API (`setBoundaryPayload` / `boundaryHostBatches`).

**velox/experimental/cudf/exec/BoundaryHostStore.h** (175 lines, new) — the
host half: thin wrapper over `exec::HybridContainer` in scattered mode
(reuses addPayload retention + dict-flatten + the extractPayloadScattered
kernels); zero-key container over a never-written dummy RowContainer;
prefix-sum map global build row id -> (batchId, rowInBatch); caller-owned
sentinel scratch so the shared build store stays immutable under
multi-driver gathers. **This file IS the "one layout, two boundaries" claim
in code — review it with that sentence in mind.**

**velox/experimental/cudf/exec/GpuRowHashTable.{cuh,cu}** (132 + 314 lines,
new) — the row-native N:M matcher:
- chained multimap: `heads[capacity]` (4B, capacity = pow2 >= 2x rows) +
  `nextFp[buildRows]` packing `[fp32 | next32]` — one 8-byte load per chain
  step yields next pointer + fingerprint;
- build: one atomicExch per build row; probe: two-pass count / cub scan /
  fill, probe key row read once, fingerprint filter, field-wise verify
  against build key rows;
- scope: fixed-width 4/8-byte keys, <= 8 composite columns; null keys not
  representable in row stores (parity with the existing path).
- Review questions: fingerprint collision handling (verify is mandatory,
  fingerprint only filters), capacity/load-factor choice, the mid-batch
  total-count sync (suspect in the one unexplained E2E corner —
  rowpack/2M/k<=2 +16-25% with all instrumented phases at parity).

**velox/experimental/cudf/exec/RowHashJoin.{h,cpp}** — orchestration:
- BuildData boundary + rowTable fields @ h:96-128;
- row-table build lambda @ **cpp:419-463** (note the guard @ 427: rowTable
  does not combine with fused probe);
- boundary build branch @ **cpp:625+** (host retention in GPU-concat order;
  keys-only store; key fields 0..K-1);
- probe dispatch @ cpp:1122-1130; matcher switch in the standard probe @
  cpp:1298-1420 (`rowTableProbe` call @ 1323);
- **`boundaryProbe()` @ cpp:1613+**: key extract-or-rowtable -> match ->
  pinned D2H of the two id arrays + ONE sync -> host gather per output
  column -> CPU RowVector out;
- `rowTableProbe()` @ cpp:1674+.
- Invariant to check: under `--row_table`, `extract_keys` never runs and no
  cudf::hash_join is constructed on either side (the wire-format-only trap:
  keys must stay rows from CPU pack through match).

**exp/2026-08-10-boundary-hybrid/** — bench + gates: `--boundary_hybrid`,
`--row_table`, `--join_keys`, `--build_key_mod` (N:M via duplicate keys),
order-independent checksum; `gate_sweep.sh` / `rowtable_gate_shootout.sh`
(debug partition), `summarize*.py`.

### Known limitations (stated, not hidden)
Inner join, no filter expr, fixed-width keys, flat non-null scan batches
(hard-fail), no fused-probe/CudfBatchConcat interop; host gather is
single-threaded per driver via DecodedVector::valueAt (known easy win);
the rowpack/2M/k<=2 E2E anomaly awaits nsys attribution.

### Measured results the code must be able to defend
- Boundary: H2D 1.65GB keys-only vs 14.6GB rowpack (8.9x); E2E 2-2.6x vs
  stock col at selective joins; crossover at bsel~50 (host gather
  dominates).
- Matcher: never loses (0.40-0.99x of cuco, 18 cells); ~2x at composite
  keys; 1.3-2.4x at every batch size in boundary (keys-only) mode; parity
  only at rowpack/2M/k<=2 (wide rows -> strided key reads).
- Correctness: checksums identical across all arms in every config, incl.
  the N:M duplicate-key 60x-multiplication gate.
