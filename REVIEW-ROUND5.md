# Review Round 5: varchar-rows branch (2026-08-21/22)

Base: b778f47b4 (reviewed). Deferral parked on wip/whole-query-deferral
(4c40249e9); rebases onto this branch after review.

## 1. Out-of-line strings in the row store
- GpuFixedRowStore.h: `FieldDesc.kind` (fixed/string), 16B `RowStrSlot`
  (len + 12 inline bytes, or len + 4B prefix + u64 heap offset; byte-
  identical to an inline Velox StringView), `chars/chars_bytes` on the
  store handle.
- GpuRowOps.cu/.cuh: `copy_field` 16B fast path; rebaseStringOffsets,
  stringsToSlots (cudf strings -> slots; int32/int64 offsets; sliced
  columns), stringHeapLayout + compactStrings (post-gather compaction into a
  fresh heap; one sync for the total), stringFieldOffsets +
  stringFieldToChars (slots -> cudf strings column).
- RowStoreVector.h: heap in the row buffer tail (pack) or a separate buffer
  (gather/concat); stringFieldOffsets().
- CudfConversion.cpp (pack): VARCHAR accepted in row mode (8-aligned slot),
  dictionary-encoded string children flattened, heap sized by a length
  pre-pass and shipped in the same H2D after the sidecar; inline views are a
  16B memcpy. `h2dBytes/toCudfBytes` now count string bytes (was a throw).
- RowHashJoin.cpp: layout-from-table handles STRING; transposeToRows returns
  the heap; build concat + rebase (row inputs) / heap from transpose (cudf
  inputs); probe heaps; output layout with string fields; compaction after
  the gather; strings column materialization at the terminal join; string
  join keys rejected loudly. REVIEW HOTSPOT: the compaction block in
  doGetOutput and makeColumnarOutput's string branch.
- CudfBatchConcat.cpp: heap concat + rebase; also carries the null sidecar
  (this base dropped it).
- Chained (row-store) probe output now attaches its null sidecar (was only
  used by the columnar terminal).

## 2. Ported from the deferral branch (independent bug fixes)
- Cross-stream event waits before the build concat (the lost-matches race).
- Empty-build guards (build with no inputs; probe against an empty build).
- CudfTpchBenchmark flags (--gpu/--row_wise/--cpu_col_to_row/--boundary_
  hybrid/--row_table/--batch_size/...), scans on CPU by default.

## 3. Scheduling / fallback / instrumentation (2026-08-21)
- DedicatedStream.h: per-operator streams for the row pack and build instead
  of cudf's 32-entry pool (aliasing made a build's synchronize() wait for
  other drivers' work -> multi-second stalls).
- OperatorAdapters/ToCudf: findRunnableAdapter (first adapter that can run
  the node on GPU); cudf join adapters + bridge translator registered behind
  the row ones, so semi/anti/filtered/outer joins run on cudf, not the CPU
  (Q21 SF100 10.8 -> 6.3 s). Row bridge translator only claims inner/
  no-filter joins.
- TpchBenchmark.cpp: prints `[extra]` stats for operators under synthetic
  plan-node ids (the -from-velox/-to-velox conversions were invisible).
- RowHashJoinProbe: host-side phase timers (prep/matcher/gather/output)
  under --log_gather_time.
- GpuResources.cpp: async pool primed with memoryPercent of free memory.
  CudfConversion: slot growth factored into ensurePinnedSlotCapacity (a
  pinned pre-warm in initialize() was tried and removed: over-provisioned,
  and irrelevant for warm runs).

## 4. Encoded-dataset study (exp/2026-08-22-string-encoding)
- encode_tpch.py: crossing string columns -> BIGINT codes (sorted dense
  rank) keeping <col>_str twins at the end; codes.json.
- TpchQueryBuilder: reads codes.json; encLit/encIn/encLikePrefix/
  encLikeSuffix/strCol helpers; trailing extra file columns exposed. One
  builder serves plain and encoded data. ABI note: TpchBenchmark.cpp must be
  recompiled with the header (make_shared size).

## 5. Measurement finding (2026-08-22)
All SF30/SF100 sweeps so far were one-query-per-process (cold): pinned
slots, the device pool, CUDA/cudf init and the file cache are all first-
query costs. Warm (3rd in-process repeat, SF100): Q22 0.63 s both arms (cold
2.6/1.9), Q10 row 2.80 vs col 2.81, Q3 2.17 vs 2.61, Q7 2.83 vs 3.35. The
"string penalty" was mostly warm-up. Proposal: report warm numbers
(REPEATS=3) for the paper tables; rerun the SF100 plain + encoded sweeps.

## Validation
- SF1 gates: 22/22 run; 21/21 deterministic match CPU (Q15 FP); fallback
  gate 8/8; encoded gate 22/22 (+Q2/Q19 after the OR-chain fix).
- SF100 sweeps (cold): results/sweep_sf100_summary.txt,
  sweep_sf100enc_summary.txt, sf100fb (fallback), warm_* (methodology).

## Queued (post-review)
0. **BATCH-LEVEL adaptive deferral (review decisions 2026-08-24,
   supersedes the cross-execution DeferralStats mode as the headline
   mechanism)**: decision variable measured AT THE MATERIALIZATION POINT.
   Chain endpoint join E = last HashJoin before a non-join consumer
   (statically known via the collectChainKeyNames plan walk). The pack
   records P (packed rows) under key E; the endpoint probe (the one whose
   emitColumnar decision fires) records S under E -- its output rows, or
   **0 when its consumer is CudfToVelox (CPU exit: materialization
   uploads nothing -> ALWAYS defer; = paper case (a))**. Pack switches
   per accumulated batch: eager until S/P <= threshold, then deferred and
   STICK (one-way). S is observable in eager mode too (same match counts),
   so the observation phase needs no deferral machinery; measuring S also
   subsumes the join-fanout caveat. First-join match rate may serve as an
   early conservative trigger (PK builds: rates only multiply down).
   Probe handles mixed batches via a two-slot ProbeLayout selected by
   input->hasProvenance(); chained outputs/materialization already
   per-batch; builds unaffected (always eager under v2). Threshold needs
   one calibration sweep (upload+gather constants). Est. ~0.5-1 day;
   implement after the current finals so always-defer / execution-adaptive
   arms have clean comparison numbers.
0b. **Direct host emission at the CPU exit** (review 2026-08-24): when the
   probe's consumer is CudfToVelox and provenance exists, emit a host
   RowVector directly (D2H gathered fixed rows + host field extraction +
   provenance gather) instead of the uniform upload-then-download path.
   Realizes paper case (a) fully (zero payload upload); REQUIRED before
   the single-join case-(a) experiment; pairs naturally with item 0 (same
   emission switch).
1. **Lazy string compaction / pointer slots** (review discussion 2026-08-22):
   replace the heap-relative u64 offset in `RowStrSlot` with an absolute
   device pointer; each RowStoreVector keeps a shared_ptr list of the
   buffers its slots reference. Drops `rebaseStringOffsets` and the
   per-batch `stringHeapLayout`+`compactStrings` (size kernel + scan + host
   sync + copy) from the join output path; concat/build only append
   keep-alive lists. Compaction becomes an opt-in policy (e.g. when
   retained bytes >> referenced bytes, or at the terminal). Cost: a chained
   join pins upstream heaps until materialization. Do BEFORE the deferral
   rebase (same seams; and a deferred host payload is the same idea with a
   host pointer).
2. Row-layout kernel study (strided access): nsys/ncu on Q3/Q10, sector
   efficiency of extract_keys / gather / transposes / compaction; candidates:
   warp-cooperative gather, tiled transposes, key pre-extraction.
3. Warm `row` vs `row_enc` on Q1/Q4/Q12/Q14/Q16/Q19 to decide whether the
   low-cardinality dictionary path (#2 in the plan) is worth doing at all.
3a. **v2 "SPINE DEFERRAL" redesign (review decision 2026-08-24)** -- the 4
   bnd winners (Q2/Q5/Q8/Q11) share one shape: fact-table probe spine
   packed at the boundary, kind-1 chain over small eager builds, ONE
   materialization at the agg. Simplify to exactly that: deferral only for
   the probe-side chain rooted at one boundary pack; builds always eager
   (values, strings ok); provenance = ONE (store, ids) pair; materialize at
   the first non-probe consumer. DELETES kind 3, makeCrossingStoreFromHost
   (both cases), GPU-resident build materialization, N-source provenance/
   terminal gather, BoundaryHostStore-for-builds. ADAPTIVE: pack always
   retains host batches; the first probe chooses per batch (references vs
   eager gather) from observed numMatches/probeRows -- eager default,
   defer on high reduction; batches self-describing via hasProvenance().
   Gives up build-side payload deferral (Q10) -- measured as fine (bnd lost
   Q10 anyway). ~1-2 days incl. validation, as the follow-up commit.
3b. **Defer-vs-eager policy (quantitative, from the bnd win/loss stats,
   2026-08-24)** -- decision variable = the REDUCTION FACTOR between the
   boundary and the materialization point: deferral wins when the chain
   shrinks cardinality a lot before materializing (Q5: keys through a 22.7M
   -row store, 728K-row round trip -> 1.06x; Q8 245K -> 1.12x; Q11 1.63x;
   Q2 1.19x) and loses when the round trip re-ships ~the whole output (Q9:
   32.6M rows + two 32.6M-row kind-3 edges -> 0.71x; Q10: 11.5M -> 0.82x).
   Plan-time: cardinality estimates; adaptive: probe falls back to eager
   output when numMatches/probeRows stays high. Analogous to the CPU-side
   hybrid's payload policy (kHybridJoinMinPayloadBytes). Pair with:
   kind-3 default -> eager (measured evidence: eager >= bnd on 13/22), the
   host-materializing variant kept behind the wide-payload policy.
   Also: extend kind 1 to any provenance-preserving op (sort/topN/limit/
   filter-on-crossing-cols); aggregation is the true barrier. Document the
   invariant in DESIGN-whole-query-deferral.md.
4. **Decline-when-all-inputs-GPU-resident** (review discussion 2026-08-23):
   the RowHashJoin adapter should decline joins where BOTH sides arrive
   GPU-resident (fed by cudf operators, e.g. Q18's agg->build), handing them
   to CudfHashJoin via findRunnableAdapter. Rationale: the row join earns
   its keep where data crosses the boundary (pack side: Q18 join[6] 44 ms
   row vs 140 ms cudf on 150M rows); on GPU-resident inputs it pays two
   col<->row conversions (harmless today: ~6 ms e2e on Q18, but becomes a
   forced host materialization under deferral). Implement WITH the deferral
   rebase, as a source-shape check in RowHashJoin{Build,Probe}Adapter::
   canRunOnGPU (plan-walk: does either source chain reach a TableScan
   without a pipeline breaker that runs on GPU?).
5. **CudfConversion.{h,cpp} reconstruction** (review request 2026-08-24):
   too many accreted rounds (merge/pack/boundary/strings/pointer slots/
   deferral). Restructure tryPinnedPack: fold the lazy-init groups
   (emitRowStore_/pinnedPackSyncMode_/boundary resolution and
   rowLayoutReady_) into ONE per-instance init; split the monolith into
   named steps (resolveConsumer, computeLayout, loadChildren, pack, ship);
   rename laterKeyNames_ reuse in the prune path (subsetPackNames_); cache
   the collectChainKeyNames plan walk per task; keep behavior identical
   (gate before/after). Do AFTER the deferral commit lands so the diff
   stays reviewable.
6. Agg rebatch A/B: upstream `concatOptimizationEnabled` (CudfBatchConcat
   before every CudfHashAggregation, default off) — measure Q1/Q4/Q17/Q18
   col+row warm, on vs off. Also note join->join small batches (Q18 join[8]:
   129 batches avg 50 rows) as a separate concat opportunity for both arms.

## 2026-08-24 (afternoon): items 0b + 0 IMPLEMENTED (uncommitted, for review)

Final paper tables are in exp/2026-08-21-whole-query-deferral/FINDINGS-factorial.md
("FINAL PAPER TABLES"). The two queued implementations landed on top:

### 0b. Direct host emission at CPU exit (RowHashJoin.{h,cpp})
- Terminal-probe resolution now also detects a CudfToVelox consumer ->
  `hostExit_`. With provenance, the probe emits a host `RowVector` directly
  (`makeHostOutput`): ONE D2H of the gathered fixed rows (+ compacted string
  heap + null sidecar), host-side strided field extraction, and the DEFERRED
  columns gathered from the BoundaryHostStore by the __rowid read out of the
  host rows -- deferred payload never touches the GPU (paper case (a)).
  CudfToVelox passes non-CudfVector inputs through untouched.

### 0. Batch-level adaptive deferral (endpoint measurement)
- DeferralPlan.h: chain walk now follows PROBE-side edges only and returns
  `endpointJoinId` -- the terminal join of the chain (= materialization point).
- DeferralStats.h: reworked to {packed, survived, reports} per endpoint id.
  The spine pack records P (rows shipped); ONLY the terminal probe records S
  (its match count) -- and a CPU exit records S=0, so single-join queries
  always defer. shouldDefer = reports>0 && S <= threshold*P.
- CudfConversion: with --deferral_adaptive the pack starts EAGER and
  re-checks per BATCH; on the first true it flips one-way to the crossing-set
  layout (resets the layout caches; stat fromVeloxDeferralSwitch).
- RowHashJoin probe: detects the new field-name signature per batch and
  recomputes input capture + output layout (stat probeLayoutSwitch), so a
  query can mix eager and deferred batches mid-stream.

### New microbench: Q32 (case-a)
TpchQueryBuilder q32: lineitem JOIN orders (o_orderdate >= 1998-07-01 AND
o_orderpriority = '1-URGENT'), 6 output cols incl. strings, NO aggregation --
the only plan whose join output goes straight to the CPU. Exercises
makeHostOutput + the S=0 endpoint rule.

Validation: gate 13910992 (0b, 22q SF1) = 63/63 PASS ex. known q15 FP tie.
Gate 13911134 (Q32 all arms + 22q batch-adaptive) pending.
Binaries: dev-named only (bin/velox_cudf_tpch_benchmark_dev3); the finals
binaries were not touched.

### Validation update (2026-08-24, gate 13911615)
25/26 PASS (only the known q15 FP tie). Q32 passes on ALL arms; hostExitRows
fired (0b executed); adaptive switched mid-query on 12/22 queries incl. the
deep chains (q9: 3 pack switches / 5 probe re-inits). Fixed en route:
- keyChannelsResolved_ was a once-guard: after the mid-query switch the
  later-join-key channels stayed classified as payload -> null srcs ->
  SEGV in packIntoSlot (q2/q5/q8/q9/q18). The switch now re-resolves.
- Q32 trimmed to 5 output columns (RowVector::toString elides past 5) and
  compare_results.py strips dictionary "[i->j] " cell annotations.
CORRECTION: the FINAL tables are SCAN-BASED (not resident) -- verified
against the resident-vs-scan comparison table; FINDINGS header fixed.
SF100 batch-level-adaptive perf runs queued: TPC-H 13911674
(sweep_finalab_g1000000/), SSB 13911675 (ssb_bla/), dev3 harness,
same protocol as finals.
