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
5. Agg rebatch A/B: upstream `concatOptimizationEnabled` (CudfBatchConcat
   before every CudfHashAggregation, default off) — measure Q1/Q4/Q17/Q18
   col+row warm, on vs off. Also note join->join small batches (Q18 join[8]:
   129 batches avg 50 rows) as a separate concat opportunity for both arms.
