# Review guide: branch `native-row-output`

Base: `17783f3e0` = the committed review state of `wip/whole-query-deferral`
(spine deferral v2 + batch-level adaptive + host-exit). Review this branch
together with `REVIEW-ROUND5.md` (deferral); the sort branch (`row-sort`)
is reviewed separately and will be rebased onto this one.

Commits on top of the snapshot:

1. `9a45469b4` Native row output at the CPU exit (`HostRowVector`,
   `--row_output_native`) — the substantive change.
2. `9ecb3b863`, `801781250`, `44c875d6a`, `38b33e54f` — cherry-picks from
   `row-sort` of benchmark plumbing only (query builder schema-by-name,
   `--synth_sort_gather`, `--synth_join_flip`, `--synth_join_build_filter`).

## Why

Fairness rule for the layout experiments: every arm delivers its result to
host memory exactly once, in its own native layout, with no conversion at
the exit. Final design (2026-08-25, after review discussion):

- **eager** = rows: the joined/sorted rows are D2H'd and handed over as a
  `HostRowVector` (zero-child `RowVector` carrying the row bytes, string
  heap, null sidecar, field layout). Extraction into Velox columns happens
  only when results are printed (`MaterializableVector::materialize()`).
- **deferred** = columns: the ids are read from the crossing rows, the
  deferred payload is gathered on the host from the retained batch, and the
  GPU-side output columns (keys, later keys, build-side values) are
  transposed ON THE DEVICE (`transposeGpuOutputColumns`, factored out of
  `makeColumnarOutput`; `transposeChunkColumns` for the sort) and brought
  over through the arrow path. No host extraction, no mixed result.
- No build-side deferral (decided: the build is packed before the probe
  can observe anything, so it cannot follow the batch-level controller;
  builds are the small side).

Before this, the row arm transposed rows to columns purely to satisfy the
`RowVector` API (eager row trailed col 0.77-0.98x on Q31-flipped; Q40 sort
at parity); the deferred exit extracted the key columns on the host.

## What

| File | What |
|---|---|
| `velox/vector/MaterializableVector.h` | Interface: `materialize()` -> ordinary Velox columns, on demand. |
| `velox/experimental/cudf/exec/HostRowVector.h` | Header-only, eager results only. `extractHostRows()` is the strided extraction used by `materialize()` and by the non-native eager exit. |
| `velox/experimental/cudf/exec/RowHashJoin.cpp` | `makeHostOutput`: deferred -> ids + host gather + device transpose -> columnar `RowVector`; eager -> `HostRowVector` under `--row_output_native`, else extraction. `transposeGpuOutputColumns` shared with `makeColumnarOutput`. |
| `velox/experimental/cudf/CudfConfig.h`, `benchmarks/CudfTpchBenchmark.cpp` | `benchmarkRowOutputNative` / `--row_output_native`. |
| `velox/benchmarks/QueryBenchmarkBase.cpp` | `copyResult=false`: the cursor no longer copies results (harness artifact; it flattened native results). |
| `velox/benchmarks/tpch/TpchBenchmark.cpp` | Under `--include_results`: native results are materialized, encoded results flattened, before printing/checksums. |

## Points worth challenging

- Zero-child `RowVector`: downstream consumers that walk children see none;
  in the benchmark the exit feeds `CudfToVelox` (pass-through) and the
  result callback (`size()` only). A production consumer needs
  `materialize()` or a row-aware sink.
- The deferred exit's device transpose costs a kernel + D2H of the same
  bytes; measured neutral-to-slightly-faster than host extraction.
- Memory: the benchmark retains all result vectors; native rows are the
  same order of bytes as the columns they replace.

## Gate and results

`exp/2026-08-21-whole-query-deferral/gate_native.sh` (job 13951687): flipped
and original Q31 on `synth_join_smoke`, TPC-H q3/q9 SF1; arms
row/rown/bnd/bndn/bndan vs cpu, multiset compare through `materialize()`.
Table-B rerun (scrambled hit-rate data, payload on the probe) with native
arms: `results/singleop_joinhit_flip_scat_native` (jobs 13953182/83; gate
13953181: 20/20 PASS). Medians of 3, seconds, `col / row / rown / bnd / bndn`:

| sel | proj 4 | proj 8 | proj 12 | proj 16 |
|---|---|---|---|---|
| 10% | 3.58 / 2.36 / 2.21 / 1.80 / 1.82 | 3.67 / 3.89 / 3.69 / 2.79 / 2.73 | 4.76 / 5.57 / 5.35 / 3.13 / 3.12 | 4.96 / 5.96 / 5.71 / 3.40 / 3.38 |
| 30% | 2.56 / 2.58 / 2.27 / 2.05 / 2.00 | 3.85 / 4.32 / 3.67 / 3.09 / 3.03 | 5.27 / 6.52 / 5.29 / 3.71 / 3.62 | 5.55 / 6.63 / 5.65 / 4.05 / 4.00 |
| 60% | 3.37 / 2.97 / 2.37 / 2.32 / 2.29 | 4.28 / 4.83 / 4.05 / 3.59 / 3.50 | 5.78 / 7.27 / 6.12 / 4.46 / 4.53 | 6.68 / 8.05 / 6.84 / 4.96 / 4.96 |
| 90% | 3.19 / 3.40 / 2.62 / 2.72 / 2.58 | 4.83 / 5.32 / 4.33 / 4.11 / 4.06 | 6.71 / 8.37 / 6.96 / 5.30 / 5.20 | 7.02 / 9.39 / 7.46 / 6.08 / 5.76 |

- Native output lifts eager row 4-30% (rown/row), more at high
  selectivity (larger output); rown ~= col (0.87-1.42x). The remaining
  eager-row deficit at wide payload is the field-major pack, not the exit.
- bndn == bnd (+-5%): the deferred exit was already host-side. Deferral
  wins every cell: col/bndn 1.19-1.97x.
- All arms are ~5-10% faster than the earlier table B (harness copy gone).

## Round 6 (2026-08-26): three items from the round-5 review — uncommitted on top of 001d18f41

1. **One layout builder** (`CudfConversion.cpp`, `computeLayoutOnce`): the
   three branches are one builder over a chosen column list. Default =
   adjacent join keys + join output columns present in the input (the scan's
   filter-only columns never cross); when no adjacent join is known or
   nothing would be pruned, every input column. Boundary = crossing set +
   `__rowid`. The old full-identity branch is gone; fallbacks preserved
   (unpackable column / width mismatch -> pinned pack declines).
2. **Probe -> build emits rows** (`RowHashJoin.cpp`, terminal detection):
   when the next operator is a `RowHashJoinBuild` (bushy plan) and deferral
   is OFF, the probe hands over `RowStoreVector` rows (the build consumes
   rows; no rows->columns->rows). Under `--boundary_hybrid` a build-feeding
   probe stays columnar: a build takes one input layout, and the adaptive
   switch would otherwise mix rows and columns (found by the 22-query gate).
   Such a probe reports survival as the chain endpoint.
3. **Transpose-first deferred exits** (`transposeGpuOutputColumns(...,
   withRowId)`): the output row buffer, with `__rowid` as one more column,
   is transposed once on the device; the ids come from that column.
   `makeColumnarOutput` loses the separate id pass; `makeHostOutput`'s
   deferred path no longer D2H's the rows at all (GPU-side columns cross
   once, via the arrow path, together with the id column).

Gates (harness native4): Q31 both orientations + q3/q9, 20/20; all 22
TPC-H queries at SF1 row/bnd/bnda vs cpu: all pass except q15 (unsupported
on every GPU arm before this round). Table-B regression rerun: see
`FINDINGS-factorial.md` (results/singleop_joinhit_flip_scat_native4).
