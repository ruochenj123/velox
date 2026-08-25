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
host memory exactly once, in its own native layout. The columnar arm does
(`CudfToVelox`: D2H + arrow import). The row arm used to transpose rows to
columns (on the GPU for the join, on the host for the sort) purely to
satisfy Velox's `RowVector` API — an API tax that made eager row trail col
whenever the exit carried payload (Q31 flipped: 0.77–0.98x; Q40).

## What

| File | What |
|---|---|
| `velox/vector/MaterializableVector.h` | Interface: `materialize()` → ordinary Velox columns, on demand. |
| `velox/experimental/cudf/exec/HostRowVector.h` | Header-only. A `RowVector` with **zero children** (allowed: `children.size() <= type->size()`) carrying the D2H'd row bytes, compacted string heap, null sidecar, the field layout, and any already-materialized (host-gathered deferred) columns. `extractHostRows()` is the strided extraction shared with the non-native path. |
| `velox/experimental/cudf/exec/RowHashJoin.cpp` | `makeHostOutput` returns a `HostRowVector` under the flag; the CPU-exit route now also takes eager rows (`hostExit_ && (provenance || native)`), so an eager row join no longer goes through `makeColumnarOutput` at a CPU exit. Stat `hostExitNativeBytes`. |
| `velox/experimental/cudf/CudfConfig.h`, `benchmarks/CudfTpchBenchmark.cpp` | `benchmarkRowOutputNative` / `--row_output_native`. |
| `velox/benchmarks/tpch/TpchBenchmark.cpp` | Under `--include_results`, results implementing `MaterializableVector` are materialized before printing/checksums — so correctness gates see real values while timed runs never extract. |

## Points worth challenging

- Zero-child `RowVector`: anything downstream that walks children would see
  none. In the benchmark the exit feeds `CudfToVelox` (passes non-Cudf
  vectors through) and the result callback (uses `size()`); nothing else.
  A production consumer would need `materialize()` or a row-aware sink.
- The deferred arm's host gather stays in the timed path (it *is* the
  materialization); only the layout transpose is removed.
- The D2H itself stays on the critical path; per the sort breakdown it is
  hidden by prefetch, but a native exit has no host work to hide behind.
  Whether it surfaces is what the rerun measures.
- Memory: the benchmark retains all result vectors; native rows are the
  same order of bytes as the columns they replace.

## Gate and results

`exp/2026-08-21-whole-query-deferral/gate_native.sh` (job 13951687): flipped
and original Q31 on `synth_join_smoke`, TPC-H q3/q9 SF1; arms
row/rown/bnd/bndn/bndan vs cpu, multiset compare through `materialize()`.
Table-B rerun (scrambled hit-rate data, payload on the probe) with native
arms: `results/singleop_joinhit_flip_scat_native` — see
`FINDINGS-factorial.md`.
