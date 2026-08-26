# Review guide: branch `row-sort` (row-wise GPU sort)

Base (2026-08-25): the review stack of the deferral work (`review-stack`,
= spine deferral v2 + batch-level adaptive + host exit + native row output).
Commits on top, in order:

1. Row-wise GPU sort (RowOrderBy)
2. global rowid base = provenance store size
3. parallel host emission; sort endpoint reports S = rows
4. REVIEW-ROWSORT.md, DESIGN-build-side-deferral.md
5. coalesce deferred payload into per-store columns while the device sorts
6. stage timers (device key sort / row gather; D2H wait, host extract, host gather)
7. an empty deferred column set resolves EAGER; per-batch deferred stat;
   'nothing to defer' decided from the actual input type
8. RowOrderBy native row output (`HostRowVector` per chunk; shared heap,
   per-column null-bit map added to `HostRowVector`)

(The old pre-rebase head is kept as branch `row-sort-pre-rebase`.)

## What to read

| File | What |
|---|---|
| `exec/RowOrderBy.{h,cpp}` | The operator. Under `--row_output_native`, `emitHostChunk` returns a `HostRowVector` (rows D2H'd, no extraction). `doAddInput` accumulates RowStoreVector (row pack / row joins, also through a gather) or CudfVector; `concatenateRowInputs` / `transposeCudfInputs` build ONE row store (+ string heap rebase, padded null sidecar, `addInt64Field` makes `__rowid` global = store base + local); `sortRows` extracts keys -> cudf columns -> `cudf::sorted_order` -> ONE `gatherRowsWarp` of whole rows; chunked (1M) emission: `emitHostChunk` (CPU exit, parallel extraction, D2H prefetch) or `emitColumnarChunk`; `gatherDeferred` materializes deferred payload from per-batch `BoundaryHostStore`s. |
| `exec/GpuRowOps.{cu,cuh}` | `addInt64Field` kernel. |
| `exec/DeferralPlan.h` | Chain walk generalized: OrderByNode adjacent/terminal, gather LocalPartition pass-through, `orderByBehindGather`. |
| `exec/CudfConversion.cpp` (resolveRowPathOnce) | Row-sort consumer (direct or behind a gather): sort keys = crossing set / null-key guard, pruning = sort output, endpoint = sort node. |
| `exec/CudfLocalPartition.cpp` | Gather passes RowStoreVector through untouched. |
| `exec/OperatorAdapters.cpp`, `CudfConfig.h`, `benchmarks/CudfTpchBenchmark.cpp` | `RowOrderByAdapter`, `benchmarkRowSort`, `--row_sort`. |
| `exec/tests/utils/TpchQueryBuilder.cpp` | `--synth_sort_gather` in Q40/Q31. The paper's sort workload is Q41 on widesort2 (pre-existing plan). |

## Design points worth challenging

- **One layout per sort input** (CHECK in `doAddInput`): the batch-level
  adaptive pack cannot flip mid-query when the chain endpoint is a sort
  (the sort reports once, at the end), so eager/deferred batches never mix.
- **Sort endpoint reports S = rows** (not the join's CPU-exit S = 0): a sort
  never reduces; deferring its payload means a full scattered host gather.
  Measured: bnd loses 0.55-0.98x on a standalone sort (Q40).
- **Global rowid across stores**: a sort input batch after a join holds only
  the matches while its rowids index the whole pack batch; bases advance by
  `store->totalRows()`.
- **Host emission** is where the row sort's e2e time goes (the device sort is
  ~1.4 s for 180M rows): 24-way over row ranges, one thread team per chunk
  for all columns, string bytes into one shared buffer per column.

## Results (exp/2026-08-21-whole-query-deferral/FINDINGS-factorial.md)

With native row output (harness sort12, no result copy for any arm; gate
13955365 12/12 ordered PASS): the eager row sort beats cudf-columnar
1.15-1.31x on the CPU-protocol plan and 1.22-1.45x on the gathered plan
(Q40, SF30); on Q41 (64-256 BIGINT payload columns) it trails 0.84-0.87x,
the field-major pack being the residual. The deferred sort is unchanged
(its exit was already host-side) and still loses on a standalone sort.
Earlier (pre-native) numbers below for reference:

- Q40 (SF30 lineitem, CPU-protocol cells): device sort a wash; e2e row
  1.0-1.45x over col on the single-driver plan, parity with a parallel scan;
  GPU 2-7x over cpu.
- Q45 (join->sort, resident SF100): row 1.47x over col.
- Q31 (synthetic single join): row ~= col; the payload is on the BUILD side,
  so probe-side deferral cannot apply -> build-side deferral proposed.

## Build (does not touch the review checkout)

`exp/2026-08-21-whole-query-deferral/wt/rebuild_cudf_exec_wt.sh` (sources
from this worktree, objects/libs into `velox/_build`), `qb_compile_wt.cmd`
+ `qb_ar.cmd`, `tpchbench_compile_wt.cmd` + `tpchbench_ar.cmd`,
`HARNESS_NAME=... build_harness_wt.sh` -> `bin/velox_cudf_tpch_benchmark_sort3`.
Sweeps: `sweep_singleop.sh` (MODE=sort|join, GATHER=1), `sweep_rowsort.sh`;
gates: `gate_rowsort.sh`, `gate_singleop.sh`; summary: `summarize_singleop.py`.
