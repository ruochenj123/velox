# Design note: build-side payload deferral (proposed, not implemented)

## Why

Q31 (the CPU paper's single-join workload) puts ALL payload on the BUILD side
(R: keys + up to 12 payloads; S: keys only). The current spine deferral only
defers PROBE-side payload, so on Q31 it defers nothing and costs a round trip
(bnd 0.63-0.94x of col). Measured Q31 cost structure (p12, sel60, row arm):
join compute 0.63 s summed; R payload H2D (pack) 18.4 s summed; 120M-row
output D2H + conversion 6.4 s. The boundary is everything.

The paper's §2 GPU design ("payloads stay host-columnar, survivor row-ids
round-trip, one survivor-only CPU gather") IS build-side deferral. It is the
GPU analog of the CPU hybrid join, cell for cell.

## Mechanism (symmetric to the probe spine)

1. **Build pack in crossing mode.** `CudfFromVelox` feeding `RowHashJoinBuild`
   packs `rightKeys` + hidden `__rowid` (BIGINT, batch-local) and retains the
   batch in a per-batch `BoundaryHostStore` (exactly what the probe-side pack
   already does; `resolveRowPathOnce` currently forces builds eager).
2. **Build accumulates stores.** `RowHashJoinBuild` concatenates its input
   batches into the GPU row table (as today) and additionally keeps
   `std::vector<shared_ptr<BoundaryHostStore>> stores` + `bases` (global
   build-row base per store) in `BuildData`; the `__rowid` field is rebased
   to global with `addInt64Field` (already exists) during concatenation --
   same bookkeeping as `RowOrderBy::concatenateRowInputs`.
   Multi-driver build: each driver contributes batches; the merged BuildData
   orders them as concatenated, so global ids are consistent.
3. **Probe output layout.** Build-side output columns not in the build's
   crossing set become DEFERRED placeholders (`{-1,0}`), and the build's
   `__rowid` is gathered into the output row as an extra hidden field
   (`outputBuildRowIdField_`), exactly like the probe rowid today.
4. **Materialization** (`makeHostOutput` / `makeColumnarOutput`): a second
   deferred source. Read the build rowids (host or D2H), map global id ->
   (store k, local id) by `upper_bound(bases)`, gather per store
   (`RowOrderBy::gatherDeferred` already implements the multi-store regroup;
   factor it into a shared helper), splice. Chained probes forward BOTH
   provenances (probe store per batch + build stores) on the RowStoreVector.
5. **Adaptive.** Build-side deferral is decided per BUILD (once, at
   build-pack resolution) from the chain endpoint's survival ratio like the
   probe: defer when S/P <= threshold. For Q31 sel=10..90 that is exactly the
   selectivity sweep -- the controller should flip between sel30 and sel60.

## Expected effect on Q31 (p12, sel60)

- Removes R payload H2D (~12 GB of the 18.4 s summed pack) and the output's
  build-payload D2H (120M x 12 cols of the 6.4 s conversion); adds a host
  gather of 120M x 12 cells (parallel, ~1-2 s at the Q40/Q44 rates).
- Should turn row ~= col into a clear win at wide payloads and low
  selectivity, and lose at sel90 (gather >= transfer) -- the same shape as
  the CPU hybrid-join figure, which is the point.

## Scope

~1 day: CudfConversion (build-side crossing mode), RowHashJoin build
(store list + rebase), probe output layout + materialization (shared
multi-store gather helper), DeferralPlan (build-side eligibility), gate
(Q31 smoke multiset, TPC-H 22q), then the Q31 sweep rerun.
