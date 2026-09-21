/*
 * GPU experiment query builder (paper: hybrid layout across the CPU-GPU
 * boundary, single-operator evaluation). Extends the TPC-H builder with the
 * GPU synthetic plans; all TPC-H/SSB/CPU-paper ids are unchanged.
 *
 *   51  gpu single join : build = R keys only (100M), probe = P (200M rows,
 *                         scrambled, keys + payloads). Payload on the PROBE:
 *                         the side whose boundary crossing the row path
 *                         optimizes. Selectivity is in the data (P's hit rate).
 *                         Keys: row_id, l_suppkey, l_orderkey, l_partkey (BIGINT;
 *                         the row matcher needs fixed-width keys).
 *   52  gpu single sort : the Velox-blog wide sort on `wide` (20M rows,
 *                         k1..k16 keys, c0..c255 payloads).
 *
 * Flags reused from the CPU builder: --synth_payload_cols (0..13),
 * --synth_join_keys (1..4), --synth_wide_sort_keys, --synth_wide_payload_cols.
 * --resident_tables applies (scans served from pre-decoded host batches).
 */
#pragma once

#include "velox/exec/tests/utils/TpchQueryBuilder.h"

namespace facebook::velox::cudf_velox {

class GpuSynthQueryBuilder : public exec::test::TpchQueryBuilder {
 public:
  static constexpr int kGpuJoin = 51;
  static constexpr int kGpuSort = 52;

  explicit GpuSynthQueryBuilder(dwio::common::FileFormat format)
      : TpchQueryBuilder(format) {}

  exec::test::TpchPlan getQueryPlan(int queryId) const override;

 private:
  exec::test::TpchPlan getGpuJoinPlan() const;
  exec::test::TpchPlan getGpuSortPlan() const;
};

} // namespace facebook::velox::cudf_velox
