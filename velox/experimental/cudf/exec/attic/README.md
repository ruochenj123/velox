# Attic: retired experimental primitives

## GpuFusedProbe.cu / .cuh (retired 2026-08-20)
The CIDR-era FUSED-PROBE arm: N:1 open-addressing GpuHashTable (combined
[key|index] single-word slots, fingerprint slots, composite-key range packing
via cudf::reduce min/max) plus the fused chain-walk probe kernels. Removed
from the build when the merged VLDB paper dropped the fusion arm; the
single-join row-native matcher (GpuRowHashTable) and its ported key packing
supersede it. The operator-side code (FusedRowHashJoinProbe, buildFusedMap)
was deleted outright -- recover it from git history at tag/commit
"Checkpoint: null sidecar, boundary-hybrid store, row-native table + key
packing" (fc4bffe49) if ever needed. Multi-join CHAIN plumbing notes: the
bench's chain wiring lived in the fused config (CudfConfig::probeFused set,
per-join-node ids); see the checkpoint commit for how plans marked fused
joins.
