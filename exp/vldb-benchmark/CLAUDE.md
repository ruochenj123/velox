# VLDB Experiment: Velox cuDF GPU Benchmark

## Overview

This experiment benchmarks 8 micro-queries (4 scan selectivity + 4 equi-join) on both CPU (Velox) and GPU (Velox+cuDF) to measure GPU acceleration for individual operators.

## Queries

### Scan Queries (Q23-Q26): `SELECT l_orderkey, l_discount, l_extendedprice FROM lineitem WHERE l_shipdate < <DATE>`

| Query ID | Selectivity | Date Cutoff | ~Output Rows (SF100) |
|----------|-------------|-------------|----------------------|
| 23 | ~1% | 1992-02-01 | ~1M (actual: 0.16%, date range starts late) |
| 24 | ~3% | 1992-04-15 | ~3-5M |
| 25 | ~10% | 1992-10-01 | ~10M |
| 26 | ~30% | 1994-03-01 | ~180M |

**Note**: The actual selectivity for Q23 is closer to 0.16% because TPC-H l_shipdate range starts ~1992-01-02. Q26 (30%) has been verified correct. Adjust dates if exact selectivity matters.

### Join Queries (Q27-Q30): `SELECT count(*) FROM <table1>, <table2> WHERE <FK join>`

| Query ID | Join | Probe Table | Build Table | Expected count(*) |
|----------|------|-------------|-------------|-------------------|
| 27 | lineitem ⋈ orders | lineitem (~600M) | orders (150M) | ~600M |
| 28 | lineitem ⋈ part | lineitem (~600M) | part (20M) | ~600M |
| 29 | lineitem ⋈ supplier | lineitem (~600M) | supplier (1M) | ~600M |
| 30 | orders ⋈ customer | orders (150M) | customer (15M) | 150M |

## Code Changes (vs upstream Velox)

Three files modified:
1. `velox/exec/tests/utils/TpchQueryBuilder.h` — 8 new method declarations (Q23-Q30)
2. `velox/exec/tests/utils/TpchQueryBuilder.cpp` — 8 query plan implementations + dispatch
3. `velox/experimental/cudf/CMakeLists.txt` — Build cudf tests/benchmarks when `VELOX_ENABLE_BENCHMARKS=ON` (not just `VELOX_BUILD_TESTING`)

## Build Instructions (Azure A100)

### Prerequisites
- NVIDIA GPU (A100 80GB or H100)
- CUDA 12.x drivers installed
- curl/git available

### Step 1: Install pixi (package manager, no root needed)
```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
```

### Step 2: Clone and build
```bash
git clone -b vldb-experiment https://github.com/ruochenj123/velox.git
cd velox
# For A100: use sm_80. For H100: change CUDAARCHS to "90" in pixi.toml
# OR set -DCMAKE_CUDA_ARCHITECTURES=80 (A100) or =90 (H100)
bash exp/vldb-benchmark/scripts/setup_build.sh
```

Build takes ~30-60 min. Produces:
- `_build/release/velox/benchmarks/tpch/velox_tpch_benchmark` (CPU)
- `_build/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark` (GPU)

### Step 3: Generate TPC-H data
```bash
pip install duckdb pyarrow
python scripts/gen_tpch_parquet.py
```
Edit `scripts/gen_tpch_parquet.py` to change `OUTPUT_DIR` and `SF` if needed.
Generates ~20GB for SF100 in `test_datasets/tpch/sf100/`.

### Step 4: Run benchmarks
```bash
bash exp/vldb-benchmark/scripts/run_all.sh
```

## Key Flags

### CPU benchmark
```bash
velox_tpch_benchmark \
  --data_path=/absolute/path/to/test_datasets/tpch/sf100 \  # MUST be absolute
  --data_format=parquet \
  --run_query_verbose=<QUERY_ID> \
  --num_drivers=32 \         # CPU: use all cores
  --num_splits_per_file=4
```

### GPU benchmark
```bash
velox_cudf_tpch_benchmark \
  --data_path=/absolute/path/to/test_datasets/tpch/sf100 \  # MUST be absolute
  --data_format=parquet \
  --run_query_verbose=<QUERY_ID> \
  --num_drivers=4 \          # GPU: 4 drivers is optimal
  --num_splits_per_file=4 \
  --cudf_chunk_read_limit=1GB \
  --cudf_memory_percent=0 \
  --cudf_memory_resource=async \
  --cudf_debug_enabled=true \  # Enable to verify GPU execution
  --v=1 --logtostderr          # Verbose logging for GPU operator verification
```

## Verifying GPU Execution

In verbose GPU logs, look for these lines from `ToCudf.cpp`:

**Scan queries** should show:
```
Operators after adapting for cuDF:
  Operator: ID 0: TableScan[0]
  Operator: ID 1: CudfFilterProject[1]
  Operator: ID 2: CudfToVelox[1-to-velox]
  Operator: ID 3: CallbackSink[N/A]
```

**Join queries** should show (3 pipelines):
```
Pipeline 1 (build): TableScan → CudfHashJoinBuild
Pipeline 2 (probe): TableScan → CudfHashJoinProbe → CudfAggregationPARTIAL → CudfLocalPartition
Pipeline 3 (final): LocalExchange → CudfAggregationFINAL → CudfToVelox → CallbackSink
```

Key indicators of full GPU execution:
- `canRunOnGPU = 1` for all compute operators
- All operators replaced: FilterProject→CudfFilterProject, HashProbe→CudfHashJoinProbe, etc.
- `CudfHiveDataSource.cpp: Adding split CudfHive:` (GPU parquet reader)
- No "CPU fallback" warnings

## Important Notes

1. **Data path MUST be absolute** — relative paths cause "No registered file system" error
2. **First GPU run is slow** (~4 min) due to CUDA JIT if compiled for wrong SM arch. Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU (80=A100, 90=H100)
3. **Schema matters**: The data generation script (`gen_tpch_parquet.py`) casts columns to match Velox's exact expected types. DuckDB's default DECIMAL(15,2) and INTEGER types cause runtime errors in Velox.
4. **pixi manages all build dependencies** (cmake, ninja, gcc, cuda toolkit, etc.) without needing root. This is the recommended approach even on Azure where you have root.
5. **Wall time is accurate for GPU timing** — GPU operators use `stream.synchronize()`, so wall time includes actual GPU kernel execution.
6. **Memory**: Data generation needs ~64GB RAM. Use a VM with sufficient memory.
7. **`test_datasets/`** directory is in `.gitignore` — data must be regenerated on each machine.

## Expected Results Structure

After `run_all.sh`, results are in `exp/vldb-benchmark/results/`:
```
results/
  cpu_q23.txt   # CPU verbose output for Q23
  cpu_q24.txt
  ...
  gpu_q23.txt   # GPU verbose output for Q23 (with debug logs)
  gpu_q24.txt
  ...
```

## Troubleshooting

- **"No registered file system"** → Use absolute path for `--data_path`
- **"multiply(DOUBLE, INTEGER) not registered"** → Data schema mismatch, regenerate with `gen_tpch_parquet.py`
- **Slow first GPU run** → CUDA JIT compiling kernels. Subsequent runs are fast.
- **OOM during data generation** → Reduce `memory_limit` in gen_tpch_parquet.py or use smaller SF
