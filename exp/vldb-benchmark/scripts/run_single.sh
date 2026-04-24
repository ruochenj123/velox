#!/bin/bash
# Run a single query on CPU or GPU with verbose output
# Usage: bash run_single.sh <cpu|gpu> <query_id> [num_drivers]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DEVICE=${1:?Usage: run_single.sh <cpu|gpu> <query_id>}
QID=${2:?Usage: run_single.sh <cpu|gpu> <query_id>}
DATA_PATH="${REPO_ROOT}/test_datasets/tpch/sf100"

case "${DEVICE}" in
    cpu)
        BIN="${REPO_ROOT}/_build/release/velox/benchmarks/tpch/velox_tpch_benchmark"
        DRIVERS=${3:-32}
        EXTRA_FLAGS=""
        ;;
    gpu)
        BIN="${REPO_ROOT}/_build/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark"
        DRIVERS=${3:-4}
        EXTRA_FLAGS="--cudf_chunk_read_limit=1GB --cudf_memory_percent=0 --cudf_memory_resource=async"
        ;;
    gpu-debug)
        BIN="${REPO_ROOT}/_build/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark"
        DRIVERS=${3:-4}
        EXTRA_FLAGS="--cudf_chunk_read_limit=1GB --cudf_memory_percent=0 --cudf_memory_resource=async --cudf_debug_enabled=true --v=1 --logtostderr"
        ;;
    *)
        echo "Unknown device: ${DEVICE}. Use cpu, gpu, or gpu-debug"
        exit 1
        ;;
esac

echo "=== Running Q${QID} on ${DEVICE} (${DRIVERS} drivers) ==="
pixi run "${BIN}" \
    --data_path="${DATA_PATH}" \
    --data_format=parquet \
    --run_query_verbose="${QID}" \
    --num_drivers="${DRIVERS}" \
    --num_splits_per_file=4 \
    ${EXTRA_FLAGS} \
    2>&1
