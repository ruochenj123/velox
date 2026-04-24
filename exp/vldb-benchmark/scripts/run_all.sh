#!/bin/bash
# Run all VLDB benchmark queries on CPU and GPU
# Usage: bash run_all.sh [SF] [NUM_RUNS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../results"
LOGS_DIR="${SCRIPT_DIR}/../logs"

SF=${1:-100}
NUM_RUNS=${2:-3}
DATA_PATH="${REPO_ROOT}/test_datasets/tpch/sf${SF}"

CPU_BIN="${REPO_ROOT}/_build/release/velox/benchmarks/tpch/velox_tpch_benchmark"
GPU_BIN="${REPO_ROOT}/_build/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark"

# Query definitions
SCAN_QUERIES="23 24 25 26"
JOIN_QUERIES="27 28 29 30"
ALL_QUERIES="${SCAN_QUERIES} ${JOIN_QUERIES}"

CPU_NUM_DRIVERS=32
GPU_NUM_DRIVERS=4

mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}"

echo "=== VLDB Benchmark: CPU + GPU ==="
echo "Data: ${DATA_PATH}"
echo "Runs per query: ${NUM_RUNS}"
echo "Results: ${RESULTS_DIR}"
echo ""

# Verify data exists
if [[ ! -d "${DATA_PATH}/lineitem" ]]; then
    echo "ERROR: Data not found at ${DATA_PATH}"
    echo "Run: bash exp/vldb-benchmark/scripts/gen_data.sh ${SF}"
    exit 1
fi

# Verify binaries
for bin in "${CPU_BIN}" "${GPU_BIN}"; do
    if [[ ! -x "${bin}" ]]; then
        echo "ERROR: Binary not found: ${bin}"
        echo "Run: bash exp/vldb-benchmark/scripts/setup_build.sh"
        exit 1
    fi
done

run_cpu() {
    local qid=$1
    local run=$2
    local outfile="${RESULTS_DIR}/cpu_q${qid}_run${run}.txt"
    echo "[CPU] Q${qid} run ${run}..."
    pixi run "${CPU_BIN}" \
        --data_path="${DATA_PATH}" \
        --data_format=parquet \
        --run_query_verbose="${qid}" \
        --num_drivers="${CPU_NUM_DRIVERS}" \
        --num_splits_per_file=4 \
        2>&1 | tee "${outfile}"
    # Extract execution time
    grep "Execution time:" "${outfile}" || true
}

run_gpu() {
    local qid=$1
    local run=$2
    local outfile="${RESULTS_DIR}/gpu_q${qid}_run${run}.txt"
    local logfile="${LOGS_DIR}/gpu_q${qid}_run${run}.log"
    echo "[GPU] Q${qid} run ${run}..."
    pixi run "${GPU_BIN}" \
        --data_path="${DATA_PATH}" \
        --data_format=parquet \
        --run_query_verbose="${qid}" \
        --num_drivers="${GPU_NUM_DRIVERS}" \
        --num_splits_per_file=4 \
        --cudf_chunk_read_limit=1GB \
        --cudf_memory_percent=0 \
        --cudf_memory_resource=async \
        --cudf_debug_enabled=true \
        --v=1 --logtostderr \
        2>"${logfile}" | tee "${outfile}"
    # Extract execution time
    grep "Execution time:" "${outfile}" || true
}

echo "========================================="
echo "  Phase 1: GPU warm-up (JIT compilation)"
echo "========================================="
echo "Running Q23 on GPU to trigger CUDA JIT..."
run_gpu 23 warmup
echo ""

echo "========================================="
echo "  Phase 2: CPU Benchmarks"
echo "========================================="
for qid in ${ALL_QUERIES}; do
    for run in $(seq 1 "${NUM_RUNS}"); do
        run_cpu "${qid}" "${run}"
        echo ""
    done
done

echo "========================================="
echo "  Phase 3: GPU Benchmarks"
echo "========================================="
for qid in ${ALL_QUERIES}; do
    for run in $(seq 1 "${NUM_RUNS}"); do
        run_gpu "${qid}" "${run}"
        echo ""
    done
done

echo "========================================="
echo "  Summary"
echo "========================================="
echo ""
echo "Query | Device | Run | Execution Time"
echo "------|--------|-----|---------------"
for qid in ${ALL_QUERIES}; do
    for device in cpu gpu; do
        for run in $(seq 1 "${NUM_RUNS}"); do
            f="${RESULTS_DIR}/${device}_q${qid}_run${run}.txt"
            if [[ -f "${f}" ]]; then
                time_str=$(grep "Execution time:" "${f}" | head -1 | sed 's/.*Execution time: //')
                printf "Q%-4s | %-6s | %d   | %s\n" "${qid}" "${device}" "${run}" "${time_str}"
            fi
        done
    done
done

echo ""
echo "Full results in: ${RESULTS_DIR}"
echo "GPU debug logs in: ${LOGS_DIR}"
