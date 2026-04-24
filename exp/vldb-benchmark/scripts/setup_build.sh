#!/bin/bash
# Build Velox + cuDF on a fresh machine with pixi
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

echo "=== Velox+cuDF Build ==="
echo "Repo: ${REPO_ROOT}"
echo "Date: $(date)"

# Detect GPU architecture
if command -v nvidia-smi &>/dev/null; then
    SM=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 | tr -d '.')
    echo "Detected GPU SM: ${SM}"
else
    SM=80
    echo "nvidia-smi not found, defaulting to SM ${SM}"
fi

# Update pixi.toml CUDAARCHS if needed
if [[ "${SM}" != "80" ]]; then
    echo "Updating pixi.toml CUDAARCHS to ${SM}"
    sed -i "s/CUDAARCHS = \"80\"/CUDAARCHS = \"${SM}\"/" pixi.toml
fi

# Install pixi if not available
if ! command -v pixi &>/dev/null; then
    echo "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

# Configure
echo "=== CMake Configure ==="
pixi run cmake -B _build/release -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVELOX_ENABLE_CUDF=ON \
  -DVELOX_ENABLE_ARROW=ON \
  -DVELOX_ENABLE_PARQUET=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${SM}" \
  -DVELOX_DEPENDENCY_SOURCE=BUNDLED \
  -DICU_SOURCE=SYSTEM \
  -DVELOX_BUILD_TESTING=OFF \
  -DVELOX_ENABLE_BENCHMARKS=ON \
  -DVELOX_ENABLE_BENCHMARKS_BASIC=ON \
  -DTREAT_WARNINGS_AS_ERRORS=OFF \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DVELOX_ENABLE_GEO=OFF \
  -Wno-dev

echo "=== CMake Configure Done: $(date) ==="

# Build
NPROC=$(nproc)
JOBS=$((NPROC > 16 ? 16 : NPROC))
echo "=== Building with -j ${JOBS} ==="
pixi run cmake --build _build/release -j "${JOBS}"

echo "=== Build Finished: $(date) ==="

# Verify binaries
echo ""
echo "=== Checking Binaries ==="
for bin in \
    _build/release/velox/benchmarks/tpch/velox_tpch_benchmark \
    _build/release/velox/experimental/cudf/benchmarks/velox_cudf_tpch_benchmark; do
    if [[ -x "${bin}" ]]; then
        echo "  OK: ${bin} ($(du -h "${bin}" | cut -f1))"
    else
        echo "  MISSING: ${bin}"
    fi
done
