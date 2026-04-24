#!/bin/bash
set -e

LOG=/users/PAS2065/ruochenj/velox/build.log
exec > "$LOG" 2>&1

echo "=== Velox Build Started: $(date) ==="
cd /users/PAS2065/ruochenj/velox

# Step 1: CMake configure
echo "=== CMake Configure ==="
pixi run cmake -B _build/release -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVELOX_ENABLE_CUDF=ON \
  -DVELOX_ENABLE_ARROW=ON \
  -DVELOX_ENABLE_PARQUET=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
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

# Step 2: Build
echo "=== Building with -j 8 ==="
pixi run cmake --build _build/release -j 8

echo "=== Build Finished Successfully: $(date) ==="
