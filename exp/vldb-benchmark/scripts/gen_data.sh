#!/bin/bash
# Generate TPC-H data for Velox benchmarks
# Requires: pip install duckdb
# Usage: bash gen_data.sh [SF] [MEMORY]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

SF=${1:-100}
MEMORY=${2:-64GB}

echo "=== Generating TPC-H SF${SF} Data ==="

cd "${REPO_ROOT}"

# Install duckdb if needed
pip install duckdb --quiet 2>/dev/null || pixi run pip install duckdb --quiet 2>/dev/null || true

python scripts/gen_tpch_parquet.py --sf "${SF}" --memory "${MEMORY}"

echo ""
echo "=== Data Generation Complete ==="
