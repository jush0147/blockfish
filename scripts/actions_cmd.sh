#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: v5_search_break on models/spatial_bc4_lin.txt (h=8 CVaR + short-horizon DS=5 PIMC) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4_lin.txt v5_search_break 5

echo ""
echo "=== Part 2: h=8 CVaR-25 on models/spatial_bc4.txt (V5 Full MLP leaf rerank) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt cvar16 5
