#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: REAL FULL-MLP 1.40x Boost + CVaR-30 on 10 Seeds (101..=110, models/spatial_bc4.txt vs 185.60, ~34m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_boost 10

echo ""
echo "=== Part 2: REAL FULL-MLP Scaled h=9 w=20 b=36 CVaR-30 on 5 Seeds (101..=105, models/spatial_bc4.txt, ~24m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_h9 5
