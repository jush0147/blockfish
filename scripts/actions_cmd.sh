#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: REAL FULL-MLP Pure Mean (cvar=0, h=8, w=16, b=28, models/spatial_bc4.txt, 5 seeds 101..=105, ~14m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_cvar0 5

echo ""
echo "=== Part 2: REAL FULL-MLP Sharp Horizon h=7 (cvar=0, w=20, b=36, models/spatial_bc4.txt, 5 seeds 101..=105, ~20m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_h7_c0 5

echo ""
echo "=== Part 3: REAL FULL-MLP Exact-Preview Horizon h=6 (cvar=0, w=20, b=40, models/spatial_bc4.txt, 5 seeds 101..=105, ~18m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_h6_c0 5

