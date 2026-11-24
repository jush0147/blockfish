#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: REAL FULL-MLP Oracle Ceiling Check (OR_p20, models/spatial_bc4.txt, 2 seeds 101..=102, ~4.5m) ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_or_p20 models/spatial_bc4.txt 2

echo ""
echo "=== Part 2: Eliminate Stage-1 Opaque 0x3FF Pruning (4-world CRN prescreen 36->12, w=16, h=8, b=28, cvar=25, 5 seeds, ~18m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_prescreen 5

echo ""
echo "=== Part 3: Top-4 2-Ply Non-Clairvoyant Expectimax Rerank (d2k4, w=12, h=8, b=24, cvar=25, 5 seeds, ~22m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_d2_exp 5
