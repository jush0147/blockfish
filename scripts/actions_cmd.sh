#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: REAL FULL-MLP DS-Target=5 PIMC (models/spatial_bc4.txt, 5 seeds 101..=105, ~20m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_ds5 5

echo ""
echo "=== Part 2: Top-2 REAL FULL-MLP Ensemble (bc4 V5 185.60 + bc6 V6 186.70) on 5 Seeds (101..=105, ~22m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt ens2:models/spatial_bc6.txt 5

