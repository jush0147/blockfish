#!/usr/bin/env bash
set -euo pipefail

echo "=== Colab Run #2: REAL FULL-MLP Risk/Cell Calibration Sweep on models/spatial_bc4.txt (5 seeds, ~28m) ==="
echo "Hardware: $(nproc) CPU cores"
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_calib 5

echo ""
echo "=== Part 2: 10-Seed (101..=110) REAL FULL-MLP Verification of realmlp_prescreen (vs 185.60, ~18m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_prescreen 10
