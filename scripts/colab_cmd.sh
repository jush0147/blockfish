#!/usr/bin/env bash
set -euo pipefail

echo "=== Colab Run #3 Part 1: 10-Seed (101..=110) REAL FULL-MLP Verification of realmlp_cov130 (vs bc4 185.60, ~18m) ==="
echo "Hardware: $(nproc) CPU cores"
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_cov130 10

echo ""
echo "=== Colab Run #3 Part 2: Fine cov/nar Calibration Sweep (cov115, cov145, cov130_nar130, cov130_c0, 5 seeds, ~26m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_cov_sweep 5
