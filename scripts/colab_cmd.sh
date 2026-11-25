#!/usr/bin/env bash
set -euo pipefail

echo "=== Colab Run #4 Part 1: 10-Seed (101..=110) REAL FULL-MLP Verification of NEW 5-Seed Champion realmlp_cov115 (vs 185.60, ~20m) ==="
echo "Hardware: $(nproc) CPU cores"
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_cov115 10

echo ""
echo "=== Colab Run #4 Part 2: 40-Seed (101..=140) Legal PB & Distribution Audit (realmlp_cov115 w=8 h=8 b=20, ~22m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_cov115_fast 40
