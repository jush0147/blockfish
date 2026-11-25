#!/usr/bin/env bash
set -euo pipefail

echo "=== Colab Run #1: Direct Legal REAL-MLP root_pimc ES Round 2 (from spatial_pimc_es1.txt, 12 gens, 45m guard) ==="
echo "Hardware: $(nproc) CPU cores"

cargo run --release -p blockfish --features race --example opt_pimc -- \
  --init models/spatial_pimc_es1.txt \
  --out models/spatial_pimc_es2.txt \
  --gens 12 \
  --pairs 3 \
  --batch-seeds 8 \
  --worlds 4 \
  --width 12 \
  --roots 8 \
  --sigma-lin 0.045 \
  --sigma-w3 0.015 \
  --lr 0.014 \
  --max-secs 2700

echo ""
echo "=== Held-Out 10-Seed (101..=110) REAL FULL-MLP Legal Verification on models/spatial_pimc_es2.txt (vs bc4 185.60) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_pimc_es2.txt fullmlp_w16 10
