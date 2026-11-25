#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Direct Legal REAL-MLP root_pimc ES (48 params from spatial_bc4.txt, strict 31m guard) ==="
cargo run --release -p blockfish --features race --example opt_pimc -- \
  --init models/spatial_bc4.txt \
  --out models/spatial_pimc_es1.txt \
  --gens 14 \
  --pairs 3 \
  --batch-seeds 8 \
  --worlds 4 \
  --width 14 \
  --roots 8 \
  --sigma-lin 0.045 \
  --sigma-w3 0.015 \
  --lr 0.015 \
  --max-secs 1900

echo ""
echo "=== Part 2: Held-Out 5-Seed REAL FULL-MLP Legal Verification on models/spatial_pimc_es1.txt (vs bc4 184.20, ~16m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_pimc_es1.txt fullmlp_w16 5
