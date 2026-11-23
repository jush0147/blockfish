#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: CVaR-Pro 5-Seed Sweep on models/spatial_bc2_lin.txt (current 186.00 record) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc2_lin.txt cvar_pro 5

echo ""
echo "=== Part 2: Fixed Direct Legal ES-3 from models/spatial_bc2_lin.txt (10 iters x 12 evals x 6 games, no clamp -> models/spatial_es3.txt) ==="
cargo run --release -p blockfish --features race --example opt_legal -- \
  --init models/spatial_bc2_lin.txt \
  --out models/spatial_es3.txt \
  --iters 10 \
  --pop 6 \
  --games 6 \
  --val-games 5 \
  --beam 64 \
  --sigma 0.012 \
  --lr 0.006 \
  --seed 20261123

echo ""
echo "=== Part 3: Held-Out 5-Seed Legal Verification of models/spatial_es3.txt ==="
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_es3.txt sp_rpimc_linfast_nope_w8_h9
