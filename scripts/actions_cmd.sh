#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-1: Full-Action-Space Afterstate Cross-Entropy BC (120g WG_ds B=1400 -> models/spatial_bc1.txt) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc1.txt \
  --games 120 \
  --wg-beam 1400 \
  --lin-epochs 16 \
  --mlp-epochs 18 \
  --batch 32 \
  --tau 0.50 \
  --cal-w 0.025 \
  --lr-lin 0.008 \
  --lr-mlp 0.0015 \
  --wd 0.002 \
  --seed-base 3000 \
  --eval-legal 5

echo ""
echo "=== Post-BC1 5-Seed Held-Out Legal Benchmarks ==="
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_bc1_lin.txt sp_rpimc_linfast_nope_w8_h9
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_bc1.txt sp_rpimc_sim_w8_h9
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc1_lin.txt deep 5
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc1_lin.txt ds4 5
