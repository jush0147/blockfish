#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-2: Full-Action-Space BC + Legal BC-DAgger (120g WG_ds B=1400 + 40g DAgger -> models/spatial_bc2.txt) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc2.txt \
  --games 120 \
  --wg-beam 1400 \
  --dagger-init models/spatial_bc1_lin.txt \
  --dagger-games 40 \
  --lin-epochs 16 \
  --mlp-epochs 16 \
  --batch 32 \
  --tau 0.50 \
  --cal-w 0.025 \
  --lr-lin 0.008 \
  --lr-mlp 0.0008 \
  --wd 0.002 \
  --wd-mlp 0.015 \
  --seed-base 3000 \
  --eval-legal 5

echo ""
echo "=== Post-BC2 5-Seed Held-Out Legal Benchmarks (h=7/8/9 Wide-World & CVaR Anti-Strategy-Fusion) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc2_lin.txt bc_cvar 5
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_bc2.txt sp_rpimc_sim_w8_h9
