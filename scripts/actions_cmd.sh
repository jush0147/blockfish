#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-6 (SPATIAL_VAL_V6 Exact 2D 3x3 Pattern NNUE 64x32): 150 WG_ds(B=1800) + 60 DAgger(bc4) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc6.txt \
  --games 150 \
  --wg-beam 1800 \
  --dagger-init models/spatial_bc4.txt \
  --dagger-games 60 \
  --lin-epochs 32 \
  --mlp-epochs 26 \
  --batch 32 \
  --tau 0.45 \
  --cal-w 0.025 \
  --lr-lin 0.008 \
  --lr-mlp 0.0009 \
  --wd 0.002 \
  --wd-mlp 0.012 \
  --dropout 0.15 \
  --label-smooth 0.02 \
  --eval-legal 5

echo ""
echo "=== Fair 10-Seed (101..=110) Held-Out Legal Verification of models/spatial_bc6.txt (fullmlp_w16 vs 187.90) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc6.txt fullmlp_w16 10
