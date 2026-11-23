#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-4 (SPATIAL_VAL_V5 + Spatial Endgame): 150 WG_ds(B=1800) + 60 DAgger(bc3_lin) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc4.txt \
  --games 150 \
  --wg-beam 1800 \
  --dagger-init models/spatial_bc3_lin.txt \
  --dagger-games 60 \
  --lin-epochs 32 \
  --mlp-epochs 24 \
  --batch 32 \
  --tau 0.45 \
  --cal-w 0.025 \
  --lr-lin 0.008 \
  --lr-mlp 0.0008 \
  --wd 0.002 \
  --wd-mlp 0.015 \
  --eval-legal 5

echo ""
echo "=== Held-Out 5-Seed Legal Verification of models/spatial_bc4_lin.txt (cvar16: pro_h9_w16_b32_cvar30) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4_lin.txt cvar16 5
