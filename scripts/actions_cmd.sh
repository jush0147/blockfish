#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-3 (SPATIAL_VAL_V4 + Upgraded Endgame): 150 WG_ds(B=1600) + 50 DAgger(es3) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc3.txt \
  --games 150 \
  --wg-beam 1600 \
  --dagger-init models/spatial_es3.txt \
  --dagger-games 50 \
  --lin-epochs 30 \
  --mlp-epochs 24 \
  --batch 32 \
  --tau 0.45 \
  --cal-w 0.025 \
  --lr-lin 0.008 \
  --lr-mlp 0.0010 \
  --wd 0.002 \
  --wd-mlp 0.012 \
  --eval-legal 5

echo ""
echo "=== Held-Out 5-Seed Legal Verification of models/spatial_bc3_lin.txt (cvar_top2) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc3_lin.txt cvar_top2 5
