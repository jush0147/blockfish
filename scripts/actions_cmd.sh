#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-5 (Wide 128x48 V5 NNUE + Dropout 0.15 + LS 0.02): 350 WG_ds(B=900) + 50 DAgger(bc4) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc5.txt \
  --games 350 \
  --wg-beam 900 \
  --dagger-init models/spatial_bc4.txt \
  --dagger-games 50 \
  --lin-epochs 28 \
  --mlp-epochs 26 \
  --batch 32 \
  --tau 0.45 \
  --cal-w 0.025 \
  --lr-lin 0.008 \
  --lr-mlp 0.0010 \
  --wd 0.002 \
  --wd-mlp 0.008 \
  --dropout 0.15 \
  --label-smooth 0.02 \
  --eval-legal 5

echo ""
echo "=== Held-Out 5-Seed Legal Verification of models/spatial_bc5.txt (fullmlp_w16: beating 181.20!) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc5.txt fullmlp_w16 5
