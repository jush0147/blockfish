#!/usr/bin/env bash
set -euo pipefail

echo "=== Round BC-7: V6 2D Pattern 64x32 | WG_ds(B=2800) x 100 + REAL-MLP Soft-Q DAgger(bc4) x 50 + Mixed Real-Hole Spawn (~45m) ==="
cargo run --release -p blockfish --features block-stacker --example train_bc -- \
  --out models/spatial_bc7.txt \
  --games 100 \
  --wg-beam 2800 \
  --dagger-init models/spatial_bc4.txt \
  --dagger-games 50 \
  --dagger-mlp \
  --mixed-spawn \
  --lin-epochs 18 \
  --mlp-epochs 24 \
  --dropout 0.15 \
  --label-smooth 0.02 \
  --tau 0.50 \
  --cal-w 0.025 \
  --seed-base 9000 \
  --eval-legal 0

