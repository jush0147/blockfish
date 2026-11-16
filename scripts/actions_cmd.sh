#!/usr/bin/env bash
set -euo pipefail

echo "=== Round 1: SpatialValueModel ExIt (140 games, H=12, B=150) ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher learned:models/crn_r3_h2stk03.txt \
  --out models/spatial_r1.txt \
  --games 140 \
  --horizon 12 \
  --beam 150 \
  --max-per-root 24 \
  --epochs 30 \
  --batch 128 \
  --lr 0.0012 \
  --wd 0.003 \
  --rank-w 2.0 \
  --diff-w 1.5 \
  --val-w 0.03 \
  --seed-base 1000 \
  --eval-oracle 5 \
  --eval-legal 0

echo ""
echo "=== Round 2: Self-Play SpatialValueModel ExIt (140 games, H=14, B=170) ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher spatial:models/spatial_r1.txt \
  --init models/spatial_r1.txt \
  --out models/spatial_r2.txt \
  --games 140 \
  --horizon 14 \
  --beam 170 \
  --max-per-root 24 \
  --epochs 30 \
  --batch 128 \
  --lr 0.0009 \
  --wd 0.003 \
  --rank-w 2.0 \
  --diff-w 1.5 \
  --val-w 0.03 \
  --seed-base 2000 \
  --eval-oracle 5 \
  --eval-legal 5

echo ""
echo "=== Post-Round2 Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_r2.txt
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_oracle5 models/spatial_r2.txt
