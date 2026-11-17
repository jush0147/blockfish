#!/usr/bin/env bash
set -euo pipefail

echo "=== Round 1: True-Outcome CRN-Terminal SpatialValueModel RL (140 games, H=12, B=140) ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher learned:models/crn_r3_h2stk03.txt \
  --out models/spatial_r1.txt \
  --games 140 \
  --horizon 12 \
  --beam 140 \
  --max-per-root 24 \
  --roll-bw 10 \
  --crn-worlds 4 \
  --cf-per-game 12 \
  --epochs 40 \
  --batch 128 \
  --lr 0.0005 \
  --wd 0.012 \
  --lin-lr-mult 0.06 \
  --rank-w 1.5 \
  --diff-w 0.30 \
  --val-w 0.05 \
  --seed-base 1000 \
  --eval-teacher \
  --eval-oracle 5 \
  --eval-legal 10

echo ""
echo "=== Post-Round1 Legal & Oracle Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_r1.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_q models/spatial_r1.txt

