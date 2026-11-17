#!/usr/bin/env bash
set -euo pipefail

echo "=== Round WG-1: Whole-Game Oracle (B=2000, 120 games) + Unbiased CRN SpatialValueModel RL ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher cons:models/crn_r3_h2stk03.txt \
  --out models/spatial_wg1.txt \
  --wg-beam 2000 \
  --games 120 \
  --cf-per-game 20 \
  --roll-horizon 22 \
  --roll-bw 24 \
  --crn-worlds 4 \
  --epochs 30 \
  --batch 128 \
  --lr 0.0008 \
  --wd 0.015 \
  --lin-lr-mult 0.35 \
  --val-w 0.08 \
  --diff-w 0.65 \
  --rank-w 2.0 \
  --cal-v-weight 0.0 \
  --no-roll-vcal \
  --seed-base 1000 \
  --eval-teacher \
  --eval-oracle 5 \
  --eval-legal 10

echo ""
echo "=== Post-WG1 Oracle & Legal Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_wg1.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_q models/spatial_wg1_ep0.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_q models/spatial_wg1.txt
