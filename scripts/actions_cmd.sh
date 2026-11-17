#!/usr/bin/env bash
set -euo pipefail

echo "=== Round 3: OOD-Safe Calibrated Finite-Horizon CRN SpatialValueModel RL (150 games, H=14, B=150) ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher learned:models/crn_r3_h2stk03.txt \
  --out models/spatial_r3.txt \
  --games 150 \
  --horizon 14 \
  --beam 150 \
  --max-per-root 24 \
  --roll-horizon 20 \
  --roll-bw 16 \
  --crn-worlds 6 \
  --cf-per-game 20 \
  --epochs 30 \
  --batch 128 \
  --lr 0.0005 \
  --wd 0.015 \
  --lin-lr-mult 0.45 \
  --val-w 0.25 \
  --diff-w 0.55 \
  --rank-w 1.1 \
  --cal-v-weight 0.55 \
  --seed-base 1000 \
  --eval-teacher \
  --eval-oracle 5 \
  --eval-legal 10

echo ""
echo "=== Post-Round3 Oracle & Legal Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_r3.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_q models/spatial_r3.txt
