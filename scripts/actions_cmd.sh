#!/usr/bin/env bash
set -euo pipefail

cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher learned:models/crn_r3_h2stk03.txt \
  --out models/spatial_v1.txt \
  --games 120 \
  --horizon 12 \
  --beam 140 \
  --max-per-root 20 \
  --epochs 40 \
  --batch 128 \
  --lr 0.0025 \
  --rank-w 2.0 \
  --seed-base 1000 \
  --eval-seeds 5

cargo run --release -p blockfish --features race --example diag_cheese -- spatial_oracle5 models/spatial_v1.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5 models/spatial_v1.txt
