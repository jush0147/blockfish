#!/usr/bin/env bash
set -euo pipefail

echo "=== Round WG-2: WG-DS cons22 (96g, B=1600) + Legal-DAgger (48g) + Conservation-Prior SpatialValueModel ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher ew:cons22 \
  --out models/spatial_wg2.txt \
  --wg-beam 1600 \
  --wg-ds \
  --games 96 \
  --legal-dagger 48 \
  --cf-per-game 16 \
  --roll-ds 6 \
  --roll-bw 24 \
  --crn-worlds 5 \
  --epochs 12 \
  --batch 128 \
  --lr 0.0005 \
  --wd 0.025 \
  --lin-lr-mult 0.35 \
  --val-w 0.12 \
  --diff-w 0.65 \
  --rank-w 2.0 \
  --cal-v-weight 0.75 \
  --no-roll-vcal \
  --seed-base 1000 \
  --eval-oracle 5 \
  --eval-legal 5

echo ""
echo "=== Post-WG2 Oracle & Legal Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_wg2.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_wg2.txt sp_rpimc_linfast_w8_h9
