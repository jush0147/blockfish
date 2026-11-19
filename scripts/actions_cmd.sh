#!/usr/bin/env bash
set -euo pipefail

echo "=== Round WG-4: WG-DS ew:cons22 (88g, B=1800) + Local-Oracle Legal-DAgger (24g) + Unflipped Expert-Anchored SpatialValueModel ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher ew:cons22 \
  --out models/spatial_wg4.txt \
  --wg-beam 1800 \
  --wg-ds \
  --games 88 \
  --legal-dagger 24 \
  --cf-per-game 14 \
  --roll-ds 8 \
  --roll-bw 36 \
  --crn-worlds 5 \
  --epochs 24 \
  --batch 128 \
  --lr 0.0010 \
  --wd 0.025 \
  --lin-lr-mult 0.35 \
  --val-w 0.12 \
  --diff-w 0.65 \
  --rank-w 2.0 \
  --cal-v-weight 0.40 \
  --no-roll-vcal \
  --seed-base 1000 \
  --eval-oracle 5 \
  --eval-legal 5

echo ""
echo "=== Post-WG4 Oracle & Legal Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_wg4.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_wg4.txt sp_rpimc_sim_w8_h9
