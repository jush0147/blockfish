#!/usr/bin/env bash
set -euo pipefail

echo "=== Round WG-3: WG-DS ew:cons22+strat_np (96g, B=1600) + Legal-DAgger (20g) + Relaxed-Bounds SpatialValueModel ==="
cargo run --release -p blockfish --features race --example train_spatial -- \
  --teacher ew:cons22 \
  --out models/spatial_wg3.txt \
  --wg-beam 1600 \
  --wg-ds \
  --games 96 \
  --legal-dagger 20 \
  --cf-per-game 14 \
  --roll-ds 6 \
  --roll-bw 24 \
  --crn-worlds 5 \
  --epochs 20 \
  --batch 128 \
  --lr 0.0010 \
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
echo "=== Post-WG3 Oracle & Legal Diagnostics ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_probe models/spatial_wg3.txt
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_wg3.txt sp_rpimc_sim_w8_h9
