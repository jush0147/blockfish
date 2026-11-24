#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Train Round V5-Q1 (Compact V5 64x32 + Multi-Move Q-Advantage adv_w=0.35 + 100% Real-Hole Spawns, ~22m) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_v5q1.txt \
  --games 120 \
  --wg-beam 1600 \
  --dagger-init models/spatial_bc4.txt \
  --dagger-games 70 \
  --dagger-mlp \
  --real-spawn \
  --v5-only \
  --clean-q \
  --adv-w 0.35 \
  --cal-w 0.04 \
  --lin-epochs 14 \
  --mlp-epochs 18 \
  --wd-mlp 0.015 \
  --dropout 0.15 \
  --label-smooth 0.02 \
  --eval-legal 0

echo ""
echo "=== Part 2: REAL FULL-MLP OR_p20 Ceiling Check on spatial_v5q1.txt (vs bc4 176.00, 2 seeds 101..=102, ~2m) ==="
cargo run --release -p blockfish --features race --example diag_cheese -- spatial_or_p20 models/spatial_v5q1.txt 2

echo ""
echo "=== Part 3: Held-Out 5-Seed REAL FULL-MLP Legal Benchmark (fullmlp_w16 on spatial_v5q1.txt vs bc4 184.20, ~17m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_v5q1.txt fullmlp_w16 5
