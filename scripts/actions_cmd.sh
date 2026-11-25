#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Round ExIt-2 (Real-Hole CRN Spawns + V5-Only + Relative Legal PIMC Q-Advantage adv_w=0.30, 64 games, ~35m) ==="
cargo run --release -p blockfish --features race --example train_exit -- \
  --init models/spatial_bc4.txt \
  --out models/spatial_exit2.txt \
  --games 64 \
  --worlds 6 \
  --horizon 8 \
  --width 20 \
  --roots 10 \
  --cvar 25 \
  --v5-only \
  --adv-w 0.30 \
  --cal-w 0.08 \
  --epochs 20 \
  --seed-base 7000

echo ""
echo "=== Part 2: Held-Out 5-Seed REAL FULL-MLP Legal Verification on models/spatial_exit2.txt (vs bc4 184.20, ~17m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_exit2.txt fullmlp_w16 5
