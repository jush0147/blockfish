#!/usr/bin/env bash
set -euo pipefail

echo "=== Round ExIt-1: Legal AlphaZero Expert Iteration (140 Legal PIMC CVaR-25 games -> models/spatial_exit1.txt) ==="
cargo run --release -p blockfish --features race --example train_exit -- \
  --init models/spatial_bc4.txt \
  --out models/spatial_exit1.txt \
  --games 140 \
  --worlds 6 \
  --horizon 8 \
  --width 20 \
  --cvar 25 \
  --epochs 24 \
  --batch 32 \
  --tau 0.45 \
  --tau-q 0.35 \
  --cal-w 0.08 \
  --lr-lin 0.0005 \
  --lr-mlp 0.0008 \
  --wd-lin 0.001 \
  --wd-mlp 0.005 \
  --dropout 0.12 \
  --seed-base 6000

echo ""
echo "=== Fair 10-Seed (101..=110) Held-Out Legal Verification of models/spatial_exit1.txt vs 187.90 Baseline ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_exit1.txt fullmlp_w16 10
