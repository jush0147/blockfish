#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Round BC-8 (Scaled Pure Compact V5 64x32: 220 WG_ds B=1800 + 80 REAL-MLP DAgger, ~30m) ==="
cargo run --release -p blockfish --features race --example train_bc -- \
  --out models/spatial_bc8.txt \
  --games 220 \
  --wg-beam 1800 \
  --dagger-init models/spatial_bc4.txt \
  --dagger-games 80 \
  --dagger-mlp \
  --v5-only \
  --lin-epochs 14 \
  --mlp-epochs 20 \
  --wd-mlp 0.015 \
  --dropout 0.15 \
  --label-smooth 0.02 \
  --eval-legal 0

echo ""
echo "=== Part 2: Held-Out 5-Seed REAL FULL-MLP Legal Benchmark on models/spatial_bc8.txt (vs bc4 184.20, ~17m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc8.txt fullmlp_w16 5
