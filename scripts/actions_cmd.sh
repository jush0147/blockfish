#!/usr/bin/env bash
set -euo pipefail

echo "=== Round ES-2: Direct Legal p=5 Antithetic-ES from spatial_bc2_lin.txt (18 iters x 16 evals x 6 games -> models/spatial_es2.txt) ==="
cargo run --release -p blockfish --features race --example opt_legal -- \
  --init models/spatial_bc2_lin.txt \
  --out models/spatial_es2.txt \
  --iters 18 \
  --pop 8 \
  --games 6 \
  --val-games 5 \
  --beam 64 \
  --sigma 0.020 \
  --lr 0.010 \
  --seed 20261122

echo ""
echo "=== Post-ES2 5-Seed Held-Out Legal CVaR-Pro Verification ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_es2.txt cvar_pro 5
