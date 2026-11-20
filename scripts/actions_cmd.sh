#!/usr/bin/env bash
set -euo pipefail

echo "=== Direct Legal p=5 Antithetic-ES Round 1 (models/spatial_es1.txt) ==="
cargo run --release -p blockfish --features race --example opt_legal -- \
  --init models/spatial_wg5_ep0.txt \
  --out models/spatial_es1.txt \
  --iters 35 \
  --pop 10 \
  --games 8 \
  --val-games 5 \
  --beam 64 \
  --sigma 0.025 \
  --lr 0.012 \
  --seed 20261120

echo ""
echo "=== Post-ES1 Held-Out 5-Seed Legal Benchmarks ==="
cargo run --release -p blockfish --features race --example bench_legal -- spatial5_1 models/spatial_es1.txt sp_rpimc_linfast_nope_w8_h9
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_es1.txt deep 5
