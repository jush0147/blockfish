#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Round BC-7 (V6 2D Soft-Q + Mixed Spawn) 10-Seed REAL FULL-MLP Verification (101..=110, ~30m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc7.txt fullmlp_w16 10

echo ""
echo "=== Part 2: 2-Model REAL FULL-MLP Ensemble (bc4 V5 + bc7 V6 2D Soft-Q) on 5 Seeds (101..=105, ~22m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt ens2:models/spatial_bc7.txt 5

