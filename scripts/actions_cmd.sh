#!/usr/bin/env bash
set -euo pipefail

echo "=== REAL FULL-MLP 10-Seed (101..=110) Showdown (after fixing wg5_lin_sweep b3/w3 zeroing bug!) ==="

echo ""
echo "=== 1. models/spatial_bc4.txt (V5 64x32 REAL FULL MLP) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt fullmlp_w16 10

echo ""
echo "=== 2. models/spatial_bc6.txt (V6 Exact 2D 3x3 Pattern 64x32 REAL FULL MLP) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc6.txt fullmlp_w16 10

echo ""
echo "=== 3. models/spatial_exit1.txt (ExIt-1 Legal Self-Play 64x32 REAL FULL MLP) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_exit1.txt fullmlp_w16 10
