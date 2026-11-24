#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Non-Clairvoyant REAL FULL-MLP Rollout PIMC (h=6 + nc=12, models/spatial_bc4.txt, 10 seeds 101..=110, ~25m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt realmlp_nc 10

echo ""
echo "=== Part 2: Legal AlphaZero ExIt-1 (models/spatial_exit1.txt) REAL FULL-MLP Verification (5 seeds 101..=105, ~25m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_exit1.txt fullmlp_w16 5

