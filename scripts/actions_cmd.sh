#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Full-MLP Inside PIMC Beam (spatial_rerank_only=false) on models/spatial_bc4.txt (V5 MLP) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt full_mlp_pimc 5

echo ""
echo "=== Part 2: Full-MLP Inside PIMC Beam (spatial_rerank_only=false) on models/spatial_bc3.txt (V4 MLP) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc3.txt full_mlp_pimc 5
