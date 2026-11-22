#!/usr/bin/env bash
set -euo pipefail

echo "=== Round NC-1: Hybrid Known-Prefix Beam + Non-Clairvoyant Rollout PIMC & BC-2 MLP + CVaR30/40 (5-seed Legal) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc2_lin.txt nc_pimc 5
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc2.txt nc_pimc_mlp 5
