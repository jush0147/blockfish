#!/usr/bin/env bash
set -euo pipefail

echo "=== Non-Clairvoyant Prefix-6 + CVaR Leaf-PIMC (total_h = 10..12, ZERO Prefix Strategy Fusion) on 10 Seeds (101..=110) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt leaf_pimc_break 10
