#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Scaling 181.20 Champion (models/spatial_bc4.txt, 64x32 native speed) to w20/w24 h8/h9 ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt fullmlp_scale 5

echo ""
echo "=== Part 2: 10-Seed (101..=110) Held-Out Legal Validation of 181.20 Champion (fullmlp_w16) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt fullmlp_w16 10
