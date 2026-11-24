#!/usr/bin/env bash
set -euo pipefail

echo "=== REAL FULL-MLP 10-Seed (101..=110) Verification of models/spatial_bc5.txt (128x48, val_ce=0.8974, ~60m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc5.txt fullmlp_w16 10
