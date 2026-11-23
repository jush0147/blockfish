#!/usr/bin/env bash
set -euo pipefail

echo "=== 10-Seed (101..=110) Legal Risk-Calibration Sweep on models/spatial_bc4.txt (fullmlp_w16 vs 187.90, ~48m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt risk_calib 10
