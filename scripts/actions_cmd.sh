#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Held-Out 5-Seed REAL FULL-MLP Legal Benchmark on models/spatial_v5q1.txt (vs bc4 184.20, ~17m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_v5q1.txt fullmlp_w16 5

echo ""
echo "=== Part 2: V5 Complementary Ensemble ens2 (spatial_bc4.txt [wide=22] + spatial_v5q1.txt [cov=20.5], 5 seeds, ~23m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4.txt ens2:models/spatial_v5q1.txt 5
