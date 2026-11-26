#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #17-CPU: 100% Legal-Physics 2-Ply Shared-Prefix PIMC"
echo "                    Same-Binary 5-seed A/B (101..=105) on beam.rs"
echo "                    base vs d2k4_w16 vs d2k6_cov115_w16 (~65 min)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building bench_legal (release) ---"
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- Running d2_expectimax (base, d2k4_w16, d2k6_cov115_w16) ---"
target/release/examples/bench_legal d2_expectimax 5 models/spatial_bc4.txt

echo "=== Colab Run #17-CPU COMPLETE ==="
