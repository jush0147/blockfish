#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #15-CPU: Anti-Strategy-Fusion Opaque-Future-Hole PIMC"
echo "                    + High-Budget w=24 h=9 b=36 on beam.rs (185.60)"
echo "                    Pure Rust CPU! 5 seeds 101..=105 (~18 min)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building bench_legal (release) ---"
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- Running opaque_pimc (same-binary base, d1, d2, w24_h9_b36) ---"
target/release/examples/bench_legal opaque_pimc 5 models/spatial_bc4.txt

echo "=== Colab Run #15-CPU COMPLETE ==="
