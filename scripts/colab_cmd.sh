#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #30-CPU: Pure BC-4 1-Ply Prior in Formal fullmlp_w16"
echo "                    Opt-in root_step1_prior_pct (0 vs 8 vs 15, exact x100 scale)"
echo "                    Validated on 40 fresh 5-ply seeds (-3.33 +/- 1.57, 95%CI=[-6.40,-0.25])"
echo "                    Strict Same-Binary 5-Seed A/B (101..=105)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- [0/2] Running library unit tests ---"
cargo test --lib --features block-stacker

echo "--- [1/2] Building bench_legal (release) ---"
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- [2/2] Running strict same-binary fullmlp_w16 step1_prior A/B (5 seeds 101..=105) ---"
target/release/examples/bench_legal step1_prior 5 models/spatial_bc4.txt

echo "=== Colab Run #30-CPU COMPLETE ==="
