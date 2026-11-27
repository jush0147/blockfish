#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #28-CPU: DS-Bucketed A* FULL-MLP PIMC (Search Paradigm Shift)"
echo "                    Exact g(n)+h(n) to identical target_ds buckets (ds=4, ds=5)"
echo "                    Cancels rem_g horizon bias vs fixed-depth h=8 fullmlp_w16"
echo "                    Strict Same-Binary 5-Seed A/B (101..=105)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building bench_legal (release) ---"
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- Running strict same-binary ds_fullmlp A/B (5 seeds 101..=105) ---"
target/release/examples/bench_legal ds_fullmlp 5 models/spatial_bc4.txt

echo "=== Colab Run #28-CPU COMPLETE ==="
