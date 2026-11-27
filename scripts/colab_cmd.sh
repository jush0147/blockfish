#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #19-CPU: Strict Same-Binary 5-seed fullmlp_w16 A/B"
echo "                    spatial_bc4.txt vs spatial_es_v3.txt (~38 min)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

MODEL="/tmp/spatial_es_v3.txt"
echo "--- Downloading spatial_es_v3.txt from Release model-spatial-es-v3 ---"
python3 scripts/download_release_asset.py model-spatial-es-v3 spatial_es_v3.txt "$MODEL"

echo ""
echo "--- Building bench_legal (release) ---"
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs es_v3 ---"
target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$MODEL"

echo "=== Colab Run #19-CPU COMPLETE ==="
