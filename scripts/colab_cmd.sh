#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #20-CPU: Complementary bc4 + es_v3 Ensemble & Cov Fix"
echo "                    Same-Binary 5-seed fullmlp_w16 (~36 min)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

MODEL="/tmp/spatial_es_v3.txt"
echo "--- Downloading spatial_es_v3.txt from Release model-spatial-es-v3 ---"
python3 scripts/download_release_asset.py model-spatial-es-v3 spatial_es_v3.txt "$MODEL"

echo ""
echo "--- Building bench_legal (release) ---"
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- Running es_v3_combo (bc4 base, ens50 bc4+esv3, esv3+bc4cov115) ---"
target/release/examples/bench_legal es_v3_combo 5 models/spatial_bc4.txt "$MODEL"

echo "=== Colab Run #20-CPU COMPLETE ==="
