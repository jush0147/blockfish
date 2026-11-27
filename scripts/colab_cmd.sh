#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #21-CPU: FIXED True 50/50 FULL-MLP Ensemble A/B"
echo "                    eval_core_ex spatial_model_2 fix verified!"
echo "                    Same-Binary 5-seed fullmlp_w16 (~36 min)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

MODEL="/tmp/spatial_es_v3.txt"
echo "--- Downloading spatial_es_v3.txt from Release model-spatial-es-v3 ---"
python3 scripts/download_release_asset.py model-spatial-es-v3 spatial_es_v3.txt "$MODEL"

echo ""
echo "--- Running unit tests & building bench_legal (release) ---"
cargo test --release -p blockfish --features block-stacker --lib test_eval_core_ex_ensemble2_uses_both_models
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- Running FIXED es_v3_combo (bc4 base, TRUE ens50 bc4+esv3, esv3+bc4cov115) ---"
target/release/examples/bench_legal es_v3_combo 5 models/spatial_bc4.txt "$MODEL"

echo "=== Colab Run #21-CPU COMPLETE ==="
