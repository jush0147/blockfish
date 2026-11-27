#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #18-CPU: Direct Legal CRN Antithetic ES (opt_pimc)"
echo "                    16-seed batches x 4 pairs x 8 gens on spatial_bc4"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (5 seeds)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building opt_pimc and bench_legal (release) ---"
cargo build --release --example opt_pimc --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_es_v3.txt"

echo ""
echo "--- [1/3] Running Direct Legal CRN Antithetic ES (max 38 min) ---"
target/release/examples/opt_pimc \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --gens 10 \
  --pairs 4 \
  --batch-seeds 16 \
  --val-seeds 16 \
  --worlds 4 \
  --width 14 \
  --roots 8 \
  --sigma-lin 0.04 \
  --sigma-w3 0.015 \
  --lr 0.015 \
  --max-secs 2280

echo ""
echo "--- [2/3] Uploading spatial_es_v3.txt to GitHub Release model-spatial-es-v3 ---"
python3 scripts/upload_release_asset.py model-spatial-es-v3 "$OUT_MODEL" "Direct Legal CRN Antithetic ES SpatialValueModel v3"

echo ""
echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs es_v3 ---"
target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"

echo "=== Colab Run #18-CPU COMPLETE ==="
