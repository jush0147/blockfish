#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #16-CPU: Pure Rust Legal p=5 Iterative Watkins TD(λ)"
echo "                    2,500 Legal games (5 iters x 500g) on spatial_bc4"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (5 seeds)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building train_spatial_td and bench_legal (release) ---"
cargo build --release --example train_spatial_td --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_td_v1.txt"
OUT_LAST="/tmp/spatial_td_v1_last.txt"

echo ""
echo "--- [1/3] Training SpatialValueModel with 2,500 Legal p=5 TD(0.92) games ---"
target/release/examples/train_spatial_td \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --iters 5 \
  --games-per-iter 500 \
  --val-games 60 \
  --beam 24 \
  --eps 3 \
  --lambda 0.92 \
  --epochs 3 \
  --batch 256 \
  --lr-lin 3e-4 \
  --lr-mlp 1.5e-4 \
  --anchor-wd 1e-3 \
  --huber-delta 4.0

echo ""
echo "--- [2/3] Uploading spatial_td_v1 checkpoints to GitHub Release model-spatial-td-v1 ---"
python3 scripts/upload_release_asset.py model-spatial-td-v1 "$OUT_MODEL" "Pure Rust Legal p=5 Watkins TD(0.92) SpatialValueModel v1"
python3 scripts/upload_release_asset.py model-spatial-td-v1 "$OUT_LAST" "Pure Rust Legal p=5 Watkins TD(0.92) SpatialValueModel v1"

echo ""
echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs td_v1 ---"
target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"

echo "=== Colab Run #16-CPU COMPLETE ==="
