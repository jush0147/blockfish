#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #24-CPU: Pure Legal p=5 2-Ply CRN-GRPO Policy Improvement"
echo "                    5,120 Legal 100L games (20 iters x 32 seeds x G=8)"
echo "                    Step-aligned CRN advantage + sibling softmax + KL"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (5 seeds)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building train_grpo and bench_legal (release) ---"
cargo build --release --example train_grpo --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_grpo_v1.txt"
OUT_LAST="/tmp/spatial_grpo_v1_last.txt"

echo ""
echo "--- [1/3] Training SpatialValueModel with 5,120 Legal p=5 CRN-GRPO games ---"
target/release/examples/train_grpo \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --iters 20 \
  --seeds-per-iter 32 \
  --group-size 8 \
  --val-games 60 \
  --tau 0.15 \
  --top-m 10 \
  --kl-beta 0.05 \
  --lr-lin 3e-4 \
  --lr-mlp 1.5e-4 \
  --anchor-wd 5e-4 \
  --batch-steps 512

echo ""
echo "--- [2/3] Uploading spatial_grpo_v1 checkpoints to Release model-spatial-grpo-v1 ---"
python3 scripts/upload_release_asset.py model-spatial-grpo-v1 "$OUT_MODEL" "Pure Legal p=5 2-Ply CRN-GRPO SpatialValueModel v1"
python3 scripts/upload_release_asset.py model-spatial-grpo-v1 "$OUT_LAST" "Pure Legal p=5 2-Ply CRN-GRPO SpatialValueModel v1"

echo ""
echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs grpo_v1 ---"
target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"

echo "=== Colab Run #24-CPU COMPLETE ==="
