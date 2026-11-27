#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #25-CPU: 5-Ply Beam CRN-GRPO + Full V6 2D 3x3 Patterns"
echo "                    3,840 Legal 5-ply games (20 iters x 24 seeds x G=8)"
echo "                    Unlocking all 1,591 V6 sparse features + fixed step_pieces"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (5 seeds)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building train_grpo and bench_legal (release) ---"
cargo build --release --example train_grpo --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_grpo_v2_5ply_v6.txt"
OUT_LAST="/tmp/spatial_grpo_v2_5ply_v6_last.txt"

echo ""
echo "--- [1/3] Training V6-1591 SpatialValueModel with 5-Ply Beam CRN-GRPO ---"
target/release/examples/train_grpo \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --iters 20 \
  --seeds-per-iter 24 \
  --group-size 8 \
  --val-games 60 \
  --beam 16 \
  --sparse-lim 1591 \
  --tau 0.15 \
  --top-m 10 \
  --kl-beta 0.015 \
  --lr-lin 3e-4 \
  --lr-mlp 1.5e-4 \
  --anchor-wd 2e-4 \
  --batch-steps 512

echo ""
echo "--- [2/3] Uploading spatial_grpo_v2_5ply_v6 checkpoints to Release model-spatial-grpo-v2 ---"
python3 scripts/upload_release_asset.py model-spatial-grpo-v2 "$OUT_MODEL" "5-Ply Beam CRN-GRPO Full-V6 SpatialValueModel v2"
python3 scripts/upload_release_asset.py model-spatial-grpo-v2 "$OUT_LAST" "5-Ply Beam CRN-GRPO Full-V6 SpatialValueModel v2"

echo ""
echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs grpo_v2 ---"
target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"

echo "=== Colab Run #25-CPU COMPLETE ==="
