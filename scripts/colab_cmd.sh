#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #29-CPU: Strong-Teacher Legal p=5 Expert Iteration (fullmlp_w16)"
echo "                    16 fullmlp_w16 games (~2,900 exact sibling Q-dists)"
echo "                    Unlocking all 1,591 V6 2D 3x3 patterns (ZERO clairvoyance)"
echo "                    Pure Relative Q-Adv + Listwise CE (ZERO scalar regression)"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (only if NEW BEST)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- [0/3] Running train_exit_w16 unit test ---"
cargo test --release --example train_exit_w16 --features block-stacker

echo "--- Building train_exit_w16 and bench_legal (release) ---"
cargo build --release --example train_exit_w16 --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_exit_w16_v1.txt"
OUT_LAST="/tmp/spatial_exit_w16_v1_last.txt"
TR_LOG="/tmp/exit_w16_v1_train.log"

echo ""
echo "--- [1/3] Collecting 16 fullmlp_w16 Legal p=5 games & training V6-1591 model ---"
target/release/examples/train_exit_w16 \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --games 16 \
  --teacher-worlds 16 \
  --teacher-width 28 \
  --spawn-worlds 4 \
  --val-games 60 \
  --val-beam 16 \
  --sparse-lim 1591 \
  --epochs 12 \
  --batch-size 32 \
  --tau-tgt 0.25 \
  --tau-pred 0.25 \
  --w-ce 1.0 \
  --w-adv 0.25 \
  --lr-lin 3e-4 \
  --lr-mlp 1.5e-4 \
  --anchor-wd 2e-4 | tee "$TR_LOG"

echo ""
echo "--- [2/3] Uploading spatial_exit_w16_v1 checkpoints to Release model-spatial-exit-w16-v1 ---"
python3 scripts/upload_release_asset.py model-spatial-exit-w16-v1 "$OUT_MODEL" "Strong-Teacher fullmlp_w16 Legal p=5 Expert Iteration V6-1591 v1"
python3 scripts/upload_release_asset.py model-spatial-exit-w16-v1 "$OUT_LAST" "Strong-Teacher fullmlp_w16 Legal p=5 Expert Iteration V6-1591 v1 (last)"

if grep -q "NEW BEST" "$TR_LOG"; then
  echo ""
  echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs exit_w16_v1 ---"
  target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"
else
  echo ""
  echo "--- [3/3] SKIPPED fullmlp_w16 A/B: no NEW BEST over base_val ---"
fi

echo "=== Colab Run #29-CPU COMPLETE ==="
