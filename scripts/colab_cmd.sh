#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #26-CPU: True Same-State Sibling-Action CRN Q-Advantage PI"
echo "                    (Zero Strategy Fusion, Pure Relative Q-Adv + Listwise CE)"
echo "                    5 PI rounds x 28 games x 5 states = 700 exact states"
echo "                    K=8 siblings x W=8 CRN Legal p=5 worlds (dG=14 lines)"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (5 seeds)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building train_sibling_q and bench_legal (release) ---"
cargo build --release --example train_sibling_q --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_sibq_v1.txt"
OUT_LAST="/tmp/spatial_sibq_v1_last.txt"

echo ""
echo "--- [1/3] Training SpatialValueModel with True Same-State Sibling CRN Q-Advantage PI ---"
target/release/examples/train_sibling_q \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --rounds 5 \
  --games-per-round 28 \
  --states-per-game 5 \
  --k-sibs 8 \
  --worlds 8 \
  --target-ds 14 \
  --outer-beam 16 \
  --roll-beam 10 \
  --val-games 60 \
  --sparse-lim 1591 \
  --epochs 8 \
  --batch-size 24 \
  --tau-tgt 0.60 \
  --tau-pred 0.60 \
  --w-ce 1.0 \
  --w-adv 0.6 \
  --lr-lin 3e-4 \
  --lr-mlp 1.5e-4 \
  --anchor-wd 2e-4

echo ""
echo "--- [2/3] Uploading spatial_sibq_v1 checkpoints to Release model-spatial-sibq-v1 ---"
python3 scripts/upload_release_asset.py model-spatial-sibq-v1 "$OUT_MODEL" "True Same-State Sibling CRN Q-Advantage PI v1"
python3 scripts/upload_release_asset.py model-spatial-sibq-v1 "$OUT_LAST" "True Same-State Sibling CRN Q-Advantage PI v1 (last)"

echo ""
echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs sibq_v1 ---"
target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"

echo "=== Colab Run #26-CPU COMPLETE ==="
