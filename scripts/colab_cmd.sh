#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #27-CPU: FIXED True Same-State Sibling-Action CRN Q-Advantage PI"
echo "                    (EMPTY_ROW + 1-to-1 CRN World Alignment Verified)"
echo "                    6 PI rounds x 32 games x 5 states = 960 exact states"
echo "                    K=8 siblings x W=8 CRN Legal p=5 worlds (dG=14, roll_beam=12)"
echo "                    + Strict Same-Binary fullmlp_w16 A/B (only if NEW BEST achieved)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- [0/3] Running train_sibling_q physics & alignment unit tests ---"
cargo test --release --example train_sibling_q --features block-stacker

echo "--- Building train_sibling_q and bench_legal (release) ---"
cargo build --release --example train_sibling_q --example bench_legal --features block-stacker

OUT_MODEL="/tmp/spatial_sibq_v2.txt"
OUT_LAST="/tmp/spatial_sibq_v2_last.txt"
TR_LOG="/tmp/sibq_v2_train.log"

echo ""
echo "--- [1/3] Training SpatialValueModel with FIXED Same-State Sibling CRN Q-Advantage PI ---"
target/release/examples/train_sibling_q \
  --init models/spatial_bc4.txt \
  --out "$OUT_MODEL" \
  --rounds 6 \
  --games-per-round 32 \
  --states-per-game 5 \
  --k-sibs 8 \
  --worlds 8 \
  --target-ds 14 \
  --outer-beam 16 \
  --roll-beam 12 \
  --val-games 60 \
  --sparse-lim 1591 \
  --epochs 6 \
  --batch-size 24 \
  --tau-tgt 0.60 \
  --tau-pred 0.60 \
  --w-ce 1.0 \
  --w-adv 0.6 \
  --lr-lin 3e-4 \
  --lr-mlp 1.5e-4 \
  --anchor-wd 2e-4 | tee "$TR_LOG"

echo ""
echo "--- [2/3] Uploading spatial_sibq_v2 checkpoints to Release model-spatial-sibq-v2 ---"
python3 scripts/upload_release_asset.py model-spatial-sibq-v2 "$OUT_MODEL" "FIXED True Same-State Sibling CRN Q-Advantage PI v2"
python3 scripts/upload_release_asset.py model-spatial-sibq-v2 "$OUT_LAST" "FIXED True Same-State Sibling CRN Q-Advantage PI v2 (last)"

if grep -q "NEW BEST" "$TR_LOG"; then
  echo ""
  echo "--- [3/3] STRICT SAME-BINARY fullmlp_w16 A/B (5 seeds 101..=105): bc4 vs sibq_v2 ---"
  target/release/examples/bench_legal compare_spatial 5 models/spatial_bc4.txt "$OUT_MODEL"
else
  echo ""
  echo "--- [3/3] SKIPPED fullmlp_w16 A/B: no NEW BEST over base_val ---"
fi

echo "=== Colab Run #27-CPU COMPLETE ==="
