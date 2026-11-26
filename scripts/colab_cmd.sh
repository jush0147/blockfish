#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #11: Joint Hybrid Residual ResNet v2"
echo "                400 WG_ds (168 pcs) + 56 Legal Self-Play = 456g"
echo "                77,649 decisions / 3.6M board states + pimc_v2!"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

WG_GZ="/tmp/wg_ds_v1_400g.bin.gz"
SP_GZ="/tmp/legal_sp_v1_56g.bin.gz"
WG_BIN="/tmp/wg_ds_v1_400g.bin"
WG_V2="/tmp/wg_ds_v1_400g_bv.bin"

if [ ! -f "$WG_GZ" ]; then
  python3 scripts/download_release_asset.py dataset-v1 wg_ds_v1_400g.bin.gz "$WG_GZ"
fi
if [ ! -f "$SP_GZ" ]; then
  python3 scripts/download_release_asset.py legal-selfplay-v1 legal_sp_v1_56g.bin.gz "$SP_GZ"
fi

echo ""
echo "--- Building annotate_base_v & eval_resnet (release) ---"
cargo build --release --example annotate_base_v --example eval_resnet --features block-stacker

if [ ! -f "$WG_V2" ]; then
  echo ""
  echo "--- Annotating 400 WG_ds games with spatial_bc4.txt base_v ---"
  gzip -dc "$WG_GZ" > "$WG_BIN"
  target/release/examples/annotate_base_v models/spatial_bc4.txt "$WG_BIN" "$WG_V2"
  rm -f "$WG_BIN"
fi

CKPT="/tmp/hybrid_resnet_v2.pt"
echo ""
echo "--- Training Hybrid Residual ResNet v2 (456 games, 12 epochs, bs=128) ---"
python3 scripts/train_hybrid_resnet.py \
  --data "$WG_V2" "$SP_GZ" \
  --channels 64 \
  --blocks 4 \
  --epochs 12 \
  --batch-size 128 \
  --max-neg 15 \
  --max-delta 8.0 \
  --temp 0.5 \
  --lr 2e-3 \
  --wd 2e-4 \
  --val-weight 0.05 \
  --reg-weight 0.005 \
  --out "$CKPT"

ls -lh "$CKPT"

echo ""
echo "--- Uploading hybrid_resnet_v2.pt to GitHub Release (tag: model-hybrid-v2) ---"
python3 scripts/upload_release_asset.py model-hybrid-v2 "$CKPT" "Blockfish Hybrid Residual ResNet v2 (456g WG+Legal)"

SOCK="/tmp/blockfish_resnet.sock"
rm -f "$SOCK"
echo ""
echo "--- Starting GPU Hybrid ResNet v2 eval server ---"
python3 scripts/resnet_eval_server.py --ckpt "$CKPT" --sock "$SOCK" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if [ -S "$SOCK" ]; then break; fi
  sleep 0.5
done

echo ""
echo ">>> [1/2] HYBRID v2 (delta_w=0.5) pimc_v2 (5 seeds 101..=105, vs BASE 193.20):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.5

echo ""
echo ">>> [2/2] HYBRID v2 (delta_w=1.0) pimc_v2 (5 seeds 101..=105, vs BASE 193.20):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 1.0

echo "=== Colab Run #11 COMPLETE ==="
