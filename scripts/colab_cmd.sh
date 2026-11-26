#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #7: Unified-Head 2D Conv ResNet v2 (C=64, B=4, 20 ep)"
echo "               Single V(s') = 1.65*rem_g + delta_v (temp=0.4)"
echo "               + OR_p20 Ceiling + Legal p=5 root_pimc (10 seeds)"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4/A100 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| BF16:', torch.cuda.is_bf16_supported())"

DATA_GZ="/tmp/wg_ds_v1_400g.bin.gz"
if [ ! -f "$DATA_GZ" ]; then
  echo ""
  echo "--- Downloading wg_ds_v1_400g.bin.gz from GitHub Release dataset-v1 ---"
  python3 scripts/download_release_asset.py dataset-v1 wg_ds_v1_400g.bin.gz "$DATA_GZ"
fi

CKPT="/tmp/resnet_v2_unified.pt"
echo ""
echo "--- Training Unified BlockfishResNet v2 (C=64, B=4, 20 ep, bs=128, max-neg=15) ---"
python3 scripts/train_resnet.py \
  --data "$DATA_GZ" \
  --channels 64 \
  --blocks 4 \
  --epochs 20 \
  --batch-size 128 \
  --max-neg 15 \
  --unified \
  --temp 0.4 \
  --lr 3e-3 \
  --wd 1e-4 \
  --val-weight 0.15 \
  --out "$CKPT"

ls -lh "$CKPT"

echo ""
echo "--- Uploading trained checkpoint to GitHub Release (tag: model-resnet-v2) ---"
python3 scripts/upload_release_asset.py model-resnet-v2 "$CKPT" "Blockfish 2D Conv ResNet v2 (Unified)"

echo ""
echo "--- Building eval_resnet (release) ---"
cargo build --release --example eval_resnet --features block-stacker

SOCK="/tmp/blockfish_resnet.sock"
rm -f "$SOCK"
echo ""
echo "--- Starting GPU ResNet eval server ---"
python3 scripts/resnet_eval_server.py --ckpt "$CKPT" --sock "$SOCK" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if [ -S "$SOCK" ]; then break; fi
  sleep 0.5
done

echo ""
echo "--- Benchmark 1: Legal p=5 1-ply Greedy (10 seeds 101..=110) ---"
target/release/examples/eval_resnet --mode greedy --sock "$SOCK" --seed-start 101 --n-seeds 10

echo ""
echo "--- Benchmark 2: Oracle OR_p20 Ceiling (h=8, b=24, mpr=6, 5 seeds 101..=105) ---"
target/release/examples/eval_resnet --mode or_p20 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --horizon 8 --beam 24 --max-per-root 6 --val-w 1.0 --logit-w 0.0

echo ""
echo "--- Benchmark 3: Legal p=5 Lockstep GPU root_pimc (w=16, h=6, b=16, cvar=25, 10 seeds 101..=110) ---"
target/release/examples/eval_resnet --mode pimc --sock "$SOCK" --seed-start 101 --n-seeds 10 \
  --worlds 16 --horizon 6 --beam 16 --max-per-root 4 --cvar 25 --val-w 1.0 --logit-w 0.0

echo "=== Colab Run #7 COMPLETE ==="
