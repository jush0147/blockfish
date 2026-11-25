#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #6e: Fast T4/L4 ResNet (C=64, B=4, 15 ep, max-neg=11)"
echo "                4x Negative Sampling + 2-min Live Log Sync + Eval"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to T4/L4/A100 GPU and click Run all!"
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

CKPT="/tmp/resnet_v1_c64b4.pt"
echo ""
echo "--- Training BlockfishResNet (C=64, B=4, 15 epochs, batch=128, max-neg=11) ---"
python3 scripts/train_resnet.py \
  --data "$DATA_GZ" \
  --channels 64 \
  --blocks 4 \
  --epochs 15 \
  --batch-size 128 \
  --max-neg 11 \
  --lr 3e-3 \
  --wd 1e-4 \
  --val-weight 0.02 \
  --out "$CKPT"

ls -lh "$CKPT"

echo ""
echo "--- Uploading trained checkpoint to GitHub Release (tag: model-resnet-v1) ---"
python3 scripts/upload_release_asset.py model-resnet-v1 "$CKPT" "Blockfish 2D Conv ResNet v1"

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
echo "--- Benchmark 2: Legal p=5 Lockstep GPU root_pimc (val_w=1.0, logit_w=0.5, 5 seeds 101..=105) ---"
target/release/examples/eval_resnet --mode pimc --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 16 --horizon 6 --beam 16 --max-per-root 4 --cvar 25 --val-w 1.0 --logit-w 0.5

echo ""
echo "--- Benchmark 3: Legal p=5 Lockstep GPU root_pimc (val_w=1.0, logit_w=0.0, 5 seeds 101..=105) ---"
target/release/examples/eval_resnet --mode pimc --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 16 --horizon 6 --beam 16 --max-per-root 4 --cvar 25 --val-w 1.0 --logit-w 0.0

echo "=== Colab Run #6c COMPLETE ==="
