#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #6b: Train FiLM+SE 2D Conv ResNet (C=96, B=6, 25 ep)"
echo "                Zero-Sync GPU Unpacking + Vectorized Batched CE"
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

echo ""
echo "--- Training BlockfishResNet (C=96, B=6, 25 epochs, batch=64) ---"
python3 scripts/train_resnet.py \
  --data "$DATA_GZ" \
  --channels 96 \
  --blocks 6 \
  --epochs 25 \
  --batch-size 64 \
  --lr 2e-3 \
  --wd 1e-4 \
  --val-weight 0.02 \
  --out /tmp/resnet_v1_c96b6.pt

ls -lh /tmp/resnet_v1_c96b6.pt

echo ""
echo "--- Uploading trained checkpoint to GitHub Release (tag: model-resnet-v1) ---"
python3 scripts/upload_release_asset.py model-resnet-v1 /tmp/resnet_v1_c96b6.pt "Blockfish 2D Conv ResNet v1"

echo "=== Colab Run #6b COMPLETE ==="
