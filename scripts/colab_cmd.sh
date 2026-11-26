#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #9b: Hybrid Residual 2D Conv ResNet (T4 FP16 20x Fix)"
echo "                V_hybrid = spatial_bc4 + ResNet_delta + v_bias"
echo "                ~5-6 min total on T4/L4 GPU -> Release model-hybrid-v1"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| Cap:', torch.cuda.get_device_capability(0))"

DATA_GZ="/tmp/legal_sp_v1_56g.bin.gz"
if [ ! -f "$DATA_GZ" ]; then
  echo ""
  echo "--- Downloading legal_sp_v1_56g.bin.gz from GitHub Release legal-selfplay-v1 ---"
  python3 scripts/download_release_asset.py legal-selfplay-v1 legal_sp_v1_56g.bin.gz "$DATA_GZ"
fi

CKPT="/tmp/hybrid_resnet_v1.pt"
echo ""
echo "--- Training BlockfishHybridResNet (C=64, B=4, 10 ep, bs=64, max-neg=15, v_bias) ---"
python3 scripts/train_hybrid_resnet.py \
  --data "$DATA_GZ" \
  --channels 64 \
  --blocks 4 \
  --epochs 10 \
  --batch-size 64 \
  --max-neg 15 \
  --max-delta 6.0 \
  --temp 0.5 \
  --lr 1e-3 \
  --wd 5e-4 \
  --val-weight 0.05 \
  --reg-weight 0.01 \
  --out "$CKPT"

ls -lh "$CKPT"

echo ""
echo "--- Uploading trained checkpoint to GitHub Release (tag: model-hybrid-v1) ---"
python3 scripts/upload_release_asset.py model-hybrid-v1 "$CKPT" "Blockfish Hybrid Residual ResNet v1"

echo ""
echo "--- Building eval_resnet (release) ---"
cargo build --release --example eval_resnet --features block-stacker

SOCK="/tmp/blockfish_resnet.sock"
rm -f "$SOCK"
echo ""
echo "--- Starting GPU Hybrid ResNet eval server ---"
python3 scripts/resnet_eval_server.py --ckpt "$CKPT" --sock "$SOCK" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if [ -S "$SOCK" ]; then break; fi
  sleep 0.5
done

echo ""
echo "--- A/B 1: Legal p=5 1-ply Greedy (10 seeds 101..=110) BASE (delta_w=0.0) vs HYBRID (delta_w=1.0) ---"
target/release/examples/eval_resnet --mode greedy --seed-start 101 --n-seeds 10 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.0
target/release/examples/eval_resnet --mode greedy --sock "$SOCK" --seed-start 101 --n-seeds 10 \
  --hybrid-model models/spatial_bc4.txt --delta-w 1.0

echo ""
echo "--- A/B 2: Legal p=5 Lockstep root_pimc (w=8, h=6, b=16, cvar=25, 5 seeds 101..=105) ---"
echo ">>> BASE (spatial_bc4, delta_w=0.0):"
target/release/examples/eval_resnet --mode pimc --seed-start 101 --n-seeds 5 \
  --worlds 8 --horizon 6 --beam 16 --max-per-root 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.0

echo ">>> HYBRID (delta_w=0.5):"
target/release/examples/eval_resnet --mode pimc --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 8 --horizon 6 --beam 16 --max-per-root 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.5

echo ">>> HYBRID (delta_w=1.0):"
target/release/examples/eval_resnet --mode pimc --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 8 --horizon 6 --beam 16 --max-per-root 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 1.0

echo "=== Colab Run #9 COMPLETE ==="
