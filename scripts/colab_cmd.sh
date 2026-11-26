#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #10b: Full-Strength Parallel Per-Root GPU root_pimc"
echo "                 FIX strict=False + consistent CRN world holes!"
echo "                 BASE (spatial_bc4) vs HYBRID (hybrid_resnet_v1)"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CKPT="/tmp/hybrid_resnet_v1.pt"
if [ ! -f "$CKPT" ]; then
  echo ""
  echo "--- Downloading hybrid_resnet_v1.pt from GitHub Release model-hybrid-v1 ---"
  python3 scripts/download_release_asset.py model-hybrid-v1 hybrid_resnet_v1.pt "$CKPT"
fi

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
echo ">>> [1/3] BASE (spatial_bc4, delta_w=0.0) pimc_v2 (5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.0

echo ""
echo ">>> [2/3] HYBRID (delta_w=0.5) pimc_v2 (5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.5

echo ""
echo ">>> [3/3] HYBRID (delta_w=1.0) pimc_v2 (5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 1.0

echo "=== Colab Run #10 COMPLETE ==="
