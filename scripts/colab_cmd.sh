#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #14-GPU: Full-MLP Internal pimc_v2 (w=12, h=8, b=20, k=8)"
echo "                    STRICT SAME-BINARY A/B: BASE (0.0) vs TD-v2 (0.35)"
echo "                    No retraining! ~12-14 min total on GPU"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CKPT="/tmp/hybrid_td_v2.pt"
if [ ! -f "$CKPT" ]; then
  echo ""
  echo "--- Downloading hybrid_td_v2.pt from GitHub Release model-hybrid-td-v2 ---"
  python3 scripts/download_release_asset.py model-hybrid-td-v2 hybrid_td_v2.pt "$CKPT"
fi

echo ""
echo "--- Building eval_resnet (release, full-mlp-internal + unified held+5) ---"
cargo build --release --example eval_resnet --features block-stacker

SOCK="/tmp/blockfish_resnet.sock"
rm -f "$SOCK"
echo ""
echo "--- Starting GPU Strong-Legal TD Hybrid ResNet v2 eval server ---"
python3 scripts/resnet_eval_server.py --ckpt "$CKPT" --sock "$SOCK" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if [ -S "$SOCK" ]; then break; fi
  sleep 0.5
done

echo ""
echo ">>> [1/2] SAME-BINARY FULL-MLP BASE (spatial_bc4, delta_w=0.0, 5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 8 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.0 --full-mlp-internal

echo ""
echo ">>> [2/2] FULL-MLP + TD-v2 HYBRID (delta_w=0.35, 5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 8 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.35 --full-mlp-internal

echo "=== Colab Run #14-GPU COMPLETE ==="
