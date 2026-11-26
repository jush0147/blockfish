#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #12-GPU: Pure 100% Legal p=5 TD(0.85) Hybrid ResNet"
echo "                    ZERO WG_ds! ~15s train + ~6m pimc_v2 eval"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RL_GZ="/tmp/legal_rl_v1_200g.bin.gz"
if [ ! -f "$RL_GZ" ]; then
  echo ""
  echo "--- Downloading legal_rl_v1_200g.bin.gz from GitHub Release legal-rl-v1 ---"
  python3 scripts/download_release_asset.py legal-rl-v1 legal_rl_v1_200g.bin.gz "$RL_GZ"
fi

CKPT="/tmp/hybrid_td_v1.pt"
echo ""
echo "--- Training Pure Legal TD(0.85) Hybrid ResNet (25 epochs, ZERO WG_ds!) ---"
python3 scripts/train_legal_td.py \
  --data "$RL_GZ" \
  --lam 0.85 \
  --channels 64 \
  --blocks 4 \
  --epochs 25 \
  --batch-size 1024 \
  --max-delta 8.0 \
  --lr 1.5e-3 \
  --wd 5e-4 \
  --reg-weight 0.01 \
  --out "$CKPT"

ls -lh "$CKPT"
python3 scripts/upload_release_asset.py model-hybrid-td-v1 "$CKPT" "Blockfish Pure Legal TD(0.85) Hybrid ResNet v1"

echo ""
echo "--- Building eval_resnet (release) ---"
cargo build --release --example eval_resnet --features block-stacker

SOCK="/tmp/blockfish_resnet.sock"
rm -f "$SOCK"
echo ""
echo "--- Starting GPU Pure-Legal TD Hybrid ResNet eval server ---"
python3 scripts/resnet_eval_server.py --ckpt "$CKPT" --sock "$SOCK" &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 30); do
  if [ -S "$SOCK" ]; then break; fi
  sleep 0.5
done

echo ""
echo ">>> [1/2] PURE-LEGAL TD HYBRID (delta_w=0.5) pimc_v2 (5 seeds 101..=105, vs BASE 193.20):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.5

echo ""
echo ">>> [2/2] PURE-LEGAL TD HYBRID (delta_w=1.0) pimc_v2 (5 seeds 101..=105, vs BASE 193.20):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 1.0

echo "=== Colab Run #12-GPU COMPLETE ==="
