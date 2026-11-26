#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #13-GPU: Phase 4B Strong root_pimc Legal TD(0.95) RL"
echo "                    Select best by unbiased val_mc_rmse +"
echo "                    STRICT SAME-BINARY A/B (BASE vs HYBRID)!"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RL_GZ="/tmp/legal_rl_v2_pimc72g.bin.gz"
if [ ! -f "$RL_GZ" ]; then
  echo ""
  echo "--- Downloading legal_rl_v2_pimc72g.bin.gz from GitHub Release legal-rl-v2 ---"
  python3 scripts/download_release_asset.py legal-rl-v2 legal_rl_v2_pimc72g.bin.gz "$RL_GZ"
fi

CKPT="/tmp/hybrid_td_v2.pt"
echo ""
echo "--- Training Strong-Policy Pure Legal TD(0.95) Hybrid ResNet (25 ep, best by val_mc_rmse) ---"
python3 scripts/train_legal_td.py \
  --data "$RL_GZ" \
  --lam 0.95 \
  --channels 64 \
  --blocks 4 \
  --epochs 25 \
  --batch-size 512 \
  --max-delta 8.0 \
  --lr 1e-3 \
  --wd 5e-4 \
  --reg-weight 0.01 \
  --out "$CKPT"

ls -lh "$CKPT"
python3 scripts/upload_release_asset.py model-hybrid-td-v2 "$CKPT" "Blockfish Strong-Policy Pure Legal TD(0.95) Hybrid ResNet v2"

echo ""
echo "--- Building eval_resnet (release, unified held+5) ---"
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
echo ">>> [1/3] SAME-BINARY BASE (spatial_bc4, held+5, delta_w=0.0) pimc_v2 (5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.0

echo ""
echo ">>> [2/3] STRONG-LEGAL TD-v2 HYBRID (delta_w=0.5) pimc_v2 (5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 0.5

echo ""
echo ">>> [3/3] STRONG-LEGAL TD-v2 HYBRID (delta_w=1.0) pimc_v2 (5 seeds 101..=105):"
target/release/examples/eval_resnet --mode pimc_v2 --sock "$SOCK" --seed-start 101 --n-seeds 5 \
  --worlds 12 --horizon 8 --beam 20 --top-roots 10 --leaf-top 4 --cvar 25 \
  --hybrid-model models/spatial_bc4.txt --delta-w 1.0

echo "=== Colab Run #13-GPU COMPLETE ==="
