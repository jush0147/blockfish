#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #12b: Phase 4 - Pure 100% Legal p=5 TD(lambda) RL"
echo "                 Fixed bag_mask + beam=16 (~10 min total!)"
echo "================================================================"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "[SKIP] No GPU detected (still on CPU runtime). Switch Colab to L4/T4 GPU and click Run all!"
  exit 0
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "--- Building export_legal_rl & eval_resnet (release) ---"
cargo build --release --example export_legal_rl --example eval_resnet --features block-stacker

RL_BIN="/tmp/legal_rl_v1_400g.bin"
RL_GZ="${RL_BIN}.gz"

echo ""
echo "--- Generating 360 train + 40 val eps-explored Legal p=5 games (beam=16, eps=10%) ---"
target/release/examples/export_legal_rl \
  --model models/spatial_bc4.txt \
  --train-games 360 \
  --val-games 40 \
  --train-seed-start 5001 \
  --val-seed-start 301 \
  --beam 16 \
  --eps 10 \
  --out "$RL_BIN"

gzip -c -6 "$RL_BIN" > "$RL_GZ"
ls -lh "$RL_BIN" "$RL_GZ"
python3 scripts/upload_release_asset.py legal-rl-v1 "$RL_GZ" "Blockfish Pure Legal p=5 RL Dataset v1 (400g)"
rm -f "$RL_GZ"

CKPT="/tmp/hybrid_td_v1.pt"
echo ""
echo "--- Training Pure Legal TD(0.85) Hybrid ResNet (20 epochs, ZERO WG_ds!) ---"
python3 scripts/train_legal_td.py \
  --data "$RL_BIN" \
  --lam 0.85 \
  --channels 64 \
  --blocks 4 \
  --epochs 20 \
  --batch-size 1024 \
  --max-delta 8.0 \
  --lr 1.5e-3 \
  --wd 5e-4 \
  --reg-weight 0.01 \
  --out "$CKPT"

ls -lh "$CKPT"
python3 scripts/upload_release_asset.py model-hybrid-td-v1 "$CKPT" "Blockfish Pure Legal TD(0.85) Hybrid ResNet v1"

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

echo "=== Colab Run #12 COMPLETE ==="
