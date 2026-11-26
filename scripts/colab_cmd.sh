#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #13-CPU: Phase 4B - Strong root_pimc Legal RL (72g)"
echo "                    64 train + 8 val (w=6, h=7, b=16, eps=5%)"
echo "                    ~22-25 min on cheap 8-core CPU (~0.10 CU)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building export_legal_rl (release) ---"
cargo build --release --example export_legal_rl --features block-stacker

RL_BIN="/tmp/legal_rl_v2_pimc72g.bin"
RL_GZ="${RL_BIN}.gz"

echo ""
echo "--- Generating 64 train (6001..6065) + 8 val (301..309) strong root_pimc Legal RL games ---"
target/release/examples/export_legal_rl \
  --model models/spatial_bc4.txt \
  --train-games 64 \
  --val-games 8 \
  --train-seed-start 6001 \
  --val-seed-start 301 \
  --pimc-worlds 6 \
  --pimc-horizon 7 \
  --pimc-width 16 \
  --eps 5 \
  --out "$RL_BIN"

echo ""
echo "--- Compressing & Uploading to GitHub Release (tag: legal-rl-v2) ---"
gzip -f -6 "$RL_BIN"
ls -lh "$RL_GZ"
python3 scripts/upload_release_asset.py legal-rl-v2 "$RL_GZ" "Blockfish Strong root_pimc Legal RL Dataset v2 (72g)"

rm -f "$RL_GZ"
echo "=== Colab Run #13-CPU COMPLETE ==="
