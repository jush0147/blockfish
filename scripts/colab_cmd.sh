#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #12-CPU: Export 200 Pure Legal p=5 RL Games (BFRL0001)"
echo "                    180 train + 20 val (beam=16, eps=10%)"
echo "                    Fixed bag_mask + held+5 avail_mask"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building export_legal_rl (release) ---"
cargo build --release --example export_legal_rl --features block-stacker

RL_BIN="/tmp/legal_rl_v1_200g.bin"
RL_GZ="${RL_BIN}.gz"

echo ""
echo "--- Generating 180 train (5001..5181) + 20 val (301..321) pure-Legal RL games ---"
target/release/examples/export_legal_rl \
  --model models/spatial_bc4.txt \
  --train-games 180 \
  --val-games 20 \
  --train-seed-start 5001 \
  --val-seed-start 301 \
  --beam 16 \
  --eps 10 \
  --out "$RL_BIN"

echo ""
echo "--- Compressing & Uploading to GitHub Release (tag: legal-rl-v1) ---"
gzip -f -6 "$RL_BIN"
ls -lh "$RL_GZ"
python3 scripts/upload_release_asset.py legal-rl-v1 "$RL_GZ" "Blockfish Pure Legal p=5 RL Dataset v1 (200g)"

rm -f "$RL_GZ"
echo "=== Colab Run #12-CPU COMPLETE ==="
