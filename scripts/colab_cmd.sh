#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #8: Export 56 Legal p=5 Self-Play Games (BFDS0002)"
echo "               spatial_bc4 root_pimc (w=6, h=7, b=16, cvar=25)"
echo "               ~8-12 min on 8-12 cores -> GH Release legal-selfplay-v1"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building export_legal_selfplay (release) ---"
cargo build --release --example export_legal_selfplay --features block-stacker

OUT_BIN="/tmp/legal_sp_v1_56g.bin"
OUT_GZ="${OUT_BIN}.gz"

echo ""
echo "--- Generating 50 train (2001..2051) + 6 val (201..207) Legal p=5 games ---"
target/release/examples/export_legal_selfplay \
  --model models/spatial_bc4.txt \
  --train-games 50 \
  --val-games 6 \
  --train-seed-start 2001 \
  --val-seed-start 201 \
  --worlds 6 \
  --horizon 7 \
  --width 16 \
  --out "$OUT_BIN"

echo ""
echo "--- Compressing dataset ---"
gzip -f -6 "$OUT_BIN"
ls -lh "$OUT_GZ"

echo ""
echo "--- Uploading to GitHub Release (tag: legal-selfplay-v1) ---"
python3 scripts/upload_release_asset.py legal-selfplay-v1 "$OUT_GZ" "Blockfish Legal p=5 Self-Play Dataset v1"

rm -f "$OUT_GZ"
echo "=== Colab Run #8 COMPLETE ==="
