#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #5: Export 400 WG_ds (B=1200) games (BFDS0001)"
echo "               and upload gzipped dataset to GitHub Release"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building export_dataset (release) ---"
cargo build --release --example export_dataset --features block-stacker

OUT_BIN="/tmp/wg_ds_v1_400g.bin"
OUT_GZ="${OUT_BIN}.gz"

echo ""
echo "--- Generating 360 train (1001..1361) + 40 val (201..241) games ---"
target/release/examples/export_dataset \
  --train-games 360 \
  --val-games 40 \
  --train-seed-start 1001 \
  --val-seed-start 201 \
  --wg-beam 1200 \
  --out "$OUT_BIN"

echo ""
echo "--- Compressing dataset ---"
gzip -f -6 "$OUT_BIN"
ls -lh "$OUT_GZ"

echo ""
echo "--- Uploading to GitHub Release (tag: dataset-v1) ---"
python3 scripts/upload_release_asset.py dataset-v1 "$OUT_GZ" "Blockfish WG_ds Dataset v1"

rm -f "$OUT_GZ"
echo "=== Colab Run #5 COMPLETE ==="
