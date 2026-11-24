#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: WIDE Cell-Conserving DS-Bucketed PIMC ds=5 b=80 w=12 cvar=25 + bc4 REAL FULL MLP leaf (5 seeds, ~8m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- deep_ds_pimc models/spatial_bc4.txt wide_ds5 5

echo ""
echo "=== Part 2: WIDE Cell-Conserving DS-Bucketed PIMC ds=5 b=120 w=12 cvar=25 + bc4 REAL FULL MLP leaf (5 seeds, ~12m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- deep_ds_pimc models/spatial_bc4.txt wide_ds5_b120 5

echo ""
echo "=== Part 3: WIDE Cell-Conserving DS-Bucketed PIMC ds=8 b=100 w=10 cvar=25 + bc4 REAL FULL MLP leaf (5 seeds, ~15m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- deep_ds_pimc models/spatial_bc4.txt wide_ds8_b100 5
