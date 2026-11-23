#!/usr/bin/env bash
set -euo pipefail

echo "=== Cell-Conservation nds_pen + Non-Clairvoyant 7-Bag Expectimax Sweep on models/spatial_bc4_lin.txt (~40m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_bc4_lin.txt nds_expectimax 5
