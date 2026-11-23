#!/usr/bin/env bash
set -euo pipefail

echo "=== Held-Out 5-Seed Legal Verification of models/spatial_es3.txt (cvar_top2: ~25m) ==="
cargo run --release -p blockfish --features race --example bench_legal -- wg5_lin_sweep models/spatial_es3.txt cvar_top2 5
