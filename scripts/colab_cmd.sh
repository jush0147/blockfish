#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #23-CPU: Zero-Strategy-Fusion Nested Legal p=5 Rollout"
echo "                    (New Paradigm: 100% Legal p=5 in-world rollout,"
echo "                     ZERO future hole peeking inside rollouts!)"
echo "                    5 seeds 101..=105 (~30 min on 8-core CPU)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Building bench_nc_rollout (release) ---"
cargo build --release --example bench_nc_rollout --features block-stacker

echo ""
echo ">>> [1/2] Nested Legal p=5 Rollout (k=6, w=8, h=8, b=10, cvar=25):"
target/release/examples/bench_nc_rollout \
  --model models/spatial_bc4.txt --n-seeds 5 --seed-start 101 \
  --top-k 6 --worlds 8 --horizon 8 --rollout-beam 10 --cvar 25

echo ""
echo ">>> [2/2] Nested Legal p=5 Rollout (k=6, w=12, h=8, b=10, cvar=0 pure mean):"
target/release/examples/bench_nc_rollout \
  --model models/spatial_bc4.txt --n-seeds 5 --seed-start 101 \
  --top-k 6 --worlds 12 --horizon 8 --rollout-beam 10 --cvar 0

echo "=== Colab Run #23-CPU COMPLETE ==="
