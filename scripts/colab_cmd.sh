#!/usr/bin/env bash
set -euo pipefail

echo "================================================================"
echo " Colab Run #22-CPU: TRUE Diverse FULL-MLP 50/50 Ensemble Showdown"
echo "                    (First real test after eval_core_ex fix!)"
echo "                    bc4 vs ens(bc4+bc6 V6) vs ens(bc4+v5q1 SoftQ)"
echo " Cores: $(nproc) | RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo "================================================================"

echo "--- Running ensemble2 unit test & building bench_legal (release) ---"
cargo test --release -p blockfish --features block-stacker --lib test_eval_core_ex_ensemble2_uses_both_models
cargo build --release --example bench_legal --features block-stacker

echo ""
echo "--- Running true_ens_showdown (5 seeds 101..=105) ---"
target/release/examples/bench_legal true_ens_showdown 5

echo "=== Colab Run #22-CPU COMPLETE ==="
