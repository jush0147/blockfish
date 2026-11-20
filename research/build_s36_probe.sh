#!/usr/bin/env bash
# Run at the repository root with the workspace Rust environment active.
set -eu
mkdir -p research/bin
for model in classic learned; do
    for backup in single robust; do
        features=generated-budget,fast-value
        if [[ $model == learned ]]; then features+=,learned-value; fi
        if [[ $backup == robust ]]; then features+=,robust-backup; fi
        cargo build --release --locked -p blockfish --bin blockfish-search-probe --features "$features"
        cp target/release/blockfish-search-probe "research/bin/s36-$model-$backup"
    done
done
