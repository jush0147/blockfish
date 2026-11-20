#!/usr/bin/env bash
set -eu
mkdir -p research/bin
for variant in control candidate; do
    features=path-probe
    if [[ $variant == candidate ]]; then features+=,terminal-priority; fi
    cargo build --release --locked -p blockfish --no-default-features --bin blockfish-path-probe --features "$features"
    cp target/release/blockfish-path-probe "research/bin/s37-path-$variant"
done
python3 research/build.py robust-terminal-priority
