#!/usr/bin/env python3
"""Download artifacts from one or more BC-4 Mass Legal GitHub Actions runs and produce
a cumulative deduplicated summary report.

Usage:
  python3 scripts/bc4_mass_legal/collect_runs.py \
    --repo jush0147/blockfish \
    --run-ids 45922368524,45923298337 \
    --work-dir ./bc4_collected \
    --out-dir ./bc4_cumulative_report
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & aggregate multiple BC-4 GitHub Actions runs")
    parser.add_argument("--repo", default="jush0147/blockfish", help="GitHub owner/repo")
    parser.add_argument("--run-ids", required=True, help="Comma-separated GitHub Actions run IDs")
    parser.add_argument("--work-dir", required=True, help="Directory to cache downloaded artifacts")
    parser.add_argument("--out-dir", required=True, help="Directory to write cumulative summary files")
    parser.add_argument("--seed-range", help="Optional expected seed range start:end for missing-seed check")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    input_dirs = []
    for rid in run_ids:
        rdir = work_dir / f"run_{rid}"
        if not rdir.exists() or not any(rdir.iterdir()):
            rdir.mkdir(parents=True, exist_ok=True)
            print(f"Downloading artifacts for run {rid} into {rdir}...", file=sys.stderr)
            subprocess.run(
                ["gh", "run", "download", rid, "-R", args.repo, "-D", str(rdir)],
                check=True,
            )
        else:
            print(f"Using cached artifacts for run {rid} in {rdir}", file=sys.stderr)
        # Only feed shard artifacts to aggregate.py (not -summary merged files) to preserve exact per-shard counts
        for sub in sorted(rdir.iterdir()):
            if sub.is_dir() and "-shard-" in sub.name:
                input_dirs.append(str(sub))

    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / "aggregate.py"),
        "--out-json",
        str(out_dir / "cumulative_summary.json"),
        "--out-md",
        str(out_dir / "cumulative_summary.md"),
        "--out-merged-jsonl",
        str(out_dir / "cumulative_results.jsonl"),
    ]
    for d in input_dirs:
        cmd.extend(["--input-dir", d])
    if args.seed_range:
        cmd.extend(["--seed-range", args.seed_range])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
