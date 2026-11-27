#!/usr/bin/env python3
"""Aggregate, validate, deduplicate, and summarize BC-4 Mass Legal 100L simulation results.

Supports combining multiple shard JSONL files across one or more batches,
strictly validating pinned model SHA-256 and search parameters, deduplicating by seed
(ensuring reruns never double-count), separating completed 100L clears from top-outs,
and computing full statistics (mean, min, max, median, P1/P5/P95/P99, top-10 seeds,
PB progression, and per-batch distributions).
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PINNED_MODEL_NAME = "models/spatial_bc4.txt"
PINNED_MODEL_SHA256 = "d99ae1f07e629e95e115f30fb3e46fecee20eb3d14c5b9e2a94dc52da1238412"
PINNED_BASELINE_COMMIT = "939cb506a2e5fb9e86f289b86a51162d05afbf2d"
REQUIRED_SEARCH = {
    "mode": "REAL_FULL_MLP",
    "legal_preview": 5,
    "root_pimc_worlds": 16,
    "root_pimc_horizon": 8,
    "root_pimc_width": 28,
    "root_pimc_cvar_pct": 25,
    "spatial_rerank_only": False,
}


def percentile(sorted_vals: List[float], p: float) -> Optional[float]:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def validate_record(rec: Dict[str, Any]) -> Tuple[bool, str]:
    if rec.get("schema_version") != "bc4_mass_legal_v1":
        return False, f"unexpected schema_version={rec.get('schema_version')}"
    if rec.get("model_sha256") != PINNED_MODEL_SHA256:
        return False, f"model_sha256 mismatch: {rec.get('model_sha256')}"
    if rec.get("baseline_commit") != PINNED_BASELINE_COMMIT:
        return False, f"baseline_commit mismatch: {rec.get('baseline_commit')}"
    sc = rec.get("search_config") or {}
    for k, expected in REQUIRED_SEARCH.items():
        if sc.get(k) != expected:
            return False, f"search_config.{k}={sc.get(k)} != {expected}"
    status = rec.get("status")
    completed = rec.get("completed_100l")
    ds_cleared = rec.get("ds_cleared", 0)
    pieces = rec.get("pieces", 0)
    if status == "completed":
        if not completed or ds_cleared < 100 or pieces <= 0:
            return False, f"invalid completed record: completed={completed}, ds={ds_cleared}, pcs={pieces}"
    elif status == "topout":
        if completed or ds_cleared >= 100:
            return False, f"invalid topout record: completed={completed}, ds={ds_cleared}"
    else:
        return False, f"unknown status={status}"
    return True, ""


def compute_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    completed = [r for r in records if r["status"] == "completed"]
    topouts = [r for r in records if r["status"] == "topout"]
    pcs = sorted(r["pieces"] for r in completed)

    mean_pcs = (sum(pcs) / len(pcs)) if pcs else None
    topout_rate = (len(topouts) / total) if total > 0 else 0.0

    # Top 10 best seeds (lowest pieces, tie-break by seed)
    top10 = sorted(completed, key=lambda r: (r["pieces"], r["seed"]))[:10]
    top10_summary = [
        {
            "seed": r["seed"],
            "pieces": r["pieces"],
            "lines_cleared": r["lines_cleared"],
            "ds_cleared": r["ds_cleared"],
            "elapsed_sec": round(r["elapsed_sec"], 2),
            "batch_id": r.get("batch_id"),
            "shard_id": r.get("shard_id"),
            "runner_job_id": r.get("runner_job_id"),
        }
        for r in top10
    ]

    # Cumulative PB progression ordered by seed
    pb_progression = []
    best_so_far = None
    for idx, r in enumerate(sorted(records, key=lambda x: x["seed"]), start=1):
        if r["status"] != "completed":
            continue
        if best_so_far is None or r["pieces"] < best_so_far:
            best_so_far = r["pieces"]
            pb_progression.append(
                {
                    "games_seen": idx,
                    "seed": r["seed"],
                    "pieces": r["pieces"],
                    "batch_id": r.get("batch_id"),
                }
            )

    # Sub-threshold counts
    le_161 = sum(1 for p in pcs if p <= 161)
    le_170 = sum(1 for p in pcs if p <= 170)
    le_175 = sum(1 for p in pcs if p <= 175)
    le_180 = sum(1 for p in pcs if p <= 180)

    return {
        "total_games": total,
        "completed_100l": len(completed),
        "topouts": len(topouts),
        "topout_rate": round(topout_rate, 6),
        "mean_pieces": round(mean_pcs, 4) if mean_pcs is not None else None,
        "min_pieces": pcs[0] if pcs else None,
        "max_pieces": pcs[-1] if pcs else None,
        "median_pieces": percentile(pcs, 50),
        "p1_pieces": percentile(pcs, 1),
        "p5_pieces": percentile(pcs, 5),
        "p95_pieces": percentile(pcs, 95),
        "p99_pieces": percentile(pcs, 99),
        "count_le_161": le_161,
        "count_le_170": le_170,
        "count_le_175": le_175,
        "count_le_180": le_180,
        "top10_seeds": top10_summary,
        "pb_progression": pb_progression,
    }


def render_markdown(summary: Dict[str, Any]) -> str:
    ov = summary["overall"]
    lines = [
        "# BC-4 Mass Legal 100L Cheese Race Simulation Summary",
        "",
        "> **Record Type**: Local Legal Simulation (`block-stacker` Guideline SRS, `p=5`, 100L Cheese `min_h=4, max_h=9`). Distinct from Jstris official leaderboard records.",
        "",
        "## Pinned Version",
        f"- **Baseline Commit**: `{PINNED_BASELINE_COMMIT}`",
        f"- **Model**: `{PINNED_MODEL_NAME}`",
        f"- **Model SHA-256**: `{PINNED_MODEL_SHA256}`",
        "- **Search**: `beam.rs` REAL FULL-MLP (`w=16, h=8, b=28, roots=32, fine=10, cvar=25, p=5`)",
        f"- **Input Files Processed**: {summary['files_processed']} | **Raw Records**: {summary['raw_records']} | **Duplicates Removed**: {summary['duplicates_removed']} | **Invalid Excluded**: {summary['invalid_records']}",
        "",
        "## Overall Statistics",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total Valid Unique Seeds | {ov['total_games']} |",
        f"| Completed 100L | {ov['completed_100l']} |",
        f"| Top Outs | {ov['topouts']} ({ov['topout_rate']*100:.2f}%) |",
        f"| Mean Pieces (Completed) | {ov['mean_pieces'] if ov['mean_pieces'] is not None else 'N/A'} |",
        f"| Min Pieces (PB) | {ov['min_pieces'] if ov['min_pieces'] is not None else 'N/A'} |",
        f"| Max Pieces | {ov['max_pieces'] if ov['max_pieces'] is not None else 'N/A'} |",
        f"| Median (P50) | {ov['median_pieces'] if ov['median_pieces'] is not None else 'N/A'} |",
        f"| P1 / P5 | {ov['p1_pieces']} / {ov['p5_pieces']} |",
        f"| P95 / P99 | {ov['p95_pieces']} / {ov['p99_pieces']} |",
        f"| <=161 / <=170 / <=175 / <=180 | {ov['count_le_161']} / {ov['count_le_170']} / {ov['count_le_175']} / {ov['count_le_180']} |",
        "",
        "## Top 10 Best Seeds",
        "| Rank | Seed | Pieces | Lines | DS | Time (s) | Batch | Shard |",
        "|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for rank, r in enumerate(ov["top10_seeds"], start=1):
        lines.append(
            f"| {rank} | `{r['seed']}` | **{r['pieces']}** | {r['lines_cleared']} | {r['ds_cleared']} | {r['elapsed_sec']} | `{r['batch_id']}` | {r['shard_id']} |"
        )

    lines.extend([
        "",
        "## Cumulative PB Progression (by Seed Order)",
        "| Games Seen | Seed | New PB (Pieces) | Batch |",
        "|---:|---:|---:|---|",
    ])
    for pb in ov["pb_progression"]:
        lines.append(
            f"| {pb['games_seen']} | `{pb['seed']}` | **{pb['pieces']}** | `{pb['batch_id']}` |"
        )

    lines.extend([
        "",
        "## Per-Batch Distribution",
        "| Batch ID | Total | Completed | Top Outs | Mean | Min | Median | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for bid, bs in sorted(summary["batches"].items()):
        lines.append(
            f"| `{bid}` | {bs['total_games']} | {bs['completed_100l']} | {bs['topouts']} | {bs['mean_pieces']} | {bs['min_pieces']} | {bs['median_pieces']} | {bs['max_pieces']} |"
        )

    if summary.get("missing_seeds_from_manifests"):
        ms = summary["missing_seeds_from_manifests"]
        lines.extend([
            "",
            f"## Missing Seeds Detected ({len(ms)})",
            f"First 50 missing seeds: `{','.join(str(s) for s in ms[:50])}`",
        ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate BC-4 Mass Legal simulation JSONL outputs")
    parser.add_argument("--input-dir", action="append", default=[], help="Directory to recursively scan for *.jsonl and *manifest*.json")
    parser.add_argument("--input-glob", action="append", default=[], help="Glob pattern for JSONL files")
    parser.add_argument("--out-json", required=True, help="Output summary JSON path")
    parser.add_argument("--out-md", required=True, help="Output summary Markdown path")
    parser.add_argument("--out-merged-jsonl", help="Optional deduplicated merged JSONL output path")
    parser.add_argument("--seed-range", help="Optional expected seed range start:end (inclusive) to check completeness")
    args = parser.parse_args()

    jsonl_files: List[Path] = []
    manifest_files: List[Path] = []
    for d in args.input_dir:
        p = Path(d)
        if p.exists():
            jsonl_files.extend(sorted(p.rglob("*.jsonl")))
            manifest_files.extend(sorted(p.rglob("*manifest*.json")))
    for g in args.input_glob:
        for match in sorted(glob.glob(g, recursive=True)):
            jsonl_files.append(Path(match))

    # Deduplicate file paths
    jsonl_files = sorted(set(jsonl_files))
    manifest_files = sorted(set(manifest_files))

    raw_records = 0
    invalid_records = 0
    duplicates_removed = 0
    consistency_mismatches = 0
    by_seed: Dict[int, Dict[str, Any]] = {}

    for fp in jsonl_files:
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                raw_records += 1
                try:
                    rec = json.loads(s)
                except json.JSONDecodeError:
                    invalid_records += 1
                    continue
                ok, reason = validate_record(rec)
                if not ok:
                    invalid_records += 1
                    print(f"WARN: excluding invalid record {fp}:{line_no}: {reason}", file=sys.stderr)
                    continue
                seed = int(rec["seed"])
                if seed in by_seed:
                    duplicates_removed += 1
                    prev = by_seed[seed]
                    if prev["pieces"] != rec["pieces"] or prev["status"] != rec["status"]:
                        consistency_mismatches += 1
                        print(
                            f"WARN: duplicate seed {seed} mismatch! prev=({prev['status']},{prev['pieces']}) vs new=({rec['status']},{rec['pieces']})",
                            file=sys.stderr,
                        )
                else:
                    by_seed[seed] = rec

    unique_records = [by_seed[s] for s in sorted(by_seed.keys())]

    # Group by batch
    by_batch: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in unique_records:
        by_batch[str(r.get("batch_id", "unknown"))].append(r)

    # Check missing seeds from manifests or explicit --seed-range
    missing_seeds = set()
    for mf in manifest_files:
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            for s in m.get("missing_seeds", []):
                if int(s) not in by_seed:
                    missing_seeds.add(int(s))
        except Exception:
            pass
    if args.seed_range:
        s0, s1 = [int(x) for x in args.seed_range.split(":")]
        for s in range(s0, s1 + 1):
            if s not in by_seed:
                missing_seeds.add(s)

    summary = {
        "schema_version": "bc4_mass_legal_summary_v1",
        "record_type": "local_legal_sim_block_stacker",
        "pinned_baseline_commit": PINNED_BASELINE_COMMIT,
        "pinned_model_name": PINNED_MODEL_NAME,
        "pinned_model_sha256": PINNED_MODEL_SHA256,
        "files_processed": len(jsonl_files),
        "manifests_processed": len(manifest_files),
        "raw_records": raw_records,
        "unique_valid_records": len(unique_records),
        "duplicates_removed": duplicates_removed,
        "consistency_mismatches": consistency_mismatches,
        "invalid_records": invalid_records,
        "missing_seeds_from_manifests": sorted(missing_seeds),
        "overall": compute_stats(unique_records),
        "batches": {bid: compute_stats(recs) for bid, recs in sorted(by_batch.items())},
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    md = render_markdown(summary)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md, encoding="utf-8")

    if args.out_merged_jsonl:
        Path(args.out_merged_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_merged_jsonl).open("w", encoding="utf-8") as f:
            for r in unique_records:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    print(md)


if __name__ == "__main__":
    main()
