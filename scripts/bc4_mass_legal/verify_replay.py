#!/usr/bin/env python3
"""Re-run and deterministically verify a saved BC-4 Mass Legal replay file."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify determinism of a BC-4 replay JSON")
    parser.add_argument("replay_json", help="Path to seed_<N>_pcs_<P>.replay.json")
    parser.add_argument(
        "--binary",
        default="target/release/examples/bc4_mass_legal",
        help="Path to compiled bc4_mass_legal binary",
    )
    args = parser.parse_args()

    orig = json.loads(Path(args.replay_json).read_text(encoding="utf-8"))
    rec = orig["record"]
    seed = int(rec["seed"])

    with tempfile.TemporaryDirectory() as tmp:
        out_jsonl = Path(tmp) / "verify.jsonl"
        replay_dir = Path(tmp) / "replays"
        cmd = [
            args.binary,
            "--seeds",
            str(seed),
            "--output",
            str(out_jsonl),
            "--replay-dir",
            str(replay_dir),
            "--replay-threshold",
            "9999",
            "--batch-id",
            "verify",
            "--threads",
            "1",
        ]
        subprocess.run(cmd, check=True)
        replays = list(replay_dir.glob("*.replay.json"))
        assert len(replays) == 1, f"Expected 1 replay file, found {len(replays)}"
        repro = json.loads(replays[0].read_text(encoding="utf-8"))

        assert repro["record"]["status"] == rec["status"], "status mismatch"
        assert repro["record"]["pieces"] == rec["pieces"], "pieces mismatch"
        assert repro["record"]["ds_cleared"] == rec["ds_cleared"], "ds_cleared mismatch"
        assert repro["record"]["lines_cleared"] == rec["lines_cleared"], "lines_cleared mismatch"
        assert repro["moves"] == orig["moves"], "move-by-move trace mismatch!"

        print(
            f"VERIFIED EXACT MATCH: seed={seed}, status={rec['status']}, "
            f"pieces={rec['pieces']}, moves={len(orig['moves'])}, "
            f"model_sha256={rec['model_sha256']}"
        )


if __name__ == "__main__":
    main()
