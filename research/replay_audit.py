#!/usr/bin/env python3
"""Replay every completed race listed in JSONL files, without running the AI."""
import argparse
import hashlib
import json
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--binary', default='research/bin/learned-fast')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    binary = ROOT / args.binary
    records = []
    for source in args.inputs:
        for line in (ROOT / source).read_text().splitlines():
            row = json.loads(line)
            if row['status'] != 'complete':
                raise ValueError(f"Cannot audit incomplete race: {source}, seed {row['seed']}")
            trace = ROOT / row['trace']
            data = json.loads(trace.read_text())
            assert data['seed'] == row['seed']
            assert len(data['ds']) == row['pieces']
            result = subprocess.run([str(binary), '--replay-file', str(trace)],
                                    capture_output=True, text=True, timeout=30)
            records.append(dict(source=source, trace=row['trace'], seed=row['seed'],
                                pieces=row['pieces'], returncode=result.returncode,
                                sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                                stdout=result.stdout.strip(), stderr=result.stderr.strip()))
    report = dict(verifier=args.binary,
                  verifier_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  total=len(records), passed=sum(r['returncode'] == 0 for r in records),
                  records=records)
    (ROOT / args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(f"Replayed {report['passed']}/{report['total']} successfully; {args.out}")
    if report['passed'] != report['total']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
