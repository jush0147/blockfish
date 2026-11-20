#!/usr/bin/env python3
"""Observe value errors on executed states; never fit on this diagnostic cohort."""
import argparse
import collections
import json
import pathlib
import statistics as st
import subprocess

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--out', required=True)
    p.add_argument('--binary', default='research/bin/cheese-quality')
    args = p.parse_args()
    fit = json.loads((ROOT/'research/results/value-fit.json').read_text())
    weights = fit['selected']['weights']
    price = fit['garbage_price']
    results = []
    for line in (ROOT/args.input).read_text().splitlines():
        row = json.loads(line)
        if row['status'] != 'complete':
            raise ValueError('Incomplete game; report failures separately.')
        trace = ROOT/row['trace']
        feature_path = trace.with_suffix('.features.json')
        subprocess.run([str(ROOT/args.binary), '--replay-file', str(trace),
                        '--replay-features-out', str(feature_path)],
                       check=True, capture_output=True, text=True, timeout=30)
        samples = json.loads(feature_path.read_text())
        errors = dict(legacy=[], learned=[])
        for i,s in enumerate(samples):
            unseen = 100-s['cleared']-s['visible_garbage']
            if unseen < 20:
                continue
            target = row['pieces']-i-price*unseen
            x = s['features']
            errors['legacy'].append(.5*x[1]+x[2]+x[3]-target)
            errors['learned'].append(max(0,sum(a*b for a,b in zip(x,weights)))-target)
        results.append(dict(seed=row['seed'], binary=row['binary'], config=row['config'],
                            pieces=row['pieces'], samples=len(errors['legacy']),
                            errors={k:dict(bias=st.mean(e), mse=st.mean(v*v for v in e))
                                    for k,e in errors.items()}))
    groups = collections.defaultdict(list)
    for r in results:
        groups[f"{pathlib.Path(r['binary']).name}:{r['config']}"].append(r)
    summary = {name:dict(games=len(rs), mean_pieces=st.mean(r['pieces'] for r in rs),
                        errors={k:{metric:st.mean(r['errors'][k][metric] for r in rs)
                                   for metric in ['bias','mse']} for k in ['legacy','learned']})
               for name,rs in groups.items()}
    report = dict(garbage_price_from_training=price, summary=summary, records=results,
                  limitation='Executed states only; errors include policy-dependent returns. '
                             'This does not measure search-leaf error or prove ranking causality.')
    (ROOT/args.out).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
