#!/usr/bin/env python3
"""Summarize completions first; paired CI uses only jointly completed seeds."""
import argparse
import collections
import json
import random
import statistics as st

def main():
    p = argparse.ArgumentParser()
    p.add_argument('files', nargs='+')
    p.add_argument('--baseline', help='binary:config, e.g. original:5/5,10,10,10')
    p.add_argument('--allow-budget-difference', action='store_true')
    a = p.parse_args()
    groups = collections.defaultdict(dict)
    for path in a.files:
        for line in open(path):
            r = json.loads(line)
            key = r['binary'].split('/')[-1] + ':' + r['config']
            if r.get('move_ms') is not None: key += '@' + str(r['move_ms']) + 'ms'
            groups[key][r['seed']] = r
    base = groups.get(a.baseline, {})
    for key, rows in sorted(groups.items()):
        wins = [r for r in rows.values() if r['status'] == 'complete']
        result = dict(variant=key, completed=len(wins), total=len(rows))
        if wins:
            pcs = [r['pieces'] for r in wins]
            result.update(mean=st.mean(pcs), median=st.median(pcs), minimum=min(pcs),
                          mean_seconds=st.mean(r['wall_seconds'] for r in rows.values()))
        diffs = [r['pieces']-base[s]['pieces'] for s,r in rows.items()
                 if s in base and r['status'] == base[s]['status'] == 'complete'
                 and (a.allow_budget_difference or (r['config'].split('/')[0] == base[s]['config'].split('/')[0]
                      and r.get('move_ms') == base[s].get('move_ms')))]
        if diffs:
            rng = random.Random(173)
            means = sorted(st.mean(rng.choices(diffs, k=len(diffs))) for _ in range(10000))
            result.update(paired_n=len(diffs), paired_delta=st.mean(diffs),
                          bootstrap95=[means[250], means[9749]])
        print(json.dumps(result))

if __name__ == '__main__':
    main()
