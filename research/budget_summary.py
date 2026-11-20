#!/usr/bin/env python3
"""Summarize complete paired cohorts and cross-budget response, without filtering seeds."""
import argparse
import collections
import itertools
import json
import pathlib
import random
import statistics as st


def paired(a, b):
    seeds = sorted(a.keys() & b.keys())
    diffs = [b[s]['pieces']-a[s]['pieces'] for s in seeds
             if a[s]['status'] == b[s]['status'] == 'complete']
    if not diffs:
        return dict(paired_seeds=len(seeds), completed_pairs=0)
    rng = random.Random(173)
    means = sorted(sum(rng.choices(diffs, k=len(diffs)))/len(diffs) for _ in range(10000))
    return dict(paired_seeds=len(seeds), completed_pairs=len(diffs),
                delta_b_minus_a=st.mean(diffs), bootstrap95=[means[250], means[9749]],
                b_wins=sum(d<0 for d in diffs), ties=sum(d==0 for d in diffs),
                b_losses=sum(d>0 for d in diffs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--budget-interaction', action='store_true',
                   help='paired change in learned-minus-baseline advantage, 50k to 200k')
    p.add_argument('--interaction-variants', nargs=2, metavar=('CONTROL', 'CANDIDATE'),
                   help='binary names for paired change in candidate-minus-control across two budgets')
    p.add_argument('--interaction-budgets', nargs=2, type=int, default=[5, 50])
    args = p.parse_args()
    groups = collections.defaultdict(dict)
    for path in args.inputs:
        for line in pathlib.Path(path).read_text().splitlines():
            r = json.loads(line)
            name = f"{pathlib.Path(r['binary']).name}:{r['config']}:{r.get('move_ms')}"
            if r.get('weights_sha256'):
                name += ':weights=' + r['weights_sha256']
            assert r['seed'] not in groups[name], (name, r['seed'])
            groups[name][r['seed']] = r
    assert all(len(g)==args.expected_count for g in groups.values()), 'Incomplete cohort'
    assert len({tuple(sorted(g)) for g in groups.values()}) == 1, 'Unpaired cohorts'
    result = dict(groups={}, pairs=[])
    for name,g in sorted(groups.items()):
        rows = list(g.values())
        done = [r for r in rows if r['status']=='complete']
        d = dict(total=len(rows), complete=len(done), seeds=sorted(g),
                 binary_sha256=sorted({r['sha256'] for r in rows}))
        if done:
            d.update(mean_pieces=st.mean(r['pieces'] for r in done),
                     min_pieces=min(r['pieces'] for r in done))
            cpu = [r['cpu_seconds'] for r in done if 'cpu_seconds' in r]
            d.update(cpu_records=len(cpu), mean_cpu_seconds=st.mean(cpu) if cpu else None)
        result['groups'][name] = d
    for a,b in itertools.combinations(sorted(groups),2):
        result['pairs'].append(dict(a=a,b=b,**paired(groups[a],groups[b])))
    if args.budget_interaction:
        deltas = {}
        for budget in [50,200]:
            assert f'quality-baseline:{budget}/5,10,10,10:None' in groups
            assert f'learned-fast:{budget}/5,10,10,10:None' in groups
            baseline = groups[f'quality-baseline:{budget}/5,10,10,10:None']
            learned = groups[f'learned-fast:{budget}/5,10,10,10:None']
            deltas[budget] = {s:dict(status='complete' if baseline[s]['status']==learned[s]['status']=='complete'
                                     else 'failed', pieces=learned[s]['pieces']-baseline[s]['pieces']
                                     if baseline[s]['pieces'] is not None and learned[s]['pieces'] is not None else 0)
                             for s in baseline}
        result['budget_interaction'] = dict(
            definition='(learned200-baseline200) - (learned50-baseline50); positive means advantage shrinks',
            **paired(deltas[50],deltas[200]))
    if args.interaction_variants:
        control, candidate = args.interaction_variants
        low, high = args.interaction_budgets
        effects = {}
        for budget in [low, high]:
            a = groups[f'{control}:{budget}/5,10,10,10:None']
            b = groups[f'{candidate}:{budget}/5,10,10,10:None']
            assert a and b
            effects[budget] = {seed: dict(
                status='complete' if a[seed]['status']==b[seed]['status']=='complete' else 'failed',
                pieces=b[seed]['pieces']-a[seed]['pieces']
                    if a[seed]['pieces'] is not None and b[seed]['pieces'] is not None else 0)
                for seed in a}
        result['budget_interaction'] = dict(
            definition=f'({candidate}-{control}) at {high}k minus {low}k; positive means advantage shrinks',
            **paired(effects[low], effects[high]))
    pathlib.Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    for name,g in result['groups'].items():
        print(name, g['complete'], '/',g['total'], 'mean', g.get('mean_pieces'),
              'CPU',g.get('mean_cpu_seconds'))
    for r in result['pairs']: print(json.dumps(r))
    if 'budget_interaction' in result: print(json.dumps(result['budget_interaction']))


if __name__=='__main__': main()
