#!/usr/bin/env python3
"""Describe support concentration and prove the probe did not change decisions."""
import argparse
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--probe',default='research/results/s16-probe.jsonl')
p.add_argument('--control',default='research/results/s16-dev.jsonl')
p.add_argument('--out',default='research/results/s16-support-diagnostic.json')
a=p.parse_args()
controls={r['seed']:r for r in map(json.loads,(ROOT/a.control).read_text().splitlines()) if Path(r['binary']).name=='learned-robust'}
rows=list(map(json.loads,(ROOT/a.probe).read_text().splitlines()))
result=[]
for r in rows:
    assert r['status']=='complete'
    stats=[json.loads(l.removeprefix('SUPPORT_PROBE ')) for l in r['stderr'].splitlines() if l.startswith('SUPPORT_PROBE ')]
    assert len(stats)==r['pieces']
    old=json.loads((ROOT/controls[r['seed']]['trace']).read_text())
    new=json.loads((ROOT/r['trace']).read_text())
    assert old['inputs']==new['inputs'] and old['ds']==new['ds']
    full=sum(s['full_roots'] for s in stats)
    unique=sum(s['full_unique_second_actions'] for s in stats)
    single=sum(s['full_single_second_roots'] for s in stats)
    result.append(dict(seed=r['seed'],decisions=len(stats),full_roots=full,
        full_unique_second_actions=unique,full_single_second_roots=single,
        mean_unique_second_actions=unique/full,fraction_single_second=single/full,
        identical_inputs=True,identical_clears=True))
full=sum(r['full_roots'] for r in result)
report=dict(games=result,weighted_mean_unique_second_actions=sum(r['full_unique_second_actions'] for r in result)/full,
    weighted_fraction_single_second=sum(r['full_single_second_roots'] for r in result)/full,
    caveat='Descriptive three-game mechanism probe; root observations are correlated, not independent games.')
(ROOT/a.out).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
