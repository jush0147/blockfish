#!/usr/bin/env python3
"""Audit S24 per-decision allocation and report game-weighted diagnostics."""
import json
import statistics as st
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
rows=[json.loads(s) for s in (ROOT/'research/results/s24-dev.jsonl').read_text().splitlines()]
rows=[r for r in rows if Path(r['binary']).name=='robust-halving']
assert len(rows)==32 and {r['seed'] for r in rows}==set(range(76800,76832))
records=[]
for r in rows:
    assert r['status']=='complete'
    stats=[json.loads(s[len('ROOT_ALLOCATION '):]) for s in r['stderr'].splitlines()
           if s.startswith('ROOT_ALLOCATION ')]
    assert len(stats)==len(r['search'])==r['pieces']
    for a,b in zip(stats,r['search']):
        assert a['generated']==b['generated_nodes']<=50000
        assert 0<=a['scout_nodes']<=a['generated']
        assert a['rated_roots']<=a['roots']
        if a['scout_complete']:
            assert a['survivors']<=a['rated_roots']
            assert b['root_moves']==a['survivors']
        else: assert b['root_moves']==a['rated_roots']
    records.append(dict(seed=r['seed'],decisions=len(stats),
        scout_fraction=st.mean(s['scout_nodes']/s['generated'] for s in stats),
        scout_completed_fraction=st.mean(s['scout_complete'] for s in stats),
        generated_roots=st.mean(s['roots'] for s in stats),
        rated_roots=st.mean(s['rated_roots'] for s in stats),
        surviving_moves=st.mean(s['root_moves'] for s in r['search']),
        complete_sequences=st.mean(s['sequences'] for s in r['search'])))
summary={k:st.mean(r[k] for r in records) for k in records[0] if k not in ['seed','decisions']}
report=dict(games=len(records),passed=len(records),game_weighted_means=summary,records=records,
    note='Descriptive allocation on executed trajectories, not causal evidence or independent per-decision samples.')
(ROOT/'research/results/s24-allocation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(summary,indent=2))
