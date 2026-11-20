#!/usr/bin/env python3
"""Audit a preregistered paired cohort and apply its fixed mean/CI gate."""
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path
from budget_summary import paired
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--lock',required=True)
p.add_argument('--inputs',nargs='+',required=True)
p.add_argument('--out-prefix',required=True)
p.add_argument('--gate',choices=['mean','ci'],default='mean')
a=p.parse_args()
lock=json.loads((ROOT/a.lock).read_text())
rows=[json.loads(l) for name in a.inputs for l in (ROOT/name).read_text().splitlines()]
count=len(lock['seeds']);groups={name:{} for name in lock['binaries']}
assert len(rows)==count*len(groups), 'Incomplete or extra cohort records'
for r in rows:
    name=Path(r['binary']).name
    assert r['sha256']==lock['binaries'][name]
    assert r['config']==lock['config'] and r.get('move_ms') is None
    assert r['seed'] not in groups[name]
    groups[name][r['seed']]=r
    assert all(s['generated_nodes']<=lock['generated_node_cap'] for s in (r.get('search') or []))
    if r['status']=='complete':
        t=json.loads((ROOT/r['trace']).read_text())
        assert t['seed']==r['seed'] and len(t['inputs'])==r['pieces']
        assert len(r['search'])==r['pieces'], 'Missing per-decision budget records'
        assert t['search']==r['search'], 'Trace and cohort search records disagree'
        assert t['ds'][-1]==100
        for key in ['garbage_min','garbage_max']:
            if key in lock: assert t[key]==lock[key]
assert all(sorted(g)==lock['seeds'] for g in groups.values())
subprocess.run([sys.executable,'research/budget_summary.py',*a.inputs,'--expected-count',str(count),
    '--out',a.out_prefix+'-summary.json'],cwd=ROOT,check=True)
scores={}
for name,g in groups.items():
    values=[r['pieces'] for r in g.values() if r['status']=='complete']
    scores[name]=dict(complete=len(values),mean_pieces=statistics.mean(values) if values else None)
candidate=scores[lock['candidate']];checks={};comparisons={}
for name in lock['controls']:
    control=scores[name]
    checks[name]=dict(completion_no_worse=candidate['complete']>=control['complete'],
        mean_lower=candidate['mean_pieces'] is not None and control['mean_pieces'] is not None
                   and candidate['mean_pieces']<control['mean_pieces'])
    comparisons[name]=paired(groups[name],groups[lock['candidate']])
    if a.gate=='ci':
        checks[name]['complete_paired_cohort']=comparisons[name]['completed_pairs']==count
        checks[name]['negative_95_interval']=comparisons[name].get('bootstrap95',[0,0])[1]<0
passed=all(all(c.values()) for c in checks.values())
report=dict(gate=a.gate,passed=passed,lock=a.lock,scores=scores,
            candidate_minus_controls=comparisons,checks=checks)
(ROOT/(a.out_prefix+'-decision.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
