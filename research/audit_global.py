#!/usr/bin/env python3
"""Audit fixed generated budgets for S29/S30, retaining failed games explicitly."""
import argparse,collections,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--out',required=True);a=p.parse_args()
records=[];skipped=[];groups=collections.defaultdict(list)
for r in map(json.loads,Path(a.input).read_text().splitlines()):
 name=Path(r['binary']).name
 if name=='learned-robust':skipped.append(r['seed']);continue
 assert name in ('robust-global-regret','robust-global-f')
 events=[json.loads(line[len('GLOBAL_SEARCH '):]) for line in r['stderr'].splitlines() if line.startswith('GLOBAL_SEARCH ')]
 for e in events:
  assert e['absolute']==(name=='robust-global-f')
  assert 1<=e['generated']<=50000
  assert 0<=e['expanded']<=e['generated'] and 0<=e['completed']<=e['generated']
  assert 0<=e['pending_children']<=e['generated']
 if r['status']=='complete':
  t=json.loads(Path(r['trace']).read_text());assert len(events)==len(r['search'])==r['pieces']
  assert t['search']==r['search'] and t['ds'][-1]==100
  for e,s in zip(events,r['search']):
   assert e['generated']==s['generated_nodes'] and e['completed']==s['sequences'] and e['roots']==s['root_moves']
 rec=dict(binary=name,seed=r['seed'],status=r['status'],logged_decisions=len(events),
  zero_root_decisions=sum(e['roots']==0 for e in events),
  means={k:statistics.mean(e[k] for e in events) if events else None for k in ['expanded','completed','roots','pending_children']})
 records.append(rec);groups[name].append(rec)
result=dict(records=records,non_candidate_control_records=len(skipped),groups={n:dict(games=len(rs),complete=sum(r['status']=='complete' for r in rs),
 zero_root_decisions=sum(r['zero_root_decisions'] for r in rs),means={k:statistics.mean(r['means'][k] for r in rs if r['means'][k] is not None) for k in ['expanded','completed','roots','pending_children']}) for n,rs in groups.items()},
 note='All failure rows retained; per-game means descriptive, not causal quality comparisons. Control uses frozen implementation.')
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
