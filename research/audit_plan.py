#!/usr/bin/env python3
"""S31 public-plan metadata, budget, fallback and observer equivalence checks."""
import argparse,json,statistics,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--out',required=True);a=p.parse_args()
rows=list(map(json.loads,Path(a.input).read_text().splitlines()))
frozen={r['seed']:r for r in rows if Path(r['binary']).name=='learned-robust'}
records=[];groups=collections.defaultdict(list)
for r in rows:
 name=Path(r['binary']).name
 if name=='learned-robust':continue
 assert name in ['robust-plan-reuse','robust-plan-observer']
 events=[json.loads(line[len('PLAN_REUSE '):]) for line in r['stderr'].splitlines() if line.startswith('PLAN_REUSE ')]
 for i,e in enumerate(events):
  assert e['observer']==(name=='robust-plan-observer')
  assert 0<=e['followed']<=e['matched']<=e['supplied']<=6
  assert e['generated']<=50000 and e['fallback_available']
  if i==0:assert e['supplied']==0
 fallbacks=r['stderr'].splitlines().count('LEGAL_FALLBACK used');parity=None
 if r['status']=='complete':
  t=json.loads(Path(r['trace']).read_text());assert len(events)==r['pieces']==len(r['search'])
  assert r['search']==t['search'] and t['ds'][-1]==100
  for e,s in zip(events,r['search']):assert e['generated']==s['generated_nodes']
  assert fallbacks==sum(s['root_moves']==0 for s in r['search'])
  if name=='robust-plan-observer' and r['seed'] in frozen:
   old=frozen[r['seed']];assert old['status']=='complete'
   o=json.loads(Path(old['trace']).read_text());parity=(t['inputs']==o['inputs'] and t['ds']==o['ds'])
   assert parity
   assert [(s['generated_nodes'],s['sequences']) for s in t['search']]==[(s['generated_nodes'],s['sequences']) for s in o['search']]
 rec=dict(binary=name,seed=r['seed'],status=r['status'],fallbacks=fallbacks,observer_parity=parity,
  eligible_fraction=statistics.mean(e['matched']>0 for e in events) if events else None,
  mean_followed=statistics.mean(e['followed'] for e in events) if events else None)
 records.append(rec);groups[name].append(rec)
result=dict(records=records,groups={name:dict(attempts=len(rs),complete=sum(r['status']=='complete' for r in rs),
 fallbacks=sum(r['fallbacks'] for r in rs),parity_passes=sum(r['observer_parity'] is True for r in rs),
 eligible_fraction=statistics.mean(r['eligible_fraction'] for r in rs if r['eligible_fraction'] is not None),
 mean_followed=statistics.mean(r['mean_followed'] for r in rs if r['mean_followed'] is not None)) for name,rs in groups.items()},
 note='Metadata audit and observer full-history equivalence; hint fractions descriptive only. Fallback uses an already charged legal root child, not an invented completed sequence.')
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
