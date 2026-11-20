#!/usr/bin/env python3
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
records=[]
for source in a.inputs:
 for r in map(json.loads,Path(source).read_text().splitlines()):
  events=[json.loads(s[len('ROOT_DAG '):]) for s in r['stderr'].splitlines() if s.startswith('ROOT_DAG ')]
  assert r['status']=='complete' and len(events)==len(r['search'])==r['pieces']
  merging=Path(r['binary']).name=='robust-dag'
  for e,s in zip(events,r['search']):
   assert e['merging']==merging
   assert e['generated']==s['generated_nodes']<=50000
   assert 0<e['unique_expanded']<=e['generated']
   assert 0<=e['duplicate_expansions']<e['generated']
  records.append(dict(binary=Path(r['binary']).name,seed=r['seed'],
    mean_duplicates=statistics.mean(e['duplicate_expansions'] for e in events),
    mean_unique_expanded=statistics.mean(e['unique_expanded'] for e in events),
    mean_sequences=statistics.mean(s['sequences'] for s in r['search'])))
result=dict(passed=len(records),records=records,means={b:{k:statistics.mean(r[k] for r in records if r['binary']==b)
 for k in ['mean_duplicates','mean_unique_expanded','mean_sequences']} for b in sorted({r['binary'] for r in records})})
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
