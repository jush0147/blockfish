#!/usr/bin/env python3
"""Collect independent counterfactual training games with resource accounting."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import random
import subprocess
import time

ROOT=pathlib.Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--start',type=int,default=73000)
p.add_argument('--count',type=int,default=80)
p.add_argument('--jobs',type=int,default=4)
p.add_argument('--out',default='research/results/s-rank-data.jsonl')
a=p.parse_args()
binary=ROOT/'research/bin/rank-data'
digest=hashlib.sha256(binary.read_bytes()).hexdigest()
out=ROOT/a.out
old=[json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
done={r['seed'] for r in old if r['sha256']==digest and r['status']=='collected'}
seeds=[s for s in range(a.start,a.start+a.count) if s not in done]
random.Random(173).shuffle(seeds)

def collect(seed):
    path=out.with_name(f'{out.stem}-{digest[:8]}-{seed}.json')
    usage=path.with_suffix('.usage.json')
    start=time.monotonic()
    result=subprocess.run([str(ROOT/'research/bin/measure'),str(usage),str(binary),
                           '--seed',str(seed),'--out',str(path)],capture_output=True,text=True)
    row=dict(seed=seed,sha256=digest,binary=str(binary.relative_to(ROOT)),
             data=str(path.relative_to(ROOT)),returncode=result.returncode,stderr=result.stderr,
             wall_seconds=time.monotonic()-start,status='failed')
    if result.returncode==0:
        data=json.loads(path.read_text())
        candidates=[c for probe in data['probes'] for c in probe['candidates']]
        row.update(status='collected',source_complete=data['source']['ds'][-1]==100,
                   source_pieces=len(data['source']['ds']),probes=len(data['probes']),
                   rollouts=len(candidates),failed_rollouts=sum(not c['complete'] for c in candidates))
    if usage.exists():
        u=json.loads(usage.read_text());row.update(cpu_seconds=u['user_seconds']+u['system_seconds'],max_rss_kib=u['max_rss_kib'])
    return row

with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(a.jobs) as pool:
    for row in pool.map(collect,seeds):
        f.write(json.dumps(row)+'\n');f.flush();print(json.dumps(row),flush=True)
