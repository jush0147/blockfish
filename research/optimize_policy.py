#!/usr/bin/env python3
"""Bounded, deterministic mirrored search on actual complete-race piece counts."""
import hashlib
import json
import math
import pathlib
import random
import statistics
import subprocess
import sys

ROOT=pathlib.Path(__file__).resolve().parent.parent
out=ROOT/'research/results/s-direct-train.jsonl'
binary=ROOT/'research/bin/learned-tunable'
digest=hashlib.sha256(binary.read_bytes()).hexdigest()
prior=json.loads((ROOT/'research/results/value-fit.json').read_text())['selected']['weights']
active=[2,4,8,9,10,11]
seeds=list(range(74000,74032))
records={};offset=0;ledger={}

def refresh():
    global offset
    if not out.exists():return
    with out.open() as f:
        f.seek(offset)
        while line:=f.readline():
            r=json.loads(line)
            if r['sha256']==digest and r['config']=='5/5,10,10,10':
                records[(r['weights_sha256'],r['seed'])]={k:r.get(k) for k in ['status','pieces','cpu_seconds']}
        offset=f.tell()

def evaluate(vector):
    weights=list(prior)
    for i,v in zip(active,vector):weights[i]*=math.exp(v)
    encoded=json.dumps(weights,separators=(',',':'))
    key=hashlib.sha256(encoded.encode()).hexdigest()
    if key in ledger:return ledger[key]
    if not all((key,s) in records for s in seeds):
        subprocess.run([sys.executable,'research/bench.py','--binary','research/bin/learned-tunable',
                        '--configs','5/5,10,10,10','--start','74000','--count','32','--jobs','4',
                        '--value-weights',encoded,'--out',str(out.relative_to(ROOT))],
                       cwd=ROOT,stdout=subprocess.DEVNULL,check=True)
        refresh()
    rows=[records[(key,s)] for s in seeds]
    completed=[r for r in rows if r['status']=='complete']
    result=dict(weights_sha256=key,vector=vector,weights=weights,
                failures=len(rows)-len(completed),
                mean_pieces=statistics.mean(r['pieces'] for r in completed) if completed else 1e9,
                cpu_seconds=sum(r['cpu_seconds'] or 0 for r in rows))
    ledger[key]=result
    print(json.dumps({k:result[k] for k in ['weights_sha256','failures','mean_pieces','cpu_seconds']}),flush=True)
    return result

refresh();rng=random.Random(173)
incumbent=evaluate([0.]*6)
initial=incumbent
generations=[]
for generation in range(6):
    sigma=max(.08,.25*.8**generation)
    center=incumbent['vector']
    candidates=[incumbent]
    for _ in range(4):
        direction=[rng.gauss(0,1) for _ in range(6)]
        for sign in [1,-1]:
            vector=[max(-.7,min(.7,v+sign*sigma*d)) for v,d in zip(center,direction)]
            candidates.append(evaluate(vector))
    incumbent=min(candidates,key=lambda r:(r['failures'],r['mean_pieces']))
    generations.append(dict(generation=generation,sigma=sigma,selected=incumbent['weights_sha256'],
                            candidates=[r['weights_sha256'] for r in candidates]))
    report=dict(method='six-generation mirrored bounded hill climb',rng_seed=173,
                binary_sha256=digest,training_seeds=[74000,74031],config='5/5,10,10,10',
                active_features=active,log_bounds=[-.7,.7],initial=initial,selected=incumbent,
                generations=generations,models=list(ledger.values()),complete=generation==5)
    (ROOT/'research/results/s-direct-fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('GENERATION',generation,'SELECTED',incumbent['mean_pieces'],flush=True)

(ROOT/'blockfish-engine/src/ai/direct_weights.rs').write_text(
    '// Direct full-race optimization on training seeds74000..74031 only.\n'
    'const WEIGHTS: [f64;14] = ['+', '.join(format(v,'.17e') for v in incumbent['weights'])+'];\n')
