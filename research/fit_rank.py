#!/usr/bin/env python3
"""Fit paired action cost differences, regularized toward the frozen evaluator."""
import argparse
import json
import pathlib
import numpy as np
from scipy.optimize import nnls

ROOT=pathlib.Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--data',default='research/results/s-rank-data.jsonl')
p.add_argument('--out',default='research/results/s-rank-fit.json')
p.add_argument('--weights-out',default='blockfish-engine/src/ai/rank_weights.rs')
a=p.parse_args()
rows=[json.loads(x) for x in (ROOT/a.data).read_text().splitlines()]
assert len(rows)==80 and {r['seed'] for r in rows}==set(range(73000,73080))
assert all(r['status']=='collected' for r in rows)
prior_fit=json.loads((ROOT/'research/results/value-fit.json').read_text())
prior=np.array(prior_fit['selected']['weights'])

def dataset(lo,hi):
    x,y,w=[],[],[]
    for r in rows:
        if not lo<=r['seed']<hi:continue
        data=json.loads((ROOT/r['data']).read_text());pairs=[]
        for probe in data['probes']:
            base=probe['candidates'][0]
            for c in probe['candidates'][1:]:
                pairs.append((np.array(c['features'])[1:]-np.array(base['features'])[1:],
                              c['pieces_remaining']-base['pieces_remaining']))
        assert pairs,'No usable probes in a training game'
        for dx,dy in pairs:x.append(dx);y.append(dy);w.append(1/len(pairs))
    return np.array(x),np.array(y),np.array(w)

x,y,w=dataset(73000,73064)
vx,vy,vw=dataset(73064,73080)
scale=np.maximum(np.sqrt(np.average(x*x,axis=0,weights=w)),1e-8)
z=x/scale
models=[]
for ridge in [.01,.1,1.,10.,100.,1000.]:
    design=np.vstack([z*np.sqrt(w)[:,None],np.eye(13)*np.sqrt(ridge)])
    target=np.r_[y*np.sqrt(w),prior[1:]*scale*np.sqrt(ridge)]
    beta,_=nnls(design,target,maxiter=10000)
    weights=np.r_[prior[0],beta/scale]
    pred=vx@weights[1:]
    models.append(dict(ridge=ridge,validation_mse=float(np.average((pred-vy)**2,weights=vw)),weights=weights.tolist()))
best=min(models,key=lambda m:m['validation_mse'])
report=dict(training_seeds=[73000,73063],validation_seeds=[73064,73079],
            features=prior_fit['features'],training_pairs=len(y),validation_pairs=len(vy),
            failure_penalty=500,failed_rollouts=sum(r['failed_rollouts'] for r in rows),
            collector_sha256=sorted({r['sha256'] for r in rows}),
            prior_validation_mse=float(np.average((vx@prior[1:]-vy)**2,weights=vw)),
            models=models,selected=best)
(ROOT/a.out).write_text(json.dumps(report,indent=2)+'\n')
(ROOT/a.weights_out).write_text('// Paired action labels: seeds73000..73079, see research/fit_rank.py.\n'
                               'const WEIGHTS: [f64;14] = ['+', '.join(format(v,'.17e') for v in best['weights'])+'];\n')
print(json.dumps(report,indent=2))
