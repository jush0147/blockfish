#!/usr/bin/env python3
"""Train a root-only residual scorer, retaining the frozen search's lookahead rating."""
import json
import pathlib
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parent.parent
rows=[json.loads(x) for x in (ROOT/'research/results/s-rank-data.jsonl').read_text().splitlines()]
assert len(rows)==80 and {r['seed'] for r in rows}==set(range(73000,73080))
prior=np.r_[1.,np.zeros(13)]

def games(lo,hi):
    out=[]
    for r in rows:
        if not lo<=r['seed']<hi:continue
        data=json.loads((ROOT/r['data']).read_text());probes=[]
        for probe in data['probes']:
            cs=probe['candidates']
            if len(cs)<2 or any(c['rating']<20 for c in cs):continue
            if data['source']['ds'][probe['step']-1]>=80:continue
            x=np.array([[c['rating']/20]+c['features'][1:] for c in cs])
            y=np.array([c['pieces_remaining'] for c in cs])
            probes.append((x,y))
        assert probes
        out.append(probes)
    return out

train=games(73000,73064);valid=games(73064,73080)
x,y,w=[],[],[]
for game in train:
    n=sum(len(cost)-1 for _,cost in game)
    for features,cost in game:
        for j in range(1,len(cost)):
            x.append(features[j]-features[0]);y.append(cost[j]-cost[0]);w.append(1/n)
x,y,w=np.array(x),np.array(y),np.array(w)
scale=np.maximum(np.sqrt(np.average(x*x,axis=0,weights=w)),1e-8)
models=[]
for ridge in [10.,100.,1000.]:
    design=np.vstack([(x/scale)*np.sqrt(w)[:,None],np.eye(14)*np.sqrt(ridge)])
    target=np.r_[y*np.sqrt(w),prior*scale*np.sqrt(ridge)]
    beta=np.linalg.lstsq(design,target,rcond=None)[0]
    weights=beta/scale
    for margin in [0.,1.]:
        deltas=[];changed=0;count=0
        for game in valid:
            game_delta=[]
            for features,cost in game:
                pred=features@weights
                best=int(np.argmin(pred))
                if pred[0]-pred[best]<=margin:best=0
                changed+=best!=0;count+=1;game_delta.append(float(cost[best]-cost[0]))
            deltas.append(float(np.mean(game_delta)))
        models.append(dict(ridge=ridge,margin=margin,validation_delta=float(np.mean(deltas)),
                           changed=changed,validation_probes=count,per_game_deltas=deltas,weights=weights.tolist()))
# Fewer changed actions breaks exact ties; full races still decide adoption.
best=min(models,key=lambda m:(m['validation_delta'],m['changed']))
report=dict(training_seeds=[73000,73063],validation_seeds=[73064,73079],
            training_pairs=len(y),models=models,selected=best,
            applicability='Best three frozen-search roots only; cleared<80, all ratings>=20. '
                          'Features exclude future replenishment. Three projections charged to node cap.')
(ROOT/'research/results/s-rerank-fit.json').write_text(json.dumps(report,indent=2)+'\n')
(ROOT/'blockfish-engine/src/ai/rerank_weights.rs').write_text(
    '// Root-only residual model, research/fit_rerank.py, training seeds73000..73079.\n'
    'const WEIGHTS: [f64;14] = ['+', '.join(format(v,'.17e') for v in best['weights'])+'];\n'
    'pub const MARGIN: f64 = '+format(best['margin'],'.17e')+';\n')
print(json.dumps(report,indent=2))
