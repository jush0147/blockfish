#!/usr/bin/env python3
"""Fixed S20 public-Hold residual fit. Prediction evidence is not policy evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser()
parser.add_argument('--td7', action='store_true')
args = parser.parse_args()
prefix = 's21' if args.td7 else 's20'
rows = [json.loads(s) for s in (ROOT/'research/results/s20-data.jsonl').read_text().splitlines()]
assert len(rows) == 160 and {r['seed'] for r in rows} == set(range(76400,76560))
assert all(r['status'] == 'complete' for r in rows)
assert len({r['sha256'] for r in rows}) == 1
assert all(r['config'] == '5/5,10,10,10' and r['collect_features'] for r in rows)
train = sorted((r for r in rows if r['seed'] < 76528), key=lambda r:r['seed'])
valid = sorted((r for r in rows if r['seed'] >= 76528), key=lambda r:r['seed'])
if args.td7:
    valid = [json.loads(s) for s in (ROOT/'research/results/s21-data.jsonl').read_text().splitlines()]
    assert len(valid)==32 and {r['seed'] for r in valid}==set(range(76600,76632))
    assert all(r['status']=='complete' and r['sha256']==rows[0]['sha256'] and r['config']=='5/5,10,10,10' for r in valid)
    valid.sort(key=lambda r:r['seed'])
prior = np.array(json.loads((ROOT/'research/results/value-fit.json').read_text())['selected']['weights'])
price = np.mean([r['pieces']/100 for r in train])
indices = [0,1,2,3,11]
types = 'IJLOSTZ'

def dataset(records):
    xs, ys, ws, bases, seeds = [], [], [], [], []
    for r in records:
        samples = json.loads((ROOT/r['trace']).read_text())['samples']
        assert len(samples) == r['pieces']
        eligible = [(i,s) for i,s in enumerate(samples)
                    if 100-s['cleared']-s['visible_garbage'] >= 20]
        assert eligible
        for i,s in eligible:
            f = np.array(s['features'])
            contrast = np.zeros(7)
            if s['hold'] is not None:
                contrast[:] = -1/7
                contrast[types.index(s['hold'])] += 1
            xs.append(np.outer(contrast, f[indices]).ravel())
            unseen = 100-s['cleared']-s['visible_garbage']
            target = r['pieces']-i-price*unseen
            if args.td7 and i+7 < len(samples):
                future = samples[i+7]
                future_unseen = 100-future['cleared']-future['visible_garbage']
                target = 7 + max(0., np.array(future['features'])@prior) + price*(future_unseen-unseen)
            ys.append(target)
            bases.append(max(0., f@prior))
            ws.append(1/len(eligible));seeds.append(r['seed'])
    return tuple(map(np.array,(xs,ys,ws,bases,seeds)))

x,y,w,base,_ = dataset(train)
scale = np.maximum(np.sqrt(np.average(x*x,axis=0,weights=w)),1e-8)
ridge = 10.
design = np.vstack([x/scale*np.sqrt(w[:,None]),np.eye(x.shape[1])*np.sqrt(ridge)])
target = np.r_[(y-base)*np.sqrt(w),np.zeros(x.shape[1])]
weights = np.linalg.lstsq(design,target,rcond=None)[0]/scale
# Collapse centered one-hot into seven rows; exact zero-mean type effect.
weights = weights.reshape(7,5)
weights -= weights.mean(axis=0)
raw = weights.ravel()
reports = {}
for label,records in [('training',train),('prediction_validation',valid)]:
    dx,dy,dw,db,ds = dataset(records)
    pred = np.maximum(0,db+dx@raw)
    game_deltas = [float(np.mean(((pred-dy)**2-(db-dy)**2)[ds==r['seed']])) for r in records]
    boot = np.random.default_rng(173).choice(game_deltas,(10000,len(records)),replace=True).mean(axis=1)
    reports[label] = dict(games=len(records),samples=len(dy),baseline_mse=float(np.average((db-dy)**2,weights=dw)),
        context_mse=float(np.average((pred-dy)**2,weights=dw)),per_game_mse_deltas=game_deltas,
        mean_mse_delta=float(np.mean(game_deltas)),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
report = dict(ridge=ridge,training_seeds=[76400,76527],prediction_validation_seeds=([76600,76631] if args.td7 else [76528,76559]),target=('7-step frozen-value bootstrap' if args.td7 else 'full-game return'),
    public_features=['held_type','bias','height','piece_estimate','i_dependencies','wells'],
    feature_indices=indices,types=types,garbage_price=float(price),weights=weights.tolist(),results=reports,
    prior_fit_sha256=hashlib.sha256((ROOT/'research/results/value-fit.json').read_bytes()).hexdigest(),
    validation_data_sha256=hashlib.sha256((ROOT/('research/results/s21-data.jsonl' if args.td7 else 'research/results/s20-data.jsonl')).read_bytes()).hexdigest(),
    source_sha256=rows[0]['sha256'],source_jsonl_sha256=hashlib.sha256((ROOT/'research/results/s20-data.jsonl').read_bytes()).hexdigest(),
    predictive_gate_passed=reports['prediction_validation']['mean_mse_delta']<0,
    limitation='Observational full-preview policy returns; not causal Hold values and not search-leaf distribution. No policy promotion from MSE.')
(ROOT/f'research/results/{prefix}-hold-fit.json').write_text(json.dumps(report,indent=2)+'\n')
out = f'// {prefix} fixed ridge10; training76400..76527 only. No full-game validation used.\n'
out += 'const HOLD_WEIGHTS: [[f64; 5]; 7] = [\n'
out += ',\n'.join('['+', '.join(format(v,'.17e') for v in row)+']' for row in weights)+'\n];\n'
(ROOT/f'research/results/{prefix}-hold-weights.rs').write_text(out)
print(json.dumps({k:v for k,v in report.items() if k not in ['weights','results']},indent=2))
for k,v in reports.items():print(k,json.dumps({a:b for a,b in v.items() if a!='per_game_mse_deltas'}))
