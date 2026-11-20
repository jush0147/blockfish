#!/usr/bin/env python3
"""Post-hoc diagnostics, not a new model-selection or independent test procedure."""
import json
import pathlib
import random
import statistics as st
ROOT = pathlib.Path(__file__).resolve().parent.parent
fit = json.loads((ROOT/'research/results/s-direct-fit.json').read_text())
train = {}
for r in map(json.loads, (ROOT/'research/results/s-direct-train.jsonl').read_text().splitlines()):
    assert r['status'] == 'complete'
    train.setdefault(r['weights_sha256'], {})[r['seed']] = r['pieces']
a = train[fit['initial']['weights_sha256']]
b = train[fit['selected']['weights_sha256']]
validation = {}
for r in map(json.loads, (ROOT/'research/results/s-direct-validation.jsonl').read_text().splitlines()):
    if r['config'] == '5/5,10,10,10':
        assert r['status'] == 'complete'
        validation.setdefault(pathlib.Path(r['binary']).name, {})[r['seed']] = r['pieces']
v = validation['learned-fast']; w = validation['learned-direct']
assert len(a)==len(b)==len(v)==len(w)==32
assert a.keys()==b.keys() and v.keys()==w.keys() and not a.keys() & v.keys()
td = [b[s]-a[s] for s in sorted(a)]
vd = [w[s]-v[s] for s in sorted(v)]
rng = random.Random(173)
gaps = sorted(st.mean(rng.choices(vd,k=32))-st.mean(rng.choices(td,k=32)) for _ in range(10000))
def stats(xs):
    return dict(mean=st.mean(xs), sd=st.stdev(xs), wins=sum(x<0 for x in xs),
                ties=xs.count(0), losses=sum(x>0 for x in xs), deltas=xs)
report = dict(training=stats(td), validation=stats(vd),
    descriptive_generalization_gap=st.mean(vd)-st.mean(td),
    conditional_bootstrap95=[gaps[250],gaps[9749]],
    caveat='Training selected among49 adaptive candidates; bootstrap does not correct selection bias. Descriptive only.',
    training_extremes=sorted([dict(seed=s,delta=b[s]-a[s]) for s in a],key=lambda r:r['delta'])[:6],
    training_half_means=[st.mean(td[:16]),st.mean(td[16:])],
    prior_train_mean=st.mean(a.values()),prior_validation_mean=st.mean(v.values()),
    selected_weights_sha256=fit['selected']['weights_sha256'])
(ROOT/'research/results/s11-generalization-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
