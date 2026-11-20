#!/usr/bin/env python3
"""Plot complete paired S11 cohorts; validation and final remain separate."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
records=list(map(json.loads,(ROOT/'research/results/s11-final.jsonl').read_text().splitlines()))
control={r['seed']:r['pieces'] for r in records if r['binary'].endswith('/learned-fast')}
candidate={r['seed']:r['pieces'] for r in records if r['binary'].endswith('/learned-robust')}
assert len(control)==len(candidate)==64 and control.keys()==candidate.keys()
assert all(r['status']=='complete' for r in records)
budget=json.loads((ROOT/'research/results/s11-budget-summary.json').read_text())
final=json.loads((ROOT/'research/results/s11-final-summary.json').read_text())
points=[]
for cap in [5,50]:
    points.append(next(p for p in budget['pairs'] if p['a']==f'learned-fast:{cap}/5,10,10,10:None' and p['b']==f'learned-robust:{cap}/5,10,10,10:None'))
points.append(final['pairs'][0])
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
seeds=sorted(control);xs=[control[s] for s in seeds];ys=[candidate[s] for s in seeds]
ax.scatter(xs,ys,c=['#137d68' if y<x else '#bc674a' for x,y in zip(xs,ys)],alpha=.7,s=32)
lo=min(xs+ys)-3;hi=max(xs+ys)+3;ax.plot([lo,hi],[lo,hi],color='#888',linestyle='--')
ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Frozen learned baseline: pieces',ylabel='Robust backup: pieces',title='Final 50k: 64 paired seeds, all complete')
ax.set_aspect('equal');ax.grid(alpha=.15)
for i,p in enumerate(points):
    mean=p['delta_b_minus_a'];low,high=p['bootstrap95']
    bx.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=5,color='#137d68' if i==2 else '#557b9c')
    bx.annotate(f'{mean:+.2f} [{low:+.2f}, {high:+.2f}]',(mean,i),xytext=(0,13),textcoords='offset points',ha='center',fontsize=9)
bx.axvline(0,color='#888',linestyle='--');bx.set_yticks(range(3),['Validation 5k (48)','Validation 50k (48)','Final 50k (64)']);bx.invert_yaxis();bx.set_ylim(2.6,-.6)
bx.set(xlabel='Candidate minus control pieces; negative is better',title='Paired means and 95% bootstrap intervals');bx.grid(axis='x',alpha=.15)
fig.savefig(ROOT/'research/results/s11-outcomes.svg')
fig.savefig(ROOT/'research/results/s11-outcomes.png',dpi=160)
