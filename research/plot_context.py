#!/usr/bin/env python3
"""Display exploratory hold ablations separately from fresh policy validation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
stages=[('s19-dev','Preserve Hold: development (32)'),
        ('s21-dev','Hold model + preserve: development (32)'),
        ('s22-dev','Hold model + original horizon: development (32)'),
        ('s22-validation','Hold model + original horizon: validation (48)'),
        ('s22-final','Hold model + original horizon: final (64)')]
fig,ax=plt.subplots(figsize=(11,5.7),layout='constrained')
for i,(stage,label) in enumerate(stages):
    report=json.loads((ROOT/f'research/results/{stage}-decision.json').read_text())
    p=report['candidate_minus_controls']['learned-robust']
    mean=p['delta_b_minus_a'];low,high=p['bootstrap95']
    ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=5,
                color=('#ae643a' if stage.endswith('final') else
                       '#426a91' if stage.endswith('validation') else '#888888'))
    ax.annotate(f'{mean:+.3f} [{low:+.3f}, {high:+.3f}]',(mean,i),xytext=(0,13),
                textcoords='offset points',ha='center',fontsize=9)
ax.axvline(0,color='#777777',linestyle='--')
ax.axhline(2.5,color='#aaaaaa',linewidth=.8)
ax.set_yticks(range(len(stages)),[label for _,label in stages])
ax.set_ylim(len(stages)-.4,-.6)
ax.set(xlabel='Candidate minus frozen robust: pieces (negative is better)',
       title='Hold-conditioned evaluation: full 100L outcomes at 50k nodes\nPaired means and 95% bootstrap intervals; development is exploratory')
ax.grid(axis='x',alpha=.15)
for suffix in ['svg','png']:
    fig.savefig(ROOT/f'research/results/s22-outcomes.{suffix}',dpi=160)
