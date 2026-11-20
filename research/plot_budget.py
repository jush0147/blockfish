#!/usr/bin/env python3
"""Plot a completed four-arm budget experiment from its audited summary."""
import argparse
import json
import os
import pathlib

p = argparse.ArgumentParser()
p.add_argument('summary')
args = p.parse_args()
path = pathlib.Path(args.summary)
r = json.loads(path.read_text())
os.environ.setdefault('MPLCONFIGDIR','/tmp/blockfish-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1,2,figsize=(10,4),layout='constrained')
for name,label,color in [('quality-baseline','Original evaluator','#b8784b'),
                         ('learned-fast','Frozen learned evaluator','#237d72')]:
    gs = [r['groups'][f'{name}:{b}/5,10,10,10:None'] for b in [50,200]]
    assert all(g['complete']==g['total'] for g in gs)
    axes[0].plot([50,200],[g['mean_pieces'] for g in gs],marker='o',label=label,color=color)
    axes[1].plot([g['mean_cpu_seconds'] for g in gs],[g['mean_pieces'] for g in gs],
                 marker='o',label=label,color=color)
axes[0].set(xlabel='Generated node cap per move (thousands)',ylabel='Mean pieces / 100L',xticks=[50,200])
axes[1].set(xlabel='Mean CPU seconds / 100L',ylabel='Mean pieces / 100L')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(alpha=.15)
axes[0].legend(frameon=False,fontsize=9)
n=next(iter(r['groups'].values()))['total']
fig.suptitle(f'100L Cheese: {n} paired seeds | fixed information and rules')
fig.savefig(path.with_suffix('.png'),dpi=180)
