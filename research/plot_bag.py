#!/usr/bin/env python3
"""Show S18 cohort separation; never pool development with held-out evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
stages = [('dev', 'Development (32)'), ('validation', 'Validation (48)'),
          ('final', 'Final (64)')]
fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
for i, (stage, label) in enumerate(stages):
    summary = json.loads((ROOT / f'research/results/s18-{stage}-summary.json').read_text())
    pair = next(p for p in summary['pairs']
                if p['a'].startswith('robust-bag-placebo:')
                and p['b'].startswith('robust-bag:'))
    mean = pair['delta_b_minus_a']
    low, high = pair['bootstrap95']
    ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', capsize=5,
                color='#888888' if stage == 'dev' else '#426a91')
    ax.annotate(f'{mean:+.3f} [{low:+.3f}, {high:+.3f}]', (mean, i),
                xytext=(0, 14), textcoords='offset points', ha='center')
ax.axvline(0, color='#777777', linestyle='--')
ax.set_yticks(range(3), [label for _, label in stages])
ax.set_ylim(2.6, -.6)
ax.set(xlabel='Inference minus robust control: pieces (negative is better)',
       title='S18: development gain did not reproduce\n50k nodes; paired means and 95% bootstrap intervals')
ax.grid(axis='x', alpha=.15)
for suffix in ['svg', 'png']:
    fig.savefig(ROOT / f'research/results/s18-outcomes.{suffix}', dpi=160)
