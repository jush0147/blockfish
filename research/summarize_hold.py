#!/usr/bin/env python3
"""S19 descriptive allocation diagnostic; only initial states are matched."""
import json
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
files = {'baseline': 's16-control-dev', 'preserve_hold': 's19-dev'}
groups = {k: [json.loads(s) for s in (ROOT / f'research/results/{p}.jsonl').read_text().splitlines()]
          for k, p in files.items()}
report = {}
for k, rows in groups.items():
    assert len(rows) == 32 and all(r['status'] == 'complete' for r in rows)
    assert len({r['seed'] for r in rows}) == 32
    report[k] = {'games': len(rows)}
    for name in ['sequences', 'root_moves']:
        report[k][f'mean_{name}_per_move_game_weighted'] = statistics.mean(
            statistics.mean(s[name] for s in r['search']) for r in rows)
        report[k][f'mean_initial_{name}'] = statistics.mean(r['search'][0][name] for r in rows)
a = {r['seed']: r for r in groups['baseline']}
b = {r['seed']: r for r in groups['preserve_hold']}
assert a.keys() == b.keys()
report['first_move_changed'] = sum(
    json.loads((ROOT / a[s]['trace']).read_text())['inputs'][0] !=
    json.loads((ROOT / b[s]['trace']).read_text())['inputs'][0] for s in a)
report['note'] = ('Only first decisions share identical states. Later move averages are '
                  'descriptive of diverging trajectories; these are not causal evidence.')
(ROOT / 'research/results/s19-allocation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
