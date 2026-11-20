#!/usr/bin/env python3
"""Describe training coverage only; do not confuse states with independent games."""
import collections
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
rows=[json.loads(s) for s in (ROOT/'research/results/s20-data.jsonl').read_text().splitlines()]
counts=collections.Counter();games=collections.Counter()
train=[r for r in rows if 76400<=r['seed']<=76527]
assert len(train)==128 and len({r['seed'] for r in train})==128
for r in train:
    seen=set()
    for sample in json.loads((ROOT/r['trace']).read_text())['samples']:
        if 100-sample['cleared']-sample['visible_garbage']<20:continue
        hold=sample['hold'] or 'empty';counts[hold]+=1;seen.add(hold)
    games.update(seen)
report=dict(training_seeds=[76400,76527],samples_by_hold=dict(sorted(counts.items())),
            games_containing_hold=dict(sorted(games.items())),
            note='Correlated executed states; sample count is not independent sample size.')
(ROOT/'research/results/s20-hold-coverage.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
