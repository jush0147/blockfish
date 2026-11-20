#!/usr/bin/env python3
"""Apply the preregistered three-arm validation gate without tuning or seed filtering."""
import json
from pathlib import Path
import statistics
import subprocess
import sys
ROOT=Path(__file__).resolve().parent.parent
lock=json.loads((ROOT/'research/results/s17-validation-lock.json').read_text())
inputs=['research/results/s17-validation.jsonl','research/results/s17-legacy-validation.jsonl']
rows=[json.loads(l) for name in inputs for l in (ROOT/name).read_text().splitlines()]
assert len(rows)==144, 'Wait for all predefined validation games.'
groups={name:{} for name in lock['binaries']}
for r in rows:
    name=Path(r['binary']).name
    assert r['sha256']==lock['binaries'][name]
    assert r['config']==lock['config'] and r.get('move_ms') is None
    assert r['seed'] not in groups[name]
    groups[name][r['seed']]=r
    assert all(s['generated_nodes']<=lock['generated_node_cap'] for s in r.get('search',[]))
assert all(sorted(g)==lock['seeds'] for g in groups.values())
subprocess.run([sys.executable,'research/budget_summary.py',*inputs,'--expected-count','48',
    '--out','research/results/s17-validation-summary.json'],cwd=ROOT,check=True)
scores={name:dict(complete=sum(r['status']=='complete' for r in g.values()),
    mean_pieces=statistics.mean(r['pieces'] for r in g.values() if r['status']=='complete'))
    for name,g in groups.items()}
candidate=scores[lock['candidate']]
checks={name:dict(completion_no_worse=candidate['complete']>=scores[name]['complete'],
                 mean_lower=candidate['mean_pieces']<scores[name]['mean_pieces']) for name in lock['controls']}
advance=all(all(c.values()) for c in checks.values())
report=dict(advance_to_final=advance,rule=lock['gate'],scores=scores,checks=checks,
            final_seeds_if_advanced=[75200,75263])
(ROOT/'research/results/s17-validation-decision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
