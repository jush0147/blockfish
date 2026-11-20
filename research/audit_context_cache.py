#!/usr/bin/env python3
"""Check S23 full policy equivalence; report pilot resource costs separately."""
import hashlib
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
rows=[json.loads(s) for s in (ROOT/'research/results/s23-parity.jsonl').read_text().splitlines()]
assert len(rows)==6 and all(r['status']=='complete' for r in rows)
groups={name:{} for name in ['robust-hold-order','robust-hold-cached']}
for row in rows:
    name=Path(row['binary']).name
    assert row['seed'] not in groups[name]
    groups[name][row['seed']]=row
    assert row['config']=='50/5,10,10,10' and row.get('move_ms') is None
    assert len(row['search'])==row['pieces']
    assert all(s['generated_nodes']<=50000 for s in row['search'])
assert all(sorted(g)==list(range(76000,76003)) for g in groups.values())
records=[]
for seed in range(76000,76003):
    a,b=[groups[name][seed] for name in groups]
    ta,tb=[json.loads((ROOT/r['trace']).read_text()) for r in [a,b]]
    assert ta['inputs']==tb['inputs'] and ta['ds']==tb['ds']
    counts=lambda r:[{k:v for k,v in s.items() if k!='seconds'} for s in r['search']]
    assert counts(a)==counts(b)
    records.append(dict(seed=seed,pieces=a['pieces'],inputs_clears_search_counts_equal=True,
                        original_cpu=a['cpu_seconds'],cached_cpu=b['cpu_seconds']))
summary={name:dict(mean_cpu=statistics.mean(r['cpu_seconds'] for r in g.values()),
                   mean_max_rss_kib=statistics.mean(r['max_rss_kib'] for r in g.values()),
                   sha256=next(iter(g.values()))['sha256']) for name,g in groups.items()}
weights=hashlib.sha256((ROOT/'blockfish-engine/src/ai/hold_weights.rs').read_bytes()).hexdigest()
lock=json.loads((ROOT/'research/results/s22-final-lock.json').read_text())
assert weights==lock['weights_sha256']
assert summary['robust-hold-order']['sha256']==lock['binaries']['robust-hold-order']
report=dict(paired_games=3,passed=3,records=records,summary=summary,weights_sha256=weights,
    note='Three development seeds, interleaved one worker with other fixed-node jobs running. Pilot CPU/RSS only; not independent piece-count improvement evidence.')
(ROOT/'research/results/s23-parity-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
