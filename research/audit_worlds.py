#!/usr/bin/env python3
"""Audit actual public history and sampled futures, with independent JSON hash checks."""
import argparse,collections,hashlib,json,statistics,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--out',required=True);a=p.parse_args()
records=[];groups=collections.defaultdict(lambda:collections.defaultdict(list))
for line in (ROOT/a.input).read_text().splitlines():
    r=json.loads(line);binary=Path(r['binary']);assert binary.name in ('world-rollout','world-rollout-placebo')
    assert hashlib.sha256((ROOT/binary).read_bytes()).hexdigest()==r['sha256']
    trace=ROOT/r['trace'];t=json.loads(trace.read_text());assert r['status']=='complete'
    run=subprocess.run([str(ROOT/binary),'--world-audit',str(trace)],capture_output=True,text=True)
    assert run.returncode==0,(r['seed'],run.stdout,run.stderr)
    paths=0;g=groups[binary.name]
    for d in t['world_events']:
        h=0xcbf29ce484222325
        for b in json.dumps(d['observation'],separators=(',',':'),ensure_ascii=False).encode():
            h=((h^b)*0x100000001b3)&((1<<64)-1)
        assert h==d['public_hash']
        assert d['placebo']==binary.name.endswith('placebo')
        g['reranked'].append(d['chosen']!=0);g['base_nodes'].append(d['base_nodes'])
        g['rollout_nodes'].append(d['rollout_nodes']);g['base_roots'].append(d['base_roots'])
        g['has_two_candidates'].append(len(d['candidates'])==2)
        if len(d['candidates'])==2:
            c0,c1=d['candidates'];deltas=[b['cost']-a['cost'] for a,b in zip(c0['samples'],c1['samples'])]
            g['world_preference_disagreement'].append((deltas[0]<0)!=(deltas[1]<0))
            g['sample_delta_spread'].append(abs(deltas[0]-deltas[1]))
            g['absolute_mean_margin'].append(abs(statistics.mean(deltas)))
        for c in d['candidates']:
            g['within_candidate_world_cost_spread'].append(abs(c['samples'][0]['cost']-c['samples'][1]['cost']))
        for c in d['candidates']:
            for s in c['samples']:
                paths+=1;g['truncated'].append(s['truncated']);g['topout'].append(s['topout'])
                g['replenished'].append(s['cleared']+s['visible_garbage']>d['observation']['garbage_generated'])
                g['locks'].append(len(s['actions']))
    records.append(dict(binary=binary.name,seed=r['seed'],decisions=len(t['world_events']),paths=paths,
        trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest()))
result=dict(games=len(records),passed=len(records),records=records,
    means={k:{f:statistics.mean(v) for f,v in g.items()} for k,g in groups.items()},
    note='Rust audit replays actual public observations and all hypothetical paths; Python independently recomputes public JSON FNV hash. No fresh quality evidence from these correctness checks.')
(ROOT/a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
