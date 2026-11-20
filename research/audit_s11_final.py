#!/usr/bin/env python3
"""Audit the fixed S11 cohort against its pre-result lock, then summarize/replay."""
import json
import pathlib
import subprocess
import sys
ROOT=pathlib.Path(__file__).resolve().parent.parent
lock=json.loads((ROOT/'research/results/s11-final-lock.json').read_text())
rows=list(map(json.loads,(ROOT/'research/results/s11-final.jsonl').read_text().splitlines()))
assert len(rows)==128
seen=set()
for r in rows:
    name=pathlib.Path(r['binary']).name
    assert r['sha256']==lock['binaries'][name]
    assert r['config']==lock['config'] and r.get('move_ms') is None
    key=name,r['seed']; assert key not in seen;seen.add(key)
    trace=json.loads((ROOT/r['trace']).read_text())
    assert trace['seed']==r['seed']
    assert trace['garbage_min']==lock['garbage_min'] and trace['garbage_max']==lock['garbage_max']
    assert len(trace['inputs'])==r['pieces']
    assert all(s['generated_nodes']<=lock['generated_node_cap'] for s in trace['search'])
assert seen=={(name,s) for name in lock['binaries'] for s in lock['seeds']}
for cmd in [
    ['research/budget_summary.py','research/results/s11-final.jsonl','--expected-count','64','--out','research/results/s11-final-summary.json'],
    ['research/replay_audit.py','research/results/s11-final.jsonl','--out','research/results/s11-final-audit.json']]:
    subprocess.run([sys.executable]+cmd,cwd=ROOT,check=True)
summary=json.loads((ROOT/'research/results/s11-final-summary.json').read_text())
a=summary['groups']['learned-fast:50/5,10,10,10:None'];b=summary['groups']['learned-robust:50/5,10,10,10:None'];pair=summary['pairs'][0]
promote=b['complete']>=a['complete'] and pair.get('bootstrap95',[0,0])[1]<0
report=dict(cohort_and_budget_audit='passed',promote=promote,
            decision_rule=lock['rule'],paired_result=pair,
            total_cpu_seconds=sum(r.get('cpu_seconds',0) for r in rows))
(ROOT/'research/results/s11-final-decision.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
