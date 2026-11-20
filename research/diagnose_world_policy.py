#!/usr/bin/env python3
"""Describe continuation-policy mismatch on identical public states; no quality gate."""
import argparse,json,subprocess,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--binary',default='research/bin/world-diagnostic');p.add_argument('--out',required=True);a=p.parse_args()
records=[]
for line in Path(a.input).read_text().splitlines():
 r=json.loads(line)
 if Path(r['binary']).name!='world-rollout-placebo':continue
 d=json.loads(subprocess.check_output([a.binary,r['trace']],text=True));d['seed']=r['seed'];records.append(d)
keys=['examined','matched_public_states','different_inputs','different_outcomes'];totals={k:sum(r[k] for r in records) for k in keys}
result=dict(records=records,totals=totals,binary_sha256=hashlib.sha256(Path(a.binary).read_bytes()).hexdigest(),
 outcome_mismatch_fraction=totals['different_outcomes']/totals['matched_public_states'],
 note='Post-hoc diagnostic, only first follow-up after original base action; sampled states must exactly match actual next board/current/preview/Hold. Outcomes normalize alternative input paths. Samples are not independent games. Mismatch does not prove which action is better or that more rollout budget would improve 100L.')
Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
