#!/usr/bin/env python3
"""Bounded S28 descriptive diagnostics. Source states, not paths, are sampling units."""
import hashlib,json,statistics,subprocess
from pathlib import Path
rows=list(map(json.loads,Path('research/results/s28-labels.jsonl').read_text().splitlines()))
lock=json.loads(Path('research/results/s28-label-lock.json').read_text())
assert sorted(r['seed'] for r in rows)==lock['seeds']
assert all(r['status']=='complete' for r in rows)
assert hashlib.sha256(Path('research/bin/public-labels').read_bytes()).hexdigest()==lock['binary_sha256']
records=[];complete=0;paths=0;strong_nodes=0;weak_nodes=0;state_nodes=0;base_nodes=0;weak_truncated=0
sign=lambda x:(x>0)-(x<0)
for r in rows:
 assert hashlib.sha256(Path(r['output']).read_bytes()).hexdigest()==r['output_sha256']
 subprocess.run(['research/bin/public-labels','--audit',r['output']],check=True,capture_output=True,text=True)
 d=json.loads(Path(r['output']).read_text());base_nodes+=d['base_nodes']
 src=json.loads(Path(d['source']).read_text());assert d['roots'][0]['inputs']==src['inputs'][100]
 h=0xcbf29ce484222325
 for b in json.dumps(d['observation'],separators=(',',':'),ensure_ascii=False).encode():h=((h^b)*0x100000001b3)&((1<<64)-1)
 assert h==d['public_hash']
 for root in d['roots']:
  for sample in root['samples']:
   paths+=1;complete+=sample['strong']['complete'];weak_truncated+=sample['weak']['truncated']
   strong_nodes+=sum(sample['strong']['search']);weak_nodes+=sum(sample['weak']['search'])
   state_nodes+=2+len(sample['strong']['actions'])+len(sample['weak']['actions'])
 if len(d['roots'])<2:
  records.append(dict(seed=r['seed'],paired=False));continue
 a,b=d['roots']
 full=[y['strong']['cost']-x['strong']['cost'] for x,y in zip(a['samples'],b['samples'])]
 strong=[y['strong']['short_cost']-x['strong']['short_cost'] for x,y in zip(a['samples'],b['samples'])]
 weak=[y['weak']['cost']-x['weak']['cost'] for x,y in zip(a['samples'],b['samples'])]
 delta=statistics.mean(full);left=statistics.mean(full[:2]);right=statistics.mean(full[2:])
 records.append(dict(seed=r['seed'],paired=True,full_world_deltas=full,strong_short_world_deltas=strong,weak_short_world_deltas=weak,
  full_mean_delta=delta,strong_short_mean_delta=statistics.mean(strong),weak_short_mean_delta=statistics.mean(weak),
  half_means=[left,right],half_signs=[sign(left),sign(right)],
  strict_half_agreement=sign(left)==sign(right)!=0,
  strong_short_agrees=sign(statistics.mean(strong))==sign(delta)!=0,
  weak_short_agrees=sign(statistics.mean(weak))==sign(delta)!=0,
  full_tie=delta==0,short_policy_difference=statistics.mean(weak)-statistics.mean(strong)))
paired=[r for r in records if r['paired']]
result=dict(source_states=len(rows),paired_states=len(paired),full_paths=paths,full_completed=complete,
 weak_truncated=weak_truncated,base_nodes=base_nodes,strong_nodes=strong_nodes,weak_nodes=weak_nodes,simulated_state_nodes=state_nodes,
 total_generated_accounting=base_nodes+strong_nodes+weak_nodes+state_nodes,
 cpu_seconds=sum(r['cpu_seconds'] for r in rows),max_rss_kib=max(r['max_rss_kib'] for r in rows),
 strict_half_agreement=sum(r['strict_half_agreement'] for r in paired),
 strong_short_agreement=sum(r['strong_short_agrees'] for r in paired),
 weak_short_agreement=sum(r['weak_short_agrees'] for r in paired),full_mean_ties=sum(r['full_tie'] for r in paired),
 records=records,note='Post-preregistered fixed pilot, descriptive only. Four sampled worlds do not establish true action values; common simulator/model bias remains. No deployed policy selected, no test-set access, no treating conditional paths as independent games.')
Path('research/results/s28-label-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
