import json,subprocess
from pathlib import Path
rows=sorted((r for r in map(json.loads,Path('research/results/s26-dev.jsonl').read_text().splitlines()) if r['binary'].endswith('world-rollout-placebo')),key=lambda r:r['seed'])
assert [r['seed'] for r in rows]==list(range(76800,76832))
obs=[];ids=[]
for row in rows:
    events=json.loads(Path(row['trace']).read_text())['world_events']
    for step in (50,100):
        obs.append(json.dumps(events[step]['observation']))
        ids.append((row['seed'],step))
p=subprocess.run(['target/release/blockfish-mirror-probe'],input='\n'.join(obs)+'\n',text=True,capture_output=True,check=True)
records=list(map(json.loads,p.stdout.splitlines()))
assert len(records)==64 and all(r['generated']<=50000 for r in records)
for r,(seed,step) in zip(records,ids):r.update(seed=seed,step=step)
matched=sum(r['matched']>0 for r in records);changed=sum(r['changed'] for r in records)
report=dict(matched=matched,changed=changed,passed=matched>=48 and changed>0,records=records)
Path('research/results/s34-probe.json').write_text(json.dumps(report,indent=2))
print(matched,changed,report['passed'])
