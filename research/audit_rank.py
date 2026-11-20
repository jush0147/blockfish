#!/usr/bin/env python3
"""Verify every full source/branch trace, preserving its nested dataset location."""
import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

ROOT=pathlib.Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser()
p.add_argument('--data',default='research/results/s-rank-data.jsonl')
p.add_argument('--out',default='research/results/s-rank-replay-audit.json')
a=p.parse_args()
results=[]
binary=ROOT/'research/bin/learned-fast'
rows=[json.loads(x) for x in (ROOT/a.data).read_text().splitlines()]
assert len(rows)==80 and all(r['status']=='collected' for r in rows)
with tempfile.TemporaryDirectory(prefix='blockfish-rank-audit-') as tmp:
    path=pathlib.Path(tmp)/'trace.json'
    for r in rows:
        data=json.loads((ROOT/r['data']).read_text())
        traces=[('source',data['source'],r['source_complete'])]
        traces += [(f'probes/{i}/candidates/{j}/trace',c['trace'],c['complete'])
                   for i,probe in enumerate(data['probes']) for j,c in enumerate(probe['candidates'])]
        for location,trace,expected_complete in traces:
            encoded=json.dumps(trace,sort_keys=True).encode()
            path.write_bytes(encoded)
            proc=subprocess.run([str(binary),'--replay-file',str(path)],capture_output=True,text=True,timeout=30)
            passed=(proc.returncode==0 if expected_complete else
                    proc.returncode!=0 and 'replay did not clear 100 garbage rows' in proc.stderr)
            results.append(dict(data=r['data'],location=location,seed=r['seed'],
                                expected_complete=expected_complete,passed=passed,
                                trace_sha256=hashlib.sha256(encoded).hexdigest(),stderr=proc.stderr))
report=dict(total=len(results),passed=sum(r['passed'] for r in results),records=results,
            verifier_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
(ROOT/a.out).write_text(json.dumps(report,indent=2)+'\n')
print(report['passed'],'/',report['total'],'source and counterfactual traces verified')
assert report['passed']==report['total']
