"""Learn a local correction from paired positions in archived training games."""
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

def geometry(cells):
    rows = []
    for (r,c), _ in cells:
        while len(rows) <= r:
            rows.append(0)
        rows[r] |= 1 << c
    holes = []
    for row in rows:
        gap = 1023 ^ row
        if gap.bit_count() != 1:
            break
        holes.append(gap.bit_length()-1)
    if not holes:
        return [0]*6
    c = holes[-1]
    above = rows[len(holes):]
    cover = [i+1 for i,row in enumerate(above) if row & (1<<c)]
    sides = [min(4,max([0]+[i+1 for i,row in enumerate(above) if row & (1<<j)]))
             if 0<=j<10 else 4 for j in (c-1,c+1)]
    return [len(cover),max([0]+cover),min(sides),max(sides),int(c in (0,9)),
            abs(c-holes[-2]) if len(holes)>1 else 0]

def fit():
    results=Path('research/results')
    weights=np.array(json.loads((results/'value-fit.json').read_text())['selected']['weights'])
    rows=[r for r in map(json.loads,(results/'s26-dev.jsonl').read_text().splitlines())
          if Path(r['binary']).name=='world-rollout-placebo']
    assert sorted(r['seed'] for r in rows)==list(range(76800,76832))
    designs=[]; targets=[]
    with tempfile.TemporaryDirectory() as directory:
        samples=Path(directory)/'samples.json'
        for r in sorted(rows,key=lambda r:r['seed']):
            subprocess.run(['research/bin/context-source','--replay-file',r['trace'],'--replay-features-out',str(samples)],check=True,stdout=subprocess.DEVNULL)
            obs=json.loads(Path(r['trace']).read_text())['world_events']
            features=json.loads(samples.read_text())
            assert len(obs)==len(features)==r['pieces']
            phi=np.array([geometry(e['observation']['cells']) for e in obs])
            base=np.maximum(0,np.array([e['features'] for e in features])@weights)
            unseen=np.array([100-e['observation']['garbage_generated'] for e in obs])
            residual=r['pieces']-np.arange(r['pieces'])-base-2.083984375*unseen
            scale=np.sqrt(len(obs)-7)
            designs.append((phi[:-7]-phi[7:])/scale)
            targets.append((residual[:-7]-residual[7:])/scale)
    x=np.concatenate(designs); y=np.concatenate(targets)
    rms=np.maximum(1e-8,np.sqrt(np.sum(x*x,axis=0)/len(rows)))
    z=x/rms
    beta=np.linalg.solve(z.T@z+4*np.eye(6),z.T@y)/rms
    (results/'s32-fit.json').write_text(json.dumps(dict(weights=beta.tolist(),games=len(rows),pairs=len(y)),indent=2))
    Path('blockfish-engine/src/ai/dig_weights.rs').write_text('const DIG: [f64; 6] = '+repr(beta.tolist())+';\n')
if __name__=='__main__':
    fit()
