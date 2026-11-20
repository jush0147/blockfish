#!/usr/bin/env python3
import concurrent.futures,hashlib,json,os,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
os.chdir(ROOT)
rows=[json.loads(x) for x in Path('research/results/s28-source.jsonl').read_text().splitlines()]
assert sorted(r['seed'] for r in rows)==list(range(76900,76908))
assert all(r['status']=='complete' and r['config']=='50/5,10,10,10' for r in rows)
binary=Path('research/bin/public-labels');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lock=dict(seeds=list(range(76900,76908)),source_sha256={str(r['seed']):sha(r['trace']) for r in rows},binary_sha256=sha(binary),source_binary_sha256=rows[0]['sha256'],prefix_locks=100,worlds=4,roots=2,strong_budget=50000,weak_budget=1040,short_horizon=4,max_locks=400)
lockpath=Path('research/results/s28-label-lock.json')
if lockpath.exists():assert json.loads(lockpath.read_text())==lock
else:lockpath.write_text(json.dumps(lock,indent=2)+'\n')
out=Path('research/results/s28-labels.jsonl');done={r['seed'] for r in map(json.loads,out.read_text().splitlines())} if out.exists() else set()
def run(r):
 path=Path(f"research/results/s28-label-{r['seed']}.json");usage=path.with_suffix('.usage.json')
 env=os.environ.copy();env.pop('BLOCKFISH_VALUE_WEIGHTS',None)
 cmd=['research/bin/measure',str(usage),str(binary),r['trace'],str(path)]
 start=time.monotonic();p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,env=env)
 try:
  stdout,stderr=p.communicate(timeout=600);status='complete' if p.returncode==0 else 'error'
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate();status='timeout'
 result=dict(seed=r['seed'],status=status,output=str(path),wall_seconds=time.monotonic()-start,returncode=p.returncode,stdout=stdout,stderr=stderr,binary_sha256=lock['binary_sha256'])
 if usage.exists():
  u=json.loads(usage.read_text().splitlines()[-1]);result.update(cpu_seconds=u['user_seconds']+u['system_seconds'],max_rss_kib=u['max_rss_kib'])
 if status=='complete':
  d=json.loads(path.read_text());source=json.loads(Path(r['trace']).read_text())
  assert d['roots'][0]['inputs']==source['inputs'][100], 'strong base action differs from frozen source policy'
  result['output_sha256']=sha(path);result['base_action_matches_frozen']=True
 return result
with out.open('a') as f,concurrent.futures.ThreadPoolExecutor(4) as pool:
 for result in pool.map(run,[r for r in rows if r['seed'] not in done]):
  f.write(json.dumps(result)+'\n');f.flush();print(json.dumps({k:result[k] for k in ['seed','status','wall_seconds']}),flush=True)
