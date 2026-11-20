"""Run the fixed paired geometric-value study and verify all complete replays."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

results = Path("research/results")
names = ["robust-dig-geometry", "learned-robust"]
def run(script,*args):
    subprocess.run([sys.executable,"research/"+script,*map(str,args)],check=True)
for stage,start,count,gate in [("validation",77200,24,"mean"),("final",77300,64,"ci")]:
    prefix = str(results/("s32-"+stage))
    lock_path=Path(prefix+"-lock.json")
    lock=dict(seeds=list(range(start,start+count)),config="50/5,10,10,10",generated_node_cap=50000,garbage_min=4,garbage_max=9,candidate=names[0],controls=names[1:],binaries={n:hashlib.sha256(Path("research/bin",n).read_bytes()).hexdigest() for n in names})
    if lock_path.exists(): assert json.loads(lock_path.read_text())==lock
    else: lock_path.write_text(json.dumps(lock,indent=2))
    run("bench.py","--binaries",*["research/bin/"+n for n in names],"--configs",lock["config"],"--start",start,"--count",count,"--jobs",4,"--out",prefix+".jsonl")
    run("decide_cohort.py","--lock",lock_path,"--inputs",prefix+".jsonl","--out-prefix",prefix,"--gate",gate)
    rows=list(map(json.loads,Path(prefix+".jsonl").read_text().splitlines()))
    completed=Path(prefix+"-complete.jsonl")
    completed.write_text("".join(json.dumps(r)+"\n" for r in rows if r["status"]=="complete"))
    run("replay_audit.py",completed,"--out",prefix+"-replay.json")
    if any(r["status"]!="complete" for r in rows) or not json.loads(Path(prefix+"-decision.json").read_text())["passed"]: break
