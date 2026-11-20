"""Run both schedulers on identical endgame observations and replay each returned path."""
import hashlib
import json
import resource
import subprocess
from pathlib import Path

results = Path("research/results")
audit = json.loads((results / "s34-replay-audit.json").read_text())
positions = []
for rec in sorted(audit["records"], key=lambda r:r["seed"]):
    raw = Path(rec["trace"]).read_bytes()
    assert rec["returncode"] == 0 and hashlib.sha256(raw).hexdigest() == rec["sha256"]
    events = json.loads(raw)["world_events"]
    selected = {next(i for i,e in enumerate(events)
                     if e["observation"]["garbage_generated"]==100 and e["cleared_before"]>=threshold)
                for threshold in (95,97)}
    for i in sorted(selected):
        positions.append((rec["seed"],i,events[i]["observation"],100-events[i]["cleared_before"]))
assert sorted({s for s,_,_,_ in positions}) == list(range(76800,76832))
payload = "".join(json.dumps(o)+"\n" for _,_,o,_ in positions)
summary = {}
for variant in ("control","candidate"):
    binary = Path(f"research/bin/s37-path-{variant}")
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    run = subprocess.run([str(binary)],input=payload,text=True,stdout=subprocess.PIPE,check=True)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    rows = list(map(json.loads,run.stdout.splitlines()))
    assert len(rows)==len(positions)
    for r,(seed,step,_,remaining) in zip(rows,positions):
        assert r["remaining"]==remaining
        r.update(seed=seed,step=step)
    counts=[sum(p["solved"] for p in r["paths"]) for r in rows]
    summary[variant]=dict(positions=len(rows),solved_roots=sum(counts),covered=sum(c>0 for c in counts),
        invalid=sum(not p["valid"] for r in rows for p in r["paths"]),
        cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    (results/f"s37-path-{variant}.json").write_text(json.dumps(rows)+"\n")
a,b=summary["control"],summary["candidate"]
summary["passed"]=b["solved_roots"]>a["solved_roots"] and b["covered"]>=a["covered"]
(results/"s37-path-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(summary)
