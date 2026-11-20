"""Recalculate paired diagnostics and the preregistered race gate."""
import json
import random
from pathlib import Path
from statistics import mean

base = Path("research/results")
fields = ("seed", "model", "backup", "preview", "nodes")
records = {tuple(row[k] for k in fields):row for row in json.loads((base/"s36-factorial.json").read_text())}
assert len(records) == 128
effects = {}
for i in range(1,5):
    pairs = [(v,w) for k,v in records.items() for l,w in records.items()
             if k[:i] == l[:i] and k[i+1:] == l[i+1:] and k[i] < l[i]]
    assert len(pairs) == 64 and all(a["actions"] and b["actions"] for a,b in pairs)
    effects[fields[i]] = sum(a["actions"][0] != b["actions"][0] for a,b in pairs)

rows = list(map(json.loads,(base/"s36-dev.jsonl").read_text().splitlines()))
assert len(rows)==16
variants = {}
for row in rows:
    assert row["status"]=="complete" and row["config"]=="50/5,10,10,10"
    assert row["cleared"]==100 and len(row["search"])==row["pieces"]
    assert all(s["generated_nodes"]<=50000 for s in row["search"])
    variants.setdefault(Path(row["binary"]).name,{})[row["seed"]]=row

a,b=(variants[name] for name in ("learned-robust","robust-classic-guide"))
seeds=list(range(77800,77808))
assert sorted(a)==sorted(b)==seeds
delta=[b[s]["pieces"]-a[s]["pieces"] for s in seeds]
rng=random.Random(36)
means=sorted(mean(rng.choices(delta,k=len(delta))) for _ in range(20000))
result=dict(changed_in_64_contrasts=effects,variants={
    name:dict(pieces=[games[s]["pieces"] for s in seeds],mean=mean(r["pieces"] for r in games.values()),
              mean_cpu=mean(r["cpu_seconds"] for r in games.values())) for name,games in variants.items()},
    deltas=delta,mean_delta=mean(delta),paired_percentile_ci=[means[499],means[19499]],
    passed=mean(delta)<=-1 and sum(d<0 for d in delta)>=6)
(base/"s36-summary.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
