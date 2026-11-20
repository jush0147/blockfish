"""Audit S37 path coverage and paired race outcomes; preserve preregistered gates."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("stage",choices=["dev","confirm"])
stage = parser.parse_args().stage
base = Path("research/results")
start, count = (78100,12) if stage=="dev" else (78200,64)
seeds = list(range(start,start+count))
rows = [json.loads(l) for l in (base/f"s37-{stage}.jsonl").read_text().splitlines()]
variants = ("learned-robust","robust-terminal-priority")
replay = json.loads((base/f"s37-{stage}-replay.json").read_text())
audited = {r["trace"]:r for r in replay["records"]}
assert replay["passed"] == replay["total"] == len(rows) == 2*count
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

by_variant = {name:{} for name in variants}
stats = {}
for row in rows:
    name = Path(row["binary"]).name
    seed = row["seed"]
    assert row["status"]=="complete" and row["pieces"]==len(row["search"])
    assert row["sha256"]==digest(row["binary"])
    assert audited[row["trace"]]["sha256"]==digest(row["trace"])
    trace=json.loads(Path(row["trace"]).read_text())
    assert trace["garbage_min"]==4 and trace["garbage_max"]==9
    assert row["config"]==trace["ai_config"]=="50/5,10,10,10" and trace["move_ms"] is None
    assert trace["ds"][-1]==100 and len(trace["inputs"])==row["pieces"]
    assert all(0<s["generated_nodes"]<=50000 and s["root_moves"]>0 for s in row["search"])
    by_variant[name][seed]=row

for name,results in by_variant.items():
    assert sorted(results)==seeds
    stats[name]=dict(mean_pieces=mean(r["pieces"] for r in results.values()),
                     best=min(r["pieces"] for r in results.values()),
                     mean_cpu=mean(r["cpu_seconds"] for r in results.values()))
a,b=[by_variant[name] for name in variants]
diff=[b[s]["pieces"]-a[s]["pieces"] for s in seeds]
rng=random.Random(37)
means=sorted(mean(rng.choices(diff,k=count)) for _ in range(20000))
wins=sum(d<0 for d in diff)
out=dict(stats=stats,paired_differences=diff,mean_delta=mean(diff),wins=wins,ties=diff.count(0),
         percentile_95=[means[499],means[19499]],replayed=len(rows),
         passed=(wins>=7 and mean(diff)<=-1) if stage=="dev" else means[19499]<0)
(base/f"s37-{stage}-decision.json").write_text(json.dumps(out,indent=2)+"\n")
print(out)
