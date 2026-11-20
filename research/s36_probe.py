"""Read S28 public observations only into each search process."""
import json
import subprocess
from pathlib import Path
from statistics import mean

root = Path("research/results")
sources = [json.loads((root / f"s28-label-{s}.json").read_text()) for s in range(76900,76908)]
payload = "".join(json.dumps(d["observation"])+"\n" for d in sources)
records = []
for model in ["classic", "learned"]:
    for backup in ["single", "robust"]:
        binary = f"research/bin/s36-{model}-{backup}"
        run = subprocess.run([binary], input=payload, capture_output=True, text=True, check=True)
        rows = list(map(json.loads, run.stdout.splitlines()))
        assert len(rows) == len(sources)*4
        for i, row in enumerate(rows):
            d = sources[i//4]
            roots = [r["inputs"] for r in d["roots"]]
            assert len(roots)==2
            ranks = [row["actions"].index(r) if r in row["actions"] else None for r in roots]
            choice = roots.index(row["actions"][0]) if row["actions"] and row["actions"][0] in roots else None
            records.append(dict(seed=76900+i//4,model=model,backup=backup, ranks=ranks, choice=choice,
                                world_costs=[[s["strong"]["cost"] for s in r["samples"]] for r in d["roots"]], **row))
assert all(r["choice"]==0 for r in records if r["model"]=="learned" and r["backup"]=="robust" and r["nodes"]==50000 and r["preview"]==5)
(root/"s36-factorial.json").write_text(json.dumps(records,indent=1)+"\n")
