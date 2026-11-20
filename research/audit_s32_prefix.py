"""Compare the occupancy proxy with public garbage colors on old development games."""
from pathlib import Path
import json
from collections import Counter
counts=Counter()
for row in map(json.loads,Path("research/results/s26-dev.jsonl").read_text().splitlines()):
    if Path(row["binary"]).name!="world-rollout-placebo":
        continue
    counts["games"]+=1
    for event in json.loads(Path(row["trace"]).read_text())["world_events"]:
        cells=event["observation"]["cells"]
        occupied=Counter(r for (r,c),color in cells)
        actual=max((r+1 for (r,c),color in cells if color=="G"),default=0)
        proxy=0
        while occupied[proxy]==9:
            proxy+=1
        counts["boards"]+=1
        counts["over" if proxy>actual else "under" if proxy<actual else "equal"]+=1
Path("research/results/s33-prefix.json").write_text(json.dumps(counts,indent=2))
print(counts)
