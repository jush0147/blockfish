"""Resume S38 from completed games. Run only one driver at a time from the repo root."""
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from statistics import mean, median, stdev

RESULTS = Path("research/results")
BASE = "learned-robust"
CANDIDATES = ["robust-terminal-priority", "robust-classic-guide"]
CONFIG = "50/5,10,10,10"
VERIFIER = Path("research/bin/learned-fast")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    path = RESULTS / f"s38-{name}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def read_hashes():
    hashes = {}
    for line in (RESULTS / "s38-binaries.sha256").read_text().splitlines():
        digest, binary = line.split()
        assert sha(binary) == digest
        hashes[Path(binary).name] = digest
    return hashes


def batch(stage, start, count, names, hashes):
    path = RESULTS / f"s38-{stage}.jsonl"
    subprocess.run([sys.executable, "research/bench.py", "--binaries",
                    *[f"research/bin/{n}" for n in names], "--configs", CONFIG,
                    "--start", str(start), "--count", str(count), "--jobs", "4",
                    "--timeout", "600", "--out", str(path)], check=True,
                   stdout=subprocess.DEVNULL)
    rows = {name:{} for name in names}
    audits = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        name = Path(row["binary"]).name
        seed = row["seed"]
        assert seed not in rows[name] and row["sha256"] == hashes[name]
        assert row["config"] == CONFIG and row["move_ms"] is None and row["value_weights"] is None
        rows[name][seed] = row
        if row["status"] != "complete":
            continue
        trace = Path(row["trace"])
        data = json.loads(trace.read_text())
        assert data["ai_config"] == CONFIG and data["move_ms"] is None
        assert (data["seed"], data["garbage_min"], data["garbage_max"]) == (seed,4,9)
        assert len(data["inputs"]) == len(data["ds"]) == len(data["search"]) == row["pieces"]
        assert data["ds"][-1] == row["cleared"] == 100 and data["search"] == row["search"]
        assert all(0<s["generated_nodes"]<=50000 and s["root_moves"]>0 for s in data["search"])
        result = subprocess.run([str(VERIFIER), "--replay-file", str(trace)],
                                capture_output=True, text=True, timeout=30)
        audits.append(dict(trace=str(trace), sha256=sha(trace), returncode=result.returncode,
                           stdout=result.stdout.strip(), stderr=result.stderr.strip()))
        if result.returncode:
            row["status"] = "replay_failed"
    save(f"{stage}-replays", dict(verifier_sha256=sha(VERIFIER), records=audits))
    assert all(sorted(group)==list(range(start,start+count)) for group in rows.values())
    assert all(r["status"]=="complete" for r in rows[BASE].values()), "baseline failure"
    return rows


def describe(group):
    good = [r for r in group.values() if r["status"]=="complete"]
    return dict(total=len(group), completed=len(good),
                mean_pieces=mean(r["pieces"] for r in good) if good else None,
                best=min((r["pieces"] for r in good),default=None),
                under_162=sum(r["pieces"]<162 for r in good),
                total_cpu=sum(r["cpu_seconds"] for r in group.values()))


def compare(base, candidate):
    assert base.keys() == candidate.keys()
    a, b = describe(base), describe(candidate)
    result = dict(baseline=a, candidate=b, cpu_ratio=b["total_cpu"]/a["total_cpu"])
    if b["completed"] != b["total"]:
        return dict(result, eligible=False)
    diff = [candidate[s]["pieces"] - base[s]["pieces"] for s in sorted(base)]
    rng = random.Random(38)
    means = sorted(sum(rng.choices(diff,k=len(diff)))/len(diff) for _ in range(50000))
    return dict(result, eligible=True, seeds=sorted(base), deltas=diff, mean=mean(diff),
                median=median(diff), sd=stdev(diff), wins=sum(d<0 for d in diff),
                ties=diff.count(0), ci=[means[1249],means[48749]])


def main():
    hashes = read_hashes()
    first = batch("first",78300,32,[BASE,*CANDIDATES],hashes)
    estimates = {n:compare(first[BASE],first[n]) for n in CANDIDATES}
    save("first-summary",estimates)
    extend = [n for n,c in estimates.items() if c["eligible"] and c["mean"]<0<=c["ci"][1]]
    if extend:
        second = batch("second",78332,32,[BASE,*extend],hashes)
        for n in extend:
            estimates[n] = compare({**first[BASE],**second[BASE]}, {**first[n],**second[n]})
    finalists = [n for n,c in estimates.items() if c["eligible"] and c["mean"]<0]
    selected = max(finalists, key=lambda n:-estimates[n]["mean"]/max(1,estimates[n]["cpu_ratio"]),
                   default=None)
    save("selection",dict(estimates=estimates,extended=extend,selected=selected))
    confirmed = None
    promoted = False
    if selected:
        final = batch("final",78400,64,[BASE,selected],hashes)
        confirmed = compare(final[BASE],final[selected])
        promoted = (confirmed["eligible"] and confirmed["mean"]<=-0.5 and
                    confirmed["ci"][1]<0 and confirmed["cpu_ratio"]<=2)
    save("decision",dict(selected=selected,confirmed=confirmed,promoted=promoted))
    print("S38 finished:",selected,promoted)


if __name__ == "__main__":
    main()
