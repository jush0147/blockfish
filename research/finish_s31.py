"""Resume S31's frozen confirmation stages and archive the completed study."""
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
import os
os.chdir(root)
results = Path("research/results")
def run(script, *args):
    subprocess.run([sys.executable, "research/"+script, *map(str,args)], check=True)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    return json.loads(Path(path).read_text())
def summarize(stage, gate):
    prefix = results / ("s31-"+stage)
    rows = [json.loads(line) for line in prefix.with_suffix(".jsonl").read_text().splitlines()]
    run("decide_cohort.py","--lock",str(prefix)+"-lock.json","--inputs",prefix.with_suffix(".jsonl"),"--out-prefix",prefix,"--gate",gate)
    candidate = Path(str(prefix)+"-candidate.jsonl")
    candidate.write_text("".join(json.dumps(r)+"\n" for r in rows if Path(r["binary"]).name=="robust-plan-reuse"))
    run("audit_plan.py",candidate,"--out",str(prefix)+"-audit.json")
    complete = Path(str(prefix)+"-complete.jsonl")
    complete.write_text("".join(json.dumps(r)+"\n" for r in rows if r["status"]=="complete"))
    run("replay_audit.py",complete,"--out",str(prefix)+"-replay.json")
    return read(str(prefix)+"-decision.json")

if __name__=="__main__":
    validation=summarize("val","mean")
    stages=["dev","val"]
    if validation["passed"]:
        lock=read(results/"s31-val-lock.json")
        lock["seeds"]=list(range(75400,75464))
        for name,h in lock["binaries"].items():
            assert digest(Path("research/bin")/name)==h
        path=results/"s31-final-lock.json"
        if path.exists(): assert read(path)==lock
        else: path.write_text(json.dumps(lock,indent=2)+"\n")
        run("bench.py","--binaries",*["research/bin/"+name for name in lock["binaries"]],"--configs",lock["config"],"--start",75400,"--count",64,"--jobs",4,"--out",results/"s31-final.jsonl")
        summarize("final","ci")
        stages.append("final")
    lines=["S31 完成。每步上限 50,000 個生成節點。"]
    for stage in stages:
        prefix=results/("s31-"+stage)
        d=read(str(prefix)+"-decision.json")
        costs=read(str(prefix)+"-summary.json")["groups"]
        lines.append("\n"+stage+"（配對盤面）")
        for name,sc in d["scores"].items():
            cpu=next(v["mean_cpu_seconds"] for k,v in costs.items() if k.startswith(name+":"))
            lines.append(f"- {name} 完成 {sc['complete']} 局，平均 {sc['mean_pieces']:.4f} 塊，CPU {cpu:.2f} 秒／完成局。")
        for name,pair in d["candidate_minus_controls"].items():
            lines.append(f"- 候選減 {name} 平均 {pair['delta_b_minus_a']:+.4f} 塊，95% 配對區間 {pair['bootstrap95']}。")
    promoted="final" in stages and read(results/"s31-final-decision.json")["passed"]
    lines.append("\n"+("正式雙控制門檻通過，保留新策略。" if promoted else "未達升級門檻，最佳版本繼續採用 cheese-robust。"))
    lines.append("已開啟的資料包括 "+", ".join(stages)+"。詳見各階段 lock、decision、audit 與 replay JSON。")
    report="\n".join(lines)+"\n"
    p=Path("research/PLAN_REUSE_REPORT.md")
    p.write_text(p.read_text().split("## 結果")[0]+"## 結果\n\n"+report)
    p=Path("research/CONTINUE.md")
    p.write_text("# S31 完成\n\n"+report+"\nCheckpoint 位於 research/checkpoints/s31-source.tar.gz，雜湊清單為 s31-manifest.json。\n\n"+p.read_text().split("--- 以下S30歷史完成狀態 ---")[-1])
    with Path("research/PROTOCOL.md").open("a") as f: f.write("\n"+report)
    archive=Path("research/checkpoints/s31-source.tar.gz")
    files=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"]).decode().split("\0")
    with tarfile.open(archive,"w:gz") as t:
        for name in sorted(set(files)):
            if name and Path(name).is_file() and not name.startswith(("research/checkpoints/","research/results/","research/bin/","research/__pycache__/")): t.add(name,recursive=False)
    artifacts=[archive,*results.glob("s31-*"),*[Path("research/bin")/n for n in ("robust-plan-reuse","robust-plan-observer","learned-robust","learned-fast")]]
    manifest={str(p):digest(p) for p in artifacts if p.is_file()}
    Path("research/checkpoints/s31-manifest.json").write_text(json.dumps({"promoted":promoted,"files":manifest},indent=2)+"\n")
    assert all(digest(p)==h for p,h in manifest.items())
    print(report,flush=True)
