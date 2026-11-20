"""Archive S33 after the preregistered confirmation and replay audit."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    return json.loads(Path(path).read_text())
results=Path("research/results")
report=Path("research/REVEAL_BOUNDARY_REPORT.md")
lines=["## 完整對局結果",""]
stages=["validation"]
if (results/"s33-final-lock.json").exists(): stages.append("final")
for stage in stages:
    prefix=str(results/("s33-"+stage))
    decision=load(prefix+"-decision.json")
    summary=load(prefix+"-summary.json")
    replay=load(prefix+"-replay.json")
    rows=list(map(json.loads,Path(prefix+".jsonl").read_text().splitlines()))
    assert replay["passed"]==sum(r["status"]=="complete" for r in rows)
    assert all(r["sha256"]==load(prefix+"-lock.json")["binaries"][Path(r["binary"]).name] for r in rows)
    for group,stats in summary["groups"].items():
        lines.append(f"- {stage} {group.split(':')[0]}，{stats['complete']}/{stats['total']} 完成，平均 {stats['mean_pieces']:.4f} 塊與 {stats['mean_cpu_seconds']:.2f} CPU 秒。")
    diff=decision["candidate_minus_controls"]["learned-robust"]
    lines.append(f"- 候選減基準 {diff['delta_b_minus_a']:+.4f} 塊，配對 95% 區間 {diff['bootstrap95']}。{replay['passed']} 局成功回放。")
passed=stage=="final" and decision["passed"]
lines.append("\n"+("S33 通過預註冊升級門檻，保留原 cheese-robust 作比較。" if passed else "S33 未達升級門檻。cheese-robust 繼續作為最佳版本。"))
result="\n".join(lines)+"\n"
report.write_text(report.read_text().split("## 完整對局結果")[0]+result)
p=Path("research/CONTINUE.md")
p.write_text("# S33 完成\n\n"+result+"\n77400..77423 已揭曉。正式集 77500..77563 的使用狀態見上列階段。原始逐盤紀錄、鎖定檔與回放在 results/s33-*，重現指令在 REVEAL_BOUNDARY_REPORT.md。封存檔 checkpoints/s33-source.tar.gz。\n\n"+p.read_text())
with Path('research/PROTOCOL.md').open('a') as f:f.write("\nS33 complete.\n"+result)
for name in ["s32-manifest.json","s11-20261113-manifest.json"]:
    assert all(digest(k)==v for k,v in load(Path("research/checkpoints")/name)["files"].items())
archive=Path("research/checkpoints/s33-source.tar.gz")
names=subprocess.check_output(["git","ls-files","--cached","--others","--exclude-standard","-z"]).decode().split("\0")
with tarfile.open(archive,"w:gz") as tar:
    for name in sorted(set(names)):
        if name and Path(name).is_file() and not name.startswith(("research/results/","research/checkpoints/","research/bin/","research/__pycache__/")):
            tar.add(name,recursive=False)
paths=[archive,Path("research/bin/robust-reveal-boundary"),Path("research/bin/learned-robust"),*results.glob("s33-*")]
manifest={str(p):digest(p) for p in paths if p.is_file()}
Path("research/checkpoints/s33-manifest.json").write_text(json.dumps(dict(promoted=passed,files=manifest),indent=2))
assert all(digest(k)==v for k,v in manifest.items())
print(result)
