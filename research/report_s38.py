"""Write the S38 handoff and checksummed checkpoint after run_s38.py finishes."""
import hashlib
import json
from pathlib import Path
from statistics import mean

root=Path("research/results")
def read(name):
    return json.loads((root/f"s38-{name}.json").read_text())

names={"robust-terminal-priority":"S37 終局排序", "robust-classic-guide":"S36 原版排序"}
selection=read("selection")
decision=read("decision")
pilot=read("pilot")
variants=list(names)
lines=["# S38　重新驗證固定預算搜尋候選","",
       "S37 的公開尾盤路徑數曾增加，S36 的原版估值引導也曾降低開發平均數。兩者分別只有 12 對與 8 對，這次保留原執行檔，以新的成對樣本驗證。基本單位是一枚種子的一整場 100L，平手以零差值保留。完整門檻與保留區間見 S38_PROTOCOL.md。","","| 階段 | 候選 | 配對數 | 基準／候選平均塊 | 候選減基準 | 配對 95% 區間 | 勝／平／敗 | CPU 比 |","|---|---|---:|---:|---:|---:|---:|---:|"]
def table_row(stage,n,r):
    base,alt=r["baseline"],r["candidate"]
    if not r["eligible"]:
        return f"| {stage} | {names[n]} | {base['total']} | 未全數完成 | — | — | — | {r['cpu_ratio']:.2f} |"
    total=base["total"]
    return (f"| {stage} | {names[n]} | {total} | {base['mean_pieces']:.3f}／{alt['mean_pieces']:.3f} | "
            f"{r['mean']:+.3f} | [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] | {r['wins']}／{r['ties']}／"
            f"{total-r['wins']-r['ties']} | {r['cpu_ratio']:.3f} |")
for n,r in read("first-summary").items(): lines.append(table_row("新驗證",n,r))
for n in selection["extended"]: lines.append(table_row("累計驗證",n,selection["estimates"][n]))
if decision["selected"]:
    lines.append(table_row("獨立正式",decision["selected"],decision["confirmed"]))
lines+=["","舊快篩的候選平均差分別為 "+ "、".join(f"{names[n]} {pilot[n]['mean']:+.3f} 塊" for n in variants)+"。上表各階段的數字均排除舊快篩。開發階段可能依結果延長或停止，其區間只描述這批樣本。正式批次使用獨立種子、單一候選與固定樣本數。所有區間均以種子重抽 50,000 次，為近似百分位區間。",""]
if decision["selected"]:
    name=names[decision["selected"]]
    lines.append(f"正式測試評估 {name}。"+("通過全部預設門檻，原 cheese-robust 仍凍結保存。" if decision["promoted"] else "未通過全部預設門檻，保留 cheese-robust。"))
else:
    lines.append("兩個候選均未取得正式測試資格；78400..78463 未啟用，保留 cheese-robust。")
lines += ["","## 逐局稽核與成本",""]
rows=[]
replayed=0
for stage in ("first","second","final"):
    path=root/f"s38-{stage}.jsonl"
    if path.exists():
        rows.extend(map(json.loads,path.read_text().splitlines()))
        replayed+=sum(r["returncode"]==0 for r in read(f"{stage}-replays")["records"])
done=[r for r in rows if r["status"]=="complete"]
cpu=sum(r["cpu_seconds"] for r in rows)
best=min(r["pieces"] for r in done)
hits=sum(r["pieces"]<162 for r in done)
lines += [f"本輪實際執行 {len(rows)} 局，{len(done)} 局完成，{replayed} 份完整回放通過。各階段共用的基準局只計費一次，遊戲總 CPU {cpu:.1f} 秒（{cpu/3600:.2f} 小時）。基準與候選每步上限皆為 50,000 個生成節點，本輪本地最低 {best} 塊，小於 162 的局數為 {hits}。此處使用 block-stacker 回放。","","| 執行檔 | 實際局數 | 平均 CPU 秒／局 | 最少方塊（種子） |","|---|---:|---:|---|"]
for name in ["learned-robust",*variants]:
    subset=[r for r in done if Path(r["binary"]).name==name]
    if subset:
        record=min(subset,key=lambda r:(r["pieces"],r["seed"]))
        lines.append(f"| {name} | {len(subset)} | {mean(r['cpu_seconds'] for r in subset):.2f} | {record['pieces']}（{record['seed']}） |")
lines+=["","各執行檔可能參與不同階段；跨版本平均與勝負比較應使用上方相同種子的配對表。種子隔離稽核記在 s38-seed-audit.json，執行檔鎖定於 s38-binaries.sha256。","","## 重現與續接","","在專案根目錄執行以下指令，已完成的對局會依執行檔雜湊與設定略過。","","```sh","python3 research/run_s38.py","python3 research/report_s38.py","```","","每一階段的 JSONL 保存設定、每局用量與完整操作檔路徑。重新執行會再做回放、選擇與確認判定。開發種子已揭曉，後續候選須另留獨立正式資料。"]
report=Path("research/S38_REPORT.md")
report.write_text("\n".join(lines)+"\n")
files=[p for p in sorted(root.glob("s38-*")) if p.is_file()]
files += [report,*map(Path,["research/S38_PROTOCOL.md","research/run_s38.py","research/report_s38.py",
                            "research/checkpoint_s38.py","research/bin/learned-robust",
                            "research/bin/robust-classic-guide","research/bin/robust-terminal-priority"])]
manifest=dict(decision=decision,stages=[s for s in ("first","second","final") if (root/f"s38-{s}.jsonl").exists()],
              files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
Path("research/checkpoints/s38-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
