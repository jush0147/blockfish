# 預覽邊界的未知方塊延伸（S14–S15）

接續[已驗證穩健版本](GENERALIZATION_REPORT.md)。基準固定為cheese-robust，
每步50k生成節點。正式已使用62200..62263不參與調參；75100..75147及
75200..75263仍未開啟。本報告全部是開發集62000..62011的淘汰實驗。

## S14：在已知方塊放完後做一步期望評估

假設：靜態盤面估值可能忽略預覽邊界的未知下一顆。對每個非底行目標的
終點，使用原合法落點生成器，枚舉七種下一顆，取各自最佳後盤面估值，
平均後加一顆路徑成本，代替原終點h；保留原parent-f平滑及四終局平均。

固定均勻先驗不是正確的七袋條件機率。程式不讀取種子、隱藏佇列、實際
未來或RNG；不模擬補垃圾，也不使用這些假想落點作為執行序列。七個假想
根及每個子狀態都計入50k；每個step至多生成一個節點，預算耗盡時捨棄
未完成的七種平均。另設相同額外運算但保留原評分的對照。

## S15：在強制用完Hold之前評估未知下一顆

檢查原搜尋語意後，S14的延伸發生在已知Hold也被強制用完之後。S15使用
既有preserve-hold，在預覽耗盡時保留已知Hold；每種假想下一顆均可與
已知Hold競爭，採原SRS落點。終點評分改為f(邊界)+E[f(下一步)]，讓平滑
兩端跨越這個未知決策。無Hold時仍不模擬空Hold再抽未知方塊。其餘先驗、
預算與估值完全固定；另有相同邊界與運算的原評分對照。

## 完整結果

每組12/12完成100L；四個候選共48份回放全部通過，逐步生成節點均≤50k。

| 設定 | 平均方塊 | 相對穩健基準 | CPU秒／局 |
|---|---:|---:|---:|
| 已驗證穩健基準 |195.3333|—|16.0391|
| S14同運算對照 |197.7500|+2.4167|10.3037|
| S14均勻延伸 |199.9167|+4.5833|10.4434|
| S15同邊界對照 |208.3333|+13.0000|10.0368|
| S15含已知Hold延伸 |202.0000|+6.6667|9.5889|

S14相對基準配對95%區間[-0.5833,9.8333]；相對同運算對照+2.1667，
區間[-2.5,6.3333]。S15相對基準+6.6667，區間[3.5,9.5833]；雖比自身
較弱對照少6.3333（區間[-11.25,-1]），仍明顯不如真正要擊敗的可靠版本。
不能把「打敗被削弱的對照」當作進步。兩者均未通過開發門檻，不進獨立
驗證，不調整機率、權重或延伸長度以追逐開發集結果。

S14平均每步完成約202條已知序列，原穩健版約773條；同運算對照也只有
約203條。這是搜尋資源被延伸評估占用的直接觀察，但不同對局軌跡也會
影響統計，不能僅據此斷言退步全由資源配置造成。均勻先驗、簡化Hold及
未模擬補垃圾也是模型限制。更低CPU時間不是目標，因此不採用這些版本。

目前的具體瓶頸是：逐終點做完整七種落點枚舉，在公平50k預算內壓縮已知
預覽搜尋，且這個簡化模型的估值收益不足以彌補。這不否定所有未知未來
建模；但繼續加預算或調平均權重不是本輪證據支持的方向。後續若研究
預覽邊界，應先改變運算位置（只精評少數候選）或建立經驗證的條件模型，
而非繼續逐葉增加假想分支。沒有在本輪實驗中驗證那些後續方案。

## 正確性與重現

S14共67項測試、S15共68項測試通過。涵蓋完整七種枚舉結果對照、嚴格
節點上限、操作trace不包含未知方塊，以及四行I洞中保留I確實能被七種
假想當前方塊共同選用。新程式為optional feature；三個保留執行檔的hash
與升級時的checkpoint完全一致。

```sh
source /workspace/.blockfish-env/activate.sh
python3 research/build.py robust-tail robust-tail-placebo robust-tail-hold robust-tail-hold-placebo
python3 research/bench.py --binaries research/bin/robust-tail research/bin/robust-tail-placebo --configs 50/5,10,10,10 --start 62000 --count 12 --jobs 4 --out research/results/s14-dev.jsonl
python3 research/bench.py --binaries research/bin/robust-tail-hold research/bin/robust-tail-hold-placebo --configs 50/5,10,10,10 --start 62000 --count 12 --jobs 4 --out research/results/s15-dev.jsonl
python3 research/budget_summary.py research/results/s11-dev.jsonl research/results/s15-dev.jsonl --expected-count 12 --out research/results/s15-dev-summary.json
```

完整配對差與CPU在s14/s15-dev-summary.json；配置統計在
s14-s15-allocation.json，回放證明在s14-audit.json及s15-audit.json。
