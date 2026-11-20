# S19：隔離保留 Hold 的搜尋邊界

S18 七袋推論未通過正式升級後，保留 cheese-robust。本輪不增加節點、
不重訓模型，單獨組合既有 `preserve-hold` 與穩健版，研究是否應停止
強制放完預覽邊界最後剩下的 Hold。先前 S2 與 S15 分別同時加入立即
回傳、假想下一顆搜尋，不能完全隔離此改變。新增可重建的
`robust-preserve-hold` profile；不啟用 bag、tail 或 eager 功能。

假設、種子及門檻在結果前寫入 PROTOCOL，二進位 SHA 鎖定於
`results/s19-dev-lock.json`。開發 76000..76031，50k 生成節點、當前＋5
預覽、原 SRS/Hold/補垃圾規則。對照重用同32盤凍結穩健結果；不是新的
獨立驗證，也不把跨批次 CPU 差當成速度收益。65 項引擎測試通過。

|版本|完成率|平均方塊|CPU秒／局（描述性）|
|---|---:|---:|---:|
|凍結 cheese-robust|32/32|197.1875|15.6658|
|保留最後 Hold|32/32|200.8125|16.8833|

配對差 +3.625，95% bootstrap [0.75,6.5]，10勝5平17敗。候選所有32份
完整回放通過，每步生成數均不超過50k。**拒絕；不開新驗證或正式集。**
預留76200..76247與75300..75363尚未使用。停止繼續微調此邊界。

## 描述性診斷與技術限制

同一初始盤面的平均完整序列數738.06→1149.94，獲得評分的第一步數
11.31→15.59，6/32第一步操作改變。更短的邊界確實讓更多候選得到評分，
但沒有轉化成較少方塊。不能以覆蓋或搜尋效率代理完整遊戲品質。
後續對局盤面已不同；逐步平均只能描述，不可當作受控因果比較。

現有學習模型輸入只有14個盤面特徵，並不包含保留下來的 Hold 型別。
因此同盤面、不同 Hold 的葉節點會得到相同靜態估值。這是明確的表示
能力限制，但尚未證明它就是 S19 失敗的原因。即使增加 Hold 特徵，也
需要針對這種終點分布重新取得訓練標籤；直接把實際對局的「當前＋5
預覽」回報套用到「只剩Hold」的葉節點有分布差異。先前 S15 的七種
未知方塊逐葉展開又耗盡已知序列搜尋，並非已驗證可用的解法。

下一個有意義的方向是有條件的葉節點成本或少量候選的條件 rollout，
必須使用公開七袋條件、獨立訓練種子、共同隨機數與完整對局驗證。
目前不以這些尚未實作的方向宣稱改善，也不繼續掃描平滑權重。

## 重現

```sh
source /workspace/.blockfish-env/activate.sh
python3 research/build.py robust-preserve-hold
cargo test --locked -p blockfish --lib --no-default-features --features race,cheese-robust,preserve-hold
python3 research/bench.py --binary research/bin/robust-preserve-hold --configs 50/5,10,10,10 --start 76000 --count 32 --jobs 4 --out research/results/s19-dev.jsonl
python3 research/decide_cohort.py --lock research/results/s19-dev-lock.json --inputs research/results/s16-control-dev.jsonl research/results/s19-dev.jsonl --out-prefix research/results/s19-dev --gate mean
python3 research/replay_audit.py research/results/s19-dev.jsonl --out research/results/s19-dev-replay-audit.json
python3 research/summarize_hold.py
```
