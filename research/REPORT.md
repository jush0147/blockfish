# Blockfish 100L Cheese Race 實驗紀錄

日期：2026-11-08。基準 commit：`739953f9331b7648f269fc52cff5aae330108ccd`。
已透過 `git ls-remote https://github.com/blockfish/blockfish.git HEAD` 核對上游 HEAD
就是此 commit（預載 checkout 的 origin 是 fork，但基準 commit 與上游相同）。

本次修改保留 block-stacker 的遊戲模擬、SRS、Hold、預覽資訊與合法落點／操作重建。
正式候選仍使用原版 B* 與 `5,10,10,10` 權重；採用位元評估與精確快取降低計算成本。
評估函數與搜尋策略的其他變體只作為可選實驗保留。

## 原版架構與限制

- `block-stacker`：實際遊戲狀態、七袋、垃圾生成、消行、旋轉與輸入。
- `blockfish-engine/src/place.rs`、`finesse.rs`：落點枚舉及合法操作重建。
- `ai/state.rs`：搜尋盤面、已知佇列與 Hold；不讀 seed 或尚未出現的垃圾。
- `ai/b_star.rs`：每一深度一個優先佇列，先搜尋完整已知序列，再選擇較佳中間節點展開。
  葉節點通常採自身與父節點的分數和；清掉目前最底行則優先按深度評分。
- `ai/eval.rs`：高度、基於殘留空區域的估計方塊數、I 依賴，加上落子路徑成本。
- 原版 `search_limit` 實際限制佇列存量，並非累計生成節點。新剪枝與 beam 比較
  明確使用累計生成上限；兩種預算不能混稱「同樣節點數」。
- 真正預設是 50k、權重 `5,10,10,10`、垃圾高度 9/4。已修正過時的 CLI 說明。

## 實驗設計

完整設定與預先鎖定的測試集見 [PROTOCOL.md](PROTOCOL.md)。每局清除 **100 條垃圾行**，
不是任意 100 行；當前方塊 + 5 顆預覽 + Hold。所有變體使用同一 seed 配對。
開發集 0–11、驗證集 1000–1023、獨立測試集 10000–10099。

完成率優先，失敗不混入成功局平均，也不丟棄失敗紀錄。時間模式使用相同每步軟期限，
四個工作程序隨機交錯分配在四核心配額，該批期間不做其他 CPU 密集工作。
每步 64 個搜尋步驟檢查一次期限，且至少提供一條完整序列；實際耗時另存。
固定節點模式可重現；時間模式的落子會受排程與快取影響，不保證逐次一致。

## 各輪假設、修改與結果

以下數字均為完整 100L 的平均方塊數，除另註外皆全數完成。

| 輪次 | 假設／修改 | 測試條件 | 結果與處置 |
|---|---|---|---|
| 0 | 原版基準 | seed 0，50k frontier | 189 顆，24.67 秒；保存未改動執行檔 |
| 1 | 降低高度權重可能減少過度保守 | 12 seeds × 8 組，5k frontier | 原版 210.75；權重 2 為 206.58；權重 0 僅 6/12 完成，淘汰 |
| 2 | 鄰近權重、葉節點去掉父分數、終點保留殘留成本 | 12 seeds，5k frontier | 鄰近權重未穩定超越；terminal-only 208.92；終點改分 211.33 |
| 3 | 初步優勢能否跨盤面／預算 | 24 新 seeds，5k；12 開發 seeds，50k | 高度權重 2 在驗證集 213.54 vs 原版 210.75；50k 為 207.33 vs 204.58，淘汰 |
| 4 | 相同狀態剪枝提高有效探索 | 12 seeds，相同 5k 累計生成上限，權重 2 | 剪枝 210.58 vs 未剪枝 208.17，未採用 |
| 5 | 深層空洞折扣更保守、表面平整度 | 12 seeds，5k frontier | 半折扣 217.58；表面項 212.50；原版 210.75，未採用 |
| 6 | 等價位元評估與快取節省時間 | 串行 CPU 計時與逐步軌跡核對 | 見下表；納入正式候選 |
| 7 | 加速能否在短期限轉成方塊優勢 | 24 驗證 seeds，10ms/步 | 位元版 211.38 vs 原版 211.54；配對差 -0.17，95% bootstrap CI [-4.92, 4.75]，未證實 |
| 8 | 分層 beam 能否更有效分配搜尋 | 12 seeds，50k 累計上限 | beam 平均實用約 20.7k 節點，206.17 顆；同上限 B* 204.00 |
| 9 | 加倍 beam 寬度使用更多預算 | 12 seeds，50k 累計上限 | 202.67 顆 vs B* 204.00，進一步測試 deadline robustness |
| 10 | 同時間比較正式候選與 wide beam | 24 驗證 seeds，50ms/步 | 快速 B* 206.83 vs 原版 204.08；差 +2.75，CI [-1.08, 6.63]。wide beam 0/24 完成，淘汰 |

第 10 輪證明固定預算的小幅 beam 優勢無法直接推廣到 anytime 使用：寬 beam
在期限前通常還沒完成整層／完整預覽序列，只剩 warmup 備選，導致堆高失敗。
同時，更多 B* 節點並不保證每一局更省方塊；評估函數仍可能選擇不利路線。

## 等價效能改善

1. **I 依賴**：逐欄掃描改成逐列位元遮罩，找連續三列兩側封閉的空格。
2. **連通空區域**：區間 union-find 改成位元 flood fill，保留完全相同的區域面積。
3. **精確快取**：每次 analysis 以完整 `BasicMatrix` 為 key 快取靜態分數，不使用近似 hash
   判等、不跨權重重用，也不改 Hold、預覽或排序。

| 50k、兩個配對 seed 的串行 CPU 平均 | 秒／局 |
|---|---:|
| 未改動原版 | 27.557 |
| 僅快取 | 21.241 |
| 位元評估 | 19.643 |
| 位元評估 + 快取 | 16.388 |

最後一項減少約 **40.5% CPU 時間**，此表樣本小，不能當大型速度統計。
5k、四個配對 seed：原版 2.326 秒，位元評估 1.494 秒，快取 1.986 秒。
這些對照的逐步消行軌跡完全相同。更廣的獨立集與操作序列驗證另列最終結果。
快取增加記憶體使用量；沒有宣稱它降低空間成本。

## 獨立測試：固定搜尋預算

100 個從未用於選參的 seed 10000–10099，5k frontier，每個版本各 100 局：

| 版本 | 完成率 | 平均顆數 | 最少顆數 | 平均牆鐘秒／局 |
|---|---:|---:|---:|---:|
| 原版策略、未優化評估 | 100/100 | 209.64 | 191 | 3.146 |
| 原版策略、位元評估 + 快取 | 100/100 | 209.64 | 191 | 2.050 |

**100/100 配對的完整操作序列和逐步消行序列都完全相同**。此批交錯執行的
平均牆鐘耗時下降 34.9%；不同於上面的串行 CPU 計時，兩者分別報告，不混為一談。
原始資料：`results/final-fixed.jsonl`，各局 trace 路徑包含於每一筆紀錄。

## 重現方式

```bash
source /workspace/.blockfish-env/activate.sh
cd /workspace/blockfish
python3 research/build.py original final-baseline final-optimized
python3 research/bench.py --binaries research/bin/final-baseline research/bin/final-optimized \
  --configs 5/5,10,10,10 --start 10000 --count 100 --jobs 4 \
  --out research/results/reproduce-fixed.jsonl
python3 research/bench.py --binaries research/bin/final-baseline research/bin/final-optimized \
  --configs 50/5,10,10,10 --move-ms 50 --start 10000 --count 100 --jobs 4 \
  --out research/results/reproduce-timed.jsonl
python3 research/summarize.py research/results/reproduce-timed.jsonl \
  --baseline final-baseline:50/5,10,10,10@50ms
cargo test --workspace --locked
```

`research/build.py` 可重建每一個命名 ablation；舊執行檔被覆寫前會按 SHA256 備份。
JSONL 保存執行檔 hash、seed、參數、成功／失敗、耗時與逐步統計。
每局 `.json` 保存消行與操作序列；新 trace 可用下列方式在同一模擬器無搜尋驗證：

```bash
target/release/blockfish-race --replay-file path/to/trace.json
```

## Jstris 紀錄的證據界線

目前證據是 **Blockfish 內建模擬器的實測**。沿用原版並不等於已驗證當前 Jstris
全部規則與亂數一致：兩者 seed 不互通，本機 generator 的垃圾與方塊 PRNG 初態也有相關性。
已測試實際輸入可在 block-stacker 重播；這不是 Jstris 官方 replay。

README 的歷史紀錄連結與搜尋到的歷史 170 顆說法都不足以核定目前最強成績。
直接讀取現行排行榜／指定 replay 未成功。因此本報告不宣稱打破 Jstris 世界紀錄。
後續要作此宣稱，需要實際 Jstris 盤面／方塊來源、規則一致性測試及可核驗 replay。
