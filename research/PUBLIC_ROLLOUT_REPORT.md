# S26：公開資訊條件世界與短期閉迴路模擬

## 假設與固定設計

S24 全根分配、S25 保留早期優勝根都未改善。這次不再微調評估函數或
淘汰比例，而是測試：把原搜尋未建模的補垃圾機制放進短期模擬，是否能
更準確比較第一步。候選仍由既有合法落點與 robust B* 產生。

每步同一 50,000 生成預算：37,500 用於基礎搜尋，取前兩個有完整評分的
合法根；每根兩個共同假想世界、各四次落子。其餘預算分攤給後續搜尋
及每個模擬世界／落子狀態；兩根時每次後續搜尋上限 1,040 節點。
只有一根時按實際數目分配，未用完預算不補作額外搜尋。
不使用時間截止。節點預算公平不代表 CPU 時間相等。

假想世界只接受公開盤面、當前塊、五個預覽、Hold、公開七袋歷史、
已生成垃圾數、最後公開的新垃圾洞。由公開 JSON 的 FNV64 與固定世界索引
產生種子，兩候選使用相同種子；塊流與垃圾流各自 domain-separated。
規則、SRS、Hold、重力與補垃圾直接沿用 Stacker，不重寫遊戲。

假想策略每次只讀該假想世界當下的 current+5。它可能看到模擬中新揭露的
預覽，這是條件世界的一部分，並非真正遊戲的隱藏隊列。真實遊戲只執行
最初合法候選的一步；下一步重新從公開觀察規劃。未複製真實 RNG。
原本 simulator 的塊流與垃圾流初始化相關性未被利用；因此這是刻意不
利用其私有隨機機制的模型，不能聲稱精確複製官方 Jstris 分布。

成本為已模擬落子數 + 凍結盤面估值/10 + 2.083984375×尚未生成垃圾行。
價格取自原訓練，不重新調參。完成100行成本只算落子；topout為1000+落子。
沒有在預算內取得建議時，截斷並估計當前盤面，不把它誤判為死亡。

同工作量 placebo 執行全部相同計算，仍選基礎第一名。另對照凍結50k
cheese-robust；後者重用S24相同32盤記錄，CPU跨批僅描述。
兩新版本交錯執行，4workers。所有參數先寫入 PROTOCOL.md。

## 稽核

- 引擎69測試、公開模擬器7測試通過。
- 舊76000兩版smoke完成；兩份公開歷史、所有假想操作與逐步預算回放通過。
- `--world-audit` 使用actual seed僅重建已記錄的真實觀察，逐步比對planner輸入；
  假想回放只用公開觀察及其衍生seed。planner介面不接受真實Stacker或seed。
- `audit_worlds.py` 另外以Python獨立重算JSON FNV；檢查binary SHA與placebo。
- `audit_bag_observer.py` 從公開預覽歷史獨立驗證七袋剩餘集合；不推理真實RNG。
- `frontier_nodes=0`表示多個搜尋沒有單一frontier，非沒有搜尋；`root_moves`
  是基礎已評分根數。`rerank_nodes`包含假想搜尋及模擬狀態。

## 統計邊界

開發76800..76831已用於S24/S25，並非新的確認集。必須平均優於
placebo及完整robust且完成率不下降，才打開尚未使用76700..76747驗證；
驗證通過才開75400..75463正式64盤。正式要求所有完成且兩個配對95%
區間全負，不擴樣本、不掃世界數／步數／分配比例。假想軌跡不是獨立遊戲樣本。

## 結果

| 開發32盤 | 平均方塊 | CPU秒／局 | 完成 |
|---|---:|---:|---:|
| 凍結robust50k | 196.21875 | 16.11831 | 32/32 |
| 同工作量placebo | 198.18750 | 14.75685 | 32/32 |
| 公開世界短期模擬選擇 | 209.96875 | 15.58488 | 32/32 |

候選−placebo +11.78125塊，95%配對bootstrap[8.78125,14.90625]，3勝1平28敗。
候選−robust +13.75塊，95%[11,16.375]，2勝1平29敗。拒絕、不升級。
64份完整實際回放、64份公開歷史、所有假想軌跡與逐步節點稽核通過；
另有2份smoke回放。正式與獨立驗證集仍未開啟。

候選在17.83%真實決策改選第二根；95.25%決策有兩根可比較。
假想軌跡68.42%補過垃圾，平均執行3.976步，截斷與topout皆0。
兩個世界偏好不同占雙候選決策6.73%；平均兩世界候選差異的差距0.254塊，
平均候選均值絕對差1.154塊。這些是決策層描述，不能當獨立樣本做品質推論。

## 技術瓶頸與停止條件

同工作量對照已隔離「執行額外模擬」與「依模擬值改選」：改選造成明顯
整局退步，不能只歸咎於少了12.5k基礎搜尋。模擬確實跨補垃圾轉移，且
沒有找不到動作的截斷；單憑補上更真實的動態機制，並未改善決策。
兩個世界多數一致，也沒有證據支持只增加抽樣數即可修復。

目前成本是低預算跟隨策略的短期軌跡，加上盤面估值。它不等同於較強
實際策略下的條件剩餘方塊數；更完整的世界模擬不自動使這個成本排序可靠。
固定短程成本與真正100L根動作價值之間仍有落差。現有實驗無法唯一區分
跟隨策略、端點估值與模擬分布的偏差，不宣稱已識別單一根因。
停止此固定方案的世界數／步數／預算比例掃描，不用提高預算挽救負結果。

接續研究先檢驗根動作條件成本標籤的可靠性：使用公開重建、共同隨機數、
實際凍結策略的反事實延續，區分標籤噪音與部署策略差異；先在全新訓練／
診斷集驗證排序，而非重新使用76700或75400調參。這需要專門的高成本
標籤資料實驗；本輪沒有執行或宣稱其能改善100L。

## 重現

```sh
source /workspace/.blockfish-env/activate.sh
python3 research/build.py world-rollout world-rollout-placebo
python3 research/bench.py --binaries research/bin/world-rollout research/bin/world-rollout-placebo --configs 50/5,10,10,10 --start 76800 --count 32 --jobs 4 --out research/results/s26-dev.jsonl
python3 research/decide_cohort.py --lock research/results/s26-dev-lock.json --inputs research/results/s26-dev.jsonl research/results/s24-control-dev.jsonl --out-prefix research/results/s26-dev --gate mean
python3 research/audit_worlds.py research/results/s26-dev.jsonl --out research/results/s26-world-audit.json
python3 research/audit_bag_observer.py research/results/s26-dev.jsonl --no-parity --out research/results/s26-bag-audit.json
python3 research/replay_audit.py research/results/s26-dev.jsonl --out research/results/s26-replay-audit.json
python3 research/build.py world-diagnostic
python3 research/diagnose_world_policy.py research/results/s26-dev.jsonl --out research/results/s26-policy-mismatch.json
```

候選二進位及其原始build source hash保留在bin/builds.jsonl。後續獨立診斷
binary只回放既有資料，沒有修改兩個已測候選。所有功能仍是headless研究
原型；不代表Jstris線上實測或官方榜單紀錄。

## 事後策略落差診斷

只取placebo所選原第一根的第一個模擬後續動作。12,620個可檢查樣本中，
3,732個的盤面、當前塊、完整預覽與Hold恰好等於下一個實際公開狀態。
對這些完全相同的輸入，低預算模擬策略有938個（25.13%）選擇與37.5k
實際策略不同的結果；比較已消除同落點不同操作序列的影響。

這是條件子集上的描述性結果，世界樣本彼此相關，不能當3,732個獨立
品質測試，也不能證明任一動作較優。但它直接確認短期成本是在大量
不同的後續決策下估算，並非實際較強策略之條件價值的無偏替代。
因此先研究可用標籤／策略一致性，比單純增加樣本數更有診斷意義。
獨立offline診斷程式不執行新搜尋，沒有開新seed或調整候選。
