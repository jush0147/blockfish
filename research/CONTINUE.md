# S35 完成

跨局兩步挖掘操作庫在 3,181 次開發查詢僅命中一次，未達 16 次門檻。完整結果與續接方向見 DIG_LIBRARY_REPORT.md，逐步資料見 results/s35-probe.json。凍結最佳版本仍為 cheese-robust。

# S34 完成

S34 完成。32 局舊開發來源的 64 個固定盤面，64 個有共同合法候選，0 個在排名合併後改選。低成本門檻失敗。未啟用新完整對局或正式資料，最佳仍為凍結 cheese-robust。方法及重現指令見 MIRROR_REPORT.md。

# S33 完成

## 完整對局結果

- validation learned-robust，24/24 完成，平均 196.2917 塊與 19.65 CPU 秒。
- validation robust-reveal-boundary，24/24 完成，平均 252.8333 塊與 17.79 CPU 秒。
- 候選減基準 +56.5417 塊，配對 95% 區間 [50.666666666666664, 62.666666666666664]。48 局成功回放。

S33 未達升級門檻。cheese-robust 繼續作為最佳版本。

77400..77423 已揭曉。正式集 77500..77563 的使用狀態見上列階段。原始逐盤紀錄、鎖定檔與回放在 results/s33-*，重現指令在 REVEAL_BOUNDARY_REPORT.md。封存檔 checkpoints/s33-source.tar.gz。

# S32 完成

## 完整對局結果

- validation learned-robust，24/24 完成，平均 196.5417 塊與 19.81 CPU 秒。
- validation robust-dig-geometry，24/24 完成，平均 195.0000 塊與 19.84 CPU 秒。
- 候選減基準 -1.5417 塊，配對 95% 區間 [-5.166666666666667, 2.1666666666666665]。48 局成功回放。
- final learned-robust，64/64 完成，平均 194.5156 塊與 19.73 CPU 秒。
- final robust-dig-geometry，64/64 完成，平均 196.1406 塊與 20.12 CPU 秒。
- 候選減基準 +1.6250 塊，配對 95% 區間 [-0.296875, 3.5]。128 局成功回放。

開發時鎖定的幾何修正在正式盤面未達升級門檻。cheese-robust 繼續作為最佳版本。兩階段對局只識別整體策略的差異，無法將結果歸因於其中任一個幾何特徵。

77200..77223 與 77300..77363 已揭曉。原始逐盤紀錄、鎖定檔與回放在 results/s32-*，重現指令在 DIG_GEOMETRY_REPORT.md。封存檔 checkpoints/s32-source.tar.gz。

# S31 完成

S31 完成。每步上限 50,000 個生成節點。

dev（配對盤面）
- robust-plan-reuse 完成 32 局，平均 194.1250 塊，CPU 16.04 秒／完成局。
- robust-plan-observer 完成 32 局，平均 194.7500 塊，CPU 16.23 秒／完成局。
- learned-robust 完成 32 局，平均 194.7500 塊，CPU 15.87 秒／完成局。
- 候選減 robust-plan-observer 平均 -0.6250 塊，95% 配對區間 [-2.09375, 0.8125]。
- 候選減 learned-robust 平均 -0.6250 塊，95% 配對區間 [-2.09375, 0.8125]。

val（配對盤面）
- robust-plan-reuse 完成 48 局，平均 195.2708 塊，CPU 19.62 秒／完成局。
- learned-robust 完成 48 局，平均 195.3125 塊，CPU 19.05 秒／完成局。
- learned-fast 完成 48 局，平均 196.5833 塊，CPU 19.72 秒／完成局。
- 候選減 learned-robust 平均 -0.0417 塊，95% 配對區間 [-1.3333333333333333, 1.1666666666666667]。
- 候選減 learned-fast 平均 -1.3125 塊，95% 配對區間 [-3.8541666666666665, 1.1458333333333333]。

final（配對盤面）
- robust-plan-reuse 完成 64 局，平均 197.4375 塊，CPU 20.05 秒／完成局。
- learned-robust 完成 64 局，平均 197.0625 塊，CPU 19.58 秒／完成局。
- learned-fast 完成 64 局，平均 199.9844 塊，CPU 20.62 秒／完成局。
- 候選減 learned-robust 平均 +0.3750 塊，95% 配對區間 [-1.25, 2.0625]。
- 候選減 learned-fast 平均 -2.5469 塊，95% 配對區間 [-5.03125, -0.015625]。

未達升級門檻，最佳版本繼續採用 cheese-robust。
已開啟的資料包括 dev, val, final。詳見各階段 lock、decision、audit 與 replay JSON。

Checkpoint 位於 research/checkpoints/s31-source.tar.gz，雜湊清單為 s31-manifest.json。



# 續接：S29/S30完成（2026-11-15）

## 現在狀態

無執行中實驗。最佳仍為cheese-robust；learned-robust/learned-fast及S11/S28
所有manifest檔案hash完整保留。本輪不升級，76700驗證與75400正式仍未開。
源碼checkpoint：checkpoints/s30-20261115-source.tar.gz及manifest。
本輪96筆嘗試（3版×32新開發盤），92完成並完整回放，4失敗，總CPU1644.949169秒。
結果總表results/s29-s30-completion.json；細節GLOBAL_SEARCH_REPORT.md。

## 方法與結果

新global.rs沿用State/SRS/PlaceFinder/Hold/Evaluator/Node/robust4，改為所有深度
共用frontier；完整兄弟枚舉後排程。S29累積(child.f−同父最低f)，S30絕對f。
同分先較深、較低f、先生成者。不改終點f+parent_f/短完成偏好，不新增模型或
未知資訊，沒有額外暖身。root及每個生成child含未完成批次都計入50k。
兩版各68tests含小盤面全搜尋每根最終評分等同B*、建議合法、預算上限及重設。

全新開發77000..77031，三版交錯4workers/50k/no deadline/timeout180：
- 凍結robust32/32，196.15625塊，15.75782 CPU秒／完成局。
- S29 robust-global-regret32/32，201.65625，21.22714 CPU秒。
  對robust差+5.5，95%[2.96875,7.90625]，6勝1平25敗；明確未過門檻。
- S30 robust-global-f28/32，成功者均值196.07143，15.16465 CPU秒。
  不可與robust全部32盤平均直接當收益。在同28盤對robust差+0.25，
  95%[-2.57143,2.85714]；仍是成功子集描述，完成率失敗即不升級。
兩個鎖定mean gate皆失敗；不再掃priority/tie/width或加暖身/預算救援。

四次S30失敗seed77007/77027/77000/77028在第12/187/86/180次分析，
50k已用盡但completed=0、roots=0，race CLI no suggestions panic。
未寫出最終操作trace，因此不能宣稱失敗對局已回放。stderr/費用/returncode
保留於完整s29-s30-dev.jsonl及s29-s30-failures.json；沒有刪除或重跑失敗。
complete-only.jsonl只供92成功對局回放，不供品質決策。所有64候選嘗試的
已記錄分析包含失敗分析均通過50k預算稽核。沒有timeout或增加預算。

## 必要結論，避免再延伸低收益診斷

S29每步約1831完整終局、7.62已評分根，但更多方案沒有降低方塊數。
局部差值可能錯罰必要的暫時惡化，僅是假說，不再為此負結果擴診斷。
S30缺乏有限預算下完整序列保障；這不證明所有global search都無效。
本輪只否定所測固定排程，沒有證明B*達極限，亦沒有新增可部署改善。
下一輪應選具有不同收益來源、可直接測完整100L的假說，而非繼續混合這兩種
排序值或增加S26/S28樣本。尚未指定下一候選，請自主評估成本與預期收益。

## 統計工具修正與資料邊界

decide_cohort.py對失敗紀錄的search:null改作缺失list處理；成功案例仍要求
逐步search完整。用原S27cohort核對決策JSON完全相同，見s30-summary-compatibility.json。
這不是演算法修改或門檻修改。所有S29/S30 gate都輸入完整96筆原cohort。
77000..77031現在已揭曉開發，不是新的獨立確認集；76900..7仍是S28診斷。
76700..76747與75400..75463仍未打開；其他歷史邊界見PROTOCOL.md。

## 重現

source /workspace/.blockfish-env/activate.sh
python3 research/build.py robust-global-regret robust-global-f
python3 research/decide_cohort.py --lock research/results/s29-dev-lock.json --inputs research/results/s29-s30-dev.jsonl --out-prefix research/results/s29-dev --gate mean
python3 research/decide_cohort.py --lock research/results/s30-dev-lock.json --inputs research/results/s29-s30-dev.jsonl --out-prefix research/results/s30-dev --gate mean
python3 research/audit_global.py research/results/s29-s30-dev.jsonl --out research/results/s29-s30-search-audit.json
python3 research/replay_audit.py research/results/s29-s30-complete-only.jsonl --out research/results/s29-s30-replay-audit.json

勿覆蓋frozen learned-robust/learned-fast。新候選只在指定headless profile驗證；
local模擬結果不是官方Jstris榜單紀錄。實驗源碼、原始資料、build hashes全部保留。

--- 以下歷史續接（不是執行中狀態） ---

# 最新：S29/S30全域搜尋實作完成，準備鎖定開發（2026-11-15）

原best cheese-robust完整保留；S11/S28所有manifest hashes已核對未變。
新global.rs，S29累積兄弟相對f差；S30全域絕對f。原終點評分與robust4不變。
兩候選各68tests（含完整小盤每根評分等價及合法重播）。無未知資訊或新增模型。
方法、成本、固定選擇門檻見GLOBAL_SEARCH_REPORT.md/PROTOCOL.md。
正在跑全新77000..31三版各32，合計96，4workers/50k/無deadline/timeout180。
輸出results/s29-s30-dev.jsonl；runner可按原命令續接既有紀錄。
python3 research/bench.py --binaries research/bin/robust-global-regret research/bin/robust-global-f research/bin/learned-robust --configs 50/5,10,10,10 --start 77000 --count 32 --jobs 4 --out research/results/s29-s30-dev.jsonl
完成後分別用s29-dev-lock.json、s30-dev-lock.json及同一完整96筆cohort執行
decide_cohort.py --gate mean，再audit_global.py和replay_audit.py。
若存在失敗，先保留完整cohort，回放只針對完成者另存過濾檔，不丟失失敗紀錄。
其中兩版為robust-global-regret、robust-global-f，第三為frozen learned-robust。
必須唯一候選平均/完成率勝另兩版，才開76700驗證；75400正式仍未開。
先查進程避免重複啟動；所有失敗保留，不掃參數挽救。

--- 以下S28完成狀態 ---

# 續接：S27–S28完成（2026-11-15）

## 當前狀態

沒有執行中的實驗。cheese-robust仍為唯一保留的已驗證最佳策略，
research/bin/learned-robust與S11/S26所有manifest項目未變；本輪不升級。
新checkpoint：checkpoints/s28-20261115-source.tar.gz及manifest。
43局實際完整100L全部完成並回放（32候選、3observer、8新訓練來源）。
另有64條條件完整延續與64條短期延續，均已回放；只來自8個來源，
不是64個新的獨立策略測試。成本882.882487+507.88818 CPU秒。
總結results/s27-s28-completion.json；腳本、二進位、原始操作與hash均保留。

## 綜合判斷

SYNTHESIS_S27.md先區分選模偏差、標籤/使用分布錯配、有限預算排擠、
模擬跟隨策略與端點偏差。共同代理目標問題不是已唯一識別的共同根因。
不預設B*已到極限，也不認為增加模型複雜度能修復標籤不確定性。
S28表明，即使跟隨策略一致，小樣本完整偏好與短期估值仍不穩定；
但這只是8個中盤來源的診斷，不能外推為所有rollout方法無效。

## S27：保留根身份的轉置合併

root-dag依完整State/depth/root記錄首次展開，跳過重複非終點子樹。
非終點parent_f不影響後代（succ覆蓋為當前f）；終點評分/robust4不改。
重複節點本身仍收生成費，所有實際生成≤50k，不接觸未知資訊。
68tests，小棋盤完整窮舉每根評分相同及合法路徑；3盤observer同操作/計數。
開發76800..31：197.28125 vs196.21875，32/32；差+1.0625，
95%[-1.71875,3.8125]，14勝4平14敗。未過平均門檻，不進驗證。
CPU21.93050 vs歷史控制16.11831秒/局；不是速度改善。
平均每步424.47重複被合併，唯一展開1515.20，但不等於品質改善。
停止轉置範圍/key掃描，不把省去冗餘當成降低方塊數的證據。

## S28：固定規模的公開條件標籤診斷

新訓練76900..76907，固定第100顆後公開狀態；每來源最多兩根、四共同世界。
strong50k執行至100L（上限400落子）；同軌跡第4步端點；weak1040四步端點。
真正來源seed僅用於重播過去100步，不把未來/RNG傳入假想策略。
8/8 base動作與frozen源局第101動作相同。64/64完整延續完成，weak截斷0。
全部公共輸入、FNV、操作、成本與逐次生成預算驗證通過。
完整四世界均值6非平手、2平手；強短/弱短各同意3/6非平手來源。
前後兩組世界偏好：2同向、2反向、3單側平手、1雙側平手。
不是可靠的accuracy估計，也不能把四世界均值當真值。
316406444總生成記帳、507.88818CPU秒、最高RSS54500KiB。
不擴pilot、不擬合新模型、不部署full-rollout作50k候選。
PUBLIC_LABEL_REPORT.md與results/s28-label-summary.json有完整結果。

## 資料邊界與後續

76700..76747獨立驗證、75400..75463正式仍完全未開啟。
76800..76831已揭曉開發；76900..76907現在是已揭曉訓練／診斷，不能當保留集。
不要用歷史已揭曉final調參；其他邊界沿用PROTOCOL.md。

下一輪先檢查既有研究假設是否仍值得做：有限預算搜尋誤差與根動作成本
不確定性尚未被分離。若沿用標籤方向，應預先處理平手／不確定性與狀態
分布，使用新訓練／診斷集；不能直接把本輪四世界均值當可靠偏好訓練。
也可探索其他決策方式，但不可僅增加預算或模型複雜度。沒有指定下一候選，
沒有宣稱任何尚未測試的方向有效；請自主選擇，事先固定實驗與停止規則。

## 重現

source /workspace/.blockfish-env/activate.sh
python3 research/build.py robust-dag robust-dag-observer public-labels
python3 research/decide_cohort.py --lock research/results/s27-dev-lock.json --inputs research/results/s27-dev.jsonl research/results/s24-control-dev.jsonl --out-prefix research/results/s27-dev --gate mean
python3 research/audit_dag.py research/results/s27-dev.jsonl research/results/s27-observer-parity.jsonl --out research/results/s27-combined-audit.json
python3 research/collect_public_labels.py
python3 research/analyze_public_labels.py

collector按已記錄seed續接，不重跑成功任務；error/timeout不可刪除挑結果。
public-labels --audit FILE只回放；build.py保存二進位SHA與sourcehash。
勿重建覆蓋learned-robust/learned-fast；所有新feature只在指定headless profile測過。
目前仍為本地規則/模擬結果，無官方Jstris榜單超越證據。

--- 以下歷史續接（不是執行中狀態） ---

# 最新S28標籤診斷執行中（2026-11-15）

S27已完成失敗：197.28125 vs196.21875，32/32，CI[-1.71875,3.8125]。
35回放、68tests、observer3盤精確parity通過；不升級、不開76700/75400。
SYNTHESIS_S27.md綜合過去失敗，注意共同代理目標問題並非唯一已證實根因。

S28新訓練來源76900..76907 frozenrobust50k八盤已完成並回放。
固定第100塊後公開狀態，50k提出最多兩根，各四共同公開條件世界。
強策略50k直到完成，另量測同強策略第4步端點、1040弱策略4步端點。
上限64條完整反事實延續、每條400落子，不擴樣本、不訓練模型。
正在 python3 research/collect_public_labels.py，4workers，s28-labels.jsonl。
每source完成後存s28-label-SEED.json（操作／節點），已內建完整重播audit；
collector另核對base第一動作與frozen source第101動作一致。
先查進程勿重複；script可依已記錄seed續接，timeout/error不可丟棄。
完成後 python3 research/analyze_public_labels.py，保存s28-label-summary.json。
新binary public-labels與root-dag皆optional；frozenbest所有檔案未改。

--- 以下S27歷史執行紀錄（目前已完成） ---

# 最新 S27 執行中（2026-11-15）

S11/S26全部manifest hash已核對不變，可靠best仍frozenrobust。
綜合分析與本輪預註冊：SYNTHESIS_S27.md、PROTOCOL.md。
新root-dag保留每根身份，對完整State/depth/root的重複非終點只展開一次；
終點不剪、原parent-f與robust4不改，所有重複節點本身仍計入50k。
68tests通過，包含小棋盤完整窮舉每根評分相同、完整合法路徑與重設表。
3盤76000..2 observer-only對凍結robust完整inputs/ds/counts parity與回放通過。
正在robust-dag32盤76800..31，4workers，s27-dev.jsonl，沿用s24-control-dev.jsonl對照。
observer記錄平均735重複非終點展開／步，但此為3盤描述非品質證據。
完成後以s27-dev-lock.json及s24-control-dev.jsonl執行decide_cohort.py，
再跑audit_dag.py及replay_audit.py；S27候選目前不可升級。
嚴格比平均與完成率，通過才開76700驗證；75400final仍未開。
勿重建覆蓋learned-robust/learned-fast。先查進程與輸出行數勿重複啟動。

--- 以下S26完成紀錄 ---

# 續接狀態：S24–S26 完成（2026-11-14）

## 當前狀態

沒有執行中的實驗。可靠最佳仍為凍結 cheese-robust（learned-robust，50k）。
S11 manifest所有檔案hash不變，見s26-baseline-integrity.json。
本輪165局全部完成並回放，35個seed ID（開發盤重用，不是165個獨立樣本），
2767.110916 CPU秒。見results/s24-s26-completion.json。
完整原始trace、cohort、鎖定manifest、稽核與二進位都留在research/。
源碼與腳本checkpoint：checkpoints/s26-20261114-source.tar.gz 及manifest。

## 結果：32盤開發76800..76831、同50k生成節點

- frozen robust196.21875。
- S24獨立根scout＋sequential halving199.15625；差+2.9375，95%[.0625,5.8125]。
- S25保留25%預算時已評分優勝半數根198.8125；差+2.59375，95%[-.03125,5.1875]。
- S26公開條件世界、短期閉迴路模擬209.96875；差+13.75，95%[11,16.375]。
- S26同工作量placebo198.1875；模擬選擇版比placebo差+11.78125，95%[8.78125,14.90625]。
所有版本32/32完成。三個方向皆拒絕，不升級，不擴樣本救援。
ROOT_SEARCH_REPORT.md、PUBLIC_ROLLOUT_REPORT.md有方法、成本、診斷與重現命令。

## S26實作及稽核

block-stacker/public-world：只從公開盤面/current+5/Hold/七袋歷史/公開垃圾計數與
最後洞重建世界，不接受真實Stacker或真實seed；兩種隨機流domain-separated。
既有SRS、Hold、合法落點、補垃圾沿用。69engine tests、7simulator tests通過。
world_rollout/mod.rs：37.5k基礎搜尋，前2根×2共同世界×4落子；剩餘12.5k包含
每世界根、每模擬落子、全部後續搜尋。正常兩根時每次後續搜索1040節點。
凍結盤面估值/10與原價格2.083984375；不訓練新權重。缺建議截斷，非假死亡。
同工作量placebo仍選原第一名；真實遊戲只執行原合法候選一步。
64dev+2smoke完整回放/public/world audits通過，實際publichistory逐步核對。
world-audit的真實seed只用於重建過去觀察，規劃介面不接受它。

候選17.83%決策改選；68.42%模擬路徑補垃圾；無截斷/topout。
兩世界偏好分歧6.73%，不支持把顯著退步簡化成抽樣雜訊。
額外offline world-diagnostic只讀既有placebo trace，不跑新搜索、不改候選：
第一個後續狀態恰與真實可見盤面/current/preview/Hold相同的3732樣本中，
938（25.13%）的低預算策略結果不同於37.5k策略。相關樣本，不是獨立品質證據。
證明策略落差存在，沒有證明哪個動作較好，也沒有證明提高預算即可修復。

## 尚未使用的資料與接續方向

獨立驗證76700..76747與正式75400..75463仍完全未開啟。
76800..76831是已揭曉開發；舊76000..31亦只可作歷史開發／parity。
其他歷史訓練與保留集邊界見PROTOCOL.md。不要把已揭曉final拿來調參。

目前清楚瓶頸是根動作成本排序可靠性，而不只是模擬速度或節點覆蓋。
停止S24淘汰比例與S26世界數/步數/預算比例掃描。沒有實驗唯一識別誤差根因。
下一輪可先設計新訓練／診斷集上的反事實根動作標籤，使用本輪公開重建
基礎設施、共同隨機數、凍結實際策略延續，檢查策略一致性、排序與標籤
不確定性；之後才決定是否值得訓練。這是尚未執行的方向，不是已驗證方案。
不要只增加部署搜索預算，也不要直接把短期或完整預覽狀態回報當搜尋葉真值。

## 重現入口

source /workspace/.blockfish-env/activate.sh
python3 research/build.py world-rollout world-rollout-placebo world-diagnostic
python3 research/decide_cohort.py --lock research/results/s26-dev-lock.json --inputs research/results/s26-dev.jsonl research/results/s24-control-dev.jsonl --out-prefix research/results/s26-dev --gate mean
python3 research/audit_worlds.py research/results/s26-dev.jsonl --out research/results/s26-world-audit.json
python3 research/diagnose_world_policy.py research/results/s26-dev.jsonl --out research/results/s26-policy-mismatch.json

構建world-diagnostic後新增的Cargo bin定義不改候選邏輯；原候選binary/sourcehash
保留於builds.jsonl。不要重建覆蓋learned-robust或learned-fast。
目前原型只支援headless固定節點研究，不宣稱官方Jstris規則完全一致或榜單紀錄。

--- 以下歷史續接（不是執行中狀態） ---


# 最新：S24新搜尋配置開發測試執行中

最佳cheese-robust凍結保留，S11manifest所有hash驗證未變。
S24 root-halving：獨立每根B*，先各取一條完整延續，再按最新robust評分
逐輪保留ceil-half，分配剩餘生成預算；不改估值、合法落點、Hold、預覽。
所有root枚舉/scout/refine共50k；每step至多1生成。淘汰動作透過新
Step::RemoveMove/Analysis訊息移除，避免留下過期樂觀分數。
67候選tests、66控制tests通過；halving-control3盤76000..2與frozenrobust
完整inputs/ds相同（s24-control-parity-check.json）。只測headless固定節點。

正在新DEVELOPMENT76800..76831，兩版各32，共應64筆：
python3 research/bench.py --binaries research/bin/robust-halving research/bin/learned-robust --configs 50/5,10,10,10 --start 76800 --count 32 --jobs 4 --out research/results/s24-dev.jsonl
先查進程勿重複。完成後：
python3 research/decide_cohort.py --lock research/results/s24-dev-lock.json --inputs research/results/s24-dev.jsonl --out-prefix research/results/s24-dev --gate mean
python3 research/audit_halving.py
python3 research/replay_audit.py research/results/s24-dev.jsonl research/results/s24-control-parity.jsonl --out research/results/s24-replay-audit.json

stderr每步ROOT_ALLOCATION記錄總節點、scout節點、roots、rated_roots、survivors。
不能以覆蓋率代替完整100L方塊數。若平均及完成率通過才開新76700..76747
三版驗證含learned-fast；再通過才開75400..75463正式64，雙控制95%區間全負，
不擴正式樣本。若失敗先看scout是否耗盡預算／評分支持不足，而非微調估值。
76800..76831現在是開發集，76000已反覆使用不再作本輪候選選擇。

以下上輪完整狀態（S20–S23）：

# 研究續接（2026-11-14，S20–S23全部完成）

## 現在狀態

沒有執行中的實驗。最佳仍為cheese-robust（research/bin/learned-robust，50k生成節點）。
原cheese-quality／learned-fast、S11source checkpoint及manifest所有hash核對未變。
S20–S23新增601局全部完成及完整回放，336個不同seed ID（開發seed重用，不能
當601個獨立樣本），8033.862064 CPU秒。見results/s20-s23-completion.json。

## 最新結果

S22正式75300..75363三版各64/64：
- learned-fast 198.671875塊；learned-robust 196.21875；Hold候選196.546875。
- 候選−robust +0.328125，95%[-1.515625,2.15625]，27勝6平31敗。
- 候選−legacy -2.125，95%[-4.25,0]。雙控制嚴格區間門檻失敗，**不升級**。
- 192回放、逐步metadata一致性與50k預算檢查全通過。
- robust−legacy -2.453125，95%[-4.171875,-0.671875]，再次支持既有策略。
- 不擴樣本，不合併開發挽救；正式前已約定僅通過才跑200k，因此未啟動200k。

## 候選由來／保留的失敗實驗

1. S20：新資料76400..76559，160局5k來源，全回放；128訓練76400..76527，
   32預測驗證76528..76559。公開Hold×5盤面特徵，固定ridge10，類型均值零。
   整局成本標籤驗證MSE40.28499→40.32575，未接入策略。
2. S21：同128訓練，固定7步＋凍結估值標籤；新76600..76631預測驗證32局。
   MSE2.735080→2.722706，區間跨零。沒有步數／正則掃描。
   robust-hold-context = cheese-robust + hold-context + preserve-hold。
   32dev198.15625 vs robust197.1875，拒絕。68測試，32回放通過。
3. S22：相同權重、原本耗盡Hold終點；robust-hold-order = cheese-robust
   +hold-context，影響中間估值及parent_f，不能稱純排序。
   dev76000..76031：195.96875 vs197.1875；新驗證76200..76247：196.6875
   vs197.833333，皆CI跨零但通過預設平均篩選；正式未重現，詳見上方。
4. S23：精確cache，robust-hold-cached = cheese-robust + hold-context-cache。
   每analysis盤面cache七種Hold修正，69測試；三盤完整inputs/ds/搜尋計數一致，
   6回放。CPU22.628→20.306秒，RSS26209→30996KiB；僅小型成本試驗。
   沒有減少方塊數，未替換S22正式執行檔，也不作推薦profile。

## 程式與重現

- CONTEXT_REPORT.md、PROTOCOL.md記錄完整假設／門檻／結果，s22-outcomes.svg圖。
- fit_hold.py：不帶參數為S20，--td7為S21；權重不重新選擇。
- 模型blockfish-engine/src/ai/hold_weights.rs；s21-fit-dependencies.json資料hash。
- State.held_color()只回實際Hold，空Hold不能把next().1當Hold。
- hold-context不隱含preserve-hold；S21profile明確加入，S22不加入。
- audit_context_cache.py檢查S23完全等價；diagnose_hold_coverage.py查訓練覆蓋。
- decide_cohort.py含配對seed/hash/config/每步預算/trace一致性門檻；replay_audit.py回放。
- checkpoints/s22-candidate-20261114-source.tar.gz已離線還原建置，二進位SHA與
  凍結S22完全相同（results/s22-restore-check.json）；當時是候選快照，現在正式失敗。
- checkpoints/s23-20261114-source.tar.gz保存本輪完成源碼與腳本；manifest含結果hash。

## 下一步與禁止重用

停止這組Hold殘差的係數／正則／步數／邊界微調。觀察性實際遊戲回報與
搜尋葉分布不同、七步標籤繼承凍結估值偏差，均是限制，尚未證明唯一失敗原因。
可研究少量候選的條件rollout或搜尋葉成本標籤；沿用原SRS/Hold/合法落點。
線上抽樣只能依公開七袋歷史，不可複製真實隱藏RNG後拿來評分。
未實作這些後續方法，不宣稱改善。仍優先完整100L方塊數，不以MSE或速度代替。

76200策略驗證、75300正式已開封；76400..76559與76600..76631已用資料學習。
76000..76031反覆開發，不是泛化證據。未來不同假設可用76700..76747新驗證、
75400..75463新正式，目前尚未使用。S11/S18等先前保留集也不可調參。
舊續接全文CONTINUE_HISTORY_20261114.md僅供歷史，若衝突以本檔及各lock為準。
