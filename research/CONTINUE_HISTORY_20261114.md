# 最新：S22正式測試執行中（2026-11-14）
S22新驗證76200..76247已通過平均門檻：196.6875 vs robust197.833333、legacy198.354167，
三版48/48；對robust差-1.145833 CI[-3.520833,1.4375]，仍跨零。144回放通過。
不升級，候選/權重/節點預算不變。

正在開封75300..75363三版64正式，共應192筆：
python3 research/bench.py --binaries research/bin/robust-hold-order research/bin/learned-robust research/bin/learned-fast --configs 50/5,10,10,10 --start 75300 --count 64 --jobs 4 --out research/results/s22-final.jsonl
先查進程，勿重複workers。完成後：
python3 research/decide_cohort.py --lock research/results/s22-final-lock.json --inputs research/results/s22-final.jsonl --out-prefix research/results/s22-final --gate ci
再全回放。要對兩個控制配對95%區間全負且64配對完整才升級。
不得看結果後擴樣本或改權重。若失敗保持cheese-robust，不混入開發/驗證來挽救。
目前候選CPU約22秒/局、robust15.6，所以只有同50k生成預算公平，未證明同CPU時間收益。
如果通過品質門檻，還需成本公平性與預算泛化檢查，不宣稱通用或官方紀錄。

候選source已封存並離線還原建置，與frozen binary SHA完全相同：
research/checkpoints/s22-candidate-20261114-source.tar.gz
results/s22-restore-check.json。模型及資料依賴hash另見s21-fit-dependencies.json。

附加S23精確cache實驗執行中：optional hold-context-cache、profile robust-hold-cached。
每個analysis cache盤面→7種Hold修正vector，仍按實際Hold取值，算術次序相同。
69測試通過。s23-parity.jsonl應6筆，兩版76000..76002交錯jobs1；需完整
inputs/ds及search計數（排除seconds）全一致。不得替換正式測試執行檔。
如果無收益或非等價就不使用；純實作成本實驗，不是方塊數提升證據。

以下舊記錄：

# 最新：S22新驗證執行中（2026-11-14）
S22dev通過：195.96875 vs robust197.1875，32/32，差-1.21875 CI跨零。
候選robust-hold-order、權重已固定；32回放通過。不是已證實提升。

正在新76200..76247三版48盤驗證，共應144筆：
python3 research/bench.py --binaries research/bin/robust-hold-order research/bin/learned-robust research/bin/learned-fast --configs 50/5,10,10,10 --start 76200 --count 48 --jobs 4 --out research/results/s22-validation.jsonl
先查進程再續跑，勿重複workers。完成後：
python3 research/decide_cohort.py --lock research/results/s22-validation-lock.json --inputs research/results/s22-validation.jsonl --out-prefix research/results/s22-validation --gate mean
全回放。要平均勝兩對照且完成率不差才開75300..75363三版64正式。
正式需對兩控制配對95%區間全負且全部64配對完成，不擴樣本。
驗證未完成前不看中途方塊數、不調模型。最佳cheese-robust不變。

以下舊記錄：

# 最新：S21拒絕，S22執行中（2026-11-14）
S21完整32盤198.15625 vs robust197.1875，未通過。相對S19則少2.65625，
但不能因勝過弱對照升級。32回放全通過。詳見CONTEXT_REPORT.md。

S22保持S21權重，移除保留最後Hold的邊界改變，使用原robust horizon。
只改中間狀態與parent_f估值，非純排序（parent_f也影響rating）。
hold-context不再依賴preserve-hold；build.py的S21profile明確加入preserve-hold，
其凍結binary保持不變。S22profile robust-hold-order，68測試通過。
權重hash與S21相同。32dev76000..76031、50k，results/s22-dev.jsonl執行中。
先查進程，勿重複。完成：
python3 research/decide_cohort.py --lock research/results/s22-dev-lock.json --inputs research/results/s16-control-dev.jsonl research/results/s21-dev.jsonl research/results/s22-dev.jsonl --out-prefix research/results/s22-dev --gate mean
全回放。要勝兩對照才開新策略驗證76200..76247，原正式75300仍封存。
不要再掃正則/步數/權重；這是固定模型最後的結構消融。

以下舊記錄：

# 最新續接：S20完成，S21完整對局執行中（2026-11-14）

最佳cheese-robust不變。S20新增公開Hold到BoardSample匯出，context-source
是cheese-quality，3局完整操作與凍結learned-fast相同。新種子76400..76559
160局5k source全部完成/回放通過。128訓練76400..76527，32預測驗證76528..76559。
固定ridge10，中心化Hold種類×[bias,height,piece_estimate,i_dependencies,wells]，
等遊戲權重。整局成本標籤S20驗證MSE40.28499→40.32575，失敗，未接入搜尋。

S21固定7步＋凍結board value標籤，訓練同128，新預測驗證76600..76631，
32source全完成/回放。MSE2.735080→2.722706，差-0.012374 CI跨零。
僅通過事前平均門檻，不能宣稱策略提升。fit_hold.py --td7可重現，結果
s21-hold-fit.json、weights；S20不帶--td7。沒有正則/步數掃描。

實作hold-context optional feature（依賴preserve-hold/learned-value），
在Evaluator.score對盤面cache之後加Hold修正，State.held_color只回實際Hold，
空Hold不能用next().1。68測試通過。robust-hold-context二進位已凍結，
權重blockfish-engine/src/ai/hold_weights.rs。分布差異仍存在：真實current+5
回報與只剩Hold的搜尋葉不相同；只看完整遊戲效果決定去留。

正在跑research/results/s21-dev.jsonl，32盤76000..76031、50k、4workers。
先查進程，勿重啟重複workers。完成後：
python3 research/decide_cohort.py --lock research/results/s21-dev-lock.json --inputs research/results/s16-control-dev.jsonl research/results/s19-dev.jsonl research/results/s21-dev.jsonl --out-prefix research/results/s21-dev --gate mean
再全回放。要同時勝過凍結robust及S19preservehold才進新策略驗證。
新策略驗證76200..76247、正式75300..75363仍未使用；不可重用已開S18正式。
失敗則不掃ridge/horizon，保留結果並檢討觀察性標籤與葉分布。

以下舊狀態：

# 最新續接（2026-11-14，S18/S19完成）

最佳仍為cheese-robust，50k生成節點；S11快照與全部凍結二進位hash核對未變。

S18正式75200..75263：推論195.046875、robust placebo195、legacy198.40625，三版64/64。
推論−robust+0.046875 CI[-1.90625,1.953125]，拒絕升級；192回放全部通過。
128局公開歷史稽核3628次推論，3612次後續確認、16次末步僅邏輯核對。
S18開發/驗證/正式均完整，詳見BAG_REPORT.md、s18-final-decision.json、s18-outcomes.svg。
正式75200..75263已開封，禁止調參或擴樣本。observer目前只接race入口。

S19隔離preserve-hold失敗：32盤76000..76031、50k，200.8125 vs197.1875，
差+3.625 CI[0.75,6.5]，10勝5平17敗，皆完成；32回放、65引擎測試通過。
完整序列與根覆蓋增加卻未減少方塊。詳見HOLD_REPORT.md與s19-allocation.json。
候選robust-preserve-hold可重建且保存，不升級。停止邊界微調。

沒有執行中的實驗。新預留76200..76247驗證、75300..75363正式尚未使用。
已開封S17驗證75100、S18驗證76100、S18正式75200不能當新集。
下一步應研究條件葉估值/少量候選rollout或其他基本方法；現有14維估值
看不到Hold種類，改葉邊界卻不改訓練分布可能有問題，但尚無因果證明。
不要因S18推論邏輯正確、S19覆蓋變多就宣稱策略提升。所有選擇看完整100L方塊數。

以下是舊記錄，與上方衝突以上方為準：

# 研究續接狀態（2026-11-13，S11–S15完成）

S16已拒絕：分支強制多樣性32/32敗，232.78125 vs197.1875。
S17分層版開發195.40625 vs197.1875，但新驗證75100..75147：
較早版198.875、穩健版197.75、分層198.1875，皆48/48完成。
相對穩健+0.4375，CI跨零；正式75200未開啟。停止再調平均規則。

S18七袋推論版開發已通過：76000..76031，194.625 vs placebo197.1875，
差−2.5625 CI[-4.84375,-0.375]，皆32/32完成。全部32盤placebo與凍結
基準完整操作一致。64份回放與公開歷史稽核通過：1832次推論、1818次
後續揭露確認、14次最後決策僅邏輯核對。

現在正在NEW76100..76147做三版48盤驗證，共應144筆：
python3 research/bench.py --binaries research/bin/robust-bag research/bin/robust-bag-placebo research/bin/learned-fast --configs 50/5,10,10,10 --start 76100 --count 48 --jobs 4 --out research/results/s18-validation.jsonl
先查進程再續跑。候選已固定，驗證結果尚未看。s18-validation-lock.json含hash。
完成後summary expected-count48、全回放，公開歷史稽核用：
python3 research/audit_bag_observer.py research/results/s18-validation.jsonl --no-parity --out research/results/s18-validation-public-audit.json
此輪不重做placebo凍結操作parity（開發32盤已全核對）；legacy無observer
事件，audit明確列為skipped_non_observer，不能宣稱其也作了歷史稽核。

候選完成率不差且平均同時勝過兩對照才開75200..75263三版64盤正式；
正式對兩控制配對95%區間均全負才升級。不擴大正式樣本。75100已用於S17，
不能再叫獨立驗證。75200仍未使用。

Observer只收公開Color串流：初始current+5，以及每次鎖定後新揭露1顆，
首次空Hold則2顆；剩唯一類型時加入搜尋，placebo觀察但不使用。沒有seed/
RNG/隱藏佇列資訊。僅適用已知從新七袋開局的標準遊戲；不可在未知袋起點
的任意中盤初始化。新增observer由race CLI接線，其他AI客戶端尚未自動接線。
詳見PROTOCOL S18及BAG_REPORT.md。

上一輪狀態：S14/S15新增48局全部完成／回放通過，皆拒絕。
詳見TAIL_REPORT.md：逐葉均勻未知下一顆延伸，以及保留已知Hold的
邊界延伸，平均199.9167／202.0，基準195.3333。不要因其CPU較低
或勝過自身較弱安慰劑而升級。七種枚舉壓縮已知序列搜尋約773→202。
後續需要重新分配精評位置或驗證條件模型，不要繼續微調逐葉延伸。
新推薦：cheese-robust，**每步50k生成節點**。
原cheese-quality / research/bin/learned-fast完整保留為可靠基準。

S11固定估值與B*展開，第一步按最佳4個不同終局平均評分；未訓練新權重。
- 開發12盤195.3333 vs196.9167。
- 新驗證75000..75047，48盤196.5625 vs197.47917，CI跨零。
- **正式62200..62263，64盤196.921875 vs199.15625，差−2.234375，
  95%配對bootstrap[-4.4375,-0.125]，兩版64/64完成，39勝2平23敗。**
  通過事前門檻，保留版本；區間接近零，證據只限此模擬器50k預算。
- 同驗證集5k：204.3958 vs202.5，差+1.8958，CI跨零；不推薦低預算。
- 安慰劑12盤與基準完整操作一致。新profile3盤與凍結候選完整操作一致。

S12 robust+root-complete：開發197.0 vs robust195.3333，拒絕。
S13 robust+endgame-wide：開發12/12同方塊數，195.3333，拒絕。
目前不要再調k、重複根覆蓋或收尾閾值以尋找開發收益。
預覽邊界已試S14/S15，失敗細節見TAIL_REPORT.md；更根本的候選級精評
或條件模型仍未實作／驗證。亦可增加獨立訓練種子及模型選擇集。

- 371局全完成與全回放通過，124個獨立seed，詳見s11-s13-completion.json。
- GENERALIZATION_REPORT.md完整論證、結果、圖與指令。
- audit_s11_final.py可重現正式預算／hash／seed／回放與升級判定。
- research/checkpoints/s11-20261113-source.tar.gz保存已實際解壓編譯的
  源碼快照；同名manifest.json含hash。使用race,cheese-robust，其他
  optional研究flags不屬於推薦版本。research/bin/learned-robust是凍結
  正式候選；research/bin/cheese-robust是相容profile建置。
- 62200..62263正式集已使用完，不再用於調參；75100..75147新驗證、
  75200..75263新正式仍未開啟。
- 原Jstris規則／RNG未完全核對，不能宣稱官方紀錄。

前輪S9/S10失敗：32訓練種子上49次選擇，訓練−7.53、獨立+2.22。
六盤貢獻70.5%訓練淨收益；不要把1568重跑局數當獨立樣本數。
診斷腳本diagnose_generalization.py保存描述性結果，不宣稱唯一因果根因。
