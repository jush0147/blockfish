# S29/S30：全域搜尋排程直接對局實驗

## 選擇理由與事前假設

S22–S28沒有證明B*到達極限，卻指出額外模擬與新標籤可能引入端點偏差、
策略落差與有限樣本不確定性。本輪不再延伸那些診斷；固定估值與robust4，
直接替換探索完整已知預覽的排程。實驗成本限定兩個事前固定的搜尋器。

S29以每個子節點相對同父最佳子節點的f差，沿路相加，作為探索優先值。
這是啟發式差值，不是真實regret或成本下界。兄弟全部枚舉後才確定最小值，
差值非負；相同優先值先較深、再較低f、再先生成者，以盡早取得完整方案。
假設：相對偏差可減少不同深度絕對估值偏移對排程的支配，探索局部合理的
替代方案；也可能錯罰必要的暫時惡化，必須由完整100L結果判斷。

S30同一全域frontier，直接按絕對f排序，其他同分規則與回傳不變。
它檢驗取消B*固定下探後、由所有深度自由競爭的效果。這與舊beam不同：
沒有固定寬度或逐層丟棄狀態；也不強迫所有根先獲評分或早期淘汰根。
兩版都只在合法終點用原f+parent_f與robust4排名根動作，保留短完成偏好。
不改權重、不取未知方塊、不模擬垃圾、不新增狀態剪枝或訓練。

收益預期是改變50k內找到的完整方案，而非只提升速度。開發成本96局，
包含兩候選與新盤面上的凍結robust；沒有參數掃描。即使窮舉目標相同，
有限預算品質可能不同，不能由更多終局或更低CPU替代方塊數證據。

## 正確性與計費

原State、PlaceFinder/SRS、Hold、Evaluator、Node::succ/rating與Leaves沿用。
新global.rs只負責全域frontier、完整兄弟枚舉與排程；每step至多生成一個
子節點，root亦計費。預算終止時未完成兄弟批次也已收費，不免費補算。
全部已生成節點包括terminal都計入同一50k，無額外B*暖身預算。

兩版各68tests通過，包括4欄小盤面窮舉所有根評分與原B*完全相同、
逐條建議按原合法落點重播、不同節點上限與重設狀態。
S11/S28全部checkpoint manifest檔案hash開始前核對未變。

## 固定資料與選擇規則（開啟結果前）

全新開發77000..77031，三版各32，交錯4workers，current+5、7bag、100L、
垃圾9/4、50k生成節點、無deadline。固定timeout180s，任何失敗都保留。
兩候選各用相同cohort接受mean gate：完成率不差於另兩版且平均較低於
另兩版，才取得唯一入選資格。若相同均值而無唯一勝者則不進驗證。
這是兩個預定候選的模型選擇，不冒充獨立確認。

入選後固定二進位，再開尚未使用76700..76747獨立驗證，三版為入選者、
frozenrobust、frozenlegacy。平均與完成率均勝兩控制，才開75400..75463
正式64盤，要求全完成且對兩控制配對95%區間全負。不得加樣本挽救。
失敗則保存必要統計與回放，不追加優先值混合、同分順序或寬度掃描。

## 結果

新77000..77031完整96筆嘗試均保留：

|版本|完成|平均方塊（完成者）|CPU秒／完成局|
|---|---:|---:|---:|
|凍結cheese-robust|32/32|196.15625|15.75782|
|S29全域相對差值|32/32|201.65625|21.22714|
|S30全域絕對f|28/32|196.07143|15.16465|

S29−robust +5.5，配對95%[2.96875,7.90625]，6勝1平25敗；拒絕。
S30的196.07僅來自28個成功案例，不能與基準全部32盤196.16直接比較並宣稱
改善。在相同28盤上，S30反而多0.25塊，區間[-2.57143,2.85714]；這仍是
成功子集的條件式描述，無法忽略4次失敗。S30完成率門檻明確失敗。

全部92個完成對局由凍結原模擬器完整回放通過；64個候選嘗試的每個已記錄
分析都通過50k生成上限稽核，包含失敗分析。總CPU含失敗1644.949169秒。
四次失敗在seed77007/77027/77000/77028，分別第12/187/86/180次分析
用盡50k而沒有完整可回傳序列；race CLI隨後no-suggestions panic。
它們未寫出最終操作trace，不能宣稱已回放；stderr、returncode與費用保留於
原96筆cohort及s29-s30-failures.json。沒有重跑或剔除失敗計算選擇門檻。

## 決策與必要診斷

兩個事前固定候選皆未通過門檻，沒有進入76700驗證或75400正式集。
最佳仍為原cheese-robust，不掃描局部差值權重、同分規則、寬度或增加暖身救援。

S29平均每步1831.44個完整終局、7.62個已評分根，仍明顯增加方塊數；
更多完整方案不是收益保證。局部估值差可能錯罰必要的暫時惡化，但本輪沒有
唯一識別此原因，不為這個負結果再擴診斷。S30則直接顯示此全域排程在有限
預算內缺乏完整序列保障；這不證明所有全域搜尋都無效，也不代表B*已達極限。

`decide_cohort.py`原先假設search欄位必為list，遇到失敗案例的null而中斷。
修正為對失敗的缺失search使用空list；成功案例仍要求逐步search完整。
用舊S27全部完成cohort回歸核對，決策JSON完全相同。這是統計工具修正，
不是改候選演算法、補救失敗或變更門檻。96筆原cohort是唯一品質決策輸入；
complete-only檔只供回放，不供品質選擇。

## 重現

```sh
source /workspace/.blockfish-env/activate.sh
python3 research/build.py robust-global-regret robust-global-f
python3 research/bench.py --binaries research/bin/robust-global-regret research/bin/robust-global-f research/bin/learned-robust --configs 50/5,10,10,10 --start 77000 --count 32 --jobs 4 --out research/results/s29-s30-dev.jsonl
python3 research/decide_cohort.py --lock research/results/s29-dev-lock.json --inputs research/results/s29-s30-dev.jsonl --out-prefix research/results/s29-dev --gate mean
python3 research/decide_cohort.py --lock research/results/s30-dev-lock.json --inputs research/results/s29-s30-dev.jsonl --out-prefix research/results/s30-dev --gate mean
python3 research/audit_global.py research/results/s29-s30-dev.jsonl --out research/results/s29-s30-search-audit.json
python3 research/replay_audit.py research/results/s29-s30-complete-only.jsonl --out research/results/s29-s30-replay-audit.json
```

兩候選binary/sourcehash由builds.jsonl保存，source checkpoint與結果manifest
另外封存。這是本地模擬結果，沒有官方Jstris成績或突破紀錄的證據。
