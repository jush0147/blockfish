# S34　對稱盤面的候選排名

S11 的四終局支持改善過第一步選擇。S24–S30 改變同一棵樹的搜索配置，多次失敗；S31 歷史路徑引導亦未通過正式門檻。S34 測試另一種取得支持的方式：把公開盤面、Hold 與預覽左右反射（J/L、S/Z 交換），在兩個視角各用 B* 和原 robust 評分。不同生成順序或有限預算可能產生互補的候選排名。S32 的幾何權重與 S33 的截斷均未啟用。

枚舉、計費兩側的第一步，剩餘節點平分。以實際方塊類型、Hold 及消行前四個格子的座標精確匹配。只在原盤面已有完整評分的候選中，以兩側排名倒數之和選擇，缺少的鏡像票為零，同分保留原排序。左右 SRS 可達性不必假設完全等價。執行操作由原盤面重新建構。兩側只有一側找到的高品質路徑、以及分數差的大小，都是這項排名規則可能捨棄的資訊。

使用 32 局舊開發軌跡（76800..76831），事前固定取每局鎖定第 50 與 100 塊後的公開盤面。進入完整對局需至少 48 個盤面有共同評分的可執行根，且至少一次合併後改選。預註冊與新種子切分記在 PROTOCOL.md。結果為 64/64 可匹配、0/64 改選、0 次備援。平均每步的根映射計費 90 個節點，兩視角各 24,955。原視角平均 4.30 個已評分候選，4.27 個亦有對側評分。固定合併規則沒有在篩選盤面上產生額外決策收益的路徑。這項檢查沒有比較兩側各自的第一名，也沒有比較半額與全額原搜尋。無法由此推估完整對局的方塊數差。

兩次執行的逐盤 JSON 完全相同。一次重測含資料讀取共 5.90 CPU 秒，最大 RSS 24,580 KiB。67 項引擎測試通過，包括反射兩次還原與合法格子映射。32 局來源歷史均已完整回放。依固定門檻結束 S34，77600..77623 及 77700..77763 均未啟用。

重現：

```sh
source /workspace/.blockfish-env/activate.sh
python3 research/build.py robust-mirror
cargo build --release --locked -p blockfish --features race,mirror-fusion --bin blockfish-mirror-probe
python3 research/probe_mirror.py
python3 research/replay_audit.py research/results/s34-source-replays.jsonl --out research/results/s34-replay-audit.json
```
