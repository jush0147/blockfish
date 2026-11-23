# Blockfish 100L Cheese Race Research Log

**Branch**: `rescue/independent-validation-20261116`  
**Objective**: Achieve **≤ 161 pieces** on official Jstris 100L Cheese Race (`block-stacker` `Ruleset::guideline()`, `previews = 5`, `total_lines = 100`, `min_height = 4`, `max_height = 9`, `BagTracker` only, zero peeking) — beating the 162-piece world record.

---

## 1. Exact Mathematical Laws & Constraints

1. **Zero Multi-Garbage Clears (`ds ∈ {0, 1}` Always)**:
   - Every intact garbage row has 9 filled cells and 1 empty hole, and consecutive garbage rows never share the same hole column (`offset ∈ 1..=9`).
   - Therefore, the set of empty cells across all garbage rows `0..g` has connected components of size **1**. No connected 4-cell tetromino can simultaneously occupy holes in 2 different garbage rows.
   - **Piece-count identity**:
     $$P = 100 + Z$$
     where $Z$ is the number of `0ds` moves (`ds == 0`).
   - **Cell conservation**:
     $$4P = 100 + 10K + R \implies P = 25 + 2.5K + \frac{R}{4}, \qquad Z = 2.5K + \frac{R}{4} - 75$$
     where $K$ is non-garbage lines cleared (`nds`) and $R$ is remaining stack cells at finish.
   - **Target Budget for $P \le 161$**:
     - Requires **$Z \le 61$** `0ds` moves and **$K \le 53.4$** `nds` lines (assuming $R \approx 10$).
     - **`O`-piece lower bound**: `O` (`2x2`) can never clear a 1-wide garbage row (`ds = 0` always). Across ~161 pieces, ~22–23 `O` pieces appear, forcing ~22–23 `0ds` moves alone.
     - Therefore, **non-`O` `0ds` moves must be reduced to ≤ 38–39** (from ~74 in current legal baseline and ~42.5 in `WG_ds_init` `164.60`).

2. **Exact Deterministic Windows Under Legal Observability**:
   - Under `min_height = 4, max_height = 9`:
     - On any clear (`lines > 0`), new garbage spawns only if `g_after < 4` (`num = max(0, 4 - g_after)`). Thus whenever `g >= 5`, clearing `g -> g-1` (`ds=1`) is **100% deterministic (0 spawned rows)**! Starting from `g = 9`, clearing `9 -> 8 -> 7 -> 6 -> 5 -> 4` has zero unknown hole randomness for 5 consecutive moves.
     - On a non-clear (`lines == 0`), `target_g = 9`, so `9 - g` unknown garbage rows immediately spawn if `g < 9` (unless `g == 9` or `rem_g == g`).
   - **7th-piece determinism**: On ~1/7 of turns (`bag_mask.count_ones() == 1`), `BagTracker` deterministically knows the 7th piece after `(hold, cur, next5)`. When `bag_mask.count_ones() == 2`, there are only 2 possible orderings for the next 2 pieces.

---

## 2. Verified Baseline Benchmarks

### 2.1 `ex_distill -- eval_h2_stk` (Seeds `101..=105`, prior verified run)
- `LEGAL_h2stk03_p5_b120`: `mean=196.60 min=190 max=205 top=0 nds=67.60 dt=30.91s`
- `LEGAL_h2stk03_nog0_p5`: `mean=195.40 min=190 max=200 top=0 nds=67.00 dt=19.85s`
- `ORACLE_p20_EXCH_b240` (non-legal upper bound): `mean=176.40 [175, 183, 165, 176, 183] nds=59.40 cov_nxt=12.6%`

### 2.2 `bench_legal -- baseline` (Legal 5-seed ablation `101..=105`)
- `r3_gbox_p5_b120`: `mean=199.20 [188..209] nds=68.00 0ds=99.20 (cov=35.0 O=20.2 nar=8.2 wide=31.8 g0=0.0 end=4.0) cov_nxt=17.6% avg_c=12.4 | 18.9s`
- `r4_gbox_p5_b120`: `mean=197.00 [191..204] nds=67.80 0ds=97.00 (cov=32.8 O=21.0 nar=8.2 wide=31.2 g0=0.0 end=3.8) cov_nxt=18.6% avg_c=11.7 | 31.8s`
- `h2stk03_p5_b120`: `mean=196.60 [190..205] nds=67.60 0ds=96.60 (cov=33.8 O=23.2 nar=8.4 wide=25.6 g0=0.0 end=5.6) cov_nxt=18.0% avg_c=11.7 | 24.4s`
- `h2stk03_nog0_b120`: `mean=195.40 [190..200] nds=67.00 0ds=95.40 (cov=33.0 O=21.4 nar=8.2 wide=26.2 g0=0.0 end=6.6) cov_nxt=18.2% avg_c=12.3 | 18.6s`
- `h2stk03_nog0_b480`: `mean=196.80 [192..204] nds=67.60 0ds=96.80 (cov=31.2 O=24.8 nar=6.0 wide=30.8 g0=0.0 end=4.0) cov_nxt=16.2% avg_c=12.5 | 115.2s`
  - **Key insight**: Scaling beam width 4x (`120 -> 480`) did NOT improve legal score (195.40 -> 196.80).

---

## 3. Root-Cause Diagnosis

1. **Evaluator Pathology (`models/crn_r3_h2stk03.txt`)**:
   - Model is **purely linear** (`mlp_weight = 0.0`) with many key features at **zero weight** (`h2_blocks`, `h4_blocks`, `near_full_9`, `near_full_8`, `h1_ds_clears_rg_open_h2`, `clean_skim_rows`).
   - `rem_g` weight is **`+2.1189`** per row while covering `h1` (`h1_blocks=+0.026`, `h1_open=-0.255`, `pe_net=+0.228`, etc.) totals only `~1.05–1.50` pieces! Thus clearing `h1` by covering `h2` (`ds_cov_next`) scores **better** than skimming/setting up cleanly (`cov_nxt=18.2%` legal, `12.6%` even in 20-preview Oracle).
   - `stack_cells` weight (`+0.030`) massively underprices stack mass (true conservation cost is up to `0.25` pieces/cell when not aligned with garbage clears), and `stack_cells_above1` is zero.
2. **Beam Search Physics & Lookahead Mismatch (`beam.rs`)**:
   - Main beam defaults to `simulate_cheese_spawn = false`, letting `g` hallucinate `4 -> 3 -> 2 -> 1 -> 0` (corrupting `h2/h3/h4` and triggering `g=0` features) and missing `0ds` height jumps back to `g = 9`.
   - `target_depth` stops at `total_known - 1` even when `bag_mask.count_ones() == 1` (deterministic 7th piece).
   - Endgame (`rem_g <= 9`, 100% deterministic known board) lacks a dedicated exact/deep solver (`end = 6.6` `0ds` moves still wasted).

### 3.3 Oracle Ceiling Ablation (`crn_r3_h2stk03` Plateau at ~176 Pieces)
Verified runs on seeds `101..=102` with exact `zds_o` (`placed_c == 'O'`) metric:
- `OR_p14_b160_h2nog0`: `mean=184.00 [187, 181]` (`0ds=84.00`, `cov=28.5 O=18.0 nar=8.5 wide=26.0 end=3.0`, `cov_nxt=17.5%`)
- `OR_p14_b240_h2nog0`: `mean=178.50 [179, 178]` (`cov=29.0 O=15.0 nar=5.0 wide=26.5`, `cov_nxt=15.0%`)
- `OR_p20_b240_h2nog0` (`trace101`): seed 101 = `175`
- `OR_p20_b500_h2nog0`: `mean=176.00 [175, 177]` (`nds=59.50 0ds=76.00`, `cov=23.5 O=16.0 nar=12.0 wide=22.0 end=2.5`, `cov_nxt=12.5%`, `avg_c=12.2`)
- `OR_p30_b240_mpr40`: `mean=177.50 [177, 178]` (`cov=31.0 O=17.5 nar=5.5 wide=20.5`, `cov_nxt=15.0%`, `avg_c=11.4`)
- Weight probes on `OR_p14_b240`: `rg=1.70` (`186.00`), `rg=2.60` (`180.50`), `rg=3.20` (`186.00`), `clean160` (`202.50`), `clean145` (`205.50`), `stratify_rem_g` (`191.00`), `prog_mpr` (`183.50`) — all worse than baseline `178.50`.
- **Actions Run #2 (`spatial_r1` / `spatial_r2`) postmortem**: `OR_p6=242.40` because `train_spatial.rs` trained ~97% on pairwise ranking imitating weak depth-6/12 beam preferences (`val_w=0.03`) and `B=8` rollouts using heuristic `clean_physics_cost(..., 1.88)` instead of true multi-move piece-to-finish outcomes.
- **Implication**: Breaking `<= 166` legally requires breaking the **Oracle ceiling down to <= 158–160** (cutting non-`O` `0ds` from `~60` down to `<= 42–44`). Simply widening/deepening beam with `m_h2_nog0` plateaus at `~176` because intermediate linear beam pruning (`d=2..10`) rejects temporary clean setups that save 2–4 pieces over the next 6–15 placements.

### 3.4 Whole-Game DS-Bucketed Oracle (`164.60 [160..170]`) & Round WG-1 Postmortem
- **Cell-Conservation Whole-Game DS-Bucketed Search (`diag_cheese -- wg_break`)**:
  - `WG_ds_init` (`ew_cons22` with `taper_rem=5`, `B=3000`, `strat_np=true`, seeds `101..=105`): **`mean=164.60 [160..170]`** (`per_seed=[163, 163, 160, 167, 170]`, `nds=54.80`, `0ds=64.60 (cov=19.2 O=17.8 nar=3.8 wide=22.0 end=1.8)`, `cov_nxt=13.2%`, `r_end=10.4`). Seed `103` hit **`160`**.
- **Round WG-1 (`models/spatial_wg1.txt`, GitHub Actions Run #6)**:
  - Spatial MLP improved `+41.4` pieces over its own linear skip (`OR_p6_b120_sp_lin = 247.20 -> OR_p6_b120_sp_full = 205.80`) and beat its `cons:crn_r3` teacher at `OR_p12` (`186.40/188.00` vs `192.00`).
  - Legal `p=5` (`bench_legal spatial5_q models/spatial_wg1.txt`): `sp_rpimc_w8_h9 = 191.80 [183..202]` (`cov=33.0`, `wide=29.4`, `end=4.0`).
  - **Root Causes of `+27.2` Gap (`191.80` vs `164.60`)**:
    1. Ridge regularization toward `0` collapsed `stk_cells = +0.000`, leaving linear skip dysfunctional (`OR_p6_sp_lin = 247.20`).
    2. Off-policy Oracle-only rollouts (`--roll-horizon 22`, no `--wg-ds`, `--cal-v-weight 0.0`) + overfitting after epoch 5 (`ep=5: 2.973 -> ep=30: 3.136`).
    3. `beam.rs` exact endgame only covered `rem_g0 <= 7` (`8..=9` already 100% deterministic!), `endgame_rollout_crn` used narrow `48/32` width, and `root_pimc_eval_world` checked only 4 spatial leaves.

- **Round WG-3 (`models/spatial_wg3.txt`, GitHub Actions Run #8) & Diagnostic Decomposition**:
  - `OR_p6_sp_lin = 210.00`, `OR_p6_sp_full = 203.40`, `OR_p12_best = 188.20`, `LEGAL_rpimc_flat_best = 194.60 [187..203]`, `sp_rpimc_sim_w8_h9 = 195.60 [192..200]`.
  - `diag_cheese` decomposition (`101..=102`): `WG_ds_b32_c22 = 196.00`, `WG_ds_b200_c22 = 177.50`, `OR_p30_ds16_b160_c22 = 178.00`.
  - `bench_legal fast2 models/spatial_wg3.txt ds3` (`101..=102`): `2_ew4_ds3_w6_b16 = 215.50 [210, 221]`, while `3_cons22_ds3_w6_b16 = 320.00` (1 topout, `cov=175`).
  - **Key Root Causes**:
    1. **Finite-horizon conservation pathology**: At finite horizon (`rem_g > 5`), `stack_cell = -22` with `nds_pen = 0` rewards hoarding cells upward (`cov=175` -> topout in flat/short beam), and `endgame_cell_taper` in [`fast_eval.rs`](file:///home/antigravity/jush0147/blockfish/blockfish-engine/src/ai/fast_eval.rs#L709-L715) ignores excess cells `stack_cells > rem_garbage` (which strictly cost `+0.25` pieces/cell).
    2. **MLP `±2.5` clamp & Phase 3 teacher destruction**: `mlp / (1.0 + mlp.abs() / 2.5)` in [`spatial_val.rs`](file:///home/antigravity/jush0147/blockfish/blockfish-engine/src/ai/spatial_val.rs#L898) caps nonlinear corrections at `[-2.5, +2.5]` pieces, while [`train_spatial.rs`](file:///home/antigravity/jush0147/blockfish/blockfish-engine/examples/train_spatial.rs#L3084-L3265) Phase 3 rollouts (`bw=24`) flipped **1,212** `WG_ds` winning pairs (`flip=62.1%`) and Ridge collapsed structural weights to bounds (`stk_cells` & `h1_blk`/`cov_holes` pinned to `lb`, `OR_p6_sp_lin = 210.00`).
    3. **Oracle `WG_ds` (`164.60`) still leaves ~8–10 `0ds` moves on the table**: `wide=22.0, cov=19.2, r_end=10.4` (`expand_cap` capped at `1500` in [`diag_cheese.rs`](file:///home/antigravity/jush0147/blockfish/blockfish-engine/examples/diag_cheese.rs#L3217)).

### 3.5 Rounds WG-4, WG-5 & Bottleneck Pivot to Direct Legal Policy Optimization
- **Round WG-4 (`models/spatial_wg4.txt`, Actions Run #11)**:
  - Unflipped CRN + `mlp_sat=±12.0` caused severe MLP cell-hoarding (`OR_p6_sp_full = 400.00`, 5/5 top-out; `LEGAL sp_rpimc_sim_w8_h9 = 204.40`).
- **Round WG-5 (`models/spatial_wg5.txt`, `models/spatial_wg5_ep0.txt`, Actions Run #12)**:
  - Gated MLP raw height/cells (`sparse 20..80`, `dense j==0, 3..=22`), tightened `mlp_sat` to `±5.0`, added `evaluate_linear_board_with_bag_no_pe` (`spatial_no_pe_internal`), relaxed collinear bounds, allowed CRN flip `< -0.55` (`crn_flipped=781`).
  - **Linear backbone & `no_pe` speedup succeeded**: `OR_p6_sp_lin` dropped `244.40 -> 204.40` (0 top-out); `sp_rpimc_linfast_nope_w8_h9` matched `sp_rpimc_sim_w8_h9` at `198.20` while running **2.67× faster** (`149.8s` vs `400.1s`).
  - **Remaining structural bottlenecks**:
    1. **MLP still hoards via 2D occupancy bits (`sparse 0..20`, `120..160`)**: `OR_p6_sp_full = 323.80` (3/5 top-out), `LEGAL_rpimc_flat_ep0 (194.80)` beat `best (201.00)`.
    2. **Ridge attenuation bias**: `h1_cc + h1_blk + cov_holes + blk_over_h = +0.89 < rem_g (+1.62)`, causing greedy `h2`-covering downstacks (`cov_nxt = 18–20%`, `cov = 33–36`).
    3. **Bounded cell-conservation (`cons_cap`) & Deep Determinized DS-PIMC (`deep_ds_pimc`) negative results**:
       - `cap14/cap18` (`stack_cell=-22` up to cap) still degraded Legal `ds3` (`266.50 / 310.00`).
       - `deep_c22_ds8_w8_b24` (DS-bucketed `ds_target=8` with `strat_np` + `ew_cons22` internal + `wg5_ep0` leaf) got `208.00` (0 top-out, `nar=7.5`, but `cov_nxt=20%`) — proving **Strategy Fusion** when determinizing 10+ unknown pieces in PIMC (`h=5: 195.4` → `h=8: 191.8` → `ds=8 (h≈15): 208.0`).
- **Pivot — Direct Legal `p=5` Optimization (`examples/opt_legal.rs`)**:
  - Instead of indirect supervised regression to noisy/confounded Oracle rollouts, directly optimize the pure-linear `SpatialValueModel` (99 params, `rem_g` anchored for piece-unit calibration) on **actual Legal `p=5` gameplay** via Antithetic Evolution Strategies + Adam + CRN (`spatial_no_pe_internal`, ~3s/eval on 4 cores), then stack flat `h=8` PIMC (`-15` to `-19` pcs lift) and non-clairvoyant rollouts on top.

- **Round BC-1 (`models/spatial_bc1_lin.txt`, `models/spatial_bc1.txt`, Actions Run #15 — NEW ALL-TIME LEGAL RECORD)**:
  - **Full-Action-Space Afterstate Cross-Entropy BC** (`examples/train_bc.rs`) on 120 `WG_ds B=1400` games (`167.21` mean, `19,945` decisions, `918,229` legal moves, `needs_pe = false`, ZERO Phase-3 rollouts):
    - `val_ce = 0.9158` (LIN, `top1=71.1%`, `top3=92.5%`, `rem_g=1.683`) / `0.8946` (MLP, `top1=71.3%`, `top3=92.7%`).
    - **`LEGAL_pimc_w8_bc_lin`: `188.80 [183..196]`** (0 top-out, `126.5s`) — **breaks 190-piece barrier**, beating previous all-time record `spatial_wg1` (`191.80`) by `-3.0` pcs and WG-5 (`198.20`) by **`-9.4` pcs**!
    - `sp_rpimc_linfast_nope_w8_h9` (`bc1_lin`): **`189.20 [182..195]`** (`cov=26.4`, `147.3s`).
    - `lin_nope_pre3_w12_h11_b28` (`bc1_lin`): **`189.00 [183..203]`** (`wide=23.8`, 4/5 seeds `<= 190`, but `cov` rose to `33.4` due to `h=11` Strategy Fusion).
    - **`LEGAL_pimc_w8_bc_mlp`: `191.60 [188..195]`** (0 top-out — completely eliminated WG-4/WG-5 MLP cell-hoarding!).
  - **Round BC-2 (`models/spatial_bc2_lin.txt`, `models/spatial_bc2.txt`, Actions Run #16 — NEW ALL-TIME LEGAL RECORD `186.00`)**:
    - 120 clean `WG_ds` + 40 Legal BC-DAgger games (`3,476` recovery decisions, `1,079,664` legal moves, `wd_mlp=0.015`):
    - **BC-DAgger cured MLP off-policy overfitting**: on plain `p=5` beam (`LEGAL_b120`), **`bc2_mlp` (`198.20 [191..203]`) beat `bc2_lin` (`201.20`) by `-3.0` pcs** (first time plain `p=5` beam broke `< 200`!), and `sp_rpimc_sim_w8_h9` improved from `192.20` (`bc1_mlp`) → `188.75` (4/5 seeds before 90m cutoff).
    - **CVaR Anti-Strategy-Fusion breakthrough (`bc_cvar` on `bc2_lin`)**:
      - `bc_h8_w16_b32` (no CVaR): `191.20` (`cov=33.2`, `cov_nxt=20.0%`)
      - `bc_h8_w16_b32_cvar25`: **`186.60 [177..192]`** (**`-4.6` pcs**, `cov=29.0`, `end=3.4`, **seed 103 hit `177`!**)
      - **`bc_h9_w16_b32_cvar30`**: **`186.00 [182..195]`** (**`-5.2` pcs**, `nds=63.20`, `nar=7.4`, `wide=26.2`, `cov_nxt=16.4%`, 3/5 seeds `<= 183`!)
  - **Round NC-1 (Actions Run #17) & BC-2 `spatial_probe`**:
    - `spatial_probe`: **`bc2` full MLP hit `OR_p6 = 199.40 [195..208]`** (first time `OR_p6 < 200`, `mlp = -0.01..-0.46`, zero hoarding!), beating `bc2_lin` (`200.00`) and `h2nog0` (`204.20`).
    - Non-Clairvoyant 1-ply rollout (`nc_h5_nc10 = 245.60`, `nc_h6_nc12 = 250.60`, `mlp_h6_nc12 = 254.00`): negative result — 1-ply greedy rollout is too weak and adds noise (`avg_c = 15.9`), confirming `b=32` in-world beam + CVaR (`186.00`) is strictly superior.
    - `bc2` MLP + CVaR: `mlp_h9_w16_b32_cvar40 = 188.20 [180..196]`, `cvar30 = 188.40` (`nar=5.6`).
  - **Round ES-3 & CVaR-Pro (`models/spatial_es3.txt`, Actions Run `45086995056` — NEW ALL-TIME LEGAL RECORD `185.80`)**:
    - **CVaR-Pro on `models/spatial_bc2_lin.txt`**:
      - **`pro_h9_w20_b36_cvar35`**: **`185.80 [182..193]`** (`[183, 193, 186, 185, 182]`, current all-time Legal `p=5` record!).
      - `pro_h10_w20_b40_cvar40`: regressed to `189.40` (`cov=40.0`), confirming `h=9` is the Strategy Fusion boundary.
    - **Fixed Direct Legal ES-3 (`opt_legal.rs`, unclamped `dense[3..=22]`, `b=64`, NO PIMC)**:
      - Improved training `b=64` from `200.62 -> 196.22`, but held-out `101..=105` `cvar_top2` (Run #20 `45097331474`) regressed slightly vs `bc2_lin` (`plain_p5_b120=202.40`, `h9_w16_b32_cvar30=187.20`, `h9_w20_b36_cvar35=187.80`) due to small-sample `b=64` overfitting.
  - **Round BC-3 (`SPATIAL_VAL_V4`, `models/spatial_bc3_lin.txt` & `models/spatial_bc3.txt`, Actions Run #21 `45098011902` — NEW ALL-TIME LEGAL RECORD `183.60`)**:
    - Upgraded `SpatialValueModel` to `SPATIAL_VAL_V4` (`SPARSE_DIM: 186 -> 330`, `DENSE_DIM: 74 -> 82`, 100% V1/V2/V3 backward-compatible): added adjacent column height diffs `dh[0..9]` (`63` bits), exact `h1`/`h2` 3-col contour (`32` bits), nonlinear `h1/h2/h3/holes` cover buckets (`16` bits), `row_g` skim-to-uncover-h1 & hand tactical interactions (`30` bits), and 8 dense scalars.
    - Trained on `150` `WG_ds (B=1600)` + `50` Legal DAgger games (`29,352` decisions, `1,356,107` legal moves, `needs_pe=false`):
      - **`plain_p5_b120` (ZERO PIMC)**: dropped **`-7.80` pcs** (`201.20 -> 193.40 [190..197]`)!
      - **`pro_h9_w16_b32_cvar30`**: **`183.60 [173..189]`** (**NEW ALL-TIME LEGAL RECORD**, `-2.20` pcs over `185.80`; `cov` dropped `32.2 -> 27.0`, `cov_nxt` `16.4% -> 14.8%`, **seed 103 hit `173` legally!**).
      - **`pro_h9_w20_b36_cvar35`** (4/5 seeds before 90m cutoff): `101:179, 102:188, 103:177, 104:185` (**4-seed mean = `182.25`**!).
    - Remaining gap (`183.60` vs `WG_ds 164.60`): `cov=27.0 (+7.8)`, `nar=8.6 (+4.8)`, `wide=24.8 (+2.8)`, `end=6.0 (+4.2)`, `O=17.2 (-0.6)`.
  - **Round BC-4 (`SPATIAL_VAL_V5` + Spatial-guided Endgame, `models/spatial_bc4_lin.txt` & `models/spatial_bc4.txt`, Actions Run `45116823389` — NEW ALL-TIME LEGAL RECORD `183.40`)**:
    - Upgraded to `SPATIAL_VAL_V5` (`374` sparse / `90` dense: exact `h1_clean_fit` without covering `h2`, 5-col outer contour around `h1`/`h2`, garbage height regime bits) + wired `SpatialValueModel` into `solve_endgame_ds` / `endgame_world_ds`.
    - Trained on `150` `WG_ds (B=1800)` + `60` Legal DAgger games from `spatial_bc3_lin.txt` (`30,063` decisions, `1,389,534` legal moves): `val_ce = 0.9773` (LIN, `top1=70.4%`) / `0.9662` (MLP, `top1=70.7%`, `top3=92.0%`).
    - **`LEGAL_pimc_w8_bc_lin` (`w=8, h=8, b=24, roots=8`, 149.6s)**: **`183.40 [179..190]`** (**NEW ALL-TIME LEGAL RECORD**, `[179, 183, 182, 183, 190]` — **`-5.80` pcs** faster/stronger than BC-3's `w8 = 189.20` and 5× faster than `h9_w16`!).
    - `LEGAL_b120_bc_mlp`: **`194.80 [191..203]`** (4/5 seeds `<= 194` with ZERO PIMC!); `end` in `pro_h9_w16_b32_cvar30` dropped `6.0 -> 4.2` (`-1.8` pcs saved in endgame).
    - Next: `v5_search_break` sweep scaling the `183.40` `h=8` winner (`w=16..20, b=28..32, cvar=0/25/30`), testing short-horizon DS-bucketed PIMC (`ds_target=5` with V5 internal + CVaR-30), and testing `spatial_bc4.txt` (V5 MLP) at `h=8` + CVaR-25.

---

## 4. Roadmap

1. **Direct Legal `p=5` Policy Optimization (`opt_legal.rs` → `models/spatial_es1.txt`)**:
   - Run 35+ Antithetic-ES generations directly minimizing Legal `p=5` piece count (eliminating Ridge shrinkage and teacher confounding), then verify held-out `101..=105` with both `b=120` beam and `linfast_nope` flat `h=8` PIMC.
2. **Non-Clairvoyant Rollout PIMC (Zero Strategy Fusion)**:
   - Replace clairvoyant deep determinized search with fast non-clairvoyant `p=5` rollouts inside CRN worlds so deeper lookahead improves rather than degrades Legal play.
3. **Break Legal `p=5` Through `180 -> 170 -> <= 161` Pieces**:
   - Validate all breakthroughs across `5 -> 20 -> 40` held-out seeds and continuously commit/push progress.
