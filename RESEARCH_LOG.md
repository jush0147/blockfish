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
    - **`v5_search_break` (Actions Run `45124242227`) & `OR_p20` Diagnostic (`spatial_bc4_lin.txt`)**:
      - `h8_w16_b28_r12_cvar25`: `185.00 [181..190]`; `ds5_w16_b24_r12_cvar30`: `187.40`; `bc4` MLP `cvar16`: `186.00` (seed 103 = **`173`**, seed 101 `cov=19`).
      - **Key Root-Cause Discovery (`diag_cheese spatial_or_p20 models/spatial_bc4_lin.txt`)**: Even with **20 previews + 100% exact holes + `b=240`**, `spatial_bc4_lin.txt` only hits **`176.00 [175, 177]`** (`nds=59.50`, vs `WG_ds` `163.00`, `nds=54.80`).
      - **`nds_expectimax` (Actions Run `45134409140`)**: Proved `nds_pen > 0` in finite-horizon beam is mathematically equivalent to rewarding `stack_cells` (`C_d = C_0 + 4d - ds - 10*nds`) and degrades Legal play (`avg_c: 12.2 -> 20.3`, `w8_h8: 183.40 -> 186.40 -> 208.00`). Exact 7-bag 1-ply Expectimax (`nc_exp_rh0`) improved `plain_b120` by `-4.0` pcs (`199.20 -> 195.20`) but trailed `w8_h8` (`183.40`).
      - **`full_mlp_pimc` (Actions Run `45144768300` — NEW ALL-TIME LEGAL RECORD `181.20 [175..187]`!!)**:
        - Running **Full V5 MLP (`models/spatial_bc4.txt`) INSIDE the PIMC inner beam (`spatial_rerank_only = false`)** broke the 183-piece barrier with a clean monotonic scaling curve:
          - `fullmlp_w8_h8_b24_cvar0`: `187.80` (`182.6s`)
          - `fullmlp_w12_h8_b24_cvar25`: `184.60` (`279.1s`, `-3.20` pcs)
          - **`fullmlp_w16_h8_b28_cvar25`**: **`181.20 [175..187]`** (`[101:187, 102:184, 103:180, 104:180, 105:175]`, `nds=61.40`, `wide=22.8`, `O=17.6`, `nar=6.8`, `end=4.2`, `459.1s` — **3/5 seeds `<= 180`**, `-3.80` pcs better than `bc4_lin` on the exact same config `185.00`, AND faster!).
      - **Round BC-5 (`128x48` V5 NNUE + Dropout `0.15` + LS `0.02`, `models/spatial_bc5.txt` & `models/spatial_bc5_lin.txt`, Actions Run `45152685776`)**:
        - Trained on `350` `WG_ds (B=900, mean=168.61)` + `50` DAgger (`spatial_bc4.txt`) games (`63,011` decisions, `2,907,240` legal moves): achieved all-time best `val_ce = 0.8974` (`top1=71.8%`, `top3=92.3%`, `-0.0688` CE better than BC-4!), `LEGAL_pimc_w8_bc_mlp = 186.20` (beat `bc5_lin 188.20`), but 4-seed `fullmlp_w16` averaged `187.00` (`101:183, 102:185, 103:189, 104:191`) — weaker `B=900` teacher (`+1.88` pcs worse than BC-4's `B=1800`) + lower DAgger ratio (`12.5%` vs `28.6%`).
      - **`fullmlp_scale` & 10-Seed (`101..=110`) Strict Baseline (Actions Run `45170400970`)**:
        - `spatial_bc4.txt` (`fullmlp_w16_h8_b28_cvar25`) on 10 seeds (`101..=110`): **`mean = 187.90 [175..201]`** (`101..=105: 181.20`, `106..=110: 194.60`, `cov=30.7`, `cov_nxt=17.9%`, `seed 108 cov_nxt=27%`). Scaling to `w20/w24` and `h9` stayed flat (`187.00–187.40`), confirming that Oracle-`WG_ds` 1-ply BC + PIMC scaling has plateaued at `~187` (10-seed).
      - **Round ExIt-1 (`examples/train_exit.rs`, `models/spatial_exit1.txt`, Actions Run `45203324414`, 43m)**:
        - `64` Legal CVaR-25 self-play games (`mean = 187.84`, `11,866` decisions) distilled soft Q-values + true Legal MC returns: `val_soft_ce` `1.4979 -> 1.3931`, `legal_v_mae` `10.89 -> 4.98`, `rem_g = 1.718`.
        - Held-out 10-seed (`101..=110`) `fullmlp_w16`: **`189.30 [180..200]`** (improved hard seeds `106..=110` from `194.60 -> 190.40` (`-4.20` pcs, `cov_nxt` `17.9% -> 17.0%`), but regressed on `101..=105`, staying on the ~188 plateau).
      - **Round BC-6 (`SPATIAL_VAL_V6` Exact 2D `3×3` Pattern NNUE `64x32`, `models/spatial_bc6.txt` & `models/spatial_bc6_lin.txt`, Actions Run `45238581076`, 54m)**:
        - Improved `val_ce` to `0.9529` (LIN) / `0.9443` (MLP, `top1=71.3%`) on the exact same BC-4 dataset (`150 WG_ds B=1800 + 60 DAgger`), `LEGAL_b120_bc_lin = 196.60`, `LEGAL_pimc_w8 = 185.80` (`104:177`), and `10-seed` `fullmlp_w16 = 190.00 [186..198]` (beat `bc4` on seeds `108` `-10` pcs and `109` `-15` pcs, confirming ~188–190 is the structural ceiling of `h=8` clairvoyant `root_pimc`).
      - **`leaf_pimc_break` (Actions Run `45252557365`, 16m, 10 seeds `101..=110`)**:
        - Non-Clairvoyant Prefix-6 + CVaR Leaf-PIMC (`lpimc_h10=196.00`, `h11=195.70`, `h12=194.00`) trailed `root_pimc` (`187.90`) by `+6.1` pcs because Stage-1 opaque `0x3FF` spawns poison the depth-1..5 beam (`cov` jumped `30.7 -> 36.9`, `cov_nxt` `17.9% -> 19.9%`).
      - **`risk_calib` (Actions Run `45284574903`, 1h06m) — CRITICAL BENCHMARK BUG DISCOVERED!!**:
        - `rc_mlp_150` produced **BIT-FOR-BIT IDENTICAL** scores (`187.90`) to baseline, exposing a silent bug in `bench_legal.rs` `wg5_lin_sweep` (lines 698–699): it had zeroed `sm_val.b3 = 0.0; sm_val.w3 = [0.0; ...]` on load, meaning EVERY prior `full_mlp_pimc` / `fullmlp_w16` / `fullmlp_scale` / `leaf_pimc_break` / `risk_calib` run on `bc4`, `bc5`, `bc6`, and `exit1` had been running **PURE LINEAR SKIP ONLY (`has_mlp=false`)**!
      - **REAL FULL-MLP 10-Seed (`101..=110`) Showdown (Actions Run `45295407061` — NEW ALL-TIME 10-SEED LEGAL RECORD `185.60 [176..193]`!!)**:
        - **`models/spatial_bc4.txt` (`V5 64x32` REAL FULL MLP, `has_mlp=true`)**: **`185.60 [176..193]`** (`[101:180, 102:184, 103:176, 104:188, 105:193, 106:187, 107:185, 108:186, 109:185, 110:192]`, `nds=63.10`, `cov=28.9`, **`end=3.4`**, `2030.7s`) — **`-2.30` pcs better** than linear skip (`187.90`), worst seed dropped `201 -> 193`, hard seeds `106..=110` dropped **`194.60 -> 187.00` (`-7.60` pcs!!)**!
        - **`models/spatial_bc6.txt` (`V6 2D 3×3 Pattern 64x32` REAL FULL MLP)**: **`186.70 [182..195]`** (`-3.30` pcs better than linear skip `190.00`!).
        - **`models/spatial_bc5.txt` (`V5 128x48`, `B=900` weaker teacher + `12.5%` DAgger, Actions Run `45307436422`, 51m)**: `188.90 [181..197]` — proving teacher strength (`B=1800` `166.73` vs `B=900` `168.61`) and high DAgger ratio (`28.6%`) beat raw hidden width (`-3.30` pcs faster & stronger for `bc4` `64x32`).
        - **REAL FULL-MLP Boost & Scaled Search Ablation (Actions Run `45314615547`, 59m22s, `models/spatial_bc4.txt`)**:
          - `realmlp140_w16_h8_b28_c30` (10 seeds): **`186.70 [175..192]`** (seed `103` hit **`175`**, max `192`, `cov_nxt=17.0%`, `1612.0s`) — naturally trained `1.0×` MLP (`185.60 [176..193]`) remains better calibrated than `1.40×` boost.
          - `realmlp_w20_h9_b36_c30` (5 seeds `101..=105`): **`186.80 [185..190]`** (vs `184.20` at `h=8, w=16, b=28, cvar=25`): deeper `h=9` clairvoyant world search suffers from 3-ply hallucination / strategy fusion (`cov_nxt` dropped to `14.2%`, `end=3.6`, but `wide` rose to `27.2`).
        - **Round BC-7 (`models/spatial_bc7.txt`, `V6 2D 3×3 Pattern 64x32`, Actions Run `45331301626`, 1h4m47s)**:
          - `100` clean `WG_ds (B=2800)` games (**`mean = 166.10`**, strongest teacher yet!) + `50` REAL-MLP DAgger games (`spatial_bc4.txt` `mlp=true`, `4341` recovery decisions, **`2327` with multi-move Soft-Q**) + `--mixed-spawn` (50% CRN real-hole + 50% opaque) -> `20,851` decisions, `967,107` moves.
          - Best LIN `val_ce = 0.9898`, `rem_g = 1.675`, `v_mae = 1.98`; best MLP (ep 6) `val_ce = 0.9839`, `top1 = 70.4%`, **`v_mae = 1.96`**, `needs_pe = false`.
          - **Held-out REAL FULL-MLP Verification (Actions Run `45340066535`, 58m49s)**:
            - `spatial_bc7.txt` standalone 10 seeds (`101..=110`): **`188.00 [181..193]`** (`nds=63.90`, `cov=29.6`, `wide=27.5`, `end=4.3`, `2193.3s`) — only `100` clean games on `1591` V6 sparse features caused early MLP saturation (ep 6).
            - `ens2 (bc4 V5 + bc7 V6)` 5 seeds (`101..=105`): **`184.60 [175..193]`** (`[101:180, 102:184, 103:175, 104:191, 105:193]`, **`cov=25.0`**, **`cov_nxt=15.2%`** — slashed `cov` by `-3.9` and `cov_nxt` by `-2.9%` vs `bc4` alone, while `wide=27.4` remained the #1 gap vs `WG_ds` `wide~16`!).
          - **REAL FULL-MLP `DS-Target=5` & Top-2 Ensemble Ablation (Actions Run `45348014227`, 51m36s, 5 seeds `101..=105`)**:
            - `realmlp_ds5_w16_b22_c25` (`spatial_bc4.txt`): `191.00 [182..197]` (`cov=31.6`, `wide=27.0`, `1742.0s`) — DS-target beam inside clairvoyant worlds amplifies strategy fusion over unknown future holes/pieces (`+6.8` pcs worse than `h=8`).
            - `ens2 (bc4 V5 + bc6 V6)`: **`184.40 [176..193]`** (`[101:180, 102:184, 103:176, 104:189, 105:193]`, `cov=25.0`, `cov_nxt=15.6%`) — virtually identical to `bc4` alone (`184.20`), confirming that **all 1-step clairvoyant-BC models share the same `~185` ceiling** caused by clairvoyance bias & strategy fusion.
          - **Non-Clairvoyant Rollout & ExIt-1 REAL FULL-MLP Verification (Actions Run `45369836460`, 42m34s)**:
            - `realmlp_h6_nc12_w16_b28_c25` (10 seeds `101..=110`): `236.00 [218..258]` (`1533.7s`) — 1-ply greedy rollout (`~230`-piece policy) adds massive variance that drowns root signal (abandoned).
            - `models/spatial_exit1.txt` REAL FULL MLP (5 seeds `101..=105`): `187.80 [181..192]` (`975.1s`, seed `104` hit **`181`**, `-7` pcs better than `bc4` `188`).
          - **REAL FULL-MLP Pure Mean (`cvar=0`) & Sharp Horizon (`h=8, 7, 6`) Ablation (Actions Run `45390096575`, `models/spatial_bc4.txt`, 5 seeds `101..=105`)**:
            - `realmlp_h8_w16_b28_c0`: **`184.80 [173..193]`** (`[101:183, 102:184, 103:173, 104:191, 105:193]`, `nds=62.80`, `cov=29.6`, `nar=9.4`, `wide=25.4`, **`end=2.8`**, `cov_nxt=15.8%`, `1055.6s`) — seed `103` hit **`173`** and `end=2.8` (all-time lowest endgame!), virtually tied with `cvar=25` (`184.20`).
            - `realmlp_h7_w20_b36_c0`: **`185.20 [180..189]`** (`[101:180, 102:184, 103:187, 104:186, 105:189]`, `cov=31.0`, `nar=7.4`, `wide=26.0`, `end=4.2`, `1388.4s`) — tightest max (`189`), `+0.4` pcs behind `h=8`.
            - `realmlp_h6_w20_b40_c0`: **`188.00 [183..191]`** (`cov=33.0`, `wide=27.0`, `1315.2s`) — confirms `h=8` is the true optimum of 1-ply-shared clairvoyant `root_pimc` (`h=5: 194.8 -> h=6: 188.0 -> h=7: 185.2 -> h=8: 184.2/184.8 -> h=9: 186.8`).
          - **Diagnosis of `+20.2` Gap (`184.80` vs `WG_ds 164.60`)**: `+16.0` of `+20.2` (`80%`!) is purely **`cov (+10.4)` + `nar (+5.6)`** (`O` is `-0.2`, `end` is `+1.0`, `wide` is `+3.4`).
          - **REAL FULL-MLP `OR_p20` Ceiling + CRN Prescreen + 2-Ply Non-Clairvoyant Expectimax (Actions Run `45410423499`, `models/spatial_bc4.txt`)**:
            - **`OR_p20_b240_sp_top1` (`has_mlp=true`, seeds `101..=102`)**: **`176.00 [175, 177]`** (`nds=59.00, cov=30.5, nar=5.5, wide=22.0, cov_nxt=16.0%, avg_c=12.5`) — **IDENTICAL** to linear skip `176.00`! Even with 20 previews + exact holes + `b=240`, `spatial_bc4.txt` still wastes `+13.0` pcs vs `WG_ds` (`163.00`), proving **search on `spatial_bc4` is saturated and the Evaluator itself is the #1 bottleneck**!
            - **`realmlp_pre4_r36_f12_w16_h8_c25` (5 seeds `101..=105`)**: **`183.80 [176..192]`** (**NEW ALL-TIME 5-SEED REAL FULL-MLP LEGAL RECORD**, `[101:182, 102:185, 103:176, 104:184, 105:192]`, `nds=62.40`, `cov=28.8`, `nar=7.0`, `wide=24.0`, **`cov_nxt=14.2%`** — lowest `cov_nxt` & `nds` ever, `-0.40` pcs over `184.20`).
            - **`realmlp_d2k4_w12_h8_b24_c25` (Phase 4B 2-Ply Non-Clairvoyant Expectimax, 5 seeds)**: **`184.40 [182..188]`** (`[101:183, 102:184, 103:182, 104:185, 105:188]`, `cov=27.8`, `end=3.0` — tightest max `188` ever!).
          - **Round V5-Q1 (`models/spatial_v5q1.txt`, Actions Run `45428604160`, recovered from artifact `research-outputs-42`)**:
            - Trained compact V5 `64x32` on `120` `WG_ds (B=1600, mean=166.98)` + `70` REAL-MLP DAgger (`198.47`) = `27,829` decisions (`8,354` with `ds_target=8` multi-move Soft-Q + relative Q-Advantage `adv_w=0.35`, 100% real-hole CRN spawns): `val_ce = 1.0405`, `v_mae = 2.06`, `rem_g = 1.680`.
            - **`OR_p20` (`has_mlp=true`, `101..=102`)**: `178.50 [182, 175]` — **MASSIVE `cov` BREAKTHROUGH**: `cov` plunged from **`30.5 -> 20.5` (`-10.0` `0ds` moves, matching `WG_ds`'s `19.2`!!)** and `cov_nxt` dropped `16.0% -> 14.0%`, while `wide` rose `22.0 -> 29.0` (exact complement to `spatial_bc4.txt`'s `cov=30.5, wide=22.0`!).
            - Part 3 (Legal `fullmlp_w16`) was cut off at 90m due to `clean_q` data generation taking 73m; model `models/spatial_v5q1.txt` recovered and committed.
            - **Held-out REAL FULL-MLP Legal Verification (Actions Run `45452866228`, 43m, 5 seeds `101..=105`)**:
              - `spatial_v5q1.txt` standalone (`fullmlp_w16`): **`186.20 [182..188]`** (`[101:188, 102:182, 103:187, 104:188, 105:186]`, `max=188`, `seed 105` improved `-7` pcs vs `bc4 193`, `1270.5s`).
              - `ens2 (spatial_bc4.txt + spatial_v5q1.txt)`: **`184.40 [176..193]`** (`[101:180, 102:184, 103:176, 104:189, 105:193]`, **`cov=25.0`**, `nds=62.40`, `cov_nxt=15.6%`, `1300.4s`) — **IDENTICAL** to `ens2 (bc4+bc6) = 184.40` and `ens2 (bc4+bc7) = 184.60`!
            - **Key Pivot — WIDE Cell-Conserving DS-Bucketed PIMC (Actions Run `45463827955`, `models/spatial_bc4.txt` REAL FULL MLP leaf, 5 seeds `101..=105`)**:
              - `wide_c22_ds5_w12_b80_c25`: `190.40 [185..199]` (`cov=29.8, wide=29.2, cov_nxt=17.8%, 905.6s`)
              - `wide_c22_ds5_w12_b120_c25`: `192.40 [185..202]` (`cov=32.8, wide=27.8, 1377.8s`)
              - `wide_c22_ds8_w10_b100_c25`: `191.00 [185..200]` (`cov=35.6, cov_nxt=19.2%, 1532.6s`)
              - Decisively rules out `ew_cons22` DS-bucketed PIMC inside finite-horizon clairvoyant worlds (`+6.6` to `+8.6` pcs worse than `realmlp_prescreen 183.80`).
          - **Round ExIt-2 (`models/spatial_exit2.txt`, Actions Run `45517753392`, 1h23m)**:
            - `64` Legal self-play games (`mean=188.89`, `11,930` decisions, `w=6, h=8, b=20, r=10, cvar=25`, 100% real-hole CRN spawns, `--v5-only`, `--adv-w 0.30`): `val_soft_ce` `1.4858 -> 1.3991`, `legal_v_mae` `12.71 -> 6.23`, `rem_g = 1.707`.
            - Held-out 5-seed `fullmlp_w16`: **`191.60 [186..195]`** (`cov=34.4, avg_c=13.2, 1426.8s`) — regressed `+7.4` pcs vs `bc4 (184.20)` because distilling a weaker `188.89`-piece `w=6` teacher + pushing `rem_g` up to `1.707` made the evaluator greedier.
          - **Round BC-8 (`models/spatial_bc8.txt`, Actions Run `45529620978`, 1h26m)**:
            - `220` `WG_ds (B=1800, mean=166.84)` + `80` REAL-MLP DAgger (`43,462` decisions, `2.01M` moves, pure V5 `64x32`): achieved best-ever V5 `val_ce = 0.9533` & `v_mae = 1.72`, YET held-out 5-seed `fullmlp_w16` regressed to **`188.00 [183..192]`** (`cov=29.0, wide=27.4, 955.4s`).
            - **Definitive Conclusion**: Lower 1-step `val_ce` to clairvoyant Oracle `WG_ds` (`bc5: 0.8974 -> 188.90`, `bc8: 0.9533 -> 188.00`, `bc4: 0.9662 -> 184.20/183.80`) does **NOT** improve Legal `p=5` `root_pimc` play!
          - **Pivot — Direct Legal REAL-MLP `root_pimc` End-to-End ES ([`opt_pimc.rs`](file:///home/antigravity/jush0147/blockfish/blockfish-engine/examples/opt_pimc.rs) → `models/spatial_pimc_es1.txt`, Actions Run `45541272919`)**:
            - Directly optimized 48 strategic parameters (`16` `lin_dense` + `32` `w3`) of `spatial_bc4.txt` directly on Legal `p=5` `root_pimc`:
            - **In just 1 single ES generation** (stopped at 31m time guard): fixed held-out `val8` (`6001..=6008`) dropped **`196.00 -> 195.00` (`-1.00` pcs!)**, and held-out 5-seed `fullmlp_w16` (`101..=105`) **immediately matched `bc4` at `184.20 [181..191]`** (`[101:182, 102:183, 103:181, 104:184, 105:191]`, **`cov=27.8`** vs `28.9`, `nar=7.0`, `nds=62.60`, max dropped `193 -> 191`, improving 3/5 seeds including `104: 188->184`!).
            - **Huge Signal**: Direct Legal `root_pimc` ES is the **ONLY** method after `BC-4` that immediately improved on Gen 1 without regressing!
          - **Colab Run #1 (`models/spatial_pimc_es2.txt`, 8-core Colab Auto-Polling Worker, 0.36 CU, ZERO Actions minutes)**:
            - `opt_pimc` Gen 1..3 held at `val8 = 193.12`; 10-seed (`101..=110`) `fullmlp_w16`: **`186.90 [181..194]`** (`[182, 183, 181, 184, 191, 193, 185, 190, 194, 186]`, `cov=29.4, cov_nxt=15.9%, 1373.5s` — 1.5× faster than Actions!).
          - **Colab Run #2 (`realmlp_calib` + 10-seed `realmlp_prescreen`, 8-core Colab)**:
            - **`realmlp_cov130` (5 seeds `101..=105`)**: **`184.20 [177..187]`** (`[101:187, 102:184, 103:186, 104:177, 105:187]`, `nds=62.40, nar=6.6, 632.2s`) — **all-time lowest worst-seed (`max=187`!)**, slashing `seed 104` by `-11` pcs (`188 -> 177`) and `seed 105` by `-6` pcs (`193 -> 187`)!
            - `realmlp_rg86`: `185.60 [173..195]` (`seed 103=173, seed 105=182, cov=24.2, cov_nxt=14.8%`); `realmlp_stk_low`: `187.40 [177..193]` (`cov=22.4, cov_nxt=12.8%`); `realmlp_rg92`: `188.40`.
            - `realmlp_prescreen` (10 seeds `101..=110`): `188.80 [176..200]` (`106..=110: 193.80`) — standard Stage-1 `fullmlp_w16` (`185.60 [176..193]`) remains 2× faster and `-3.20` pcs stronger on 10 seeds.
          - **Colab Run #3 (`realmlp_cov130` 10-seed + `cov/nar` fine sweep, 8-core Colab — NEW ALL-TIME 5-SEED LEGAL RECORD `183.00`!!)**:
            - **`realmlp_cov115_c25` (5 seeds `101..=105`)**: **`183.00 [177..189]`** (**NEW ALL-TIME 5-SEED REAL FULL-MLP LEGAL RECORD**, `[101:187, 102:185, 103:177, 104:177, 105:189]`, **`nds=62.00`** all-time lowest, `wide=22.8`, `nar=7.2`, `653.0s` — `-1.20` pcs over `bc4 184.20` and `-0.80` pcs faster/stronger than `prescreen 183.80`!).
            - `realmlp_cov130` (10 seeds `101..=110`): `185.90 [177..193]` (`104:177, 109:182, 110:188`); `cov130_c0` (5 seeds): `184.20 [173..191]` (`103:173`); `cov130_nar130`: `184.80`; `cov145`: `185.40`.
          - **Colab Run #4 (`realmlp_cov115` 10-seed + 40-seed Legal PB/Distribution Audit, 8-core Colab, 49m, `0.21` CU)**:
            - `realmlp_cov115_w16_h8_b28_c25` (10 seeds `101..=110`): **`186.70 [177..197]`** (`101..=105: 183.00`, `106..=110: 190.40`, `nds=63.70, 1343.0s`) — confirms original `models/spatial_bc4.txt` (`185.60 [176..193]`) remains the all-time 10-seed Legal champion.
            - `realmlp_cov115_w8_h8_b20_c25` (40 seeds `101..=140`): **`mean=190.38 [180..200]`** (`nds=64.97, cov=30.3, wide=29.9, <=166: 0, 1559.5s`) — establishes that the `64x32` handcrafted-feature MLP architecture has fully saturated at `~185` mean / `~173` single-seed PB (`OR_p20 = 176.00`).
          - **Phase 2–4 Colab Pro GPU/CPU Runs (#5–#14, `2026-11-25`–`11-26`)**:
            - **Run #5–#7 (`ResNet v1/v2`, 400 `WG_ds` games, L4 GPU)**: Achieved `val_ce = 0.7717` (`top1=75.26%`), yet `OR_p20 = 189.80` (vs `spatial_bc4` `176.00`) and Legal `p=5` `205` — confirming clairvoyant `WG_ds` BC hurts Legal play.
            - **Run #8–#11 (`Hybrid ResNet v1/v2`, L4/T4 GPU)**: Adding `WG_ds` to Legal self-play lowered mixed `val_ce` (`0.934 -> 0.852`) while monotonically worsening Legal `pimc_v2` (`193.20 -> 194.80 -> 199.00`), providing final proof that imitating clairvoyant `WG_ds` actively harms Legal `p=5`.
            - **Run #12–#14 (Pure Legal `p=5` `TD(λ)` Hybrid ResNet v1/v2, T4/L4 GPU)**: 72 strong Legal `root_pimc` games improved unbiased `val_mc_rmse` (`15.384 -> 14.877`) and `pimc_v2` linear-internal (`191.80 -> 190.00`), but full-MLP-internal `pimc_v2` regressed (`195.20 -> 199.00`) — 64–200 games is too noisy for a 367k-param ResNet, and `pimc_v2` trails `beam.rs` (`184.20`) by 8–11 pcs.
          - **Colab Run #15-CPU (`beam.rs` `opaque_pimc` + `w24_h9_b36`, 8-core CPU, 5 seeds `101..=105`)**:
            - Strict same-binary results: `fullmlp_base_w16_h8_b28 = 184.20 [176..193] (cov=25.0)` vs `opaque_d1 = 186.20 (cov=29.6)` vs `opaque_d2 = 188.60 (cov=34.6)` vs **`fullmlp_w24_h9_b36 = 186.40 [183..191] (cov=30.2)`**.
            - **Physics caveat on `opaque_d1/d2`**: `OPAQUE_GARBAGE_ROW` (`0xFBFF`) has no hole and is never materialized into a real clearable garbage row during `root_pimc_eval_world`, altering simulated physics rather than cleanly isolating Strategy Fusion.
            - **Key finding on `w24_h9_b36`**: Scaling search budget (`w=24, h=9, b=36, f=12`) worsened Legal play by `+2.20` pcs (`cov: 25.0 -> 30.2`), confirming `spatial_bc4.txt` + standard PIMC is saturated and motivating large-scale pure-Rust Legal `p=5` `TD(λ)` RL directly on `SpatialValueModel` (`train_spatial_td.rs`).
          - **Colab Run #16-CPU (`train_spatial_td.rs`, 2,500 Legal `p=5` games, 5 iters × 500g, 8-core CPU)**:
            - Fixed 60-seed greedy `beam=24` validation (`seeds 1..=60`, `top=0` on all iters): `Iter 0 BASE (spatial_bc4) = 203.10 [182..219]` → `Iter 1 = 206.37` → `Iter 2 = 206.72` → `Iter 3 = 208.43` → `Iter 4 = 212.93` → `Iter 5 = 212.25` (best checkpoint stayed `Iter 0`, `fullmlp_w16` same-binary A/B `184.20` vs `184.20`).
            - **Metric caveats**: `train_spatial_td.rs`'s `cov_turns` counts all turns where the top garbage hole is covered (not restricted to `ds == 0` like `bench_legal.rs`'s `cov`), and per-iter `val_mc_rmse` (`22.059 -> 20.139`) is evaluated on each iteration's own policy trajectory distribution rather than a fixed state set.
            - **Conclusion**: Absolute scalar `TD(λ)` regression on single-trajectory returns washes out `0.1`-piece sibling move rankings; pivoting to **100% Legal-Physics 2-Ply Shared-Prefix Expectimax** (`d2_top_k`) and/or **CRN Antithetic ES** directly on Legal `p=5` piece counts.
          - **Colab Run #17-CPU (`bench_legal d2_expectimax`, 5 seeds `101..=105`, 8-core CPU)**:
            - `fullmlp_base_w16_h8_b28 = 184.20 [176..193] (cov=25.0)` vs `d2k4_w16 = 192.00 (+7.80, cov=36.6)` vs `d2k6_cov115_w16 = 188.40 (+4.20, cov=33.6)`.
            - **Legality bug found & fixed in `beam.rs`**: `d2_prepped` generated `fp2` on `world 0` and passed it directly into `root_pimc_eval_world` for all worlds without checking collision/support against world-specific newly spawned garbage rows (`0..num_new`), invalidating Run #17's `d2` numbers. Fixed by filtering `fp2.norm_i <= num_new` in `d2_prepped` and adding `FastShape::intersects` collision + support guards in `root_pimc_eval_world`.
          - **Colab Runs #18–#19-CPU (`opt_pimc.rs` Direct Legal CRN ES → `model-spatial-es-v3`, strict same-binary `fullmlp_w16` 5 seeds `101..=105`)**:
            - `spatial_bc4.txt`: **`184.20 [176..193]`** (`[180, 184, 176, 188, 193]`, `cov=25.0, O=20.0, nar=8.6, wide=27.0, end=3.6`).
            - `spatial_es_v3.txt`: **`184.40 [179..190]`** (`[184, 179, 185, 184, 190]`, `cov=31.6, O=17.2, nar=6.4, wide=25.8, end=3.4`) — beat `bc4` on 3/5 seeds (`102:-5, 104:-4, 105:-3`), cut `O+nar+wide+end` by **`-6.4` pcs**, and tightened `max` `193 -> 190`, while losing `+6.6` on `cov` due to `dense[23/25/28]` stepping `-0.02`.
          - **Colab Run #21-CPU (`eval_core_ex` `spatial_model_2` bug fixed + verified by unit test `65/65`, strict same-binary `fullmlp_w16` 5 seeds `101..=105`)**:
            - Fixed critical `eval_core_ex` short-circuit bug that had silently bypassed `spatial_model_2` in ALL prior `allow_spatial=true` FULL-MLP `ens2` runs.
            - `fullmlp_base_bc4`: **`184.20 [176..193]`** (`[180, 184, 176, 188, 193]`, `707.5s`).
            - `fullmlp_ens50_bc4_esv3` (TRUE 50/50 FULL-MLP): `186.40 [184..191]` (`1149.4s`).
            - **`fullmlp_esv3_bc4cov115`**: **`184.20 [176..187]`** (`[187, 184, 187, 176, 187]`, `wide=24.0, O=18.0, avg_c=12.2, 671.0s`) — tied `bc4` mean while slashing `seed 104: 188->176 (-12)` and `seed 105: 193->187 (-6)` (`max=187`, tied all-time lowest worst-seed!).
          - **Colab Run #22-CPU (`true_ens_showdown`, first true diverse FULL-MLP 50/50 ensembles after `eval_core_ex` fix, 5 seeds `101..=105`, 8-core CPU)**:
            - `fullmlp_base_bc4`: **`184.20 [176..193]`** (`[180, 184, 176, 188, 193]`, `cov=25.0, cov_nxt=15.0%, 703.1s`).
            - `fullmlp_TRUE_ens50_bc4_bc6` (V5 + V6 2D `3×3`): `185.80 [181..192]` (`[181, 185, 192, 188, 183]`, `cov=26.2, cov_nxt=17.0%, 1213.2s`).
            - `fullmlp_TRUE_ens50_bc4_v5q1` (V5 + Soft-Q): `185.00 [182..187]` (`[183, 187, 186, 182, 187]`, `cov=24.6, cov_nxt=14.8%, 1144.9s`) — lowest `cov`/`cov_nxt` and tightest `[182..187]` range, confirming all `spatial_bc4`-family variants cluster in `183–186`.

---

## 4. Roadmap

1. **Direct Legal `p=5` Policy Optimization (`opt_legal.rs` → `models/spatial_es1.txt`)**:
   - Run 35+ Antithetic-ES generations directly minimizing Legal `p=5` piece count (eliminating Ridge shrinkage and teacher confounding), then verify held-out `101..=105` with both `b=120` beam and `linfast_nope` flat `h=8` PIMC.
2. **Non-Clairvoyant Rollout PIMC (Zero Strategy Fusion)**:
   - Replace clairvoyant deep determinized search with fast non-clairvoyant `p=5` rollouts inside CRN worlds so deeper lookahead improves rather than degrades Legal play.
3. **Break Legal `p=5` Through `180 -> 170 -> <= 161` Pieces**:
   - Validate all breakthroughs across `5 -> 20 -> 40` held-out seeds and continuously commit/push progress.
