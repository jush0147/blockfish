# Blockfish 100L Cheese Race Research Log

**Branch**: `rescue/independent-validation-20261116`  
**Objective**: Achieve **≤ 166 pieces** on official Jstris 100L Cheese Race (`block-stacker` `Ruleset::guideline()`, `previews = 5`, `total_lines = 100`, `min_height = 4`, `max_height = 9`, `BagTracker` only, zero peeking).

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
   - **Target Budget for $P \le 166$**:
     - Requires **$Z \le 66$** `0ds` moves and **$K \le 55$** `nds` lines.
     - **`O`-piece lower bound**: `O` (`2x2`) can never clear a 1-wide garbage row (`ds = 0` always). Across ~166 pieces, ~23–24 `O` pieces appear, forcing ~22–24 `0ds` moves alone.
     - Therefore, **non-`O` `0ds` moves must be reduced to ≤ 42–44** (from ~74 in current legal baseline and ~55 in current 20-preview Oracle).

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

---

## 4. Roadmap

1. **Gap Decomposition & Trace Analysis**: Run `diag_cheese -- gap5` & move-level trace diagnostics on legal vs Oracle games to isolate piece losses by horizon, unknown holes, endgame, and evaluator errors.
2. **Exact-Physics Search & Endgame Solver**:
   - Exploit zero-spawn deterministic transitions (`g >= 5` on clears, `g == 9` on `0ds`), 1-bit `bag_mask` deterministic extension, exact 9-branch expectation when spawning 1 row (`g=4 -> 3`), and a deep/exact endgame solver for `rem_g <= 9` (especially `rem_g <= 6`).
   - Proper CRN/belief-space search over real Jstris physics (`min=4, max=9`).
3. **Stronger Nonlinear Evaluator / Deep-Teacher RL**:
   - Build spatial + tactical board representation (beyond linear 98 features: column heights relative to `g`, per-row occupancy/cavity/overhang over `h1..h4`, SRS direct-fit & skim-to-clear readiness, `O`-slot management) + MLP/residual value network trained on deep-lookahead Oracle/CRN rollouts generated via GitHub Actions.
4. **Large-Scale Independent Verification**: Evaluate every promising change on `dev40` (40 seeds) locally or via `.github/workflows/research.yml` on GitHub Actions.
