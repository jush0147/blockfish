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

---

## 4. Roadmap

1. **Break the Oracle Ceiling (`176 -> <= 158–160`)**:
   - Inspect concrete `OR_p20_b500` non-`O` `0ds` moves (`cov=23.5, wide=22.0, nar=12.0`) to identify exact structural & search causes.
   - Replace myopic 1-step linear pruning and weak-teacher imitation with: (a) genuine search-architecture improvements (e.g., rollout-evaluated beam leaves / structural diversity Pareto beam retention / multi-move skim-and-dig tactical lookahead) and (b) true-outcome (`rem_pcs`, TD(λ), wide deep-rollout counterfactual) nonlinear evaluation.
2. **Close Legal (`p=5`) vs Oracle (`p=20`) Gap**:
   - Exploit zero-spawn deterministic transitions (`g >= 5` on clears, `g == 9` on `0ds`), 1-bit `bag_mask` deterministic extension, stratified CRN belief-space worlds, and an exact/deep deterministic endgame solver (`rem_g <= 9`).
3. **Large-Scale Independent Verification on GitHub Actions**:
   - Train & benchmark on `dev20`/`dev40` (`seeds 101..=120, 201..=220`) via `.github/workflows/research.yml`, keeping all code, models, and logs pushed to `rescue/independent-validation-20261116`.

