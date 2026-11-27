# BC-4 Mass Legal 100L Cheese Race Simulation Pipeline

Completely isolated, fixed-version Monte Carlo simulation system for evaluating the verified BC-4 model (`models/spatial_bc4.txt`) with the `beam.rs` REAL FULL-MLP Legal `p=5` searcher across large sets of deterministic random seeds.

## 1. Pinned Baseline & Integrity Locks

All runs verify these SHA-256 checksums before executing (and abort on any mismatch):

| Artifact | Path | Pinned Value |
|---|---|---|
| Baseline Git Commit | `rescue/independent-validation-20261116` (read-only) | `939cb506a2e5fb9e86f289b86a51162d05afbf2d` |
| Model Weights | `models/spatial_bc4.txt` | `d99ae1f07e629e95e115f30fb3e46fecee20eb3d14c5b9e2a94dc52da1238412` |
| Search Engine | `blockfish-engine/src/ai/beam.rs` | `31ae4072dfe51ac9e3cd6aaed13bf329c028553a32af3384c7f6f116e99fb09f` |
| Spatial Evaluator | `blockfish-engine/src/ai/spatial_val.rs` | `5cc76e71265aba86c065957e6f5226b9b676503e149ee92bf5e010a36e7077c1` |
| Guideline Ruleset | `support/guideline.json` | `b196f65f0a1a748f2194c9b36a7006259a1cb0fa6ee5321f1fb9ef6197115c5e` |

### Locked Search Parameters (`bc4_mass_legal.rs`)
- Mode: **REAL FULL-MLP** (`spatial_rerank_only = false`, `spatial_no_pe_internal = false`)
- Legal preview: **`p = 5`** (`snap.queue.len() <= 6` asserted on every move)
- Root PIMC: `w = 16`, `h = 8`, `b = 28`, `roots = 32`, `fine_roots = 10`, `cvar_pct = 25`, `margin = 500`, `weight_pct = 100`
- Base Beam: `beam_width = 80`, `max_per_root = 16`, `bag_det_extend = true`, `endgame_solver = true`, `simulate_cheese_spawn = true`
- No CLI flags exist to weaken search parameters.

### Ruleset Note (Local Legal Sim vs Jstris Official)
All records are explicitly tagged `record_type: "local_legal_sim_block_stacker"`:
- Simulator: `block-stacker` (`Ruleset::guideline()`, 10x25 matrix, 20 visible rows, 5 previews, Hold enabled)
- Cheese Race: 100L (`total_lines = 100`, `max_height = 9`, `min_height = 4` on combo, no consecutive same-column holes)
- PRNG: `rand_pcg::Pcg64Mcg::seed_from_u64(seed)`
- Results represent **Local Legal Simulation Records**, distinct from **Jstris Official Leaderboard Records** (which use Jstris's own JS PRNG and engine nuances).

## 2. Isolation Guarantees

- **Branch**: `experiment/bc4-mass-legal-seeds` (never merged to `dev` or `rescue/independent-validation-20261116`).
- **Trigger**: `.github/workflows/bc4-mass-legal-sim.yml` ONLY uses `on: workflow_dispatch`. No `push`, `pull_request`, `schedule`, `workflow_run`, `workflow_call`, or `repository_dispatch`.
- **Concurrency**: `group: bc4-mass-legal-${{ inputs.batch_id || github.run_id }}` with `cancel-in-progress: false`.
- **Permissions**: `contents: read` (never commits or pushes back to Git; all results are stored in GitHub Actions Artifacts with 90-day retention).

## 3. Manual Dispatch & Scaling

Trigger a batch via GitHub CLI:

```bash
gh workflow run bc4-mass-legal-sim.yml \
  -R jush0147/blockfish \
  --ref experiment/bc4-mass-legal-seeds \
  -f batch_id=batch-001 \
  -f seed_start=100001 \
  -f seed_count=200 \
  -f num_shards=20 \
  -f replay_threshold=176
```

### Recommended Non-Overlapping Seed Plan
- Calibration / Smoke: `101..=105` (known 5-seed benchmark, avg `184.20`, seed `103` = `176`)
- Batch 001 (200 seeds, 20 shards x 10 seeds, ~20–25 min): `100001..=100200`
- Batch 002 (200 seeds): `100201..=100400`
- Batch 003 (200 seeds): `100401..=100600`
- Batch 004 (200 seeds): `100601..=100800`
- Batch 005 (200 seeds): `100801..=101000` (Reaches 1,000 seeds!)
- Subsequent batches: `101001..=102000`, etc. (scales to 10,000 / 100,000+ without overlap).

### Rerunning Missing Seeds
If any shard hits the time guard or runner preemption, inspect `summary.json` -> `missing_seeds_from_manifests` and dispatch a targeted rerun:

```bash
gh workflow run bc4-mass-legal-sim.yml \
  -R jush0147/blockfish \
  --ref experiment/bc4-mass-legal-seeds \
  -f batch_id=batch-001-rerun \
  -f explicit_seeds="100012,100045,100099" \
  -f num_shards=3
```

## 4. Multi-Batch Aggregation & Replay Verification

Download artifacts from multiple runs into a local directory and aggregate with automatic deduplication:

```bash
python3 scripts/bc4_mass_legal/aggregate.py \
  --input-dir ./downloaded_artifacts \
  --out-json ./cumulative_summary.json \
  --out-md ./cumulative_summary.md \
  --out-merged-jsonl ./cumulative_results.jsonl
```

Independently verify any saved replay file move-for-move:

```bash
python3 scripts/bc4_mass_legal/verify_replay.py path/to/seed_103_pcs_176.replay.json
```
