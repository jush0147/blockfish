//! Independent, fixed-version BC-4 Mass Legal 100L Cheese Race Simulator.
//!
//! Pinned configuration (read-only baseline from `rescue/independent-validation-20261116`):
//! - Baseline commit: 939cb506a2e5fb9e86f289b86a51162d05afbf2d
//! - Model: models/spatial_bc4.txt
//! - Model SHA-256: d99ae1f07e629e95e115f30fb3e46fecee20eb3d14c5b9e2a94dc52da1238412
//! - Engine: blockfish-engine/src/ai/beam.rs (REAL FULL-MLP, w=16, h=8, b=28, cvar=25)
//! - Legal rules: block-stacker Guideline SRS, p=5 previews, 100L cheese (min_h=4, max_h=9)
//!
//! This binary intentionally locks all AI search parameters and verifies the model SHA-256
//! before running so search quality can never be silently degraded.

use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{BeamConfig, FastAI, SpatialValueModel},
    Input, StackerExt as _,
};
use serde::Serialize;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

const SCHEMA_VERSION: &str = "bc4_mass_legal_v1";
const RECORD_TYPE: &str = "local_legal_sim_block_stacker";
const PINNED_BASELINE_COMMIT: &str = "939cb506a2e5fb9e86f289b86a51162d05afbf2d";
const PINNED_MODEL_NAME: &str = "models/spatial_bc4.txt";
const PINNED_MODEL_SHA256: &str =
    "d99ae1f07e629e95e115f30fb3e46fecee20eb3d14c5b9e2a94dc52da1238412";

// Minimal self-contained SHA-256 (FIPS 180-4) so the binary verifies the pinned model
// without adding any external crate dependency to Cargo.toml.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut msg = Vec::with_capacity(((data.len() + 9 + 63) / 64) * 64);
    msg.extend_from_slice(data);
    msg.push(0x80);
    while (msg.len() % 64) != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in &h {
        out.push_str(&format!("{:08x}", word));
    }
    out
}

fn input_name(inp: Input) -> &'static str {
    match inp {
        Input::Left => "Left",
        Input::Right => "Right",
        Input::CW => "CW",
        Input::CCW => "CCW",
        Input::SD => "SD",
        Input::HD => "HD",
        Input::Hold => "Hold",
    }
}

fn garbage_rows(stacker: &Stacker) -> u16 {
    let mut max_g = 0u16;
    let mut any_g = false;
    for ((r, _), cc) in stacker.matrix() {
        if cc == 'G' {
            any_g = true;
            max_g = max_g.max(r + 1);
        }
    }
    if any_g {
        max_g
    } else {
        0
    }
}

fn h1_info(stacker: &Stacker) -> (Option<u16>, usize, usize, usize) {
    let g = garbage_rows(stacker);
    if g == 0 {
        return (None, 0, 0, 0);
    }
    let mut row_g_minus_1 = [false; 10];
    let mut col_h = [0u16; 10];
    let mut blocks_over = 0usize;
    let mut cells_above = 0usize;
    for ((r, c), _) in stacker.matrix() {
        if r == g - 1 {
            row_g_minus_1[c as usize] = true;
        }
        if r >= g {
            cells_above += 1;
        }
        col_h[c as usize] = col_h[c as usize].max(r + 1);
    }
    let h1_col = (0..10u16).find(|&c| !row_g_minus_1[c as usize]);
    let mut chute_min = 0usize;
    if let Some(c1) = h1_col {
        for ((r, c), _) in stacker.matrix() {
            if r >= g && c == c1 {
                blocks_over += 1;
            }
        }
        let lh = if c1 == 0 {
            99
        } else {
            col_h[(c1 - 1) as usize].saturating_sub(g) as usize
        };
        let rh = if c1 == 9 {
            99
        } else {
            col_h[(c1 + 1) as usize].saturating_sub(g) as usize
        };
        chute_min = lh.min(rh);
    }
    (h1_col, blocks_over, chute_min, cells_above)
}

#[derive(Clone, Serialize)]
struct SearchConfigMeta {
    engine: &'static str,
    mode: &'static str,
    legal_preview: usize,
    beam_width: usize,
    max_per_root: usize,
    rollout_top_k: usize,
    bag_det_extend: bool,
    endgame_solver: bool,
    simulate_cheese_spawn: bool,
    spatial_rerank_only: bool,
    spatial_no_pe_internal: bool,
    root_pimc_prescreen_worlds: usize,
    root_pimc_roots: usize,
    root_pimc_fine_roots: usize,
    root_pimc_worlds: usize,
    root_pimc_horizon: usize,
    root_pimc_width: usize,
    root_pimc_margin: i64,
    root_pimc_weight_pct: i64,
    root_pimc_steady_g4: bool,
    root_pimc_cvar_pct: usize,
    root_pimc_opaque_after_d: usize,
    root_pimc_d2_top_k: usize,
}

impl SearchConfigMeta {
    fn pinned() -> Self {
        Self {
            engine: "blockfish-engine/src/ai/beam.rs",
            mode: "REAL_FULL_MLP",
            legal_preview: 5,
            beam_width: 80,
            max_per_root: 16,
            rollout_top_k: 0,
            bag_det_extend: true,
            endgame_solver: true,
            simulate_cheese_spawn: true,
            spatial_rerank_only: false,
            spatial_no_pe_internal: false,
            root_pimc_prescreen_worlds: 0,
            root_pimc_roots: 32,
            root_pimc_fine_roots: 10,
            root_pimc_worlds: 16,
            root_pimc_horizon: 8,
            root_pimc_width: 28,
            root_pimc_margin: 500,
            root_pimc_weight_pct: 100,
            root_pimc_steady_g4: false,
            root_pimc_cvar_pct: 25,
            root_pimc_opaque_after_d: 0,
            root_pimc_d2_top_k: 0,
        }
    }
}

#[derive(Clone, Serialize)]
struct RulesetMeta {
    simulator: &'static str,
    ruleset: &'static str,
    cols: usize,
    rows: usize,
    visible_rows: usize,
    previews: usize,
    garbage_total: usize,
    garbage_min_height: usize,
    garbage_max_height: usize,
    prng: &'static str,
}

impl RulesetMeta {
    fn pinned() -> Self {
        Self {
            simulator: "block-stacker",
            ruleset: "guideline (support/guideline.json)",
            cols: 10,
            rows: 25,
            visible_rows: 20,
            previews: 5,
            garbage_total: 100,
            garbage_min_height: 4,
            garbage_max_height: 9,
            prng: "rand_pcg::Pcg64Mcg::seed_from_u64(seed)",
        }
    }
}

#[derive(Clone, Serialize)]
struct Diagnostics {
    nds: usize,
    zds: usize,
    ds1: usize,
    ds_cov_next: usize,
    zds_h1_cov: usize,
    zds_g0: usize,
    zds_end: usize,
    zds_open_o: usize,
    zds_open_nar: usize,
    zds_open_wide: usize,
    avg_stack_cells: f64,
    holds_used: usize,
}

#[derive(Clone, Serialize)]
struct GameRecord {
    schema_version: &'static str,
    record_type: &'static str,
    seed: u64,
    status: &'static str,
    completed_100l: bool,
    pieces: usize,
    lines_cleared: usize,
    ds_cleared: usize,
    elapsed_sec: f64,
    model_name: &'static str,
    model_sha256: &'static str,
    baseline_commit: &'static str,
    git_commit: String,
    batch_id: String,
    shard_id: usize,
    runner_job_id: String,
    search_config: SearchConfigMeta,
    ruleset: RulesetMeta,
    diagnostics: Diagnostics,
}

#[derive(Clone, Serialize)]
struct MoveStep {
    step: usize,
    cur: char,
    hold_before: Option<char>,
    queue_p5: String,
    did_hold: bool,
    placed_piece: char,
    inputs: Vec<&'static str>,
    lines_cleared: usize,
    ds_cleared: usize,
    ds_total: usize,
    g_rows_before: u16,
    rem_g_before: u16,
    h1_col_before: Option<u16>,
    blocks_over_h1: usize,
    cells_above_g: usize,
}

#[derive(Clone, Serialize)]
struct GameReplay {
    record: GameRecord,
    moves: Vec<MoveStep>,
}

#[derive(Serialize)]
struct ShardManifest {
    schema_version: &'static str,
    batch_id: String,
    shard_id: usize,
    runner_job_id: String,
    git_commit: String,
    baseline_commit: &'static str,
    model_name: &'static str,
    model_sha256: &'static str,
    requested_count: usize,
    completed_count: usize,
    skipped_resume_count: usize,
    missing_seeds: Vec<u64>,
    stopped_by_time_guard: bool,
    wall_elapsed_sec: f64,
}

fn build_pinned_beam_config(sm: Arc<SpatialValueModel>) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.spatial_model = Some(sm);
    c.rollout_top_k = 0;
    c.beam_width = 80;
    c.max_per_root = 16;
    c.bag_det_extend = true;
    c.endgame_solver = true;
    c.simulate_cheese_spawn = true;
    c.spatial_rerank_only = false; // REAL FULL-MLP
    c.spatial_no_pe_internal = false;
    c.root_pimc_prescreen_worlds = 0;
    c.root_pimc_roots = 32;
    c.root_pimc_fine_roots = 10;
    c.root_pimc_worlds = 16;
    c.root_pimc_horizon = 8;
    c.root_pimc_width = 28;
    c.root_pimc_margin = 500;
    c.root_pimc_weight_pct = 100;
    c.root_pimc_steady_g4 = false;
    c.root_pimc_cvar_pct = 25;
    c
}

fn play_legal_seed_with_trace(
    seed: u64,
    cfg: &BeamConfig,
    meta: &RunMeta,
) -> (GameRecord, Vec<MoveStep>) {
    let t0 = Instant::now();
    let rules = Ruleset::guideline();
    assert_eq!(rules.previews, 5, "legal preview count must be 5");
    assert_eq!(rules.cols, 10, "matrix cols must be 10");

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    assert_eq!(game_cfg.garbage.min_height, 4);
    assert_eq!(game_cfg.garbage.max_height, 9);

    let mut stacker = Stacker::new(rules.into(), game_cfg);
    assert_eq!(stacker.prng_seed(), seed);
    let mut ai = FastAI::new(cfg.clone());

    let mut pieces = 0usize;
    let mut lines_total = 0usize;
    let mut ds_total = 0usize;
    let mut nds = 0usize;
    let mut zds = 0usize;
    let mut ds1 = 0usize;
    let mut ds_cov_next = 0usize;
    let mut zds_h1_cov = 0usize;
    let mut zds_g0 = 0usize;
    let mut zds_end = 0usize;
    let mut zds_open_o = 0usize;
    let mut zds_open_nar = 0usize;
    let mut zds_open_wide = 0usize;
    let mut cells_sum = 0usize;
    let mut holds_used = 0usize;

    let mut moves: Vec<MoveStep> = Vec::with_capacity(240);

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().expect("snapshot available");
        // Strict legal p=5 check: 1 current piece + at most 5 next previews = 6.
        assert!(
            snap.queue.len() <= 6,
            "illegal preview length: {}",
            snap.queue.len()
        );
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let (h1_col, blocks_over, chute_min, cells_above) = h1_info(&stacker);
        cells_sum += cells_above;

        let cur_c = snap.queue[0].as_char();
        let hold_before = snap.hold.map(|c| c.as_char());
        let eff_hold_c = snap
            .hold
            .or_else(|| snap.queue.get(1).copied())
            .map(|c| c.as_char());
        let queue_p5: String = snap.queue.iter().map(|c| c.as_char()).collect();

        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };
        let input_names: Vec<&'static str> = res.inputs.iter().map(|&i| input_name(i)).collect();
        let mut inputs = res.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        let did_hold = inputs.first() == Some(&Input::Hold);
        if did_hold {
            holds_used += 1;
        }
        let placed_c = if did_hold {
            eff_hold_c.unwrap_or(cur_c)
        } else {
            cur_c
        };

        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        pieces += 1;
        lines_total += lines;
        ds_total += ds;
        nds += lines.saturating_sub(ds);

        if ds > 0 {
            ds1 += ds;
            let (_, new_bo, _, _) = h1_info(&stacker);
            if new_bo > 0 {
                ds_cov_next += 1;
            }
        } else {
            zds += 1;
            if rem_g <= 7 && rem_g == g_rows {
                zds_end += 1;
            } else if g_rows == 0 {
                zds_g0 += 1;
            } else if blocks_over > 0 {
                zds_h1_cov += 1;
            } else {
                let is_o = placed_c == 'O';
                let has_i = cur_c == 'I' || eff_hold_c == Some('I');
                if is_o {
                    zds_open_o += 1;
                } else if chute_min >= 1 && !has_i {
                    zds_open_nar += 1;
                } else {
                    zds_open_wide += 1;
                }
            }
        }

        moves.push(MoveStep {
            step: pieces,
            cur: cur_c,
            hold_before,
            queue_p5,
            did_hold,
            placed_piece: placed_c,
            inputs: input_names,
            lines_cleared: lines,
            ds_cleared: ds,
            ds_total,
            g_rows_before: g_rows,
            rem_g_before: rem_g,
            h1_col_before: h1_col,
            blocks_over_h1: blocks_over,
            cells_above_g: cells_above,
        });
    }

    let completed_100l = ds_total >= 100;
    let status = if completed_100l { "completed" } else { "topout" };
    let elapsed_sec = t0.elapsed().as_secs_f64();

    let record = GameRecord {
        schema_version: SCHEMA_VERSION,
        record_type: RECORD_TYPE,
        seed,
        status,
        completed_100l,
        pieces,
        lines_cleared: lines_total,
        ds_cleared: ds_total,
        elapsed_sec,
        model_name: PINNED_MODEL_NAME,
        model_sha256: PINNED_MODEL_SHA256,
        baseline_commit: PINNED_BASELINE_COMMIT,
        git_commit: meta.git_commit.clone(),
        batch_id: meta.batch_id.clone(),
        shard_id: meta.shard_id,
        runner_job_id: meta.runner_job_id.clone(),
        search_config: SearchConfigMeta::pinned(),
        ruleset: RulesetMeta::pinned(),
        diagnostics: Diagnostics {
            nds,
            zds,
            ds1,
            ds_cov_next,
            zds_h1_cov,
            zds_g0,
            zds_end,
            zds_open_o,
            zds_open_nar,
            zds_open_wide,
            avg_stack_cells: cells_sum as f64 / pieces.max(1) as f64,
            holds_used,
        },
    };

    (record, moves)
}

#[derive(Clone)]
struct RunMeta {
    batch_id: String,
    shard_id: usize,
    runner_job_id: String,
    git_commit: String,
}

struct CliOpts {
    seed_start: Option<u64>,
    seed_end: Option<u64>,
    seeds_list: Vec<u64>,
    output: PathBuf,
    manifest: Option<PathBuf>,
    replay_dir: Option<PathBuf>,
    replay_threshold: usize,
    model_path: PathBuf,
    batch_id: String,
    shard_id: usize,
    runner_job_id: String,
    git_commit: String,
    threads: Option<usize>,
    resume: bool,
    max_minutes: Option<f64>,
}

fn print_usage() {
    eprintln!(
        "Usage: bc4_mass_legal [OPTIONS]
  --seed-start <u64>        Start seed (inclusive)
  --seed-end <u64>          End seed (inclusive)
  --seeds <u64,u64,...>     Explicit comma-separated seed list (overrides start/end)
  --output <path>           Output JSONL file path (required)
  --manifest <path>         Optional shard manifest JSON path
  --replay-dir <dir>        Directory to save full replays for best games
  --replay-threshold <n>    Save full replay for any completed game <= n pieces (default: 176)
  --model-path <path>       Path to spatial_bc4.txt (default: models/spatial_bc4.txt; SHA-256 verified)
  --batch-id <str>          Batch identifier (default: local)
  --shard-id <usize>        Shard index (default: 0)
  --runner-job-id <str>     Runner / Job identifier
  --git-commit <str>        Git commit SHA
  --threads <usize>         Worker threads (default: available CPU cores)
  --resume                  Skip seeds already present in --output JSONL
  --max-minutes <f64>       Stop launching new seeds after N wall minutes (graceful checkpoint)"
    );
}

fn parse_cli() -> CliOpts {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut seed_start = None;
    let mut seed_end = None;
    let mut seeds_list = Vec::new();
    let mut output: Option<PathBuf> = None;
    let mut manifest: Option<PathBuf> = None;
    let mut replay_dir: Option<PathBuf> = None;
    let mut replay_threshold = 176usize;
    let mut model_path = PathBuf::from(PINNED_MODEL_NAME);
    let mut batch_id = String::from("local");
    let mut shard_id = 0usize;
    let mut runner_job_id = std::env::var("GITHUB_RUN_ID")
        .ok()
        .map(|rid| {
            let job = std::env::var("GITHUB_JOB").unwrap_or_else(|_| "job".to_string());
            format!("{}/{}", rid, job)
        })
        .unwrap_or_else(|| "local".to_string());
    let mut git_commit =
        std::env::var("GITHUB_SHA").unwrap_or_else(|_| PINNED_BASELINE_COMMIT.to_string());
    let mut threads = None;
    let mut resume = false;
    let mut max_minutes = None;

    let mut i = 0;
    while i < args.len() {
        let k = args[i].as_str();
        let next_val = |idx: &mut usize, flag: &str| -> String {
            *idx += 1;
            args.get(*idx)
                .unwrap_or_else(|| {
                    eprintln!("Missing value for {}", flag);
                    std::process::exit(2);
                })
                .clone()
        };
        match k {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--seed-start" => {
                seed_start = Some(next_val(&mut i, k).parse().expect("invalid --seed-start"))
            }
            "--seed-end" => {
                seed_end = Some(next_val(&mut i, k).parse().expect("invalid --seed-end"))
            }
            "--seeds" => {
                let raw = next_val(&mut i, k);
                for tok in raw.split(',') {
                    let t = tok.trim();
                    if !t.is_empty() {
                        seeds_list.push(t.parse().expect("invalid seed in --seeds"));
                    }
                }
            }
            "--output" => output = Some(PathBuf::from(next_val(&mut i, k))),
            "--manifest" => manifest = Some(PathBuf::from(next_val(&mut i, k))),
            "--replay-dir" => replay_dir = Some(PathBuf::from(next_val(&mut i, k))),
            "--replay-threshold" => {
                replay_threshold = next_val(&mut i, k)
                    .parse()
                    .expect("invalid --replay-threshold")
            }
            "--model-path" => model_path = PathBuf::from(next_val(&mut i, k)),
            "--batch-id" => batch_id = next_val(&mut i, k),
            "--shard-id" => {
                shard_id = next_val(&mut i, k).parse().expect("invalid --shard-id")
            }
            "--runner-job-id" => runner_job_id = next_val(&mut i, k),
            "--git-commit" => git_commit = next_val(&mut i, k),
            "--threads" => {
                threads = Some(next_val(&mut i, k).parse().expect("invalid --threads"))
            }
            "--resume" => resume = true,
            "--max-minutes" => {
                max_minutes = Some(next_val(&mut i, k).parse().expect("invalid --max-minutes"))
            }
            other => {
                eprintln!("Unknown argument: {}", other);
                print_usage();
                std::process::exit(2);
            }
        }
        i += 1;
    }

    let output = output.unwrap_or_else(|| {
        eprintln!("--output is required");
        print_usage();
        std::process::exit(2);
    });

    CliOpts {
        seed_start,
        seed_end,
        seeds_list,
        output,
        manifest,
        replay_dir,
        replay_threshold,
        model_path,
        batch_id,
        shard_id,
        runner_job_id,
        git_commit,
        threads,
        resume,
        max_minutes,
    }
}

fn load_existing_seeds(path: &Path) -> HashSet<u64> {
    let mut done = HashSet::new();
    if !path.exists() {
        return done;
    }
    let f = match File::open(path) {
        Ok(f) => f,
        Err(_) => return done,
    };
    for line in BufReader::new(f).lines().flatten() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(trimmed) {
            if let Some(s) = v.get("seed").and_then(|x| x.as_u64()) {
                done.insert(s);
            }
        }
    }
    done
}

fn write_replay(dir: &Path, replay: &GameReplay) {
    fs::create_dir_all(dir).expect("create replay dir");
    let fname = format!(
        "seed_{}_pcs_{}.replay.json",
        replay.record.seed, replay.record.pieces
    );
    let path = dir.join(fname);
    let f = File::create(&path).expect("create replay file");
    let mut w = BufWriter::new(f);
    serde_json::to_writer_pretty(&mut w, replay).expect("serialize replay");
    w.flush().expect("flush replay");
}

fn main() {
    let opts = parse_cli();

    // 1. Verify pinned model SHA-256 before loading
    let model_bytes = fs::read(&opts.model_path).unwrap_or_else(|e| {
        eprintln!(
            "FATAL: failed to read model {}: {}",
            opts.model_path.display(),
            e
        );
        std::process::exit(1);
    });
    let actual_sha = sha256_hex(&model_bytes);
    if actual_sha != PINNED_MODEL_SHA256 {
        eprintln!(
            "FATAL: model SHA-256 mismatch!\n  expected: {}\n  actual:   {}\n  path:     {}",
            PINNED_MODEL_SHA256,
            actual_sha,
            opts.model_path.display()
        );
        std::process::exit(1);
    }

    // 2. Build requested seed list (deduplicated, preserving order)
    let mut requested_seeds: Vec<u64> = Vec::new();
    let mut seen_req = HashSet::new();
    if !opts.seeds_list.is_empty() {
        for s in &opts.seeds_list {
            if seen_req.insert(*s) {
                requested_seeds.push(*s);
            }
        }
    } else if let (Some(s0), Some(s1)) = (opts.seed_start, opts.seed_end) {
        assert!(
            s0 <= s1,
            "--seed-start ({}) must be <= --seed-end ({})",
            s0,
            s1
        );
        for s in s0..=s1 {
            requested_seeds.push(s);
        }
    } else {
        eprintln!("FATAL: specify either --seeds or both --seed-start and --seed-end");
        std::process::exit(2);
    }

    // 3. Resume support
    let already_done = if opts.resume {
        load_existing_seeds(&opts.output)
    } else {
        HashSet::new()
    };
    let pending_seeds: Vec<u64> = requested_seeds
        .iter()
        .copied()
        .filter(|s| !already_done.contains(s))
        .collect();
    let skipped_resume_count = requested_seeds.len() - pending_seeds.len();

    if let Some(parent) = opts.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).expect("create output parent dir");
        }
    }
    if let Some(ref rdir) = opts.replay_dir {
        fs::create_dir_all(rdir).expect("create replay dir");
    }

    // 4. Load model & construct locked BC-4 REAL FULL-MLP config
    let mut sm_val = SpatialValueModel::load_from_file(
        opts.model_path.to_str().expect("valid utf-8 model path"),
    )
    .expect("load pinned spatial_bc4.txt");
    sm_val.prepare_fast();
    let cfg = build_pinned_beam_config(Arc::new(sm_val));

    let meta = RunMeta {
        batch_id: opts.batch_id.clone(),
        shard_id: opts.shard_id,
        runner_job_id: opts.runner_job_id.clone(),
        git_commit: opts.git_commit.clone(),
    };

    let workers = opts
        .threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(2)
        })
        .max(1)
        .min(pending_seeds.len().max(1));

    println!(
        "=== BC-4 Mass Legal Simulator (Pinned REAL FULL-MLP p=5 w=16 h=8 b=28 cvar=25) ==="
    );
    println!("  batch_id:         {}", meta.batch_id);
    println!("  shard_id:         {}", meta.shard_id);
    println!("  runner_job_id:    {}", meta.runner_job_id);
    println!("  baseline_commit:  {}", PINNED_BASELINE_COMMIT);
    println!("  git_commit:       {}", meta.git_commit);
    println!("  model_sha256:     {} (VERIFIED)", actual_sha);
    println!(
        "  seeds:            requested={} skipped_resume={} pending={} workers={}",
        requested_seeds.len(),
        skipped_resume_count,
        pending_seeds.len(),
        workers
    );
    println!("  output:           {}", opts.output.display());
    if let Some(m) = opts.max_minutes {
        println!("  max_minutes:      {:.1}", m);
    }
    let _ = std::io::stdout().flush();

    let out_file = OpenOptions::new()
        .create(true)
        .append(opts.resume)
        .write(true)
        .truncate(!opts.resume)
        .open(&opts.output)
        .expect("open output jsonl");
    let writer = Mutex::new(BufWriter::new(out_file));
    let completed_set = Mutex::new(already_done);
    let shard_best: Mutex<Option<GameReplay>> = Mutex::new(None);
    let stopped_by_time_guard = std::sync::atomic::AtomicBool::new(false);
    let next_idx = AtomicUsize::new(0);
    let wall_t0 = Instant::now();

    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| loop {
                if let Some(max_min) = opts.max_minutes {
                    if wall_t0.elapsed().as_secs_f64() >= max_min * 60.0 {
                        stopped_by_time_guard.store(true, Ordering::Relaxed);
                        break;
                    }
                }
                let i = next_idx.fetch_add(1, Ordering::Relaxed);
                if i >= pending_seeds.len() {
                    break;
                }
                let seed = pending_seeds[i];
                let (rec, moves) = play_legal_seed_with_trace(seed, &cfg, &meta);

                // Flush JSONL record immediately for crash/timeout resilience
                {
                    let mut w = writer.lock().unwrap();
                    serde_json::to_writer(&mut *w, &rec).expect("serialize game record");
                    writeln!(w).expect("write newline");
                    w.flush().expect("flush jsonl");
                }

                // Save replay if completed and meets threshold, and track shard best
                if rec.completed_100l {
                    let replay = GameReplay {
                        record: rec.clone(),
                        moves,
                    };
                    if let Some(ref rdir) = opts.replay_dir {
                        if rec.pieces <= opts.replay_threshold {
                            write_replay(rdir, &replay);
                        }
                    }
                    let mut best = shard_best.lock().unwrap();
                    let is_better = match best.as_ref() {
                        None => true,
                        Some(prev) => {
                            rec.pieces < prev.record.pieces
                                || (rec.pieces == prev.record.pieces
                                    && rec.seed < prev.record.seed)
                        }
                    };
                    if is_better {
                        *best = Some(replay);
                    }
                }

                completed_set.lock().unwrap().insert(seed);

                println!(
                    "  [seed {:7}] status={:<9} pcs={:3} ds={:3} lines={:3} holds={:2} game={:5.1}s wall={:6.1}s",
                    rec.seed,
                    rec.status,
                    rec.pieces,
                    rec.ds_cleared,
                    rec.lines_cleared,
                    rec.diagnostics.holds_used,
                    rec.elapsed_sec,
                    wall_t0.elapsed().as_secs_f64()
                );
                let _ = std::io::stdout().flush();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });

    // Always save the best completed game of this shard as a replay
    if let Some(ref rdir) = opts.replay_dir {
        if let Some(best) = shard_best.into_inner().unwrap() {
            if best.record.pieces > opts.replay_threshold {
                write_replay(rdir, &best);
            }
        }
    }

    let done_seeds = completed_set.into_inner().unwrap();
    let missing_seeds: Vec<u64> = requested_seeds
        .iter()
        .copied()
        .filter(|s| !done_seeds.contains(s))
        .collect();
    let completed_count = requested_seeds.len() - missing_seeds.len();
    let hit_guard = stopped_by_time_guard.load(Ordering::Relaxed);

    if let Some(ref mpath) = opts.manifest {
        if let Some(parent) = mpath.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).expect("create manifest parent dir");
            }
        }
        let manifest = ShardManifest {
            schema_version: SCHEMA_VERSION,
            batch_id: opts.batch_id,
            shard_id: opts.shard_id,
            runner_job_id: opts.runner_job_id,
            git_commit: opts.git_commit,
            baseline_commit: PINNED_BASELINE_COMMIT,
            model_name: PINNED_MODEL_NAME,
            model_sha256: PINNED_MODEL_SHA256,
            requested_count: requested_seeds.len(),
            completed_count,
            skipped_resume_count,
            missing_seeds: missing_seeds.clone(),
            stopped_by_time_guard: hit_guard,
            wall_elapsed_sec: wall_t0.elapsed().as_secs_f64(),
        };
        let f = File::create(mpath).expect("create manifest file");
        serde_json::to_writer_pretty(f, &manifest).expect("write manifest");
    }

    println!(
        "=== Shard Done: {}/{} completed (missing={}, time_guard={}, wall={:.1}s) ===",
        completed_count,
        requested_seeds.len(),
        missing_seeds.len(),
        hit_guard,
        wall_t0.elapsed().as_secs_f64()
    );
}
