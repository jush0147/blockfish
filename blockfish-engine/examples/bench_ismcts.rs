//! Fast local Legal `p=5` benchmark for Zero-Strategy-Fusion SO-ISMCTS (`ismcts.rs`)
//! vs Frozen BC-4 5-ply Beam and matched-budget Clairvoyant PIMC.

use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        BagTracker, BeamConfig, FastAI, GameContext, IsMctsConfig, IsMctsSearcher,
        NcRolloutConfig, NcRolloutSearcher, SpatialValueModel,
    },
    Input, StackerExt as _,
};
use std::sync::Arc;
use std::time::Instant;

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

fn h1_blocks_over(stacker: &Stacker) -> usize {
    let g = garbage_rows(stacker);
    if g == 0 {
        return 0;
    }
    let mut row_top = [false; 10];
    for ((r, c), _) in stacker.matrix() {
        if r == g - 1 {
            row_top[c as usize] = true;
        }
    }
    if let Some(c1) = (0..10u16).find(|&c| !row_top[c as usize]) {
        let mut bo = 0usize;
        for ((r, c), _) in stacker.matrix() {
            if r >= g && c == c1 {
                bo += 1;
            }
        }
        bo
    } else {
        0
    }
}

#[derive(Clone, Debug, Default)]
struct GameRes {
    seed: u64,
    pieces: usize,
    topout: bool,
    nds: usize,
    cov_nxt: usize,
    ds_total: usize,
    avg_evals: f64,
    elapsed_s: f64,
}

fn play_beam_seed(seed: u64, cfg: &BeamConfig) -> GameRes {
    let t0 = Instant::now();
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());

    let mut res = GameRes {
        seed,
        ..Default::default()
    };
    let mut eval_sum = 0usize;

    while res.ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(res.ds_total) as u16;
        let mv = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };
        eval_sum += mv.nodes_evaluated;
        let mut inputs = mv.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        res.pieces += 1;
        res.ds_total += ds;
        res.nds += lines.saturating_sub(ds);
        if ds > 0 && h1_blocks_over(&stacker) > 0 {
            res.cov_nxt += 1;
        }
        if res.pieces >= 400 {
            break;
        }
    }
    res.topout = res.ds_total < 100;
    res.avg_evals = eval_sum as f64 / res.pieces.max(1) as f64;
    res.elapsed_s = t0.elapsed().as_secs_f64();
    res
}

fn play_ismcts_seed(seed: u64, cfg: &IsMctsConfig) -> GameRes {
    let t0 = Instant::now();
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut searcher = IsMctsSearcher::new_srs();
    let mut bag_tracker = BagTracker::new();

    let mut res = GameRes {
        seed,
        ..Default::default()
    };
    let mut eval_sum = 0usize;

    while res.ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(res.ds_total) as u16;
        let bag_mask = bag_tracker.observe(snap.hold, &snap.queue);
        let ctx = GameContext {
            garbage_rows: Some(g_rows),
            remaining_garbage: Some(rem_g),
            next_bag_mask: Some(bag_mask),
        };
        let r = searcher.search_with_ctx(&snap, ctx, cfg);
        if r.inputs.is_empty() {
            break;
        }
        eval_sum += r.mlp_evals;
        let mut inputs = r.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        res.pieces += 1;
        res.ds_total += ds;
        res.nds += lines.saturating_sub(ds);
        if ds > 0 && h1_blocks_over(&stacker) > 0 {
            res.cov_nxt += 1;
        }
        if res.pieces >= 400 {
            break;
        }
    }
    res.topout = res.ds_total < 100;
    res.avg_evals = eval_sum as f64 / res.pieces.max(1) as f64;
    res.elapsed_s = t0.elapsed().as_secs_f64();
    res
}

fn play_nc_seed(seed: u64, cfg: &NcRolloutConfig) -> GameRes {
    let t0 = Instant::now();
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut searcher = NcRolloutSearcher::new_srs();
    let mut bag_tracker = BagTracker::new();

    let mut res = GameRes {
        seed,
        ..Default::default()
    };
    let mut eval_sum = 0usize;

    while res.ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(res.ds_total) as u16;
        let bag_mask = bag_tracker.observe(snap.hold, &snap.queue);
        let ctx = GameContext {
            garbage_rows: Some(g_rows),
            remaining_garbage: Some(rem_g),
            next_bag_mask: Some(bag_mask),
        };
        let r = searcher.search_with_ctx(&snap, ctx, cfg);
        if r.inputs.is_empty() {
            break;
        }
        eval_sum += r.mlp_evals;
        let mut inputs = r.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        res.pieces += 1;
        res.ds_total += ds;
        res.nds += lines.saturating_sub(ds);
        if ds > 0 && h1_blocks_over(&stacker) > 0 {
            res.cov_nxt += 1;
        }
        if res.pieces >= 400 {
            break;
        }
    }
    res.topout = res.ds_total < 100;
    res.avg_evals = eval_sum as f64 / res.pieces.max(1) as f64;
    res.elapsed_s = t0.elapsed().as_secs_f64();
    res
}

fn run_parallel<F>(seeds: &[u64], f: F) -> Vec<GameRes>
where
    F: Fn(u64) -> GameRes + Sync,
{
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<GameRes>> = vec![None; seeds.len()];
    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    local.push((i, f(seeds[i])));
                }
                local
            }));
        }
        for h in handles {
            for (i, r) in h.join().unwrap() {
                out[i] = Some(r);
            }
        }
    });
    out.into_iter().map(|x| x.unwrap()).collect()
}

fn summarize(label: &str, res: &[GameRes], ref_res: Option<&[GameRes]>) {
    let ok: Vec<&GameRes> = res.iter().filter(|r| !r.topout).collect();
    let top = res.len() - ok.len();
    let mean = if ok.is_empty() {
        f64::NAN
    } else {
        ok.iter().map(|r| r.pieces as f64).sum::<f64>() / ok.len() as f64
    };
    let min_p = ok.iter().map(|r| r.pieces).min().unwrap_or(0);
    let max_p = ok.iter().map(|r| r.pieces).max().unwrap_or(0);
    let nds = if ok.is_empty() {
        0.0
    } else {
        ok.iter().map(|r| r.nds as f64).sum::<f64>() / ok.len() as f64
    };
    let cov = if ok.is_empty() {
        0.0
    } else {
        ok.iter().map(|r| r.cov_nxt as f64).sum::<f64>() / ok.len() as f64
    };
    let avg_ev = res.iter().map(|r| r.avg_evals).sum::<f64>() / res.len().max(1) as f64;
    let max_t = res
        .iter()
        .map(|r| r.elapsed_s)
        .fold(0.0f64, f64::max);

    let delta_str = if let Some(rr) = ref_res {
        let mut diffs = Vec::new();
        let mut w = 0usize;
        let mut t = 0usize;
        let mut l = 0usize;
        for (a, b) in res.iter().zip(rr.iter()) {
            if !a.topout && !b.topout {
                let d = a.pieces as f64 - b.pieces as f64;
                diffs.push(d);
                if a.pieces < b.pieces {
                    w += 1;
                } else if a.pieces == b.pieces {
                    t += 1;
                } else {
                    l += 1;
                }
            }
        }
        if diffs.is_empty() {
            String::new()
        } else {
            let md = diffs.iter().sum::<f64>() / diffs.len() as f64;
            format!(" | vs_ref={:+.2} W/T/L={}/{}/{}", md, w, t, l)
        }
    } else {
        String::new()
    };

    println!(
        "{:<24} n={:2}/{:2} top={:2} mean={:6.2} [{:3}..{:3}] nds={:5.2} cov={:4.1}% ev/mv={:6.1} t={:5.1}s{}",
        label,
        ok.len(),
        res.len(),
        top,
        mean,
        min_p,
        max_p,
        nds,
        cov,
        avg_ev,
        max_t,
        delta_str
    );
    let per_seed: Vec<String> = res
        .iter()
        .map(|r| {
            if r.topout {
                format!("{}:TOP", r.seed)
            } else {
                format!("{}:{}", r.seed, r.pieces)
            }
        })
        .collect();
    println!("  seeds: [{}]", per_seed.join(", "));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("scale");
    let model_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("models/spatial_bc4.txt");
    let n_seeds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let seed_start: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(101);
    let seeds: Vec<u64> = (seed_start..seed_start + n_seeds as u64).collect();

    let sm = Arc::new(SpatialValueModel::load_from_file(model_path).expect("load spatial_bc4"));
    println!(
        "=== SO-ISMCTS (Zero Strategy Fusion) Local Legal p=5 Benchmark [mode={}] ===\nModel: {} | Seeds: {:?}\n",
        mode, model_path, seeds
    );

    let base_m = IsMctsConfig {
        simulations: 2400,
        max_depth: 12,
        max_actions: 10,
        root_max_actions: 16,
        c_puct: 1.5,
        prior_temp: 120.0,
        fpu_penalty: 80.0,
        q_scale: 250.0,
        select_by_visits: false,
        backup_mode: 1,
        spatial_model: Some(sm.clone()),
        ..Default::default()
    };

    if mode == "scale" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, depth, bm, max_a, root_a) in &[
            ("ISMCTS_s2400_d12_bell", 2400usize, 12usize, 1u8, 10usize, 16usize),
            ("ISMCTS_s2400_d12_blend", 2400, 12, 2, 10, 16),
            ("ISMCTS_s5000_d14_bell", 5000, 14, 1, 12, 18),
            ("ISMCTS_s5000_d14_blend", 5000, 14, 2, 12, 18),
            ("ISMCTS_s10000_d16_bell", 10000, 16, 1, 12, 20),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = depth;
            cfg.backup_mode = bm;
            cfg.max_actions = max_a;
            cfg.root_max_actions = root_a;
            let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "baseline10" {
        let mut b5_cfg = BeamConfig::default();
        b5_cfg.beam_width = 16;
        b5_cfg.max_per_root = 8;
        b5_cfg.spatial_model = Some(sm.clone());
        b5_cfg.spatial_no_pe_internal = true;
        b5_cfg.root_pimc_worlds = 0;
        let b5_res = run_parallel(&seeds, |s| play_beam_seed(s, &b5_cfg));
        summarize("BASE_5ply_b16", &b5_res, None);

        let mut pimc_cfg = BeamConfig::default();
        pimc_cfg.beam_width = 24;
        pimc_cfg.max_per_root = 12;
        pimc_cfg.spatial_model = Some(sm.clone());
        pimc_cfg.spatial_no_pe_internal = true;
        pimc_cfg.root_pimc_worlds = 4;
        pimc_cfg.root_pimc_horizon = 7;
        pimc_cfg.root_pimc_width = 8;
        pimc_cfg.root_pimc_roots = 10;
        pimc_cfg.root_pimc_fine_roots = 8;
        pimc_cfg.root_pimc_prescreen_worlds = 0;
        pimc_cfg.root_pimc_steady_g4 = true;
        let pimc_res = run_parallel(&seeds, |s| play_beam_seed(s, &pimc_cfg));
        summarize("PIMC_w4_h7_b8", &pimc_res, Some(&b5_res));

        for &(name, sims, depth, bm) in &[
            ("s2400_d12_mean", 2400usize, 12usize, 0u8),
            ("s5000_d12_mean", 5000, 12, 0),
            ("s5000_d12_bell", 5000, 12, 1),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = depth;
            cfg.backup_mode = bm;
            cfg.max_actions = 10;
            cfg.root_max_actions = 16;
            let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, Some(&pimc_res));
        }
        return;
    }

    if mode == "ncroll" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, k, w, h, p_plies, p_w, p_mlp) in &[
            ("NC_w18_h8_p1", 12usize, 18usize, 8usize, 1usize, 1usize, false),
            ("NC_w18_h12_p1", 12, 18, 12, 1, 1, false),
            ("NC_w18_h16_p1", 12, 18, 16, 1, 1, false),
            ("NC_w18_h8_p2w6", 12, 18, 8, 2, 6, false),
            ("NC_w18_h12_p2w6", 12, 18, 12, 2, 6, false),
            ("NC_w18_h10_p3w6", 12, 18, 10, 3, 6, false),
        ] {
            let cfg = NcRolloutConfig {
                root_candidates: k,
                worlds: w,
                horizon: h,
                policy_plies: p_plies,
                policy_width: p_w,
                steady_g4: true,
                policy_full_mlp: p_mlp,
                spatial_model: Some(sm.clone()),
                ..Default::default()
            };
            let res = run_parallel(&seeds, |s| play_nc_seed(s, &cfg));
            summarize(name, &res, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "stepcost" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, depth, bm, sc) in &[
            ("s2400_d12_bell_sc100", 2400usize, 12usize, 1u8, 100.0f64),
            ("s2400_d12_bell_sc90", 2400, 12, 1, 90.0),
            ("s2400_d12_bell_sc86", 2400, 12, 1, 86.0),
            ("s5000_d12_bell_sc90", 5000, 12, 1, 90.0),
            ("s5000_d12_mean_sc90", 5000, 12, 0, 90.0),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = depth;
            cfg.backup_mode = bm;
            cfg.step_cost_cp = sc;
            cfg.max_actions = 10;
            cfg.root_max_actions = 16;
            let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "cmp10" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, depth, bm) in &[
            ("s2400_d12_bell", 2400usize, 12usize, 1u8),
            ("s5000_d12_mean", 5000, 12, 0),
            ("s5000_d12_bell", 5000, 12, 1),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = depth;
            cfg.backup_mode = bm;
            cfg.max_actions = 10;
            cfg.root_max_actions = 16;
            let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "tune10" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, depth, bm, cp, la) in &[
            ("s2400_d12_bell_la0", 2400usize, 12usize, 1u8, 1.5f64, 0usize),
            ("s2400_d12_bell_la2", 2400, 12, 1, 1.5, 2),
            ("s2400_d12_bell_cp08", 2400, 12, 1, 0.8, 0),
            ("s2400_d12_bell_cp25", 2400, 12, 1, 2.5, 0),
            ("s5000_d12_bell_la0", 5000, 12, 1, 1.5, 0),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = depth;
            cfg.backup_mode = bm;
            cfg.c_puct = cp;
            cfg.leaf_obs_lookahead = la;
            cfg.max_actions = 10;
            cfg.root_max_actions = 16;
            let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "clean10" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, depth, bm) in &[
            ("s2400_d8_mean", 2400usize, 8usize, 0u8),
            ("s2400_d8_bellman", 2400, 8, 1),
            ("s2400_d8_consbell", 2400, 8, 3),
            ("s2400_d12_bellman", 2400, 12, 1),
            ("s2400_d12_consbell", 2400, 12, 3),
            ("s5000_d8_consbell", 5000, 8, 3),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = depth;
            cfg.backup_mode = bm;
            cfg.max_actions = 10;
            cfg.root_max_actions = 16;
            let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    // Default quick comparison vs 5-ply beam and PIMC
    let mut b5_cfg = BeamConfig::default();
    b5_cfg.beam_width = 16;
    b5_cfg.max_per_root = 8;
    b5_cfg.spatial_model = Some(sm.clone());
    b5_cfg.spatial_no_pe_internal = true;
    b5_cfg.root_pimc_worlds = 0;
    let b5_res = run_parallel(&seeds, |s| play_beam_seed(s, &b5_cfg));
    summarize("BASE_5ply_b16", &b5_res, None);

    let res = run_parallel(&seeds, |s| play_ismcts_seed(s, &base_m));
    summarize("ISMCTS_s2400_d12_bell", &res, Some(&b5_res));
}
