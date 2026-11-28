//! Legal `p=5` benchmark for Zero-Strategy-Fusion SO-ISMCTS (`ismcts.rs`)
//! reporting true batch Wall Time, games/hour, states/hour, average expansion depth,
//! post-spawn effective depth, topout%, and paired delta ± SE / 95% CI.

use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        BagTracker, BeamConfig, FastAI, GameContext, IsMctsConfig, IsMctsSearcher,
        SpatialValueModel,
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
    avg_depth: f64,
    post_spawn_depth: f64,
    eg_moves: usize,
    avg_trans_hits: f64,
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
    let mut depth_sum = 0.0f64;
    let mut depth_cnt = 0usize;
    let mut ps_depth_sum = 0.0f64;
    let mut ps_depth_cnt = 0usize;
    let mut trans_sum = 0usize;

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
        if r.used_endgame_solver {
            res.eg_moves += 1;
        } else {
            depth_sum += r.avg_sim_depth;
            depth_cnt += 1;
            if r.spawn_sims > 0 {
                ps_depth_sum += r.post_spawn_avg_depth;
                ps_depth_cnt += 1;
            }
            trans_sum += r.trans_hits;
        }

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
    res.avg_depth = if depth_cnt > 0 {
        depth_sum / depth_cnt as f64
    } else {
        0.0
    };
    res.post_spawn_depth = if ps_depth_cnt > 0 {
        ps_depth_sum / ps_depth_cnt as f64
    } else {
        0.0
    };
    res.avg_trans_hits = if depth_cnt > 0 {
        trans_sum as f64 / depth_cnt as f64
    } else {
        0.0
    };
    res.elapsed_s = t0.elapsed().as_secs_f64();
    res
}

fn run_parallel<F>(seeds: &[u64], f: F) -> (Vec<GameRes>, f64)
where
    F: Fn(u64) -> GameRes + Sync,
{
    let batch_t0 = Instant::now();
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
    let wall_s = batch_t0.elapsed().as_secs_f64();
    (out.into_iter().map(|x| x.unwrap()).collect(), wall_s)
}

fn summarize(label: &str, res: &[GameRes], wall_s: f64, ref_res: Option<&[GameRes]>) {
    let n_all = res.len();
    let ok: Vec<&GameRes> = res.iter().filter(|r| !r.topout).collect();
    let top = n_all - ok.len();
    let top_pct = 100.0 * top as f64 / n_all.max(1) as f64;
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
    let total_states: usize = res.iter().map(|r| r.pieces).sum();
    let games_hr = (n_all as f64) * 3600.0 / wall_s.max(1e-6);
    let states_hr = (total_states as f64) * 3600.0 / wall_s.max(1e-6);
    let avg_d = res.iter().map(|r| r.avg_depth).sum::<f64>() / n_all.max(1) as f64;
    let ps_d = res.iter().map(|r| r.post_spawn_depth).sum::<f64>() / n_all.max(1) as f64;
    let eg_mv = res.iter().map(|r| r.eg_moves as f64).sum::<f64>() / n_all.max(1) as f64;
    let tr_hit = res.iter().map(|r| r.avg_trans_hits).sum::<f64>() / n_all.max(1) as f64;

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
            let n = diffs.len() as f64;
            let md = diffs.iter().sum::<f64>() / n;
            let var = if diffs.len() > 1 {
                diffs.iter().map(|&x| (x - md) * (x - md)).sum::<f64>() / (n - 1.0)
            } else {
                0.0
            };
            let se = (var / n).sqrt();
            format!(
                " | vs_ref={:+.2}±{:.2} 95%CI=[{:+.2},{:+.2}] W/T/L={}/{}/{}",
                md,
                se,
                md - 1.96 * se,
                md + 1.96 * se,
                w,
                t,
                l
            )
        }
    } else {
        String::new()
    };

    println!(
        "{:<24} n={:2}/{:2} top={:4.1}% mean={:6.2} [{:3}..{:3}] nds={:5.2} cov={:4.1}% | wall={:5.1}s g/hr={:6.1} st/hr={:8.0} | avg_d={:4.2} ps_d={:4.2} eg={:4.1} tr={:5.1}{}",
        label,
        ok.len(),
        n_all,
        top_pct,
        mean,
        min_p,
        max_p,
        nds,
        cov,
        wall_s,
        games_hr,
        states_hr,
        avg_d,
        ps_d,
        eg_mv,
        tr_hit,
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
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("ablate");
    let model_path = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("models/spatial_bc4.txt");
    let n_seeds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);
    let seed_start: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20001);
    let seeds: Vec<u64> = (seed_start..seed_start + n_seeds as u64).collect();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);

    let sm = Arc::new(SpatialValueModel::load_from_file(model_path).expect("load spatial_bc4"));
    println!(
        "=== SO-ISMCTS Legal p=5 Benchmark [mode={}] ===\nModel: {} | Seeds: {}..={} (n={}) | CPU threads: {}\n",
        mode,
        model_path,
        seed_start,
        seed_start + n_seeds as u64 - 1,
        n_seeds,
        workers
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
        step_cost_cp: 100.0,
        split_tail_preview: false,
        use_endgame_solver: false,
        use_transposition: false,
        spatial_model: Some(sm.clone()),
        ..Default::default()
    };

    if mode == "psd20" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, max_a, root_a) in &[
            ("FixedDAG_a10_s2400", 2400usize, 10usize, 16usize),
            ("FixedDAG_a6_s2400", 2400, 6, 16),
            ("FixedDAG_a4_s2400", 2400, 4, 16),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_actions = max_a;
            cfg.root_max_actions = root_a;
            cfg.split_tail_preview = false;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "walltime" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims, trans) in &[
            ("NoDAG_EG_s2400", 2400usize, false),
            ("FixedDAG_EG_s2400", 2400, true),
            ("FixedDAG_EG_s2650_eqW", 2650, true),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.split_tail_preview = false;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = trans;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "dag20" {
        let mut cfg = base_m.clone();
        cfg.split_tail_preview = false;
        cfg.use_endgame_solver = true;
        cfg.use_transposition = true;
        let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
        summarize("D_noSplit_EG_TransDAG", &res, wall_s, None);
        return;
    }

    if mode == "ablate" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, split_tail, eg, trans) in &[
            ("A_base_noSplit_noEG", false, false, false),
            ("B_fullNext5_splitTail", true, false, false),
            ("C_noSplit_EG", false, true, false),
            ("D_noSplit_EG_TransDAG", false, true, true),
            ("E_fullNext5_EG_TransDAG", true, true, true),
        ] {
            let mut cfg = base_m.clone();
            cfg.split_tail_preview = split_tail;
            cfg.use_endgame_solver = eg;
            cfg.use_transposition = trans;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    // Default: compare BASE_5ply_b16 vs SO-ISMCTS
    let mut b5_cfg = BeamConfig::default();
    b5_cfg.beam_width = 16;
    b5_cfg.max_per_root = 8;
    b5_cfg.spatial_model = Some(sm.clone());
    b5_cfg.spatial_no_pe_internal = true;
    b5_cfg.root_pimc_worlds = 0;
    let (b5_res, b5_wall) = run_parallel(&seeds, |s| play_beam_seed(s, &b5_cfg));
    summarize("BASE_5ply_b16", &b5_res, b5_wall, None);

    let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &base_m));
    summarize("ISMCTS_s2400_d12_bell", &res, wall_s, Some(&b5_res));
}
