//! Legal `p=5` benchmark for Zero-Strategy-Fusion SO-ISMCTS (`ismcts.rs`)
//! reporting true batch Wall Time, games/hour, states/hour, average expansion depth,
//! post-spawn effective depth, topout%, and paired delta ± SE / 95% CI.

use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard},
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

fn run_parallel<T, F>(seeds: &[u64], f: F) -> (Vec<T>, f64)
where
    T: Send,
    F: Fn(u64) -> T + Sync,
{
    let batch_t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<T>> = (0..seeds.len()).map(|_| None).collect();
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

    if mode == "nocpw_scale_b1" || mode == "nocpw_scale_b4" {
        let bm: u8 = if mode == "nocpw_scale_b1" { 1 } else { 4 };
        let tag = if bm == 1 { "B1_BellMin" } else { "B4_VisitW" };
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &sims in &[1200usize, 2400, 4800] {
            let name = format!("TruncTailApprox_noCPW_{}_s{}", tag, sims);
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = 16;
            cfg.backup_mode = bm;
            cfg.split_tail_preview = false;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_alpha = 0.0;
            cfg.cpw_tail_alpha = 0.0;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(&name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "steady_ab20" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sg4) in &[
            ("TruncTailApprox_noCPW_steadyG4_true", true),
            ("TruncTailApprox_noCPW_exactJstrisG9", false),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = 2400;
            cfg.max_depth = 16;
            cfg.backup_mode = 1;
            cfg.steady_g4 = sg4;
            cfg.split_tail_preview = false;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_alpha = 0.0;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "td_causal_audit" {
        let extract_vis = |snap: &blockfish::ai::Snapshot| -> (Option<u8>, [u8; 6], usize) {
            let q_len = snap.queue.len().min(6);
            let mut vis = [0u8; 6];
            for i in 0..q_len {
                vis[i] = color_to_idx(snap.queue[i]) as u8;
            }
            let hold = snap.hold.map(|c| color_to_idx(c) as u8);
            (hold, vis, q_len)
        };

        #[derive(Clone, Default)]
        struct StepRec {
            d_place: f64,
            d_spawn: f64,
            d_queue: f64,
            delta: f64,
            n_spawn: usize,
            lines: usize,
            ds: usize,
            rem_g: u16,
            rows: u8,
        }
        #[derive(Clone, Default)]
        struct GameTraj {
            steps: Vec<StepRec>,
            k_drifts: [Vec<f64>; 4], // k=1,2,4,8
            cal_errs: Vec<f64>,
            audit_states: Vec<(blockfish::ai::Snapshot, GameContext)>,
        }

        let mut roll_cfg = base_m.clone();
        roll_cfg.simulations = 1200;
        roll_cfg.max_depth = 16;
        roll_cfg.backup_mode = 1;
        roll_cfg.split_tail_preview = false;
        roll_cfg.use_endgame_solver = true;
        roll_cfg.use_transposition = true;
        roll_cfg.dag_propagate_parents = true;
        roll_cfg.cpw_alpha = 0.0;

        let sm_c = sm.clone();
        let seed0 = seeds[0];
        let (trajs, wall_s) = run_parallel(&seeds, |sd| {
            let mut game_cfg = BSConfig::default();
            game_cfg.garbage.total_lines = Some(100);
            game_cfg.prng_seed = Some(sd);
            let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
            let mut searcher = IsMctsSearcher::new_srs();
            let mut bt = BagTracker::new();
            let mut ds_tot = 0usize;
            let mut ply = 0usize;
            let mut gt = GameTraj::default();
            let mut v_seq: Vec<f64> = Vec::new();

            while ds_tot < 100 && stacker.current_piece_type().is_some() && ply < 350 {
                let snap0 = stacker.snapshot().unwrap();
                let g0 = garbage_rows(&stacker);
                let rem0 = 100usize.saturating_sub(ds_tot) as u16;
                let bm0 = bt.observe(snap0.hold, &snap0.queue[..snap0.queue.len().min(6)]);
                let fb0 = FastBoard::from_basic_matrix(&snap0.matrix, g0);
                let (h0, vis0, vl0) = extract_vis(&snap0);
                let v0 = IsMctsSearcher::eval_board_v_bc4(&sm_c, &fb0, rem0, h0, &vis0[..vl0], bm0);
                let ctx0 = GameContext {
                    garbage_rows: Some(g0),
                    remaining_garbage: Some(rem0),
                    next_bag_mask: Some(bm0),
                };
                if sd <= seed0 + 2 && ply % 10 == 5 && rem0 > 12 {
                    gt.audit_states.push((snap0.clone(), ctx0));
                }
                let r = searcher.search_with_ctx(&snap0, ctx0, &roll_cfg);
                if r.inputs.is_empty() { break; }
                let mut ins = r.inputs;
                ins.pop();
                stacker.run(ins);
                let (lines, ds) = stacker.hard_drop();
                ds_tot += ds;
                v_seq.push(v0);

                if ds_tot < 100 && stacker.current_piece_type().is_some() {
                    let snap1 = stacker.snapshot().unwrap();
                    let g1 = garbage_rows(&stacker);
                    let rem1 = 100usize.saturating_sub(ds_tot) as u16;
                    let bm1 = {
                        let mut bt_peek = bt.clone();
                        bt_peek.observe(snap1.hold, &snap1.queue[..snap1.queue.len().min(6)])
                    };
                    let fb_after = FastBoard::from_basic_matrix(&snap1.matrix, g1);
                    let g_no_spawn = (g0 as usize).saturating_sub(ds);
                    let n_spawn = (g1 as usize).saturating_sub(g_no_spawn);
                    // Reconstruct exact intermediate board B_placed (after placement/clear, BEFORE spawn)
                    let mut fb_placed = fb_after;
                    if n_spawn > 0 && (fb_after.rows as usize) >= n_spawn {
                        let new_r = (fb_after.rows as usize) - n_spawn;
                        for rr in 0..new_r {
                            fb_placed.data[rr] = fb_after.data[rr + n_spawn];
                        }
                        for rr in new_r..(fb_after.rows as usize) {
                            fb_placed.data[rr] = 0;
                        }
                        fb_placed.rows = new_r as u8;
                        fb_placed.garbage_rows = fb_after.garbage_rows.saturating_sub(n_spawn as u8);
                    }
                    let (h1, vis1, vl1) = extract_vis(&snap1);
                    let v1 = IsMctsSearcher::eval_board_v_bc4(&sm_c, &fb_placed, rem1, h0, &vis0[..vl0], bm0);
                    let v2 = IsMctsSearcher::eval_board_v_bc4(&sm_c, &fb_after, rem1, h0, &vis0[..vl0], bm0);
                    let v3 = IsMctsSearcher::eval_board_v_bc4(&sm_c, &fb_after, rem1, h1, &vis1[..vl1], bm1);
                    let d_place = (100.0 + v1) - v0;
                    let d_spawn = v2 - v1;
                    let d_queue = v3 - v2;
                    let delta = (100.0 + v3) - v0;
                    assert!((d_place + d_spawn + d_queue - delta).abs() < 1e-6);
                    gt.steps.push(StepRec {
                        d_place,
                        d_spawn,
                        d_queue,
                        delta,
                        n_spawn,
                        lines,
                        ds,
                        rem_g: rem0,
                        rows: fb0.rows,
                    });
                }
                ply += 1;
            }
            if ds_tot >= 100 && v_seq.len() >= 10 {
                let t_tot = v_seq.len();
                for t in 0..t_tot {
                    gt.cal_errs.push(v_seq[t] - 100.0 * ((t_tot - t) as f64));
                }
                for (ki, &k) in [1usize, 2, 4, 8].iter().enumerate() {
                    for t in 0..t_tot.saturating_sub(k) {
                        gt.k_drifts[ki].push(((100.0 * k as f64 + v_seq[t + k]) - v_seq[t]) / (k as f64));
                    }
                }
            }
            gt
        });

        let cluster_stats = |per_game_vals: &[Vec<f64>]| -> (usize, usize, f64, f64) {
            let mut g_means = Vec::new();
            let mut n_tot = 0usize;
            for gv in per_game_vals {
                if !gv.is_empty() {
                    n_tot += gv.len();
                    g_means.push(gv.iter().sum::<f64>() / gv.len() as f64);
                }
            }
            let g_act = g_means.len();
            if g_act == 0 {
                return (0, 0, 0.0, 0.0);
            }
            let mean = g_means.iter().sum::<f64>() / g_act as f64;
            let se = if g_act > 1 {
                let var = g_means.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (g_act as f64 - 1.0);
                (var / g_act as f64).sqrt()
            } else {
                0.0
            };
            (g_act, n_tot, mean, se)
        };

        let report_decomp = |label: &str, pred: &dyn Fn(&StepRec) -> bool| {
            let mut pg_tot = Vec::new();
            let mut pg_pl = Vec::new();
            let mut pg_sp = Vec::new();
            let mut pg_qu = Vec::new();
            for gt in &trajs {
                let mut vt = Vec::new();
                let mut vp = Vec::new();
                let mut vs = Vec::new();
                let mut vq = Vec::new();
                for s in &gt.steps {
                    if pred(s) {
                        vt.push(s.delta);
                        vp.push(s.d_place);
                        vs.push(s.d_spawn);
                        vq.push(s.d_queue);
                    }
                }
                pg_tot.push(vt);
                pg_pl.push(vp);
                pg_sp.push(vs);
                pg_qu.push(vq);
            }
            let (g_act, n_st, mt, set) = cluster_stats(&pg_tot);
            let (_, _, mp, sep) = cluster_stats(&pg_pl);
            let (_, _, ms, ses) = cluster_stats(&pg_sp);
            let (_, _, mq, seq) = cluster_stats(&pg_qu);
            println!(
                "  {:28} | G={:2} n={:4} | total={:+6.2}±{:4.2} | place={:+6.2}±{:4.2} | spawn={:+6.2}±{:4.2} | queue={:+6.2}±{:4.2}",
                label, g_act, n_st, mt, set, mp, sep, ms, ses, mq, seq
            );
        };

        println!("=== Part 1: Exact Searcher-Encoding Causal TD Decomposition [CPW=OFF, G={} games, wall={:.1}s] ===", trajs.len(), wall_s);
        println!("  (All SEs are Game-Clustered across G independent games; delta = d_place + d_spawn + d_queue)");
        report_decomp("ALL transitions", &|_| true);
        println!("  -- Joint Stratification (n_spawn, ds, lines) --");
        report_decomp("sp=0, lines=0 (stack)", &|s| s.n_spawn == 0 && s.lines == 0);
        report_decomp("sp=0, ds>0 (no-spawn ds)", &|s| s.n_spawn == 0 && s.ds > 0);
        report_decomp("sp=0, nds>0 (non-ds clr)", &|s| s.n_spawn == 0 && s.lines > 0 && s.ds == 0);
        report_decomp("sp=1, ds=1", &|s| s.n_spawn == 1 && s.ds == 1);
        report_decomp("sp=1, ds>=2", &|s| s.n_spawn == 1 && s.ds >= 2);
        report_decomp("sp>=2, ds>=2", &|s| s.n_spawn >= 2 && s.ds >= 2);
        println!("  -- Spawn Events (n_spawn>=1) by rem_g & Board Height --");
        report_decomp("sp>=1, rem_g>60 (early)", &|s| s.n_spawn >= 1 && s.rem_g > 60);
        report_decomp("sp>=1, rem_g 31..=60 (mid)", &|s| s.n_spawn >= 1 && s.rem_g > 30 && s.rem_g <= 60);
        report_decomp("sp>=1, rem_g<=30 (late)", &|s| s.n_spawn >= 1 && s.rem_g <= 30);
        report_decomp("sp>=1, rows<=8", &|s| s.n_spawn >= 1 && s.rows <= 8);
        report_decomp("sp>=1, rows 9..=11", &|s| s.n_spawn >= 1 && s.rows >= 9 && s.rows <= 11);
        report_decomp("sp>=1, rows>=12", &|s| s.n_spawn >= 1 && s.rows >= 12);
        println!("  -- Multi-Step Per-Ply Drift & Global Calibration (Game-Clustered SE) --");
        for (ki, &k) in [1usize, 2, 4, 8].iter().enumerate() {
            let pg: Vec<Vec<f64>> = trajs.iter().map(|gt| gt.k_drifts[ki].clone()).collect();
            let (ga, ns, m, se) = cluster_stats(&pg);
            println!("  k={} per-ply drift           | G={:2} n={:4} | mean={:+6.2} ± {:4.2} cp/ply", k, ga, ns, m, se);
        }
        {
            let pg: Vec<Vec<f64>> = trajs.iter().map(|gt| gt.cal_errs.clone()).collect();
            let (ga, ns, m, se) = cluster_stats(&pg);
            println!("  V_bc4(s_t) - 100*(T-t)       | G={:2} n={:4} | mean={:+7.2} ± {:5.2} cp", ga, ns, m, se);
        }

        // Part 2: Non-Circular Candidate Pruning Recall (Fair minimum visits to ALL legal root cands!)
        let mut audit_states = Vec::new();
        for gt in &trajs {
            audit_states.extend(gt.audit_states.iter().cloned());
        }
        let n_st = audit_states.len().min(45);
        audit_states.truncate(n_st);
        println!("\n=== Part 2: Fair Non-Circular Candidate Pruning Recall (root_max=64, min_visits=40/cand, s=3000, CPW=OFF, n={}) ===", n_st);
        let mut searcher = IsMctsSearcher::new_srs();
        let mut rec_top16 = 0usize;
        let mut rec_top10 = 0usize;
        let mut tail_beats_0 = 0usize;
        let mut tail_beats_25 = 0usize;
        let mut tail_beats_50 = 0usize;
        let mut q_gap_sum = 0.0f64;
        let mut tot_cands = 0usize;

        for (snap, ctx) in &audit_states {
            let mut cfg_fair = base_m.clone();
            cfg_fair.simulations = 3000;
            cfg_fair.max_depth = 16;
            cfg_fair.root_max_actions = 64;
            cfg_fair.root_min_visits_per_action = 40;
            cfg_fair.backup_mode = 1;
            cfg_fair.split_tail_preview = false;
            cfg_fair.use_endgame_solver = false;
            cfg_fair.use_transposition = true;
            cfg_fair.dag_propagate_parents = true;
            cfg_fair.cpw_alpha = 0.0;
            let r = searcher.search_with_ctx(snap, *ctx, &cfg_fair);
            tot_cands += r.root_candidates.len();
            // Rank by 1-ply linear prior descending
            let mut by_prior = r.root_candidates.clone();
            by_prior.sort_by(|a, b| b.3.total_cmp(&a.3));
            let best_all = r
                .root_candidates
                .iter()
                .filter(|x| x.1 >= 40)
                .min_by(|a, b| a.2.total_cmp(&b.2))
                .unwrap();
            let best_top16_q = by_prior
                .iter()
                .take(16)
                .filter(|x| x.1 >= 40)
                .map(|x| x.2)
                .fold(f64::INFINITY, f64::min);
            let rank = by_prior
                .iter()
                .position(|x| x.0.did_hold == best_all.0.did_hold && x.0.tf == best_all.0.tf)
                .unwrap_or(99);
            if rank < 16 { rec_top16 += 1; }
            if rank < 10 { rec_top10 += 1; }
            let gap = best_top16_q - best_all.2; // > 0 means tail beat top-16!
            if gap > 1e-6 { tail_beats_0 += 1; }
            if gap > 25.0 { tail_beats_25 += 1; }
            if gap > 50.0 { tail_beats_50 += 1; }
            q_gap_sum += gap.max(0.0);
        }
        println!(
            "  Avg legal cands={:.1} | Fair Best-Q recall: top-16={:.1}% ({}/{}), top-10={:.1}% ({}/{})",
            tot_cands as f64 / n_st as f64,
            100.0 * rec_top16 as f64 / n_st as f64, rec_top16, n_st,
            100.0 * rec_top10 as f64 / n_st as f64, rec_top10, n_st
        );
        println!(
            "  Tail (>top-16) beats top-16: >0cp={:.1}% ({}/{}), >25cp={:.1}% ({}/{}), >50cp={:.1}% ({}/{}) | mean Q regret={:.2} cp",
            100.0 * tail_beats_0 as f64 / n_st as f64, tail_beats_0, n_st,
            100.0 * tail_beats_25 as f64 / n_st as f64, tail_beats_25, n_st,
            100.0 * tail_beats_50 as f64 / n_st as f64, tail_beats_50, n_st,
            q_gap_sum / n_st as f64
        );
        return;
    }

    if mode == "nocpw_diag" {
        let eval_v_bc4 = |snap: &blockfish::ai::Snapshot, g_rows: u16, rem_g: u16, bmask: u8| -> f64 {
            let fb = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
            let q_len = snap.queue.len().min(6);
            let mut vis = [0u8; 6];
            for i in 0..q_len {
                vis[i] = color_to_idx(snap.queue[i]) as u8;
            }
            let hold = snap.hold.map(|c| color_to_idx(c) as u8);
            IsMctsSearcher::eval_board_v_bc4(&sm, &fb, rem_g, hold, &vis[..q_len], bmask)
        };

        // Part A & B: Play 10 full CPW-OFF games (30001..=30010, s=1200) to collect:
        //   - Same-path 1-step & k-step TD residuals stratified by spawn, clear, and board height
        //   - 45 midgame states (from 30001..=30003) for CPW-OFF root action stability & recall audit
        println!("=== Part 1: Same-Path TD Residual delta_t = (100 + V_bc4(s_t+1)) - V_bc4(s_t) [CPW=OFF, TruncTailApprox] ===");
        #[derive(Default)]
        struct Stat { n: usize, sum: f64, sum2: f64 }
        impl Stat {
            fn push(&mut self, x: f64) { self.n += 1; self.sum += x; self.sum2 += x * x; }
            fn report(&self, label: &str) {
                if self.n == 0 {
                    println!("  {:28}: n=   0", label);
                    return;
                }
                let m = self.sum / self.n as f64;
                let var = if self.n > 1 {
                    ((self.sum2 - self.n as f64 * m * m) / (self.n as f64 - 1.0)).max(0.0)
                } else { 0.0 };
                let sd = var.sqrt();
                let se = sd / (self.n as f64).sqrt();
                println!("  {:28}: n={:4}  mean={:+7.2} cp  SE={:5.2}  SD={:6.2}", label, self.n, m, se, sd);
            }
        }
        let mut st_all = Stat::default();
        let mut st_sp0 = Stat::default();
        let mut st_sp1 = Stat::default();
        let mut st_sp2 = Stat::default();
        let mut st_clr0 = Stat::default();
        let mut st_ds = Stat::default();
        let mut st_nds = Stat::default();
        let mut st_h_lo = Stat::default();
        let mut st_h_mid = Stat::default();
        let mut st_h_hi = Stat::default();
        let mut st_h_dan = Stat::default();
        let mut st_k = [Stat::default(), Stat::default(), Stat::default(), Stat::default()]; // k=1,2,4,8
        let mut st_cal = Stat::default(); // V_bc4(s_t) - 100*(T - t)
        let mut audit_states = Vec::new();

        let mut roll_cfg = base_m.clone();
        roll_cfg.simulations = 1200;
        roll_cfg.max_depth = 16;
        roll_cfg.backup_mode = 1;
        roll_cfg.split_tail_preview = false;
        roll_cfg.use_endgame_solver = true;
        roll_cfg.use_transposition = true;
        roll_cfg.dag_propagate_parents = true;
        roll_cfg.cpw_alpha = 0.0;

        let mut searcher = IsMctsSearcher::new_srs();
        for &sd in &seeds {
            let mut game_cfg = BSConfig::default();
            game_cfg.garbage.total_lines = Some(100);
            game_cfg.prng_seed = Some(sd);
            let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
            let mut bt = BagTracker::new();
            let mut ds_tot = 0usize;
            let mut ply = 0usize;
            let mut traj_v: Vec<f64> = Vec::new();
            let mut traj_meta: Vec<(usize, usize, usize, u8)> = Vec::new(); // (n_spawn, lines, ds, rows_before)

            while ds_tot < 100 && stacker.current_piece_type().is_some() && ply < 350 {
                let snap = stacker.snapshot().unwrap();
                let g_rows = garbage_rows(&stacker);
                let rem_g = 100usize.saturating_sub(ds_tot) as u16;
                let bmask = bt.observe(snap.hold, &snap.queue[..snap.queue.len().min(6)]);
                let fb_before = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
                let v_before = eval_v_bc4(&snap, g_rows, rem_g, bmask);
                let ctx = GameContext {
                    garbage_rows: Some(g_rows),
                    remaining_garbage: Some(rem_g),
                    next_bag_mask: Some(bmask),
                };
                if sd <= seeds[0] + 2 && ply % 10 == 5 && rem_g > 12 {
                    audit_states.push((snap.clone(), ctx));
                }
                let r = searcher.search_with_ctx(&snap, ctx, &roll_cfg);
                if r.inputs.is_empty() { break; }
                let mut ins = r.inputs;
                ins.pop();
                stacker.run(ins);
                let (lines, ds) = stacker.hard_drop();
                ds_tot += ds;
                let g_after = garbage_rows(&stacker);
                let g_no_spawn = (g_rows as usize).saturating_sub(ds);
                let n_spawn = (g_after as usize).saturating_sub(g_no_spawn);
                traj_v.push(v_before);
                traj_meta.push((n_spawn, lines, ds, fb_before.rows));
                ply += 1;
            }
            if ds_tot >= 100 && traj_v.len() >= 10 {
                let t_tot = traj_v.len();
                for t in 0..t_tot {
                    st_cal.push(traj_v[t] - 100.0 * ((t_tot - t) as f64));
                }
                for t in 0..(t_tot - 1) {
                    let delta = (100.0 + traj_v[t + 1]) - traj_v[t];
                    let (nsp, lines, ds, rows) = traj_meta[t];
                    st_all.push(delta);
                    if nsp == 0 { st_sp0.push(delta); }
                    else if nsp == 1 { st_sp1.push(delta); }
                    else { st_sp2.push(delta); }
                    if lines == 0 { st_clr0.push(delta); }
                    else if ds > 0 { st_ds.push(delta); }
                    else { st_nds.push(delta); }
                    if rows <= 6 { st_h_lo.push(delta); }
                    else if rows <= 10 { st_h_mid.push(delta); }
                    else if rows <= 13 { st_h_hi.push(delta); }
                    else { st_h_dan.push(delta); }
                }
                for (ki, &k) in [1usize, 2, 4, 8].iter().enumerate() {
                    for t in 0..t_tot.saturating_sub(k) {
                        let dk = (100.0 * k as f64 + traj_v[t + k]) - traj_v[t];
                        st_k[ki].push(dk / (k as f64)); // per-ply drift
                    }
                }
            }
        }
        st_all.report("ALL transitions (1-step)");
        st_cal.report("V_bc4(s_t) - 100*(T-t)");
        println!("  -- Stratified by Garbage Spawn --");
        st_sp0.report("n_spawn == 0");
        st_sp1.report("n_spawn == 1");
        st_sp2.report("n_spawn >= 2");
        println!("  -- Stratified by Line Clear / Downstack --");
        st_clr0.report("lines == 0");
        st_ds.report("ds > 0 (downstack)");
        st_nds.report("lines > 0 && ds == 0 (nds)");
        println!("  -- Stratified by Board Height (rows before move) --");
        st_h_lo.report("rows <= 6");
        st_h_mid.report("rows 7..=10");
        st_h_hi.report("rows 11..=13");
        st_h_dan.report("rows >= 14");
        println!("  -- Multi-Step Per-Ply TD Residual ((k*100 + V(t+k) - V(t))/k) --");
        for (ki, &k) in [1usize, 2, 4, 8].iter().enumerate() {
            st_k[ki].report(&format!("k={} per-ply drift", k));
        }

        // Part 2: CPW-OFF Root Action Stability (backup_mode=1 and backup_mode=4) + Candidate Pruning Recall
        let n_st = audit_states.len().min(45);
        audit_states.truncate(n_st);
        println!("\n=== Part 2: CPW-OFF Root Action Stability (n={} identical states, 1200 vs 2400 vs 4800) ===", n_st);
        for &(bm, label) in &[(1u8, "B1_BellMin"), (4u8, "B4_VisitW")] {
            let mut chg_12_24 = 0usize;
            let mut chg_24_48 = 0usize;
            let mut chg_12_48 = 0usize;
            let mut top3_ov = 0.0f64;
            let mut rho_sum = 0.0f64;
            let mut rho_cnt = 0usize;
            for (snap, ctx) in &audit_states {
                let mut res_by_s = Vec::new();
                for &sims in &[1200usize, 2400, 4800] {
                    let mut cfg = base_m.clone();
                    cfg.simulations = sims;
                    cfg.max_depth = 16;
                    cfg.backup_mode = bm;
                    cfg.split_tail_preview = false;
                    cfg.use_endgame_solver = false;
                    cfg.use_transposition = true;
                    cfg.dag_propagate_parents = true;
                    cfg.cpw_alpha = 0.0;
                    let r = searcher.search_with_ctx(snap, *ctx, &cfg);
                    let mut by_q = r.root_candidates.clone();
                    by_q.sort_by(|a, b| a.2.total_cmp(&b.2));
                    res_by_s.push((r.inputs, by_q));
                }
                if res_by_s[0].0 != res_by_s[1].0 { chg_12_24 += 1; }
                if res_by_s[1].0 != res_by_s[2].0 { chg_24_48 += 1; }
                if res_by_s[0].0 != res_by_s[2].0 { chg_12_48 += 1; }
                let t3_a: Vec<_> = res_by_s[0].1.iter().take(3).map(|x| (x.0.did_hold, x.0.tf)).collect();
                let t3_b: Vec<_> = res_by_s[2].1.iter().take(3).map(|x| (x.0.did_hold, x.0.tf)).collect();
                top3_ov += (t3_a.iter().filter(|x| t3_b.contains(x)).count() as f64) / 3.0;
                let mut ra_v = Vec::new();
                let mut rb_v = Vec::new();
                for (ra, ca) in res_by_s[0].1.iter().enumerate() {
                    let key = (ca.0.did_hold, ca.0.tf);
                    if let Some(rb) = res_by_s[2].1.iter().position(|cb| (cb.0.did_hold, cb.0.tf) == key) {
                        ra_v.push(ra as f64);
                        rb_v.push(rb as f64);
                    }
                }
                let k = ra_v.len();
                if k >= 3 {
                    let d2: f64 = ra_v.iter().zip(&rb_v).map(|(a, b)| (a - b).powi(2)).sum();
                    rho_sum += 1.0 - (6.0 * d2) / ((k as f64) * ((k * k - 1) as f64));
                    rho_cnt += 1;
                }
            }
            println!(
                "  {:12}: chg 1200->2400={:4.1}% ({:2}/{}) | 2400->4800={:4.1}% ({:2}/{}) | 1200->4800={:4.1}% ({:2}/{}) | Top3_ov={:.1}% | Spearman={:.3}",
                label,
                100.0 * chg_12_24 as f64 / n_st as f64, chg_12_24, n_st,
                100.0 * chg_24_48 as f64 / n_st as f64, chg_24_48, n_st,
                100.0 * chg_12_48 as f64 / n_st as f64, chg_12_48, n_st,
                100.0 * top3_ov / n_st as f64,
                rho_sum / rho_cnt.max(1) as f64
            );
        }

        // Part 3: Candidate Action Pruning Recall (root_max_actions=64 vs top-16 / top-10 linear prior)
        println!("\n=== Part 3: Candidate Pruning Recall (root_max_actions=64 @ s=2400 vs 1-ply linear top-16 / top-10) ===");
        let mut in_top16 = 0usize;
        let mut in_top10 = 0usize;
        let mut in_top16_by_v = 0usize;
        let mut total_cands = 0usize;
        for (snap, ctx) in &audit_states {
            let mut cfg64 = base_m.clone();
            cfg64.simulations = 2400;
            cfg64.max_depth = 16;
            cfg64.root_max_actions = 64;
            cfg64.backup_mode = 1;
            cfg64.split_tail_preview = false;
            cfg64.use_endgame_solver = false;
            cfg64.use_transposition = true;
            cfg64.dag_propagate_parents = true;
            cfg64.cpw_alpha = 0.0;
            let r64 = searcher.search_with_ctx(snap, *ctx, &cfg64);
            total_cands += r64.root_candidates.len();
            // Sort by prior descending (= exact 1-ply linear ranking!)
            let mut by_prior = r64.root_candidates.clone();
            by_prior.sort_by(|a, b| b.3.total_cmp(&a.3));
            // Best Q among visited >= max_v/4 (same rule as select_by_visits=false)
            let max_v = r64.root_candidates.iter().map(|x| x.1).max().unwrap_or(0);
            let min_req = (max_v / 4).max(1);
            let best_q_cand = r64.root_candidates.iter().filter(|x| x.1 >= min_req).min_by(|a, b| a.2.total_cmp(&b.2)).unwrap();
            let best_v_cand = &r64.root_candidates[0]; // already sorted by visits desc
            let rank_q = by_prior.iter().position(|x| x.0.did_hold == best_q_cand.0.did_hold && x.0.tf == best_q_cand.0.tf).unwrap_or(99);
            let rank_v = by_prior.iter().position(|x| x.0.did_hold == best_v_cand.0.did_hold && x.0.tf == best_v_cand.0.tf).unwrap_or(99);
            if rank_q < 16 { in_top16 += 1; }
            if rank_q < 10 { in_top10 += 1; }
            if rank_v < 16 { in_top16_by_v += 1; }
        }
        println!(
            "  Avg legal root cands={:.1} | Best-Q recall in top-16={:.1}% ({}/{}), in top-10={:.1}% ({}/{}) | Most-visited recall in top-16={:.1}% ({}/{})",
            total_cands as f64 / n_st as f64,
            100.0 * in_top16 as f64 / n_st as f64, in_top16, n_st,
            100.0 * in_top10 as f64 / n_st as f64, in_top10, n_st,
            100.0 * in_top16_by_v as f64 / n_st as f64, in_top16_by_v, n_st
        );
        return;
    }

    if mode == "audit_all" {
        // Part A: Non-uniform P=[0.90, 0.05, 0.05], V=[0, 100, 100], exact E[V]=10.0
        println!("=== Part A: Non-Uniform Chance Tree P=[0.90, 0.05, 0.05], V=[0, 100, 100], exact E[V]=10.00 ===");
        let probs = [0.90f64, 0.05, 0.05];
        let vals = [0.0f64, 100.0, 100.0];
        let exact = 10.0f64;
        let next_u64 = |s: &mut u64| -> u64 {
            let mut x = *s;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            *s = x;
            x.wrapping_mul(0x2545F4914F6CDD1D)
        };
        let draw_nu = |s: &mut u64| -> usize {
            let u = ((next_u64(s) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64));
            if u < probs[0] { 0 } else if u < probs[0] + probs[1] { 1 } else { 2 }
        };
        let trials = 400usize;
        for &(label, use_old) in &[
            ("OLD_least_visited_CPW(k_max=2)", true),
            ("NEW_TrueSample_LeafEval_CPW(0.5,0.20)", false),
        ] {
            println!("  [{}]", label);
            for &n_sims in &[20usize, 100, 500, 2000] {
                let mut est_sum = 0.0f64;
                let mut sq_sum = 0.0f64;
                let mut freq = [0usize; 3];
                let mut unexp_cnt = 0usize;
                for tr in 0..trials {
                    let mut rng = 0x517cc1b727220a95u64
                        ^ ((tr as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
                        ^ ((n_sims as u64).wrapping_mul(0xBF58476D1CE4E5B9));
                    let mut ch: Vec<(usize, u32)> = Vec::new();
                    let mut uq = 0.0f64;
                    for sim in 0..n_sims {
                        if use_old {
                            if ch.len() >= 2 {
                                let ci = if ch[0].1 <= ch[1].1 { 0 } else { 1 };
                                ch[ci].1 += 1;
                                freq[ch[ci].0] += 1;
                            } else {
                                let o = draw_nu(&mut rng);
                                freq[o] += 1;
                                if let Some(ci) = ch.iter().position(|&(oid, _)| oid == o) {
                                    ch[ci].1 += 1;
                                } else {
                                    ch.push((o, 1));
                                }
                            }
                        } else {
                            let o = draw_nu(&mut rng);
                            freq[o] += 1;
                            if let Some(ci) = ch.iter().position(|&(oid, _)| oid == o) {
                                ch[ci].1 += 1;
                            } else {
                                let k_max = (0.5 * ((sim + 1) as f64).powf(0.20)).ceil().max(1.0) as usize;
                                if ch.len() >= k_max && !ch.is_empty() {
                                    uq += vals[o];
                                    unexp_cnt += 1;
                                } else {
                                    ch.push((o, 1));
                                }
                            }
                        }
                    }
                    let num = uq + ch.iter().map(|&(o, n)| (n as f64) * vals[o]).sum::<f64>();
                    let qh = num / (n_sims as f64);
                    est_sum += qh;
                    sq_sum += (qh - exact).powi(2);
                }
                let m = est_sum / (trials as f64);
                let rmse = (sq_sum / (trials as f64)).sqrt();
                let tot = (trials * n_sims) as f64;
                println!(
                    "    N={:4}: freq=[{:.4},{:.4},{:.4}] E[V]={:6.2} bias={:+6.2} RMSE={:5.2} unexp%={:4.1}%",
                    n_sims,
                    freq[0] as f64 / tot,
                    freq[1] as f64 / tot,
                    freq[2] as f64 / tot,
                    m,
                    m - exact,
                    rmse,
                    100.0 * unexp_cnt as f64 / tot
                );
            }
        }

        // Part B: Audit backup_mode=4 root action change rate & root Q ranking across s=1200/2400/4800
        println!("\n=== Part B: backup_mode=4 Root Decision & Q-Ranking Stability (s=1200 vs 2400 vs 4800) ===");
        let mut states = Vec::new();
        for &sd in &[30001u64, 30002, 30003] {
            let mut game_cfg = BSConfig::default();
            game_cfg.garbage.total_lines = Some(100);
            game_cfg.prng_seed = Some(sd);
            let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
            let mut searcher = IsMctsSearcher::new_srs();
            let mut bt = BagTracker::new();
            let mut ds_tot = 0usize;
            let mut ply = 0usize;
            let mut roll_cfg = base_m.clone();
            roll_cfg.simulations = 800;
            roll_cfg.use_endgame_solver = true;
            roll_cfg.use_transposition = true;
            while ds_tot < 100 && stacker.current_piece_type().is_some() && ply < 180 {
                let snap = stacker.snapshot().unwrap();
                let g_rows = garbage_rows(&stacker);
                let rem_g = 100usize.saturating_sub(ds_tot) as u16;
                let bmask = bt.observe(snap.hold, &snap.queue[..snap.queue.len().min(6)]);
                let ctx = GameContext {
                    garbage_rows: Some(g_rows),
                    remaining_garbage: Some(rem_g),
                    next_bag_mask: Some(bmask),
                };
                if ply % 10 == 5 && rem_g > 12 {
                    states.push((snap.clone(), ctx));
                }
                let r = searcher.search_with_ctx(&snap, ctx, &roll_cfg);
                if r.inputs.is_empty() { break; }
                let mut ins = r.inputs;
                ins.pop();
                stacker.run(ins);
                let (_l, ds) = stacker.hard_drop();
                ds_tot += ds;
                ply += 1;
            }
        }
        let n_st = states.len().min(45);
        states.truncate(n_st);
        let mut searcher = IsMctsSearcher::new_srs();
        let mut chg_12_24 = 0usize;
        let mut chg_24_48 = 0usize;
        let mut chg_12_48 = 0usize;
        let mut top3_ov_12_48 = 0.0f64;
        let mut spearman_12_48 = 0.0f64;
        let mut spearman_cnt = 0usize;

        for (snap, ctx) in &states {
            let mut res_by_s = Vec::new();
            for &sims in &[1200usize, 2400, 4800] {
                let mut cfg = base_m.clone();
                cfg.simulations = sims;
                cfg.max_depth = 16;
                cfg.backup_mode = 4;
                cfg.split_tail_preview = false;
                cfg.use_endgame_solver = false;
                cfg.use_transposition = true;
                cfg.dag_propagate_parents = true;
                cfg.cpw_c = 1.0;
                cfg.cpw_alpha = 0.5;
                let r = searcher.search_with_ctx(snap, *ctx, &cfg);
                // Sort root candidates by Q ascending to inspect Q ranking
                let mut by_q = r.root_candidates.clone();
                by_q.sort_by(|a, b| a.2.total_cmp(&b.2));
                res_by_s.push((r.inputs, by_q));
            }
            if res_by_s[0].0 != res_by_s[1].0 { chg_12_24 += 1; }
            if res_by_s[1].0 != res_by_s[2].0 { chg_24_48 += 1; }
            if res_by_s[0].0 != res_by_s[2].0 { chg_12_48 += 1; }
            // Top-3 Q overlap between s=1200 and s=4800
            let t3_a: Vec<_> = res_by_s[0].1.iter().take(3).map(|x| (x.0.did_hold, x.0.tf)).collect();
            let t3_b: Vec<_> = res_by_s[2].1.iter().take(3).map(|x| (x.0.did_hold, x.0.tf)).collect();
            let ov = t3_a.iter().filter(|x| t3_b.contains(x)).count();
            top3_ov_12_48 += (ov as f64) / 3.0;
            // Spearman rank correlation on common root candidates
            let mut ranks_a = Vec::new();
            let mut ranks_b = Vec::new();
            for (ra, ca) in res_by_s[0].1.iter().enumerate() {
                let key = (ca.0.did_hold, ca.0.tf);
                if let Some(rb) = res_by_s[2].1.iter().position(|cb| (cb.0.did_hold, cb.0.tf) == key) {
                    ranks_a.push(ra as f64);
                    ranks_b.push(rb as f64);
                }
            }
            let k = ranks_a.len();
            if k >= 3 {
                let d2: f64 = ranks_a.iter().zip(&ranks_b).map(|(a, b)| (a - b).powi(2)).sum();
                let rho = 1.0 - (6.0 * d2) / ((k as f64) * ((k * k - 1) as f64));
                spearman_12_48 += rho;
                spearman_cnt += 1;
            }
        }
        println!(
            "  States audited: n={} | Root action change rate: 1200->2400={:.1}% ({}/{}), 2400->4800={:.1}% ({}/{}), 1200->4800={:.1}% ({}/{})",
            n_st,
            100.0 * chg_12_24 as f64 / n_st as f64, chg_12_24, n_st,
            100.0 * chg_24_48 as f64 / n_st as f64, chg_24_48, n_st,
            100.0 * chg_12_48 as f64 / n_st as f64, chg_12_48, n_st
        );
        println!(
            "  Root Q-ranking (1200 vs 4800): Top-3 overlap={:.1}% | mean Spearman rho={:.3}",
            100.0 * top3_ov_12_48 / n_st as f64,
            spearman_12_48 / (spearman_cnt.max(1) as f64)
        );
        return;
    }

    if mode == "leafcpw20" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, split_tail, sims, cpw_c, cpw_a, tc, ta) in &[
            ("TruncTailApprox_noCPW_s2400", false, 2400usize, 1.0f64, 0.0f64, 0.5f64, 0.0f64),
            ("TruncTailApprox_LeafCPW_s2400", false, 2400, 1.0, 0.5, 0.5, 0.0),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.split_tail_preview = split_tail;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = cpw_c;
            cfg.cpw_alpha = cpw_a;
            cfg.cpw_tail_c = tc;
            cfg.cpw_tail_alpha = ta;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "leafcpw20_fullp5" {
        for &(name, split_tail, sims, cpw_c, cpw_a, tc, ta) in &[
            ("TruncTailApprox_LeafCPW_s3000_eqW", false, 3000usize, 1.0f64, 0.5f64, 0.5f64, 0.0f64),
            ("FullP5Disclosure_LeafCPW_s2400", true, 2400, 1.0, 0.5, 0.35, 0.20),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.split_tail_preview = split_tail;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = cpw_c;
            cfg.cpw_alpha = cpw_a;
            cfg.cpw_tail_c = tc;
            cfg.cpw_tail_alpha = ta;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, None);
        }
        return;
    }

    if mode == "cpw20" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, split_tail, sims, cpw_c, cpw_a) in &[
            ("A_trunc_noCPW_s2400", false, 2400usize, 1.0f64, 0.0f64),
            ("B_trunc_CPW_1.0_0.5", false, 2400, 1.0, 0.5),
            ("C_fullP5_CPW_1.0_0.5", true, 2400, 1.0, 0.5),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.split_tail_preview = split_tail;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = cpw_c;
            cfg.cpw_alpha = cpw_a;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "cpw20_part2" {
        for &(name, split_tail, sims, cpw_c, cpw_a) in &[
            ("C_fullP5_CPW_1.0_0.5", true, 2400usize, 1.0f64, 0.5f64),
            ("B_trunc_CPW_s1900_eqW", false, 1900, 1.0, 0.5),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.split_tail_preview = split_tail;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = cpw_c;
            cfg.cpw_alpha = cpw_a;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, None);
        }
        return;
    }

    if mode == "fullp5_sep" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, tc, ta) in &[
            ("C1_fullP5_sepCPW_0.5_0.25", 0.5f64, 0.25f64),
            ("C2_fullP5_sepCPW_0.35_0.20", 0.35, 0.20),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = 2400;
            cfg.split_tail_preview = true;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = 1.0;
            cfg.cpw_alpha = 0.5;
            cfg.cpw_tail_c = tc;
            cfg.cpw_tail_alpha = ta;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "scale10" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims) in &[
            ("S1200_trunc_CPW_d16", 1200usize),
            ("S2400_trunc_CPW_d16", 2400),
            ("S4800_trunc_CPW_d16", 4800),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = 16;
            cfg.split_tail_preview = false;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = 1.0;
            cfg.cpw_alpha = 0.5;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "scale10_vw" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, sims) in &[
            ("S1200_VW_CPW_d16", 1200usize),
            ("S2400_VW_CPW_d16", 2400),
            ("S4800_VW_CPW_d16", 4800),
        ] {
            let mut cfg = base_m.clone();
            cfg.simulations = sims;
            cfg.max_depth = 16;
            cfg.backup_mode = 4;
            cfg.split_tail_preview = false;
            cfg.use_endgame_solver = true;
            cfg.use_transposition = true;
            cfg.dag_propagate_parents = true;
            cfg.cpw_c = 1.0;
            cfg.cpw_alpha = 0.5;
            let (res, wall_s) = run_parallel(&seeds, |s| play_ismcts_seed(s, &cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

    if mode == "beam20" {
        let mut ref_res: Option<Vec<GameRes>> = None;
        for &(name, bw, mpr) in &[
            ("BASE_5ply_b16", 16usize, 8usize),
            ("BASE_5ply_b32", 32, 12),
        ] {
            let mut b_cfg = BeamConfig::default();
            b_cfg.beam_width = bw;
            b_cfg.max_per_root = mpr;
            b_cfg.spatial_model = Some(sm.clone());
            b_cfg.spatial_no_pe_internal = true;
            b_cfg.root_pimc_worlds = 0;
            let (res, wall_s) = run_parallel(&seeds, |s| play_beam_seed(s, &b_cfg));
            summarize(name, &res, wall_s, ref_res.as_deref());
            if ref_res.is_none() {
                ref_res = Some(res);
            }
        }
        return;
    }

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
