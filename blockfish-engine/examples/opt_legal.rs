use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        spatial_val::{DENSE_DIM, SPARSE_DIM, SPATIAL_H2},
        BeamConfig, FastAI, SpatialValueModel,
    },
    Input, StackerExt as _,
};
use std::sync::Arc;
use std::time::Instant;

struct Rng64(u64);

impl Rng64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
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
    if any_g { max_g } else { 0 }
}

fn play_legal_fast(seed: u64, cfg: &BeamConfig) -> (usize, usize, bool) {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());

    let mut ds_total = 0usize;
    let mut moves = 0usize;
    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };
        let mut inputs = res.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (_lines, ds) = stacker.hard_drop();
        ds_total += ds;
        moves += 1;
        if moves >= 360 {
            break;
        }
    }
    let topout = ds_total < 100;
    let r_end = stacker.matrix().count();
    (moves, r_end, topout)
}

fn eval_batch(seeds: &[u64], cfg: &BeamConfig) -> (f64, usize, Vec<usize>) {
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let chunk_sz = (seeds.len() + n_threads - 1) / n_threads;
    let results: Vec<(usize, usize, bool)> = std::thread::scope(|sc| {
        let handles: Vec<_> = seeds
            .chunks(chunk_sz)
            .map(|sl| {
                sc.spawn(move || {
                    sl.iter()
                        .map(|&s| play_legal_fast(s, cfg))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(seeds.len());
        for h in handles {
            out.extend(h.join().unwrap());
        }
        out
    });
    let mut sum_fit = 0.0f64;
    let mut tops = 0usize;
    let mut pcs = Vec::with_capacity(results.len());
    for &(m, r_end, top) in &results {
        if top {
            tops += 1;
            sum_fit += 400.0;
            pcs.push(400);
        } else {
            sum_fit += (m as f64) + 0.002 * (r_end as f64);
            pcs.push(m);
        }
    }
    (sum_fit / (results.len().max(1) as f64), tops, pcs)
}

// Optimize lin_dense[1..DENSE_DIM] (73 params; keep lin_dense[0] = rem_g fixed for piece-unit calibration)
// plus lin_sparse[160..SPARSE_DIM] (26 key tactical sparse indicators) = 99 params total.
const SPARSE_START: usize = 160;
const N_DENSE_OPT: usize = DENSE_DIM - 1;
const N_SPARSE_OPT: usize = SPARSE_DIM - SPARSE_START;
const PARAM_DIM: usize = N_DENSE_OPT + N_SPARSE_OPT;

fn extract_params(m: &SpatialValueModel) -> [f32; PARAM_DIM] {
    let mut theta = [0.0f32; PARAM_DIM];
    for j in 1..DENSE_DIM {
        theta[j - 1] = m.lin_dense[j];
    }
    for i in SPARSE_START..SPARSE_DIM {
        theta[N_DENSE_OPT + (i - SPARSE_START)] = m.lin_sparse[i];
    }
    theta
}

fn apply_params(base: &SpatialValueModel, theta: &[f32; PARAM_DIM]) -> SpatialValueModel {
    let mut m = base.clone();
    m.b3 = 0.0;
    m.w3 = [0.0; SPATIAL_H2];
    for j in 1..DENSE_DIM {
        m.lin_dense[j] = theta[j - 1];
    }
    for i in SPARSE_START..SPARSE_DIM {
        m.lin_sparse[i] = theta[N_DENSE_OPT + (i - SPARSE_START)];
    }
    m.prepare_fast();
    m
}

fn make_beam_cfg(sm: Arc<SpatialValueModel>, beam_w: usize, mpr: usize, with_pimc: bool) -> BeamConfig {
    let mut cfg = BeamConfig::default();
    cfg.beam_width = beam_w;
    cfg.max_per_root = mpr;
    cfg.rollout_top_k = 0;
    cfg.bag_det_extend = true;
    cfg.endgame_wide = true;
    cfg.endgame_solver = true;
    cfg.simulate_cheese_spawn = true;
    cfg.spatial_model = Some(sm);
    cfg.spatial_rerank_only = true;
    cfg.spatial_fast_ew_internal = false;
    cfg.spatial_no_pe_internal = true;
    if with_pimc {
        cfg.root_pimc_worlds = 8;
        cfg.root_pimc_horizon = 8;
        cfg.root_pimc_roots = 8;
        cfg.root_pimc_width = 24;
        cfg.root_pimc_margin = 350;
        cfg.root_pimc_weight_pct = 100;
        cfg.root_pimc_steady_g4 = false;
    }
    cfg
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_wg5_ep0.txt".to_string();
    let mut out_path = "models/spatial_es1.txt".to_string();
    let mut iters: usize = 30;
    let mut pop: usize = 10; // antithetic pairs (2*pop evaluations per iter)
    let mut games: usize = 8;
    let mut val_games: usize = 5;
    let mut beam_w: usize = 64;
    let mut sigma0: f32 = 0.025;
    let mut lr0: f32 = 0.012;
    let mut seed_rng: u64 = 20261120;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--init" => { init_path = args[i + 1].clone(); i += 2; }
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            "--pop" => { pop = args[i + 1].parse().unwrap(); i += 2; }
            "--games" => { games = args[i + 1].parse().unwrap(); i += 2; }
            "--val-games" => { val_games = args[i + 1].parse().unwrap(); i += 2; }
            "--beam" => { beam_w = args[i + 1].parse().unwrap(); i += 2; }
            "--sigma" => { sigma0 = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr0 = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed_rng = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
    }

    let mut base = SpatialValueModel::load_from_file(&init_path).expect("load init model");
    base.b3 = 0.0;
    base.w3 = [0.0; SPATIAL_H2];
    base.prepare_fast();

    let mut theta = extract_params(&base);
    let mut best_theta = theta;
    let val_seeds: Vec<u64> = (101..(101 + val_games as u64)).collect();
    let train_pool: Vec<u64> = (2000..2064).collect();

    println!(
        "=== Direct Legal p=5 Antithetic-ES Optimizer ===\n  init={} out={} iters={} pop={} ({} evals/iter) games={} val_games={} beam={} sigma0={} lr0={} params={}",
        init_path, out_path, iters, pop, pop * 2, games, val_games, beam_w, sigma0, lr0, PARAM_DIM
    );

    let t_all = Instant::now();
    let init_sm = Arc::new(apply_params(&base, &theta));
    let init_cfg = make_beam_cfg(init_sm.clone(), beam_w, (beam_w / 4).max(16), false);
    let (mut best_val, init_tops, init_pcs) = eval_batch(&val_seeds, &init_cfg);
    println!(
        "  [iter  0] val={:.2} (top={}) pcs={:?} | init baseline",
        best_val, init_tops, init_pcs
    );
    apply_params(&base, &best_theta)
        .save_to_file(&out_path)
        .expect("save init best");

    let mut rng = Rng64(seed_rng ^ 0x9E3779B97F4A7C15);
    let mut adam_m = [0.0f32; PARAM_DIM];
    let mut adam_v = [0.0f32; PARAM_DIM];
    let beta1 = 0.85f32;
    let beta2 = 0.99f32;
    let eps_adam = 1e-6f32;

    for iter in 1..=iters {
        let t_it = Instant::now();
        let prog = (iter - 1) as f32 / (iters.max(1) as f32);
        let cos_m = 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
        let sigma = sigma0 * (0.25 + 0.75 * cos_m);
        let lr = lr0 * (0.20 + 0.80 * cos_m);

        // Pick a rotating CRN batch of `games` training seeds
        let mut batch_seeds = Vec::with_capacity(games);
        let offset = ((iter - 1) * games) % train_pool.len();
        for k in 0..games {
            batch_seeds.push(train_pool[(offset + k) % train_pool.len()]);
        }

        let mut grad = [0.0f32; PARAM_DIM];
        let mut pair_means = Vec::with_capacity(pop * 2);
        let mut pair_diffs = Vec::with_capacity(pop);

        for _p in 0..pop {
            let mut eps = [0.0f32; PARAM_DIM];
            let mut tp = theta;
            let mut tm = theta;
            for d in 0..PARAM_DIM {
                let z = rng.next_normal();
                eps[d] = z;
                tp[d] += sigma * z;
                tm[d] -= sigma * z;
            }
            let mp = Arc::new(apply_params(&base, &tp));
            let mm = Arc::new(apply_params(&base, &tm));
            let cfg_p = make_beam_cfg(mp, beam_w, (beam_w / 4).max(16), false);
            let cfg_m = make_beam_cfg(mm, beam_w, (beam_w / 4).max(16), false);
            let (fp, _, _) = eval_batch(&batch_seeds, &cfg_p);
            let (fm, _, _) = eval_batch(&batch_seeds, &cfg_m);
            pair_means.push(fp);
            pair_means.push(fm);
            let diff = ((fp - fm).clamp(-25.0, 25.0)) as f32;
            pair_diffs.push(diff);
            for d in 0..PARAM_DIM {
                grad[d] += diff * eps[d];
            }
        }

        let scale = 1.0 / (2.0 * (pop as f32) * sigma.max(1e-4));
        // Normalize gradient RMS to keep Adam step scale clean across diff magnitudes
        let mut g2_sum = 0.0f32;
        for d in 0..PARAM_DIM {
            grad[d] *= scale;
            g2_sum += grad[d] * grad[d];
        }
        let g_norm = (g2_sum / (PARAM_DIM as f32)).sqrt().max(1e-6);
        let b1t = 1.0 - beta1.powi(iter as i32);
        let b2t = 1.0 - beta2.powi(iter as i32);
        for d in 0..PARAM_DIM {
            let gn = grad[d] / g_norm;
            adam_m[d] = beta1 * adam_m[d] + (1.0 - beta1) * gn;
            adam_v[d] = beta2 * adam_v[d] + (1.0 - beta2) * gn * gn;
            let m_hat = adam_m[d] / b1t;
            let v_hat = adam_v[d] / b2t;
            theta[d] -= lr * m_hat / (v_hat.sqrt() + eps_adam);
        }

        let tr_mean = pair_means.iter().sum::<f64>() / (pair_means.len() as f64);
        let tr_min = pair_means.iter().cloned().fold(f64::INFINITY, f64::min);
        let abs_diff = pair_diffs.iter().map(|d| d.abs()).sum::<f32>() / (pair_diffs.len() as f32);

        let cur_sm = Arc::new(apply_params(&base, &theta));
        let cur_cfg = make_beam_cfg(cur_sm, beam_w, (beam_w / 4).max(16), false);
        let (val_f, val_tops, val_pcs) = eval_batch(&val_seeds, &cur_cfg);
        let improved = val_tops == 0 && val_f < best_val;
        if improved {
            best_val = val_f;
            best_theta = theta;
            apply_params(&base, &best_theta)
                .save_to_file(&out_path)
                .expect("save best model");
        }
        println!(
            "  [iter {:2}/{}] sig={:.4} lr={:.4} | tr_mean={:.2} tr_min={:.2} |d|={:.2} | val={:.2} (top={}) pcs={:?}{} | {:.1}s",
            iter,
            iters,
            sigma,
            lr,
            tr_mean,
            tr_min,
            abs_diff,
            val_f,
            val_tops,
            val_pcs,
            if improved { " *BEST*" } else { "" },
            t_it.elapsed().as_secs_f64()
        );
    }

    println!("\n=== Post-ES Held-Out Verification (seeds 101..={}) ===", 100 + val_games);
    let best_sm = Arc::new(apply_params(&base, &best_theta));
    let mpr_b120 = 24;
    let (v_init_120, t_i120, p_i120) = eval_batch(&val_seeds, &make_beam_cfg(init_sm.clone(), 120, mpr_b120, false));
    let (v_best_120, t_b120, p_b120) = eval_batch(&val_seeds, &make_beam_cfg(best_sm.clone(), 120, mpr_b120, false));
    println!("  LEGAL_b120_init            mean={:.2} top={} pcs={:?}", v_init_120, t_i120, p_i120);
    println!("  LEGAL_b120_best            mean={:.2} top={} pcs={:?}", v_best_120, t_b120, p_b120);

    let (v_init_pimc, t_ip, p_ip) = eval_batch(&val_seeds, &make_beam_cfg(init_sm, 64, 16, true));
    let (v_best_pimc, t_bp, p_bp) = eval_batch(&val_seeds, &make_beam_cfg(best_sm, 64, 16, true));
    println!("  LEGAL_pimc_nope_w8_init    mean={:.2} top={} pcs={:?}", v_init_pimc, t_ip, p_ip);
    println!("  LEGAL_pimc_nope_w8_best    mean={:.2} top={} pcs={:?}", v_best_pimc, t_bp, p_bp);
    println!("Saved best ES model to {} (total {:.1}s)", out_path, t_all.elapsed().as_secs_f64());
}
