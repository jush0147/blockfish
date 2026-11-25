use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{BeamConfig, FastAI, SpatialValueModel},
    Input, StackerExt as _,
};
use std::sync::Arc;
use std::time::Instant;

const STRATEGIC_DENSE: [usize; 16] = [
    3,  // stack_cells
    4,  // stack_cells_row_g
    5,  // stack_cells_above1
    7,  // upper_cells
    23, // covered_holes
    25, // blocks_over_holes
    28, // h1_blocks
    31, // h1_narrow_1
    32, // h1_narrow_2plus
    36, // h2_blocks
    42, // h3_blocks
    45, // cells_to_uncover_h1
    47, // bumpiness
    51, // i_dep_depth
    61, // clean_skim_rows
    70, // h2_only_row_g_blocked
];

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

fn play_legal_pimc(seed: u64, cfg: &BeamConfig) -> (usize, bool) {
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
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        moves += 1;
        if moves >= 340 {
            break;
        }
    }
    let topout = ds_total < 100;
    (if topout { 360 } else { moves }, topout)
}

fn make_pimc_cfg(mdl: Arc<SpatialValueModel>, worlds: usize, width: usize, roots: usize) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.beam_width = 48;
    c.max_per_root = 12;
    c.rollout_top_k = 0;
    c.bag_det_extend = true;
    c.endgame_wide = true;
    c.endgame_solver = true;
    c.simulate_cheese_spawn = true;
    c.spatial_model = Some(mdl);
    c.spatial_rerank_only = false;
    c.spatial_no_pe_internal = true;
    c.root_pimc_prescreen_worlds = 0;
    c.root_pimc_roots = roots;
    c.root_pimc_worlds = worlds;
    c.root_pimc_horizon = 8;
    c.root_pimc_width = width;
    c.root_pimc_margin = 350;
    c.root_pimc_weight_pct = 100;
    c.root_pimc_cvar_pct = 25;
    c.root_pimc_steady_g4 = false;
    c
}

fn eval_batch(
    mdl: &SpatialValueModel,
    seeds: &[u64],
    worlds: usize,
    width: usize,
    roots: usize,
) -> (f64, usize) {
    let mut m = mdl.clone();
    m.prepare_fast();
    let cfg = make_pimc_cfg(Arc::new(m), worlds, width, roots);
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let chunk_sz = (seeds.len() + n_threads - 1) / n_threads;
    let res: Vec<(usize, bool)> = std::thread::scope(|sc| {
        let hs: Vec<_> = seeds
            .chunks(chunk_sz)
            .map(|sl| {
                let c_ref = &cfg;
                sc.spawn(move || sl.iter().map(|&s| play_legal_pimc(s, c_ref)).collect::<Vec<_>>())
            })
            .collect();
        let mut out = Vec::new();
        for h in hs {
            out.extend(h.join().unwrap());
        }
        out
    });
    let mean = res.iter().map(|&(p, _)| p as f64).sum::<f64>() / (res.len().max(1) as f64);
    let tops = res.iter().filter(|&&(_, t)| t).count();
    (mean, tops)
}

fn extract_theta(mdl: &SpatialValueModel, ah2: usize) -> Vec<f32> {
    let mut th = Vec::with_capacity(STRATEGIC_DENSE.len() + ah2);
    for &j in &STRATEGIC_DENSE {
        th.push(mdl.lin_dense[j]);
    }
    for k in 0..ah2 {
        th.push(mdl.w3[k]);
    }
    th
}

fn apply_theta(base: &SpatialValueModel, th: &[f32], ah2: usize) -> SpatialValueModel {
    let mut m = base.clone();
    for (idx, &j) in STRATEGIC_DENSE.iter().enumerate() {
        m.lin_dense[j] = th[idx];
    }
    for k in 0..ah2 {
        m.w3[k] = th[STRATEGIC_DENSE.len() + k];
    }
    m.prepare_fast();
    m
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "models/spatial_pimc_es1.txt".to_string();
    let mut gens: usize = 12;
    let mut pairs: usize = 3;
    let mut batch_seeds: usize = 8;
    let mut worlds: usize = 4;
    let mut width: usize = 14;
    let mut roots: usize = 8;
    let mut sigma_lin: f32 = 0.045;
    let mut sigma_w3: f32 = 0.015;
    let mut lr: f32 = 0.015;
    let mut max_secs: u64 = 1900;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--init" => { init_path = args[i + 1].clone(); i += 2; }
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--gens" => { gens = args[i + 1].parse().unwrap(); i += 2; }
            "--pairs" => { pairs = args[i + 1].parse().unwrap(); i += 2; }
            "--batch-seeds" => { batch_seeds = args[i + 1].parse().unwrap(); i += 2; }
            "--worlds" => { worlds = args[i + 1].parse().unwrap(); i += 2; }
            "--width" => { width = args[i + 1].parse().unwrap(); i += 2; }
            "--roots" => { roots = args[i + 1].parse().unwrap(); i += 2; }
            "--sigma-lin" => { sigma_lin = args[i + 1].parse().unwrap(); i += 2; }
            "--sigma-w3" => { sigma_w3 = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i + 1].parse().unwrap(); i += 2; }
            "--max-secs" => { max_secs = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
    }

    let mut base_mdl = SpatialValueModel::load_from_file(&init_path).expect("load init model");
    base_mdl.prepare_fast();
    let ah2 = base_mdl.active_h2.max(1);
    let dim = STRATEGIC_DENSE.len() + ah2;
    let mut theta = extract_theta(&base_mdl, ah2);
    let mut sigmas = vec![sigma_w3; dim];
    for idx in 0..STRATEGIC_DENSE.len() {
        sigmas[idx] = sigma_lin;
    }

    let val_seeds: Vec<u64> = (6001..=6008).collect();
    let t0 = Instant::now();
    println!(
        "=== Direct Legal REAL-MLP root_pimc ES (opt_pimc) ===\n  init={} out={} dim={} (16 dense + {} w3) gens={} pairs={} batch_seeds={} w={} b={} r={} lr={} max_secs={}s",
        init_path, out_path, dim, ah2, gens, pairs, batch_seeds, worlds, width, roots, lr, max_secs
    );

    let (init_val, init_top) = eval_batch(&base_mdl, &val_seeds, worlds, width, roots);
    println!("  [INIT val8] mean={:.2} top={} dt={:.1}s", init_val, init_top, t0.elapsed().as_secs_f64());

    let mut best_val = init_val;
    let mut best_mdl = base_mdl.clone();
    best_mdl.save_to_file(&out_path).expect("save init out");

    let mut m_adam = vec![0.0f32; dim];
    let mut v_adam = vec![0.0f32; dim];
    let mut rng = Rng64(0x9E3779B97F4A7C15 ^ 0xBF58476D1CE4E5B9);

    for gen in 1..=gens {
        if t0.elapsed().as_secs() >= max_secs {
            println!("  [TIME GUARD] Reached {}s >= {}s, stopping ES cleanly.", t0.elapsed().as_secs(), max_secs);
            break;
        }
        let seed_off = 5000 + ((gen - 1) * batch_seeds) as u64;
        let tr_seeds: Vec<u64> = (seed_off..(seed_off + batch_seeds as u64)).collect();
        let mut grad = vec![0.0f32; dim];
        let mut pair_means = Vec::with_capacity(pairs);

        for _ in 0..pairs {
            let eps: Vec<f32> = (0..dim).map(|_| rng.next_normal()).collect();
            let th_pos: Vec<f32> = (0..dim).map(|d| theta[d] + sigmas[d] * eps[d]).collect();
            let th_neg: Vec<f32> = (0..dim).map(|d| theta[d] - sigmas[d] * eps[d]).collect();

            let m_pos = apply_theta(&base_mdl, &th_pos, ah2);
            let m_neg = apply_theta(&base_mdl, &th_neg, ah2);
            let (f_pos, _) = eval_batch(&m_pos, &tr_seeds, worlds, width, roots);
            let (f_neg, _) = eval_batch(&m_neg, &tr_seeds, worlds, width, roots);
            pair_means.push(0.5 * (f_pos + f_neg));

            // Minimize piece count: gradient points toward lower f
            let diff = ((f_pos - f_neg) as f32).clamp(-12.0, 12.0);
            for d in 0..dim {
                grad[d] += diff * eps[d];
            }
        }

        let inv_p = 1.0 / (2.0 * pairs as f32);
        let prog = (gen - 1) as f32 / (gens.max(1) as f32);
        let lr_cur = lr * (0.2 + 0.8 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos()));
        let b1t = 1.0 - 0.8f32.powi(gen as i32);
        let b2t = 1.0 - 0.99f32.powi(gen as i32);

        for d in 0..dim {
            let g = grad[d] * inv_p;
            m_adam[d] = 0.8 * m_adam[d] + 0.2 * g;
            v_adam[d] = 0.99 * v_adam[d] + 0.01 * g * g;
            let mh = m_adam[d] / b1t;
            let vh = v_adam[d] / b2t;
            let step = (lr_cur * sigmas[d] / 0.03) * mh / (vh.sqrt() + 1e-6);
            theta[d] -= step.clamp(-0.08, 0.08);
        }

        let cur_mdl = apply_theta(&base_mdl, &theta, ah2);
        let tr_avg = pair_means.iter().sum::<f64>() / (pair_means.len().max(1) as f64);

        if gen % 2 == 0 || gen == gens || t0.elapsed().as_secs() + 120 >= max_secs {
            let (v_mean, v_top) = eval_batch(&cur_mdl, &val_seeds, worlds, width, roots);
            let star = if v_mean < best_val {
                best_val = v_mean;
                best_mdl = cur_mdl.clone();
                best_mdl.save_to_file(&out_path).expect("save best es model");
                " *NEW BEST*"
            } else {
                ""
            };
            println!(
                "  [ES gen {:2}/{:2}] tr_mean={:.2} | val8={:.2} (best={:.2}, top={}){} | dt={:.1}s",
                gen, gens, tr_avg, v_mean, best_val, v_top, star, t0.elapsed().as_secs_f64()
            );
        } else {
            println!(
                "  [ES gen {:2}/{:2}] tr_mean={:.2} | dt={:.1}s",
                gen, gens, tr_avg, t0.elapsed().as_secs_f64()
            );
        }
    }

    println!(
        "Finished opt_pimc: init_val8={:.2} -> best_val8={:.2} (saved to {}) in {:.1}s",
        init_val, best_val, out_path, t0.elapsed().as_secs_f64()
    );
}
