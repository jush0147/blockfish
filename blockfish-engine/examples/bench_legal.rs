use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{cheese_val::FEAT_DIM, BeamConfig, FastAI, LearnedValueModel, SpatialValueModel},
    Input, StackerExt as _,
};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
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

fn load_model(path: &str) -> LearnedValueModel {
    let f = File::open(path).expect("open model file");
    let mut lines = BufReader::new(f).lines().map(|l| l.unwrap());
    let mut m = LearnedValueModel::default();
    let l1: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    m.linear_bias = l1[0];
    m.mlp_weight = l1[1];
    let l2: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let n_f = (l2.len() / 3).min(FEAT_DIM);
    for j in 0..n_f {
        m.mean[j] = l2[j * 3];
        m.inv_std[j] = l2[j * 3 + 1];
        m.linear_w[j] = l2[j * 3 + 2];
    }
    m.b2 = lines.next().unwrap().trim().parse().unwrap();
    for h in 0..blockfish::ai::cheese_val::MLP_HIDDEN {
        if let Some(line) = lines.next() {
            let lh: Vec<f32> = line
                .split_whitespace()
                .map(|s| s.parse().unwrap())
                .collect();
            m.b1[h] = lh[0];
            m.w2[h] = lh[1];
            let n_w = lh.len().saturating_sub(2).min(FEAT_DIM);
            for j in 0..n_w {
                m.w1[h][j] = lh[2 + j];
            }
        }
    }
    m.prepare_fast();
    m
}

#[derive(Clone, Debug, Default)]
struct GameStats {
    seed: u64,
    pieces: usize,
    topout: bool,
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

fn play_legal_seed(seed: u64, cfg: &BeamConfig) -> GameStats {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());

    let mut st = GameStats {
        seed,
        ..Default::default()
    };
    let mut ds_total = 0usize;
    let mut cells_sum = 0usize;

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
        cells_sum += cells_above;
        let cur_c = snap.queue[0].as_char();
        let hold_c = snap
            .hold
            .or_else(|| snap.queue.get(1).copied())
            .map(|c| c.as_char());

        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };
        let mut inputs = res.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        st.pieces += 1;
        ds_total += ds;
        st.nds += lines.saturating_sub(ds);
        if ds > 0 {
            st.ds1 += ds;
            let (_, new_bo, _, _) = h1_info(&stacker);
            if new_bo > 0 {
                st.ds_cov_next += 1;
            }
        } else {
            st.zds += 1;
            if rem_g <= 7 && rem_g == g_rows {
                st.zds_end += 1;
            } else if g_rows == 0 {
                st.zds_g0 += 1;
            } else if blocks_over > 0 {
                st.zds_h1_cov += 1;
            } else {
                let has_o = cur_c == 'O' || hold_c == Some('O');
                let has_i = cur_c == 'I' || hold_c == Some('I');
                if has_o {
                    st.zds_open_o += 1;
                } else if chute_min >= 1 && !has_i {
                    st.zds_open_nar += 1;
                } else {
                    st.zds_open_wide += 1;
                }
            }
        }
    }

    st.topout = ds_total < 100;
    if st.topout {
        st.pieces = 400;
    }
    st.avg_stack_cells = cells_sum as f64 / st.pieces.max(1) as f64;
    st
}

fn eval_seeds(label: &str, seeds: &[u64], cfg: &BeamConfig) -> Vec<GameStats> {
    let t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut results: Vec<Option<GameStats>> = vec![None; seeds.len()];

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
                    local.push((i, play_legal_seed(seeds[i], cfg)));
                }
                local
            }));
        }
        for h in handles {
            for (i, st) in h.join().unwrap() {
                results[i] = Some(st);
            }
        }
    });

    let stats: Vec<GameStats> = results.into_iter().map(|x| x.unwrap()).collect();
    let dt = t0.elapsed().as_secs_f64();
    let n = stats.len() as f64;
    let tops = stats.iter().filter(|s| s.topout).count();
    let ok: Vec<&GameStats> = stats.iter().filter(|s| !s.topout).collect();
    let mean = stats.iter().map(|s| s.pieces).sum::<usize>() as f64 / n;
    let mn = ok.iter().map(|s| s.pieces).min().unwrap_or(999);
    let mx = ok.iter().map(|s| s.pieces).max().unwrap_or(999);
    let nds = stats.iter().map(|s| s.nds).sum::<usize>() as f64 / n;
    let zds = stats.iter().map(|s| s.zds).sum::<usize>() as f64 / n;
    let cov = stats.iter().map(|s| s.zds_h1_cov).sum::<usize>() as f64 / n;
    let z_o = stats.iter().map(|s| s.zds_open_o).sum::<usize>() as f64 / n;
    let z_nar = stats.iter().map(|s| s.zds_open_nar).sum::<usize>() as f64 / n;
    let z_wide = stats.iter().map(|s| s.zds_open_wide).sum::<usize>() as f64 / n;
    let z_g0 = stats.iter().map(|s| s.zds_g0).sum::<usize>() as f64 / n;
    let z_end = stats.iter().map(|s| s.zds_end).sum::<usize>() as f64 / n;
    let ds_cnt: usize = stats.iter().map(|s| s.ds1).sum();
    let ds_cov: usize = stats.iter().map(|s| s.ds_cov_next).sum();
    let avg_c = stats.iter().map(|s| s.avg_stack_cells).sum::<f64>() / n;
    let le166 = ok.iter().filter(|s| s.pieces <= 166).count();

    println!(
        "{:<26} n={:2} mean={:6.2} min={:3} max={:3} <=166:{:2} top={:2} | nds={:5.2} 0ds={:5.2} (cov={:.1} O={:.1} nar={:.1} wide={:.1} g0={:.1} end={:.1}) cov_nxt={:.1}% avg_c={:.1} | {:.1}s",
        label,
        stats.len(),
        mean,
        mn,
        mx,
        le166,
        tops,
        nds,
        zds,
        cov,
        z_o,
        z_nar,
        z_wide,
        z_g0,
        z_end,
        100.0 * ds_cov as f64 / ds_cnt.max(1) as f64,
        avg_c,
        dt
    );
    if stats.len() <= 10 {
        let per_s: Vec<String> = stats
            .iter()
            .map(|s| format!("{}:{}", s.seed, if s.topout { 999 } else { s.pieces }))
            .collect();
        println!("  seeds: [{}]", per_s.join(", "));
    }
    let _ = std::io::stdout().flush();
    stats
}

fn make_cfg(m: LearnedValueModel) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.beam_width = 120;
    c.max_per_root = 24;
    c.rollout_top_k = 12;
    c.endgame_wide = true;
    c.no_g_credit = 0;
    c.value_fast_micro = 0;
    c.value_model = Some(Arc::new(m));
    c
}

fn make_spatial_cfg(m: Arc<SpatialValueModel>) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.beam_width = 120;
    c.max_per_root = 24;
    c.rollout_top_k = 12;
    c.endgame_wide = true;
    c.no_g_credit = 0;
    c.value_fast_micro = 0;
    c.spatial_model = Some(m);
    c
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "baseline".to_string());
    let m_h2 = load_model(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/crn_r3_h2stk03.txt"
    ));
    let mut m_h2_nog0 = m_h2.clone();
    m_h2_nog0.linear_w[2] = 0.0;
    m_h2_nog0.prepare_fast();

    let m_r3 = load_model(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/crn_r3_gbox.txt"
    ));
    let m_r4 = load_model(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/crn_r4_gbox.txt"
    ));

    let seeds5: Vec<u64> = (101..=105).collect();
    let train20: Vec<u64> = (101..=120).collect();
    let dev20: Vec<u64> = (1..=20).collect();

    if mode == "baseline" {
        println!("=== Legal 5-seed ablation (101..=105) ===");
        eval_seeds("r3_gbox_p5_b120", &seeds5, &make_cfg(m_r3.clone()));
        eval_seeds("r4_gbox_p5_b120", &seeds5, &make_cfg(m_r4.clone()));
        eval_seeds("h2stk03_p5_b120", &seeds5, &make_cfg(m_h2.clone()));
        eval_seeds("h2stk03_nog0_b120", &seeds5, &make_cfg(m_h2_nog0.clone()));

        let mut c_b480 = make_cfg(m_h2_nog0.clone());
        c_b480.beam_width = 480;
        c_b480.max_per_root = 96;
        c_b480.rollout_top_k = 24;
        eval_seeds("h2stk03_nog0_b480", &seeds5, &c_b480);

        let mut c_sim = make_cfg(m_h2_nog0.clone());
        c_sim.simulate_cheese_spawn = true;
        eval_seeds("h2stk03_nog0_sim1w", &seeds5, &c_sim);

        let mut c_rpimc_real = make_cfg(m_h2_nog0.clone());
        c_rpimc_real.root_pimc_worlds = 12;
        c_rpimc_real.root_pimc_horizon = 99;
        c_rpimc_real.root_pimc_roots = 10;
        c_rpimc_real.root_pimc_width = 32;
        c_rpimc_real.root_pimc_margin = 1400;
        c_rpimc_real.root_pimc_weight_pct = 100;
        c_rpimc_real.root_pimc_steady_g4 = false;
        eval_seeds("h2_rpimc_realphys_w12", &seeds5, &c_rpimc_real);
        return;
    }

    if mode == "rpimc_h12" {
        println!("=== Legal 5-seed Long-Horizon CRN Root-PIMC (101..=105) ===");
        let mut c_h9 = make_cfg(m_h2_nog0.clone());
        c_h9.root_pimc_worlds = 8;
        c_h9.root_pimc_horizon = 8; // 9 plies total
        c_h9.root_pimc_roots = 8;
        c_h9.root_pimc_width = 24;
        c_h9.root_pimc_margin = 350;
        c_h9.root_pimc_weight_pct = 100;
        c_h9.root_pimc_steady_g4 = false;
        eval_seeds("h2_rpimc_w8_h9_pw100", &seeds5, &c_h9);

        let mut c_h12 = c_h9.clone();
        c_h12.root_pimc_horizon = 11; // 12 plies total
        eval_seeds("h2_rpimc_w8_h12_pw100", &seeds5, &c_h12);

        let mut c_h12_mix = c_h12.clone();
        c_h12_mix.root_pimc_weight_pct = 75;
        eval_seeds("h2_rpimc_w8_h12_pw75", &seeds5, &c_h12_mix);
        return;
    }

    if mode == "ablate" {
        println!("=== Legal 5-seed Search & Calibration Ablation (101..=105) ===");
        let mut c0 = make_cfg(m_h2_nog0.clone());
        eval_seeds("0_base_h2nog0_b120", &seeds5, &c0);

        c0.bag_det_extend = true;
        eval_seeds("1_+bag_det_extend", &seeds5, &c0);

        c0.endgame_solver = true;
        eval_seeds("2_+endgame_solver", &seeds5, &c0);

        c0.fog_spawn = true;
        eval_seeds("3_+fog_spawn", &seeds5, &c0);

        let mut c4 = c0.clone();
        c4.rollout_top_k = 48;
        c4.rollout_per_root = 3;
        eval_seeds("4_+roll_k48_pr3", &seeds5, &c4);

        let mut c5 = c4.clone();
        c5.fog_spawn = false;
        eval_seeds("5_no_fog_k48_pr3", &seeds5, &c5);

        // Calibrated heuristic patch over crn_r3_h2stk03
        let mut m_cal = m_h2_nog0.clone();
        let set_raw = |m: &mut LearnedValueModel, idx: usize, raw_w: f32| {
            if m.inv_std[idx] > 1e-6 {
                m.linear_w[idx] = raw_w / m.inv_std[idx];
            }
        };
        set_raw(&mut m_cal, 0, 1.65);   // rem_g
        set_raw(&mut m_cal, 9, 0.08);   // stack_cells
        set_raw(&mut m_cal, 11, 0.06);  // stack_cells_above1
        set_raw(&mut m_cal, 36, 0.22);  // h2_blocks
        set_raw(&mut m_cal, 39, -0.15); // h2_open
        set_raw(&mut m_cal, 57, -0.18); // near_full_9
        set_raw(&mut m_cal, 63, 0.0);   // hand_O
        set_raw(&mut m_cal, 88, 0.25);  // h2_only_row_g_blocked
        set_raw(&mut m_cal, 92, -0.12); // clean_skim_rows
        m_cal.prepare_fast();
        let mut c6 = c0.clone();
        c6.value_model = Some(Arc::new(m_cal));
        eval_seeds("6_cal_probe+fog", &seeds5, &c6);
        return;
    }

    if mode == "spatial5" || mode == "spatial20" || mode == "spatial40" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_v1.txt".to_string());
        let sm = Arc::new(SpatialValueModel::load_from_file(&model_path).expect("load spatial model"));
        let target_seeds: Vec<u64> = if mode == "spatial5" {
            seeds5.clone()
        } else if mode == "spatial20" {
            (201..=220).collect()
        } else {
            let mut s = Vec::new();
            s.extend_from_slice(&dev20);
            s.extend_from_slice(&train20);
            s
        };
        println!("=== Legal {} ({}) ===", mode, model_path);
        let c_nosp = make_spatial_cfg(sm.clone());
        if mode == "spatial5" {
            eval_seeds("sp_nosp_b120", &target_seeds, &c_nosp);
        }

        let mut c_crn = make_spatial_cfg(sm.clone());
        c_crn.simulate_cheese_spawn = true;
        if mode == "spatial5" {
            eval_seeds("sp_crn1w_b120", &target_seeds, &c_crn);
        }

        let mut c_crn_be = c_crn.clone();
        c_crn_be.bag_det_extend = true;
        c_crn_be.endgame_solver = true;
        eval_seeds("sp_crn1w_bdet_end", &target_seeds, &c_crn_be);

        let mut c_rpimc = make_spatial_cfg(sm.clone());
        c_rpimc.bag_det_extend = true;
        c_rpimc.endgame_solver = true;
        c_rpimc.root_pimc_worlds = 12;
        c_rpimc.root_pimc_horizon = 99;
        c_rpimc.root_pimc_roots = 10;
        c_rpimc.root_pimc_width = 32;
        c_rpimc.root_pimc_margin = 1400;
        c_rpimc.root_pimc_weight_pct = 100;
        c_rpimc.root_pimc_steady_g4 = false;
        eval_seeds("sp_rpimc_w12_b120", &target_seeds, &c_rpimc);

        if mode == "spatial5" {
            let mut c_rpimc_b240 = c_rpimc.clone();
            c_rpimc_b240.beam_width = 240;
            c_rpimc_b240.max_per_root = 48;
            c_rpimc_b240.rollout_top_k = 24;
            eval_seeds("sp_rpimc_w12_b240", &target_seeds, &c_rpimc_b240);
        }
        return;
    }

    if mode == "dev40" {
        let mut seeds40 = Vec::new();
        seeds40.extend_from_slice(&dev20);
        seeds40.extend_from_slice(&train20);
        println!("=== Legal 40-seed baseline (1..=20 + 101..=120) ===");
        eval_seeds("h2stk03_nog0_b120", &seeds40, &make_cfg(m_h2_nog0));
        return;
    }

    if mode == "trace" {
        let seed: u64 = std::env::args()
            .nth(2)
            .and_then(|s| s.parse().ok())
            .unwrap_or(101);
        let cfg = make_cfg(m_h2_nog0);
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai = FastAI::new(cfg);
        let mut ds_total = 0usize;
        let mut step = 0usize;
        let mut printed = 0usize;
        println!("=== Legal Trace seed={} ===", seed);
        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let (h1_c, bo, chute, cells_ab) = h1_info(&stacker);
            let cur_c = snap.queue[0].as_char();
            let hold_c = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|c| c.as_char());
            let res = ai.best_move(&snap, Some(g_rows), Some(rem_g)).unwrap();
            let top_cands = ai.last_top_candidates().to_vec();
            let top_leaves = ai.last_top_leaf_info().to_vec();
            let mut inputs = res.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            step += 1;
            ds_total += ds;
            let (_, new_bo, _, _) = h1_info(&stacker);
            let is_cov_nxt = ds > 0 && new_bo > 0;
            let is_wide = ds == 0 && rem_g > 7 && g_rows > 0 && bo == 0 && cur_c != 'O' && hold_c != Some('O') && chute == 0;
            if (is_cov_nxt || is_wide) && printed < 24 {
                printed += 1;
                let tag = if is_cov_nxt { "COV_NXT" } else { "WIDE_0DS" };
                let q_str: String = snap.queue.iter().map(|c| c.as_char()).collect();
                println!(
                    "step={:3} {:8} rem_g={:3} g={} h1={:?} bo={} cells={} hold={:?} q={} -> ds={} lines={} new_bo={}",
                    step, tag, rem_g, g_rows, h1_c, bo, cells_ab, snap.hold.map(|c| c.as_char()), q_str, ds, lines, new_bo
                );
                for (ci, c) in top_cands.iter().take(4).enumerate() {
                    let li = top_leaves.get(ci);
                    println!("   cand#{}: ({},{},{},r{},hold={}) rating={} leaf={:?}", ci, c.0, c.1, c.2, c.3, c.4, c.5, li);
                }
            }
        }
        println!("Finished seed={} in {} pieces", seed, step);
        return;
    }
}
