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
        let did_hold = inputs.first() == Some(&Input::Hold);
        let placed_c = if did_hold {
            hold_c.unwrap_or(cur_c)
        } else {
            cur_c
        };
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
                let is_o = placed_c == 'O';
                let has_i = cur_c == 'I' || hold_c == Some('I');
                if is_o {
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
                    let st = play_legal_seed(seeds[i], cfg);
                    println!(
                        "  [seed {:3}] pcs={:3} nds={:2} 0ds={:2} (cov={:2} O={:2} nar={:2} wide={:2} end={:2}) cov_nxt={:4.1}% | {:.1}s",
                        st.seed,
                        if st.topout { 999 } else { st.pieces },
                        st.nds,
                        st.zds,
                        st.zds_h1_cov,
                        st.zds_open_o,
                        st.zds_open_nar,
                        st.zds_open_wide,
                        st.zds_end,
                        100.0 * st.ds_cov_next as f64 / st.ds1.max(1) as f64,
                        t0.elapsed().as_secs_f64()
                    );
                    let _ = std::io::stdout().flush();
                    local.push((i, st));
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

fn make_ew_cons_cfg() -> BeamConfig {
    let mut c = BeamConfig::default();
    c.beam_width = 120;
    c.max_per_root = 24;
    c.rollout_top_k = 0;
    c.endgame_wide = true;
    c.endgame_solver = true;
    c.bag_det_extend = true;
    c.value_model = None;
    c.spatial_model = None;
    let w = &mut c.weights;
    w.garbage_row = 165;
    w.piece_estimate = 0;
    w.stack_cell = 6;
    w.upper_cell = 5;
    w.stack_h_sq = 0;
    w.stack_over2_sq = 4;
    w.max_stack_h_excess = 14;
    w.covered_hole = 55;
    w.hole_row = 20;
    w.blocks_over_hole = 20;
    w.h1_block = 32;
    w.h1_empty_blocking = 8;
    w.h1_height = 10;
    w.h2_block = 26;
    w.h2_extra_empty = 6;
    w.h3_block = 10;
    w.h3_extra_empty = 3;
    w.h4_block = 4;
    w.h1_narrow_1 = 16;
    w.h1_narrow_2plus = 26;
    w.h1_fit_scarcity = 8;
    w.h1_unfit_hand = 28;
    w.zero_o_spot_pen = 14;
    w.bag_no_i_chute_pen = 18;
    w.g0_row0_cell = 0;
    w.open_h1_bonus = 24;
    w.combo2_bonus = 22;
    w.h1_h2_adj = 4;
    w.bumpiness = 3;
    w.sharp_diff = 5;
    w.i_dep = 35;
    w.two_well = 10;
    w.i_dep_multi = 50;
    w.i_dep_chute = 12;
    w.row_transition = 3;
    w.near_full_9_bonus = 22;
    w.near_full_8_bonus = 6;
    w.row_g_aligned_bonus = 20;
    w.hold_bonus = [-10, 4, -8, 4, 4, -2, -2];
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

    if mode == "spatial5_q" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_r2.txt".to_string());
        let sm = Arc::new(SpatialValueModel::load_from_file(&model_path).expect("load spatial model"));
        println!("=== Legal spatial5_q ({}) ===", model_path);

        let mut c_nosp = make_spatial_cfg(sm.clone());
        c_nosp.bag_det_extend = true;
        c_nosp.endgame_solver = true;
        eval_seeds("sp_nosp_bdet_end", &seeds5, &c_nosp);

        let mut c_sim = c_nosp.clone();
        c_sim.simulate_cheese_spawn = true;
        eval_seeds("sp_sim1w_bdet_end", &seeds5, &c_sim);

        let mut c_rpimc = c_nosp.clone();
        c_rpimc.root_pimc_worlds = 8;
        c_rpimc.root_pimc_horizon = 8;
        c_rpimc.root_pimc_roots = 8;
        c_rpimc.root_pimc_width = 24;
        c_rpimc.root_pimc_margin = 350;
        c_rpimc.root_pimc_weight_pct = 100;
        c_rpimc.root_pimc_steady_g4 = false;
        eval_seeds("sp_rpimc_w8_h9", &seeds5, &c_rpimc);

        let mut c_rpimc_sim = c_rpimc.clone();
        c_rpimc_sim.simulate_cheese_spawn = true;
        eval_seeds("sp_rpimc_sim_w8_h9", &seeds5, &c_rpimc_sim);

        let mut c_rpimc_linfast = c_rpimc_sim.clone();
        c_rpimc_linfast.spatial_rerank_only = true;
        eval_seeds("sp_rpimc_linfast_w8_h9", &seeds5, &c_rpimc_linfast);
        return;
    }

    if mode == "sp1_probe" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_r1.txt".to_string());
        let which = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        let sm = Arc::new(SpatialValueModel::load_from_file(&model_path).expect("load spatial model"));
        println!("=== Legal sp1_probe ({}, which={}) ===", model_path, which);

        let mut c_rr = make_cfg(m_h2_nog0.clone());
        c_rr.spatial_model = Some(sm.clone());
        c_rr.spatial_rerank_only = true;
        c_rr.bag_det_extend = true;
        c_rr.endgame_solver = true;
        if which == "all" || which == "1" {
            eval_seeds("rr_h2beam_spleaf", &seeds5, &c_rr);
        }

        let mut c_rr_sim = c_rr.clone();
        c_rr_sim.simulate_cheese_spawn = true;
        if which == "all" || which == "2" {
            eval_seeds("rr_sim1w_spleaf", &seeds5, &c_rr_sim);
        }

        let mut c_rr_rp = c_rr.clone();
        c_rr_rp.root_pimc_worlds = 8;
        c_rr_rp.root_pimc_horizon = 8;
        c_rr_rp.root_pimc_roots = 8;
        c_rr_rp.root_pimc_width = 24;
        c_rr_rp.root_pimc_margin = 350;
        c_rr_rp.root_pimc_weight_pct = 100;
        if which == "all" || which == "3" {
            eval_seeds("rr_rpimc_w8_h9", &seeds5, &c_rr_rp);
        }
        return;
    }

    if mode == "spatial5_1" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_r2.txt".to_string());
        let cfg_name = std::env::args()
            .nth(3)
            .unwrap_or_else(|| "sp_rpimc_sim_w8_h9".to_string());
        let mut sm_val = SpatialValueModel::load_from_file(&model_path).expect("load spatial model");
        if cfg_name.contains("_lin") && !cfg_name.contains("_linfast") {
            sm_val.b3 = 0.0;
            sm_val.w3 = [0.0; blockfish::ai::spatial_val::SPATIAL_H2];
            sm_val.prepare_fast();
        }
        let sm = Arc::new(sm_val);
        println!("=== Legal spatial5_1 ({} / {}) ===", model_path, cfg_name);

        let mut c = make_spatial_cfg(sm);
        c.bag_det_extend = true;
        c.endgame_solver = true;
        c.root_pimc_worlds = 8;
        c.root_pimc_horizon = 8;
        c.root_pimc_roots = 8;
        c.root_pimc_width = 24;
        c.root_pimc_margin = 350;
        c.root_pimc_weight_pct = 100;
        c.root_pimc_steady_g4 = false;
        if cfg_name.contains("_sim") || cfg_name.contains("_linfast") {
            c.simulate_cheese_spawn = true;
        }
        if cfg_name.contains("_linfast") {
            c.spatial_rerank_only = true;
        }
        if cfg_name.contains("_w9") {
            c.root_pimc_worlds = 9;
        }
        if cfg_name.contains("_w12") {
            c.root_pimc_worlds = 12;
        }
        if cfg_name.contains("_h12") {
            c.root_pimc_horizon = 11;
        }
        if cfg_name.contains("_m500") {
            c.root_pimc_margin = 500;
        }
        eval_seeds(&cfg_name, &seeds5, &c);
        return;
    }

    if mode == "end5" {
        println!("=== Legal 5-seed Endgame Solver Check (101..=105) ===");
        let mut c1 = make_cfg(m_h2_nog0.clone());
        c1.bag_det_extend = true;
        eval_seeds("1_bagdet_noend", &seeds5, &c1);

        let mut c2 = c1.clone();
        c2.endgame_solver = true;
        eval_seeds("2_bagdet_new_end", &seeds5, &c2);
        return;
    }

    if mode == "legal_search_probe" {
        println!("=== Legal 5-seed Search Probe (101..=105) ===");
        let mut c_base = make_cfg(m_h2_nog0.clone());
        c_base.bag_det_extend = true;
        c_base.endgame_solver = true;

        let mut c_ann = c_base.clone();
        c_ann.anneal_mid_pen = 35;
        eval_seeds("anneal_35", &seeds5, &c_ann);

        let mut c_k3 = c_base.clone();
        c_k3.rollout_top_k = 0;
        c_k3.root_top_k_mean = 3;
        eval_seeds("top3mean_noroll", &seeds5, &c_k3);

        let mut c_ann_k3 = c_k3.clone();
        c_ann_k3.anneal_mid_pen = 35;
        eval_seeds("ann35_top3mean", &seeds5, &c_ann_k3);

        let mut c_rp = c_base.clone();
        c_rp.anneal_mid_pen = 35;
        c_rp.root_pimc_worlds = 8;
        c_rp.root_pimc_horizon = 11;
        c_rp.root_pimc_roots = 8;
        c_rp.root_pimc_width = 24;
        c_rp.root_pimc_margin = 350;
        c_rp.root_pimc_weight_pct = 100;
        eval_seeds("ann35_rpimc_w8_h12", &seeds5, &c_rp);
        return;
    }

    if mode == "sh_rpimc" {
        let model_arg = std::env::args().nth(2).unwrap_or_else(|| "h2nog0".to_string());
        let which = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        let m_use = if model_arg == "h2nog0" {
            m_h2_nog0.clone()
        } else {
            load_model(&model_arg)
        };
        println!("=== Legal 5-seed Successive-Halving CRN-PIMC (model={}, which={}) ===", model_arg, which);

        let mut c_base = make_cfg(m_use.clone());
        c_base.bag_det_extend = true;
        c_base.endgame_solver = true;
        c_base.simulate_cheese_spawn = false;
        if which == "all" || which == "base" {
            eval_seeds("base_end7_nosp", &seeds5, &c_base);
        }

        let mut c_fast = c_base.clone();
        c_fast.beam_width = 60;
        c_fast.max_per_root = 12;
        c_fast.root_pimc_prescreen_worlds = 3;
        c_fast.root_pimc_roots = 28;
        c_fast.root_pimc_fine_roots = 6;
        c_fast.root_pimc_worlds = 10;
        c_fast.root_pimc_horizon = 9;
        c_fast.root_pimc_width = 20;
        c_fast.root_pimc_weight_pct = 100;
        if which == "all" || which == "fast" {
            eval_seeds("sh_w10_h9_pw20_k6", &seeds5, &c_fast);
        }

        let mut c_med = c_base.clone();
        c_med.beam_width = 60;
        c_med.max_per_root = 12;
        c_med.root_pimc_prescreen_worlds = 4;
        c_med.root_pimc_roots = 36;
        c_med.root_pimc_fine_roots = 8;
        c_med.root_pimc_worlds = 12;
        c_med.root_pimc_horizon = 9;
        c_med.root_pimc_width = 24;
        c_med.root_pimc_weight_pct = 100;
        if which == "all" || which == "med" {
            eval_seeds("sh_w12_h9_pw24_k8", &seeds5, &c_med);
        }
        return;
    }

    if mode == "ds_probe" {
        let which = std::env::args().nth(2).unwrap_or_else(|| "all".to_string());
        let sm = Arc::new(
            SpatialValueModel::load_from_file(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../models/spatial_wg1.txt"
            ))
            .expect("load spatial_wg1"),
        );
        println!("=== Legal ds_probe (which={}) ===", which);

        let mut c_ew = make_ew_cons_cfg();
        c_ew.beam_width = 64;
        c_ew.max_per_root = 12;
        c_ew.rollout_top_k = 0;
        c_ew.weights.garbage_row = 220;
        c_ew.weights.stack_cell = -22;
        c_ew.weights.upper_cell = 3;
        c_ew.weights.stack_over2_sq = 3;
        c_ew.weights.blocks_over_hole = 14;
        c_ew.weights.zero_o_spot_pen = 0;
        c_ew.weights.bumpiness = 11;
        c_ew.weights.near_full_9_bonus = 0;
        c_ew.weights.near_full_8_bonus = 0;
        c_ew.weights.row_g_aligned_bonus = 0;
        c_ew.weights.h1_contig_cover = 140;
        c_ew.weights.h2_contig_cover = 95;
        c_ew.weights.h3_block = 22;
        c_ew.weights.h4_block = 8;

        c_ew.root_pimc_prescreen_worlds = 3;
        c_ew.root_pimc_roots = 32;
        c_ew.root_pimc_fine_roots = 8;
        c_ew.root_pimc_worlds = 10;
        c_ew.root_pimc_horizon = 10;
        c_ew.root_pimc_width = 16;
        c_ew.root_pimc_weight_pct = 100;
        c_ew.root_pimc_ds_target = 5;

        if which == "fast_cmp" || which == "fast_ew4" {
            // 1. Ultra-fast ew_sc4 internal + spatial_wg1 leaf, flat w8 h9
            let mut c_ew4_sp = make_ew_cons_cfg();
            c_ew4_sp.beam_width = 64;
            c_ew4_sp.max_per_root = 16;
            c_ew4_sp.spatial_model = Some(sm.clone());
            c_ew4_sp.spatial_rerank_only = true;
            c_ew4_sp.spatial_fast_ew_internal = true;
            c_ew4_sp.weights.garbage_row = 225;
            c_ew4_sp.weights.stack_cell = 4;
            c_ew4_sp.weights.upper_cell = 5;
            c_ew4_sp.weights.blocks_over_hole = 14;
            c_ew4_sp.weights.zero_o_spot_pen = 0;
            c_ew4_sp.weights.bumpiness = 11;
            c_ew4_sp.weights.near_full_9_bonus = 0;
            c_ew4_sp.weights.near_full_8_bonus = 0;
            c_ew4_sp.weights.row_g_aligned_bonus = 0;
            c_ew4_sp.weights.h1_contig_cover = 120;
            c_ew4_sp.weights.h2_contig_cover = 60;
            c_ew4_sp.root_pimc_worlds = 8;
            c_ew4_sp.root_pimc_horizon = 9;
            c_ew4_sp.root_pimc_roots = 8;
            c_ew4_sp.root_pimc_width = 24;
            c_ew4_sp.root_pimc_margin = 400;
            c_ew4_sp.root_pimc_weight_pct = 100;
            eval_seeds("ew4_sp_flat_w8_h9", &seeds5, &c_ew4_sp);

            // 2. Same ew4 + ds_target = 3 (fast pure-bitwise internal)
            let mut c_ew4_ds3 = c_ew4_sp.clone();
            c_ew4_ds3.root_pimc_ds_target = 3;
            c_ew4_ds3.root_pimc_horizon = 10;
            c_ew4_ds3.root_pimc_width = 16;
            eval_seeds("ew4_sp_ds3_w8_b16", &seeds5, &c_ew4_ds3);

            // 3. ew4 + ds_target = 3 with ds-conservation stack_cell=-22 ONLY? Wait: Stage 1 uses weights too, so test ds_target=4
            if which == "fast_cmp" {
                let mut c_h2_sp = make_cfg(m_h2_nog0.clone());
                c_h2_sp.beam_width = 64;
                c_h2_sp.max_per_root = 16;
                c_h2_sp.spatial_model = Some(sm.clone());
                c_h2_sp.spatial_rerank_only = true;
                c_h2_sp.bag_det_extend = true;
                c_h2_sp.endgame_solver = true;
                c_h2_sp.rollout_top_k = 0;
                c_h2_sp.root_pimc_worlds = 8;
                c_h2_sp.root_pimc_horizon = 9;
                c_h2_sp.root_pimc_roots = 6;
                c_h2_sp.root_pimc_width = 16;
                c_h2_sp.root_pimc_margin = 350;
                c_h2_sp.root_pimc_weight_pct = 100;
                eval_seeds("h2_sp_flat_w8_h9_b16", &seeds5, &c_h2_sp);

                let mut c_h2_ds3 = c_h2_sp.clone();
                c_h2_ds3.root_pimc_ds_target = 3;
                c_h2_ds3.root_pimc_horizon = 10;
                eval_seeds("h2_sp_ds3_w8_b16", &seeds5, &c_h2_ds3);
            }
            return;
        }
        if which == "ds_fix" {
            let mut s1w = make_ew_cons_cfg().weights;
            s1w.garbage_row = 225;
            s1w.stack_cell = 4;
            s1w.upper_cell = 5;
            s1w.blocks_over_hole = 14;
            s1w.zero_o_spot_pen = 0;
            s1w.bumpiness = 11;
            s1w.near_full_9_bonus = 0;
            s1w.near_full_8_bonus = 0;
            s1w.row_g_aligned_bonus = 0;
            s1w.h1_contig_cover = 130;
            s1w.h2_contig_cover = 80;

            let mut c_ds3 = make_ew_cons_cfg();
            c_ds3.beam_width = 64;
            c_ds3.max_per_root = 16;
            c_ds3.rollout_top_k = 0;
            c_ds3.stage1_weights = Some(s1w.clone());
            c_ds3.weights.garbage_row = 225;
            c_ds3.weights.stack_cell = -22;
            c_ds3.weights.upper_cell = 3;
            c_ds3.weights.stack_over2_sq = 3;
            c_ds3.weights.blocks_over_hole = 14;
            c_ds3.weights.zero_o_spot_pen = 0;
            c_ds3.weights.bumpiness = 11;
            c_ds3.weights.near_full_9_bonus = 0;
            c_ds3.weights.near_full_8_bonus = 0;
            c_ds3.weights.row_g_aligned_bonus = 0;
            c_ds3.weights.h1_contig_cover = 140;
            c_ds3.weights.h2_contig_cover = 95;
            c_ds3.weights.h3_block = 22;
            c_ds3.weights.h4_block = 8;
            c_ds3.weights.endgame_cell_taper = 47;
            c_ds3.nds_pen = 0;
            c_ds3.root_pimc_prescreen_worlds = 3;
            c_ds3.root_pimc_roots = 24;
            c_ds3.root_pimc_fine_roots = 8;
            c_ds3.root_pimc_worlds = 8;
            c_ds3.root_pimc_horizon = 11;
            c_ds3.root_pimc_width = 12;
            c_ds3.root_pimc_weight_pct = 100;
            c_ds3.root_pimc_ds_target = 3;
            eval_seeds("ew22_ds3_s1p4_w8", &seeds5, &c_ds3);

            let mut c_ds3_sp = c_ds3.clone();
            c_ds3_sp.spatial_model = Some(sm.clone());
            c_ds3_sp.spatial_rerank_only = true;
            c_ds3_sp.spatial_fast_ew_internal = true;
            eval_seeds("ew22_ds3_sp_wg1_s1p4", &seeds5, &c_ds3_sp);
            return;
        }
        if which == "nds_cal" {
            let mut c_ew4_sp = make_ew_cons_cfg();
            c_ew4_sp.beam_width = 64;
            c_ew4_sp.max_per_root = 16;
            c_ew4_sp.spatial_model = Some(sm.clone());
            c_ew4_sp.spatial_rerank_only = true;
            c_ew4_sp.spatial_fast_ew_internal = true;
            c_ew4_sp.weights.garbage_row = 225;
            c_ew4_sp.weights.stack_cell = -18;
            c_ew4_sp.weights.upper_cell = 4;
            c_ew4_sp.weights.stack_over2_sq = 3;
            c_ew4_sp.weights.blocks_over_hole = 14;
            c_ew4_sp.weights.zero_o_spot_pen = 0;
            c_ew4_sp.weights.bumpiness = 11;
            c_ew4_sp.weights.near_full_9_bonus = 0;
            c_ew4_sp.weights.near_full_8_bonus = 0;
            c_ew4_sp.weights.row_g_aligned_bonus = 0;
            c_ew4_sp.weights.h1_contig_cover = 130;
            c_ew4_sp.weights.h2_contig_cover = 80;
            c_ew4_sp.weights.endgame_cell_taper = 43;
            c_ew4_sp.nds_pen = 70;
            c_ew4_sp.root_pimc_worlds = 8;
            c_ew4_sp.root_pimc_horizon = 10;
            c_ew4_sp.root_pimc_roots = 8;
            c_ew4_sp.root_pimc_width = 16;
            c_ew4_sp.root_pimc_margin = 400;
            c_ew4_sp.root_pimc_weight_pct = 100;
            c_ew4_sp.root_pimc_ds_target = 3;
            eval_seeds("ew18_sp_ds3_nds70", &seeds5, &c_ew4_sp);

            let mut c_sp_ds3 = make_spatial_cfg(sm.clone());
            c_sp_ds3.bag_det_extend = true;
            c_sp_ds3.endgame_solver = true;
            c_sp_ds3.beam_width = 64;
            c_sp_ds3.max_per_root = 16;
            c_sp_ds3.root_pimc_worlds = 8;
            c_sp_ds3.root_pimc_horizon = 10;
            c_sp_ds3.root_pimc_roots = 8;
            c_sp_ds3.root_pimc_width = 16;
            c_sp_ds3.root_pimc_margin = 350;
            c_sp_ds3.root_pimc_weight_pct = 100;
            c_sp_ds3.root_pimc_ds_target = 3;
            eval_seeds("sp_wg1_ds3_nds0", &seeds5, &c_sp_ds3);

            let mut c_sp_ds3_n80 = c_sp_ds3.clone();
            c_sp_ds3_n80.nds_pen = 80;
            eval_seeds("sp_wg1_ds3_nds80", &seeds5, &c_sp_ds3_n80);
            return;
        }
        if which == "all" || which == "ew" {
            eval_seeds("ew_ds5_w10_b16", &seeds5, &c_ew);
            let mut c_r12 = c_ew.clone();
            c_r12.rollout_top_k = 12;
            eval_seeds("ew_ds5_w10_b16_r12", &seeds5, &c_r12);
        }
        if which == "all" || which == "sp" {
            let mut c_sp_ds = c_ew.clone();
            c_sp_ds.spatial_model = Some(sm.clone());
            c_sp_ds.spatial_rerank_only = true;
            c_sp_ds.spatial_fast_ew_internal = true;
            eval_seeds("sp_ew_ds5_w10_b16", &seeds5, &c_sp_ds);

            let mut c_sp_flat = c_sp_ds.clone();
            c_sp_flat.root_pimc_ds_target = 0;
            c_sp_flat.root_pimc_width = 24;
            eval_seeds("sp_ew_flat_w10_h10", &seeds5, &c_sp_flat);
        }
        return;
    }

    if mode == "ew_sh" {
        let which = std::env::args().nth(2).unwrap_or_else(|| "all".to_string());
        println!("=== Legal 5-seed Pure-Bitwise EvalWeights + SH CRN-PIMC (which={}) ===", which);
        let c_base = make_ew_cons_cfg();
        if which == "all" || which == "base" {
            eval_seeds("ew_base_b120_end7", &seeds5, &c_base);
        }

        if which == "contig" || which == "all" {
            let mut c_g220 = c_base.clone();
            c_g220.weights.garbage_row = 220;
            c_g220.weights.stack_cell = -18;
            c_g220.weights.upper_cell = 6;
            c_g220.weights.blocks_over_hole = 14;
            c_g220.weights.zero_o_spot_pen = 0;
            c_g220.weights.bumpiness = 11;
            c_g220.weights.h1_contig_cover = 140;
            c_g220.weights.h2_contig_cover = 50;
            eval_seeds("ew_nosp_g220_sc-18_c140", &seeds5, &c_g220);

            let mut c_g240_sc4 = c_g220.clone();
            c_g240_sc4.weights.garbage_row = 225;
            c_g240_sc4.weights.stack_cell = 4;
            c_g240_sc4.weights.upper_cell = 5;
            c_g240_sc4.weights.h1_contig_cover = 120;
            c_g240_sc4.weights.h2_contig_cover = 40;
            eval_seeds("ew_nosp_g225_sc4_c120", &seeds5, &c_g240_sc4);

            let mut c_g220_roll = c_g220.clone();
            c_g220_roll.rollout_top_k = 12;
            eval_seeds("ew_nosp_g220_sc-18_r12", &seeds5, &c_g220_roll);
        }

        let mut c_sh = c_base.clone();
        c_sh.weights.garbage_row = 220;
        c_sh.weights.stack_cell = -18;
        c_sh.weights.upper_cell = 6;
        c_sh.weights.blocks_over_hole = 14;
        c_sh.weights.zero_o_spot_pen = 0;
        c_sh.weights.bumpiness = 11;
        c_sh.weights.h1_contig_cover = 140;
        c_sh.weights.h2_contig_cover = 50;
        c_sh.beam_width = 60;
        c_sh.max_per_root = 12;
        c_sh.root_pimc_prescreen_worlds = 4;
        c_sh.root_pimc_roots = 48;
        c_sh.root_pimc_fine_roots = 10;
        c_sh.root_pimc_worlds = 12;
        c_sh.root_pimc_horizon = 11;
        c_sh.root_pimc_width = 24;
        c_sh.root_pimc_weight_pct = 100;
        if which == "all" || which == "sh" {
            eval_seeds("ew_sh_w12_h11_k10", &seeds5, &c_sh);
        }

        let mut c_cvar = c_sh.clone();
        c_cvar.root_pimc_cvar_pct = 25;
        if which == "all" || which == "cvar" {
            eval_seeds("ew_sh_w12_h11_cvar25", &seeds5, &c_cvar);
        }
        return;
    }

    if mode == "opt_legal" {
        let iters: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(50);
        let t0 = Instant::now();
        let mut base_cfg = make_ew_cons_cfg();
        base_cfg.beam_width = 80;
        base_cfg.max_per_root = 16;
        base_cfg.rollout_top_k = 0;
        base_cfg.endgame_solver = false; // fast inner eval
        base_cfg.simulate_cheese_spawn = false;
        base_cfg.weights.garbage_row = 220;
        base_cfg.weights.stack_cell = -18;
        base_cfg.weights.upper_cell = 6;
        base_cfg.weights.blocks_over_hole = 14;
        base_cfg.weights.zero_o_spot_pen = 0;
        base_cfg.weights.bumpiness = 11;
        base_cfg.weights.h1_contig_cover = 120;
        base_cfg.weights.h2_contig_cover = 40;

        let eval_fast5 = |cfg: &BeamConfig| -> (i64, f64, usize, f64) {
            let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).min(5);
            let next_idx = std::sync::atomic::AtomicUsize::new(0);
            let mut results: Vec<Option<GameStats>> = vec![None; 5];
            std::thread::scope(|sc| {
                let mut hs = Vec::new();
                for _ in 0..workers {
                    hs.push(sc.spawn(|| {
                        let mut loc = Vec::new();
                        loop {
                            let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if i >= 5 { break; }
                            loc.push((i, play_legal_seed(seeds5[i], cfg)));
                        }
                        loc
                    }));
                }
                for h in hs {
                    for (i, st) in h.join().unwrap() { results[i] = Some(st); }
                }
            });
            let stats: Vec<GameStats> = results.into_iter().map(|x| x.unwrap()).collect();
            let sum_pcs: usize = stats.iter().map(|s| s.pieces).sum();
            let sum_cov: usize = stats.iter().map(|s| s.zds_h1_cov).sum();
            let ds_c: usize = stats.iter().map(|s| s.ds1).sum();
            let cov_n: usize = stats.iter().map(|s| s.ds_cov_next).sum();
            let cov_pct = 100.0 * cov_n as f64 / ds_c.max(1) as f64;
            let obj = (sum_pcs as i64) * 100_000 + (sum_cov as i64) * 500 + (cov_n as i64) * 200;
            (obj, sum_pcs as f64 / 5.0, sum_pcs, cov_pct)
        };

        let mut cur = base_cfg.weights.clone();
        let (mut best_obj, mut best_mean, _, mut best_cov) = eval_fast5(&base_cfg);
        println!("=== Legal 5-seed EvalWeights Optimizer (b={}, iters={}) init mean={:.2} cov_nxt={:.1}% ===",
            base_cfg.beam_width, iters, best_mean, best_cov);
        let _ = std::io::stdout().flush();

        let mut rng: u64 = 0xD1B54A32D192ED03u64;
        let mut next_u64 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            rng
        };

        for iter in 1..=iters {
            let mut cand = cur.clone();
            let k_mut = 1 + ((next_u64() >> 32) % 3) as usize;
            for _ in 0..k_mut {
                let idx = ((next_u64() >> 32) % 24) as usize;
                let sign: i64 = if (next_u64() & 1) == 0 { 1 } else { -1 };
                let scale = if iter <= iters / 2 { 2 } else { 1 };
                let delta = sign * scale * (1 + ((next_u64() >> 40) % 4) as i64);
                match idx {
                    0 => cand.garbage_row = (cand.garbage_row + delta * 10).clamp(120, 340),
                    1 => cand.stack_cell = (cand.stack_cell + delta * 3).clamp(-26, 20),
                    2 => cand.upper_cell = (cand.upper_cell + delta).clamp(0, 20),
                    3 => cand.stack_over2_sq = (cand.stack_over2_sq + delta).clamp(0, 20),
                    4 => cand.max_stack_h_excess = (cand.max_stack_h_excess + delta * 2).clamp(0, 40),
                    5 => cand.covered_hole = (cand.covered_hole + delta * 6).clamp(20, 220),
                    6 => cand.hole_row = (cand.hole_row + delta * 4).clamp(0, 100),
                    7 => cand.blocks_over_hole = (cand.blocks_over_hole + delta * 3).clamp(0, 60),
                    8 => cand.h1_block = (cand.h1_block + delta * 5).clamp(10, 160),
                    9 => cand.h1_contig_cover = (cand.h1_contig_cover + delta * 10).clamp(0, 320),
                    10 => cand.h1_empty_blocking = (cand.h1_empty_blocking + delta * 2).clamp(0, 40),
                    11 => cand.h1_height = (cand.h1_height + delta * 2).clamp(0, 50),
                    12 => cand.h2_block = (cand.h2_block + delta * 4).clamp(0, 100),
                    13 => cand.h2_contig_cover = (cand.h2_contig_cover + delta * 8).clamp(0, 200),
                    14 => cand.h1_narrow_1 = (cand.h1_narrow_1 + delta * 3).clamp(0, 60),
                    15 => cand.h1_narrow_2plus = (cand.h1_narrow_2plus + delta * 4).clamp(0, 70),
                    16 => cand.open_h1_bonus = (cand.open_h1_bonus + delta * 5).clamp(0, 120),
                    17 => cand.combo2_bonus = (cand.combo2_bonus + delta * 4).clamp(0, 100),
                    18 => cand.bumpiness = (cand.bumpiness + delta).clamp(0, 20),
                    19 => cand.sharp_diff = (cand.sharp_diff + delta).clamp(0, 20),
                    20 => cand.i_dep = (cand.i_dep + delta * 5).clamp(0, 90),
                    21 => cand.near_full_9_bonus = (cand.near_full_9_bonus + delta * 3).clamp(-40, 50),
                    22 => cand.row_g_aligned_bonus = (cand.row_g_aligned_bonus + delta * 4).clamp(-40, 50),
                    _ => {
                        let hi = ((next_u64() >> 48) % 7) as usize;
                        cand.hold_bonus[hi] = (cand.hold_bonus[hi] + delta * 3).clamp(-35, 35);
                    }
                }
            }
            let mut c_try = base_cfg.clone();
            c_try.weights = cand.clone();
            let (obj, mean, _, cov_pct) = eval_fast5(&c_try);
            if obj < best_obj {
                best_obj = obj;
                best_mean = mean;
                best_cov = cov_pct;
                cur = cand;
                println!("  [iter {:2} {:.1}s] NEW BEST mean={:.2} cov_nxt={:.1}% obj={}",
                    iter, t0.elapsed().as_secs_f64(), best_mean, best_cov, best_obj);
                let _ = std::io::stdout().flush();
            }
        }
        println!("=== Best Legal EvalWeights ({:.1}s) mean={:.2} ===", t0.elapsed().as_secs_f64(), best_mean);
        println!("{:#?}", cur);

        let mut c_ver = make_ew_cons_cfg();
        c_ver.weights = cur.clone();
        eval_seeds("opt_legal_b120_end7", &seeds5, &c_ver);

        let mut c_ver_r = c_ver.clone();
        c_ver_r.rollout_top_k = 12;
        eval_seeds("opt_legal_b120_r12", &seeds5, &c_ver_r);

        let mut c_sh = c_ver_r.clone();
        c_sh.beam_width = 60;
        c_sh.max_per_root = 12;
        c_sh.root_pimc_prescreen_worlds = 4;
        c_sh.root_pimc_roots = 48;
        c_sh.root_pimc_fine_roots = 10;
        c_sh.root_pimc_worlds = 12;
        c_sh.root_pimc_horizon = 11;
        c_sh.root_pimc_width = 24;
        c_sh.root_pimc_weight_pct = 100;
        eval_seeds("opt_legal_sh_w12_h11", &seeds5, &c_sh);
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

    if mode == "trace_end" {
        let seed: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(101);
        let mut c = make_cfg(m_h2_nog0.clone());
        c.bag_det_extend = true;
        c.endgame_solver = true;
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(seed);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai = FastAI::new(c);
        let mut ds_total = 0usize;
        let mut step = 0usize;
        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
                Some(r) => r,
                None => {
                    println!("step={} NONE! g={} rem_g={}", step, g_rows, rem_g);
                    break;
                }
            };
            let mut inputs = res.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            step += 1;
            ds_total += ds;
            if rem_g <= 12 {
                let top_c = ai.last_top_candidates();
                println!(
                    "step={:3} rem_g={:2} g={} -> ds={} lines={} rating={} top0={:?} top1={:?}",
                    step, rem_g, g_rows, ds, lines, res.rating, top_c.get(0), top_c.get(1)
                );
            }
        }
        println!("Done seed={} pcs={} ds_total={}", seed, step, ds_total);
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
