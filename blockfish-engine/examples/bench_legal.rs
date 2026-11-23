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
        c_rpimc.rollout_top_k = 0;
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
        c_rr_rp.rollout_top_k = 0;
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
        if (cfg_name.contains("_lin") && !cfg_name.contains("_linfast")) || cfg_name.contains("_nomlp") {
            sm_val.b3 = 0.0;
            sm_val.w3 = [0.0; blockfish::ai::spatial_val::SPATIAL_H2];
            sm_val.prepare_fast();
        }
        let sm = Arc::new(sm_val);
        println!("=== Legal spatial5_1 ({} / {}) ===", model_path, cfg_name);

        let mut c = make_spatial_cfg(sm);
        c.rollout_top_k = 0;
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
            c.rollout_top_k = 0;
        }
        if cfg_name.contains("_nope") {
            c.spatial_no_pe_internal = true;
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

    if mode == "wg5_lin_sweep" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_wg5_ep0.txt".to_string());
        let which = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        let n_seeds: usize = std::env::args()
            .nth(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let seeds: Vec<u64> = (101..(101 + n_seeds as u64)).collect();
        let mut sm_val = SpatialValueModel::load_from_file(&model_path).expect("load spatial model");
        sm_val.b3 = 0.0;
        sm_val.w3 = [0.0; blockfish::ai::spatial_val::SPATIAL_H2];
        sm_val.prepare_fast();
        let sm = Arc::new(sm_val);
        println!(
            "=== Legal wg5_lin_sweep ({}, which={}, n_seeds={}) ===",
            model_path, which, n_seeds
        );

        let mut c_base = make_spatial_cfg(sm.clone());
        c_base.rollout_top_k = 0;
        c_base.bag_det_extend = true;
        c_base.endgame_solver = true;
        c_base.simulate_cheese_spawn = true;
        c_base.spatial_rerank_only = true;
        c_base.spatial_no_pe_internal = true;
        c_base.root_pimc_worlds = 8;
        c_base.root_pimc_horizon = 8;
        c_base.root_pimc_roots = 8;
        c_base.root_pimc_width = 24;
        c_base.root_pimc_margin = 350;
        c_base.root_pimc_weight_pct = 100;
        c_base.root_pimc_steady_g4 = false;

        if which == "all" || which == "base" {
            eval_seeds("lin_nope_w8_h8_r8", &seeds, &c_base);
        }

        let mut c_deep = c_base.clone();
        c_deep.beam_width = 80;
        c_deep.max_per_root = 16;
        c_deep.root_pimc_prescreen_worlds = 3;
        c_deep.root_pimc_roots = 32;
        c_deep.root_pimc_fine_roots = 10;
        c_deep.root_pimc_worlds = 12;
        c_deep.root_pimc_horizon = 11;
        c_deep.root_pimc_width = 28;
        c_deep.root_pimc_margin = 500;
        if which == "all" || which == "deep" {
            eval_seeds("lin_nope_pre3_w12_h11_b28", &seeds, &c_deep);
        }

        let mut c_ds3 = c_deep.clone();
        c_ds3.root_pimc_ds_target = 3;
        c_ds3.root_pimc_width = 24;
        if which == "all" || which == "ds3" {
            eval_seeds("lin_nope_ds3_w12_b24", &seeds, &c_ds3);
        }

        let mut c_ds4 = c_ds3.clone();
        c_ds4.root_pimc_ds_target = 4;
        c_ds4.root_pimc_horizon = 12;
        if which == "all" || which == "ds4" {
            eval_seeds("lin_nope_ds4_w12_b24", &seeds, &c_ds4);
        }

        let mut c_ds4_nds60 = c_ds4.clone();
        c_ds4_nds60.nds_pen = 60;
        if which == "all" || which == "nds" {
            eval_seeds("lin_nope_ds4_nds60", &seeds, &c_ds4_nds60);
        }

        let mut c_ds4_big = c_ds4.clone();
        c_ds4_big.root_pimc_prescreen_worlds = 4;
        c_ds4_big.root_pimc_roots = 40;
        c_ds4_big.root_pimc_fine_roots = 12;
        c_ds4_big.root_pimc_worlds = 16;
        c_ds4_big.root_pimc_width = 32;
        c_ds4_big.root_pimc_margin = 600;
        if which == "all" || which == "big" {
            eval_seeds("lin_nope_ds4_w16_b32", &seeds, &c_ds4_big);
        }

        if which == "all" || which == "wg1_ref" {
            let sm_wg1 = Arc::new(
                SpatialValueModel::load_from_file("models/spatial_wg1.txt")
                    .expect("load spatial_wg1"),
            );
            let mut c_wg1 = make_spatial_cfg(sm_wg1.clone());
            c_wg1.rollout_top_k = 0;
            c_wg1.bag_det_extend = true;
            c_wg1.endgame_solver = true;
            c_wg1.simulate_cheese_spawn = true;
            c_wg1.spatial_rerank_only = true;
            c_wg1.spatial_no_pe_internal = true;
            c_wg1.root_pimc_worlds = 8;
            c_wg1.root_pimc_horizon = 8;
            c_wg1.root_pimc_roots = 8;
            c_wg1.root_pimc_width = 24;
            c_wg1.root_pimc_margin = 350;
            c_wg1.root_pimc_weight_pct = 100;
            c_wg1.root_pimc_steady_g4 = false;
            eval_seeds("wg1_ref_linfast_nope_w8_h8", &seeds, &c_wg1);

            let mut c_wg1_ds4 = c_ds4.clone();
            c_wg1_ds4.spatial_model = Some(sm_wg1);
            eval_seeds("wg1_ref_ds4_w12_b24", &seeds, &c_wg1_ds4);
        }
        if which == "bc_cvar" {
            let mut c_h7 = c_base.clone();
            c_h7.beam_width = 80;
            c_h7.max_per_root = 16;
            c_h7.root_pimc_prescreen_worlds = 3;
            c_h7.root_pimc_roots = 32;
            c_h7.root_pimc_fine_roots = 10;
            c_h7.root_pimc_worlds = 16;
            c_h7.root_pimc_horizon = 7;
            c_h7.root_pimc_width = 32;
            c_h7.root_pimc_margin = 500;
            eval_seeds("bc_h7_w16_b32", &seeds, &c_h7);

            let mut c_h8 = c_h7.clone();
            c_h8.root_pimc_horizon = 8;
            eval_seeds("bc_h8_w16_b32", &seeds, &c_h8);

            let mut c_h8_cvar = c_h8.clone();
            c_h8_cvar.root_pimc_cvar_pct = 25;
            eval_seeds("bc_h8_w16_b32_cvar25", &seeds, &c_h8_cvar);

            let mut c_h9_cvar = c_h8.clone();
            c_h9_cvar.root_pimc_horizon = 9;
            c_h9_cvar.root_pimc_cvar_pct = 30;
            eval_seeds("bc_h9_w16_b32_cvar30", &seeds, &c_h9_cvar);
        }
        if which == "nc_pimc" {
            let mut c_nc = c_base.clone();
            c_nc.beam_width = 80;
            c_nc.max_per_root = 16;
            c_nc.root_pimc_prescreen_worlds = 3;
            c_nc.root_pimc_roots = 32;
            c_nc.root_pimc_fine_roots = 10;
            c_nc.root_pimc_worlds = 16;
            c_nc.root_pimc_horizon = 5;
            c_nc.root_pimc_width = 24;
            c_nc.root_pimc_margin = 500;
            c_nc.root_pimc_nc_steps = 10;
            c_nc.root_pimc_cvar_pct = 25;
            eval_seeds("nc_h5_nc10_w16_cvar25", &seeds, &c_nc);

            let mut c_nc2 = c_nc.clone();
            c_nc2.root_pimc_horizon = 6;
            c_nc2.root_pimc_nc_steps = 12;
            c_nc2.root_pimc_cvar_pct = 30;
            eval_seeds("nc_h6_nc12_w16_cvar30", &seeds, &c_nc2);
        }
        if which == "nc_pimc_mlp" {
            let sm_full = Arc::new(
                SpatialValueModel::load_from_file(&model_path).expect("load full spatial model"),
            );
            let mut c_mlp = make_spatial_cfg(sm_full);
            c_mlp.rollout_top_k = 0;
            c_mlp.beam_width = 80;
            c_mlp.max_per_root = 16;
            c_mlp.bag_det_extend = true;
            c_mlp.endgame_solver = true;
            c_mlp.simulate_cheese_spawn = true;
            c_mlp.spatial_rerank_only = true;
            c_mlp.spatial_no_pe_internal = true;
            c_mlp.root_pimc_prescreen_worlds = 3;
            c_mlp.root_pimc_roots = 32;
            c_mlp.root_pimc_fine_roots = 10;
            c_mlp.root_pimc_worlds = 16;
            c_mlp.root_pimc_horizon = 9;
            c_mlp.root_pimc_width = 32;
            c_mlp.root_pimc_margin = 500;
            c_mlp.root_pimc_weight_pct = 100;
            c_mlp.root_pimc_steady_g4 = false;
            c_mlp.root_pimc_cvar_pct = 30;
            eval_seeds("mlp_h9_w16_b32_cvar30", &seeds, &c_mlp);

            let mut c_mlp_c40 = c_mlp.clone();
            c_mlp_c40.root_pimc_cvar_pct = 40;
            eval_seeds("mlp_h9_w16_b32_cvar40", &seeds, &c_mlp_c40);

            let mut c_mlp_nc = c_mlp.clone();
            c_mlp_nc.root_pimc_horizon = 6;
            c_mlp_nc.root_pimc_width = 24;
            c_mlp_nc.root_pimc_nc_steps = 12;
            c_mlp_nc.root_pimc_cvar_pct = 30;
            eval_seeds("mlp_h6_nc12_w16_cvar30", &seeds, &c_mlp_nc);
        }
        if which == "cvar_pro" {
            let mut c1 = c_base.clone();
            c1.beam_width = 80;
            c1.max_per_root = 16;
            c1.root_pimc_prescreen_worlds = 3;
            c1.root_pimc_roots = 32;
            c1.root_pimc_fine_roots = 10;
            c1.root_pimc_worlds = 16;
            c1.root_pimc_horizon = 9;
            c1.root_pimc_width = 32;
            c1.root_pimc_margin = 500;
            c1.root_pimc_cvar_pct = 30;
            eval_seeds("pro_h9_w16_b32_cvar30", &seeds, &c1);

            let mut c2 = c1.clone();
            c2.root_pimc_prescreen_worlds = 4;
            c2.root_pimc_roots = 36;
            c2.root_pimc_fine_roots = 12;
            c2.root_pimc_worlds = 20;
            c2.root_pimc_width = 36;
            c2.root_pimc_cvar_pct = 35;
            eval_seeds("pro_h9_w20_b36_cvar35", &seeds, &c2);

            let mut c3 = c2.clone();
            c3.root_pimc_horizon = 10;
            c3.root_pimc_width = 40;
            c3.root_pimc_cvar_pct = 40;
            eval_seeds("pro_h10_w20_b40_cvar40", &seeds, &c3);
        }
        if which == "cvar_top2" {
            let mut c0 = c_base.clone();
            c0.beam_width = 120;
            c0.max_per_root = 24;
            c0.root_pimc_worlds = 0;
            eval_seeds("plain_p5_b120", &seeds, &c0);

            let mut c1 = c_base.clone();
            c1.beam_width = 80;
            c1.max_per_root = 16;
            c1.root_pimc_prescreen_worlds = 3;
            c1.root_pimc_roots = 32;
            c1.root_pimc_fine_roots = 10;
            c1.root_pimc_worlds = 16;
            c1.root_pimc_horizon = 9;
            c1.root_pimc_width = 32;
            c1.root_pimc_margin = 500;
            c1.root_pimc_cvar_pct = 30;
            eval_seeds("pro_h9_w16_b32_cvar30", &seeds, &c1);

            let mut c2 = c1.clone();
            c2.root_pimc_prescreen_worlds = 4;
            c2.root_pimc_roots = 36;
            c2.root_pimc_fine_roots = 12;
            c2.root_pimc_worlds = 20;
            c2.root_pimc_width = 36;
            c2.root_pimc_cvar_pct = 35;
            eval_seeds("pro_h9_w20_b36_cvar35", &seeds, &c2);
        }
        return;
    }

    if mode == "cons_cap" {
        let which = std::env::args().nth(2).unwrap_or_else(|| "all".to_string());
        let n_seeds: usize = std::env::args()
            .nth(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let seeds: Vec<u64> = (101..(101 + n_seeds as u64)).collect();
        println!("=== Legal cons_cap (which={}, n_seeds={}) ===", which, n_seeds);

        let mut c_ew4 = make_ew_cons_cfg();
        c_ew4.rollout_top_k = 0;
        c_ew4.beam_width = 80;
        c_ew4.max_per_root = 16;
        c_ew4.bag_det_extend = true;
        c_ew4.endgame_solver = true;
        c_ew4.simulate_cheese_spawn = true;
        c_ew4.weights.garbage_row = 225;
        c_ew4.weights.piece_estimate = 0;
        c_ew4.weights.stack_cell = 4;
        c_ew4.weights.upper_cell = 5;
        c_ew4.weights.stack_h_sq = 0;
        c_ew4.weights.stack_over2_sq = 3;
        c_ew4.weights.max_stack_h_excess = 14;
        c_ew4.weights.covered_hole = 55;
        c_ew4.weights.hole_row = 20;
        c_ew4.weights.blocks_over_hole = 14;
        c_ew4.weights.h1_block = 32;
        c_ew4.weights.h1_contig_cover = 130;
        c_ew4.weights.h1_empty_blocking = 8;
        c_ew4.weights.h1_height = 10;
        c_ew4.weights.h2_block = 26;
        c_ew4.weights.h2_contig_cover = 80;
        c_ew4.weights.h2_extra_empty = 6;
        c_ew4.weights.h3_block = 22;
        c_ew4.weights.h3_extra_empty = 3;
        c_ew4.weights.h4_block = 8;
        c_ew4.weights.h1_narrow_1 = 16;
        c_ew4.weights.h1_narrow_2plus = 26;
        c_ew4.weights.h1_fit_scarcity = 8;
        c_ew4.weights.h1_unfit_hand = 28;
        c_ew4.weights.zero_o_spot_pen = 0;
        c_ew4.weights.bag_no_i_chute_pen = 18;
        c_ew4.weights.open_h1_bonus = 24;
        c_ew4.weights.combo2_bonus = 22;
        c_ew4.weights.h1_h2_adj = 4;
        c_ew4.weights.bumpiness = 11;
        c_ew4.weights.sharp_diff = 5;
        c_ew4.weights.i_dep = 35;
        c_ew4.weights.two_well = 10;
        c_ew4.weights.i_dep_multi = 50;
        c_ew4.weights.i_dep_chute = 12;
        c_ew4.weights.row_transition = 3;
        c_ew4.weights.near_full_9_bonus = 0;
        c_ew4.weights.near_full_8_bonus = 0;
        c_ew4.weights.row_g_aligned_bonus = 0;
        c_ew4.weights.endgame_cell_taper = 23;
        c_ew4.weights.endgame_cell_taper_rem = 5;
        c_ew4.weights.hold_bonus = [-10, 4, -8, 4, 4, -2, -2];
        c_ew4.root_pimc_prescreen_worlds = 3;
        c_ew4.root_pimc_roots = 24;
        c_ew4.root_pimc_fine_roots = 8;
        c_ew4.root_pimc_worlds = 8;
        c_ew4.root_pimc_horizon = 10;
        c_ew4.root_pimc_ds_target = 3;
        c_ew4.root_pimc_width = 24;
        c_ew4.root_pimc_margin = 400;
        c_ew4.root_pimc_weight_pct = 100;
        c_ew4.root_pimc_steady_g4 = false;

        if which == "all" || which == "ew4" {
            eval_seeds("ew4_ds3_w8_b24", &seeds, &c_ew4);
        }

        let make_cap = |cap: i64, excess: i64, rg: i64, sc: i64, ds_t: usize, nds_p: i64| {
            let mut c = c_ew4.clone();
            c.weights.garbage_row = rg;
            c.weights.stack_cell = sc;
            c.weights.stack_cell_credit_cap = cap;
            c.weights.stack_cell_excess = excess;
            c.weights.upper_cell = 4;
            c.weights.h1_contig_cover = 140;
            c.weights.h2_contig_cover = 95;
            c.weights.endgame_cell_taper = 0;
            c.root_pimc_ds_target = ds_t;
            c.nds_pen = nds_p;
            c
        };

        if which == "all" || which == "cap" {
            eval_seeds("cap14_ex8_rg168_ds3", &seeds, &make_cap(14, 8, 168, -22, 3, 0));
            eval_seeds("cap18_ex8_rg168_ds3", &seeds, &make_cap(18, 8, 168, -22, 3, 0));
            eval_seeds("cap16_ex12_rg168_ds3", &seeds, &make_cap(16, 12, 168, -22, 3, 0));
            eval_seeds("cap16_ex12_rg168_flat", &seeds, &make_cap(16, 12, 168, -22, 0, 0));
            eval_seeds("cap16_ex12_rg168_nds40", &seeds, &make_cap(16, 12, 168, -22, 3, 40));
        }
        return;
    }

    if mode == "deep_ds_pimc" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_wg5_ep0.txt".to_string());
        let which = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        let n_seeds: usize = std::env::args()
            .nth(4)
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        let seeds: Vec<u64> = (101..(101 + n_seeds as u64)).collect();
        let mut sm_val = SpatialValueModel::load_from_file(&model_path).expect("load spatial model");
        sm_val.b3 = 0.0;
        sm_val.w3 = [0.0; blockfish::ai::spatial_val::SPATIAL_H2];
        sm_val.prepare_fast();
        let sm = Arc::new(sm_val);
        println!(
            "=== Legal deep_ds_pimc ({}, which={}, n_seeds={}) ===",
            model_path, which, n_seeds
        );

        let mut ew_c22 = blockfish::ai::fast_eval::EvalWeights::default();
        ew_c22.garbage_row = 165;
        ew_c22.piece_estimate = 0;
        ew_c22.stack_cell = -22;
        ew_c22.upper_cell = 3;
        ew_c22.stack_h_sq = 0;
        ew_c22.stack_over2_sq = 3;
        ew_c22.max_stack_h_excess = 14;
        ew_c22.covered_hole = 55;
        ew_c22.hole_row = 20;
        ew_c22.blocks_over_hole = 14;
        ew_c22.h1_block = 32;
        ew_c22.h1_contig_cover = 140;
        ew_c22.h1_empty_blocking = 8;
        ew_c22.h1_height = 10;
        ew_c22.h2_block = 26;
        ew_c22.h2_contig_cover = 95;
        ew_c22.h2_extra_empty = 6;
        ew_c22.h3_block = 22;
        ew_c22.h3_extra_empty = 3;
        ew_c22.h4_block = 8;
        ew_c22.h1_narrow_1 = 16;
        ew_c22.h1_narrow_2plus = 26;
        ew_c22.h1_fit_scarcity = 8;
        ew_c22.h1_unfit_hand = 28;
        ew_c22.zero_o_spot_pen = 0;
        ew_c22.bag_no_i_chute_pen = 18;
        ew_c22.open_h1_bonus = 24;
        ew_c22.combo2_bonus = 22;
        ew_c22.h1_h2_adj = 4;
        ew_c22.bumpiness = 11;
        ew_c22.sharp_diff = 5;
        ew_c22.i_dep = 35;
        ew_c22.two_well = 10;
        ew_c22.i_dep_multi = 50;
        ew_c22.i_dep_chute = 12;
        ew_c22.row_transition = 3;
        ew_c22.near_full_9_bonus = 0;
        ew_c22.near_full_8_bonus = 0;
        ew_c22.row_g_aligned_bonus = 0;
        ew_c22.endgame_cell_taper = 47;
        ew_c22.endgame_cell_taper_rem = 5;
        ew_c22.hold_bonus = [-10, 4, -8, 4, 4, -2, -2];

        let mut ew4 = ew_c22.clone();
        ew4.garbage_row = 225;
        ew4.stack_cell = 4;
        ew4.upper_cell = 5;
        ew4.h1_contig_cover = 130;
        ew4.h2_contig_cover = 80;
        ew4.endgame_cell_taper = 23;

        let make_deep = |ds_t: usize, worlds: usize, width: usize, fine_r: usize| {
            let mut c = make_spatial_cfg(sm.clone());
            c.rollout_top_k = 0;
            c.beam_width = 80;
            c.max_per_root = 16;
            c.bag_det_extend = true;
            c.endgame_solver = true;
            c.simulate_cheese_spawn = true;
            c.spatial_rerank_only = true;
            c.spatial_fast_ew_internal = true;
            c.weights = ew_c22.clone();
            c.stage1_weights = Some(ew4.clone());
            c.root_pimc_prescreen_worlds = 3;
            c.root_pimc_roots = 28;
            c.root_pimc_fine_roots = fine_r;
            c.root_pimc_worlds = worlds;
            c.root_pimc_horizon = 15;
            c.root_pimc_ds_target = ds_t;
            c.root_pimc_width = width;
            c.root_pimc_margin = 450;
            c.root_pimc_weight_pct = 100;
            c.root_pimc_steady_g4 = false;
            c
        };

        if which == "all" || which == "ds8" {
            eval_seeds("deep_c22_ds8_w8_b24", &seeds, &make_deep(8, 8, 24, 6));
        }
        if which == "all" || which == "ds12" {
            eval_seeds("deep_c22_ds12_w8_b24", &seeds, &make_deep(12, 8, 24, 6));
        }
        if which == "all" || which == "ds12_b48" {
            eval_seeds("deep_c22_ds12_w8_b48", &seeds, &make_deep(12, 8, 48, 6));
        }
        if which == "all" || which == "ds16" {
            eval_seeds("deep_c22_ds16_w8_b32", &seeds, &make_deep(16, 8, 32, 6));
        }
        return;
    }

    if mode == "spatial_ablate" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_wg2.txt".to_string());
        let which = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        let sm = Arc::new(SpatialValueModel::load_from_file(&model_path).expect("load spatial model"));
        println!("=== Legal spatial_ablate ({}, which={}) ===", model_path, which);

        let mut c_base = make_spatial_cfg(sm);
        c_base.bag_det_extend = true;
        c_base.endgame_solver = true;
        c_base.simulate_cheese_spawn = true;
        c_base.spatial_rerank_only = true;
        c_base.root_pimc_worlds = 8;
        c_base.root_pimc_horizon = 9;
        c_base.root_pimc_roots = 8;
        c_base.root_pimc_width = 24;
        c_base.root_pimc_margin = 350;
        c_base.root_pimc_weight_pct = 100;
        c_base.root_pimc_steady_g4 = false;

        // Match cons22 fast eval weights when testing spatial_fast_ew_internal
        c_base.weights.garbage_row = 165;
        c_base.weights.piece_estimate = 0;
        c_base.weights.stack_cell = -22;
        c_base.weights.upper_cell = 3;
        c_base.weights.stack_h_sq = 0;
        c_base.weights.stack_over2_sq = 3;
        c_base.weights.max_stack_h_excess = 14;
        c_base.weights.covered_hole = 55;
        c_base.weights.hole_row = 20;
        c_base.weights.blocks_over_hole = 14;
        c_base.weights.h1_block = 32;
        c_base.weights.h1_contig_cover = 140;
        c_base.weights.h1_empty_blocking = 8;
        c_base.weights.h1_height = 10;
        c_base.weights.h2_block = 26;
        c_base.weights.h2_contig_cover = 95;
        c_base.weights.h2_extra_empty = 6;
        c_base.weights.h3_block = 22;
        c_base.weights.h3_extra_empty = 3;
        c_base.weights.h4_block = 8;
        c_base.weights.h1_narrow_1 = 16;
        c_base.weights.h1_narrow_2plus = 26;
        c_base.weights.h1_fit_scarcity = 8;
        c_base.weights.h1_unfit_hand = 28;
        c_base.weights.zero_o_spot_pen = 0;
        c_base.weights.bag_no_i_chute_pen = 18;
        c_base.weights.open_h1_bonus = 24;
        c_base.weights.combo2_bonus = 22;
        c_base.weights.h1_h2_adj = 4;
        c_base.weights.bumpiness = 11;
        c_base.weights.sharp_diff = 5;
        c_base.weights.i_dep = 35;
        c_base.weights.two_well = 10;
        c_base.weights.i_dep_multi = 50;
        c_base.weights.i_dep_chute = 12;
        c_base.weights.row_transition = 3;
        c_base.weights.near_full_9_bonus = 0;
        c_base.weights.near_full_8_bonus = 0;
        c_base.weights.row_g_aligned_bonus = 0;
        c_base.weights.endgame_cell_taper = 47;
        c_base.weights.endgame_cell_taper_rem = 5;
        c_base.weights.hold_bonus = [-10, 4, -8, 4, 4, -2, -2];

        let mut c_cons_ds4 = c_base.clone();
        c_cons_ds4.spatial_model = None;
        c_cons_ds4.spatial_rerank_only = false;
        c_cons_ds4.rollout_top_k = 0;
        c_cons_ds4.beam_width = 80;
        c_cons_ds4.max_per_root = 16;
        c_cons_ds4.root_pimc_prescreen_worlds = 3;
        c_cons_ds4.root_pimc_roots = 40;
        c_cons_ds4.root_pimc_fine_roots = 10;
        c_cons_ds4.root_pimc_worlds = 12;
        c_cons_ds4.root_pimc_horizon = 12;
        c_cons_ds4.root_pimc_ds_target = 4;
        c_cons_ds4.root_pimc_width = 24;
        if which == "0_cons22_ds4" {
            eval_seeds("0_cons22_ds4_w12", &seeds5, &c_cons_ds4);
        }

        if which == "all" || which == "1_linfast" {
            eval_seeds("1_linfast_w8_h9_r12", &seeds5, &c_base);
        }
        let mut c_r0 = c_base.clone();
        c_r0.rollout_top_k = 0;
        if which == "all" || which == "2_r0" {
            eval_seeds("2_linfast_w8_h9_r0", &seeds5, &c_r0);
        }
        let mut c_pre = c_r0.clone();
        c_pre.beam_width = 80;
        c_pre.max_per_root = 16;
        c_pre.root_pimc_prescreen_worlds = 3;
        c_pre.root_pimc_roots = 40;
        c_pre.root_pimc_fine_roots = 10;
        c_pre.root_pimc_worlds = 12;
        c_pre.root_pimc_horizon = 10;
        c_pre.root_pimc_width = 24;
        if which == "all" || which == "3_pre" {
            eval_seeds("3_pre3_k40_f10_w12_h10", &seeds5, &c_pre);
        }
        let mut c_ds4 = c_pre.clone();
        c_ds4.root_pimc_ds_target = 4;
        c_ds4.root_pimc_horizon = 12;
        if which == "all" || which == "4_ds4" {
            eval_seeds("4_pre3_ds4_w12", &seeds5, &c_ds4);
        }
        let mut c_ew_pre = c_pre.clone();
        c_ew_pre.spatial_fast_ew_internal = true;
        if which == "all" || which == "5_ewint" {
            eval_seeds("5_ewint_pre3_k40_w12", &seeds5, &c_ew_pre);
        }
        let mut c_ew_ds4 = c_ds4.clone();
        c_ew_ds4.spatial_fast_ew_internal = true;
        if which == "all" || which == "6_ew_ds4" {
            eval_seeds("6_ewint_ds4_w12", &seeds5, &c_ew_ds4);
        }
        let mut c_cvar = c_ds4.clone();
        c_cvar.root_pimc_cvar_pct = 25;
        if which == "all" || which == "7_cvar" {
            eval_seeds("7_ds4_w12_cvar25", &seeds5, &c_cvar);
        }
        let mut c_r12_pre = c_pre.clone();
        c_r12_pre.rollout_top_k = 12;
        if which == "all" || which == "8_pre_r12" {
            eval_seeds("8_pre3_k40_w12_r12", &seeds5, &c_r12_pre);
        }
        let mut c_full_mlp = c_r0.clone();
        c_full_mlp.spatial_rerank_only = false;
        if which == "all" || which == "9_fullmlp" {
            eval_seeds("9_fullmlp_w8_h9_r0", &seeds5, &c_full_mlp);
        }
        let mut c_full_ds3 = c_r0.clone();
        c_full_ds3.spatial_rerank_only = false;
        c_full_ds3.beam_width = 64;
        c_full_ds3.max_per_root = 16;
        c_full_ds3.root_pimc_prescreen_worlds = 3;
        c_full_ds3.root_pimc_roots = 24;
        c_full_ds3.root_pimc_fine_roots = 8;
        c_full_ds3.root_pimc_worlds = 8;
        c_full_ds3.root_pimc_horizon = 10;
        c_full_ds3.root_pimc_ds_target = 3;
        c_full_ds3.root_pimc_width = 14;
        if which == "all" || which == "full_ds3" {
            eval_seeds("10_fullmlp_ds3_w8_pw14", &seeds5, &c_full_ds3);
        }
        let mut c_full_ds4_n80 = c_full_ds3.clone();
        c_full_ds4_n80.nds_pen = 60;
        if which == "full_ds3" {
            eval_seeds("11_fullmlp_ds3_nds60", &seeds5, &c_full_ds4_n80);
        }
        return;
    }

    if mode == "fast2" {
        let model_path = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "models/spatial_wg2.txt".to_string());
        let which = std::env::args().nth(3).unwrap_or_else(|| "all".to_string());
        let seeds2: Vec<u64> = vec![101, 102];
        let sm = Arc::new(SpatialValueModel::load_from_file(&model_path).expect("load spatial model"));
        println!("=== Legal fast2 ({}, which={}, seeds=101..102) ===", model_path, which);

        let mut c_cons = make_ew_cons_cfg();
        c_cons.rollout_top_k = 0;
        c_cons.beam_width = 64;
        c_cons.max_per_root = 16;
        c_cons.weights.garbage_row = 225;
        c_cons.weights.stack_cell = -22;
        c_cons.weights.upper_cell = 3;
        c_cons.weights.stack_over2_sq = 3;
        c_cons.weights.blocks_over_hole = 14;
        c_cons.weights.zero_o_spot_pen = 0;
        c_cons.weights.bumpiness = 11;
        c_cons.weights.near_full_9_bonus = 0;
        c_cons.weights.near_full_8_bonus = 0;
        c_cons.weights.row_g_aligned_bonus = 0;
        c_cons.weights.h1_contig_cover = 140;
        c_cons.weights.h2_contig_cover = 95;
        c_cons.weights.h3_block = 22;
        c_cons.weights.h4_block = 8;
        c_cons.weights.endgame_cell_taper = 47;
        c_cons.weights.endgame_cell_taper_rem = 5;

        let mut c_ew4 = c_cons.clone();
        c_ew4.weights.garbage_row = 225;
        c_ew4.weights.stack_cell = 4;
        c_ew4.weights.upper_cell = 5;
        c_ew4.weights.endgame_cell_taper = 23;

        if which == "all" || which == "base" {
            eval_seeds("0_ew4_b64_nosp", &seeds2, &c_ew4);
            eval_seeds("1_cons22_b64_nosp", &seeds2, &c_cons);
        }

        let mut c_ew4_ds3 = c_ew4.clone();
        c_ew4_ds3.simulate_cheese_spawn = true;
        c_ew4_ds3.root_pimc_prescreen_worlds = 3;
        c_ew4_ds3.root_pimc_roots = 32;
        c_ew4_ds3.root_pimc_fine_roots = 8;
        c_ew4_ds3.root_pimc_worlds = 6;
        c_ew4_ds3.root_pimc_horizon = 9;
        c_ew4_ds3.root_pimc_ds_target = 3;
        c_ew4_ds3.root_pimc_width = 16;
        c_ew4_ds3.root_pimc_weight_pct = 100;
        c_ew4_ds3.root_pimc_steady_g4 = false;

        let mut c_cons_ds3 = c_ew4_ds3.clone();
        c_cons_ds3.weights = c_cons.weights.clone();

        if which == "all" || which == "ds3" {
            eval_seeds("2_ew4_ds3_w6_b16", &seeds2, &c_ew4_ds3);
            eval_seeds("3_cons22_ds3_w6_b16", &seeds2, &c_cons_ds3);
        }

        let mut c_sp_ewint = c_ew4_ds3.clone();
        c_sp_ewint.spatial_model = Some(sm.clone());
        c_sp_ewint.spatial_rerank_only = true;
        c_sp_ewint.spatial_fast_ew_internal = true;

        let mut c_sp_c22 = c_cons_ds3.clone();
        c_sp_c22.spatial_model = Some(sm.clone());
        c_sp_c22.spatial_rerank_only = true;
        c_sp_c22.spatial_fast_ew_internal = true;

        let mut c_sp_flat = c_ew4_ds3.clone();
        c_sp_flat.root_pimc_ds_target = 0;
        c_sp_flat.root_pimc_horizon = 9;
        c_sp_flat.spatial_model = Some(sm.clone());
        c_sp_flat.spatial_rerank_only = true;
        c_sp_flat.spatial_fast_ew_internal = true;

        if which == "all" || which == "sp" {
            eval_seeds("4_sp_ew4_ds3_ewint", &seeds2, &c_sp_ewint);
            eval_seeds("4b_sp_c22_ds3_ewint", &seeds2, &c_sp_c22);
            eval_seeds("5_sp_ew4_flat9_ewint", &seeds2, &c_sp_flat);
        }

        let mut c_sp_direct = c_ew4.clone();
        c_sp_direct.spatial_model = Some(sm.clone());
        c_sp_direct.spatial_rerank_only = false;
        c_sp_direct.simulate_cheese_spawn = true;
        c_sp_direct.root_pimc_worlds = 0;
        if which == "all" || which == "sp" {
            eval_seeds("6_sp_direct_b64_sim", &seeds2, &c_sp_direct);
        }
        if which == "sp2" {
            let sm_ep0 = Arc::new(
                SpatialValueModel::load_from_file("models/spatial_wg2_ep0.txt")
                    .expect("load spatial_wg2_ep0"),
            );
            let mut c_ep0_b64 = c_sp_direct.clone();
            c_ep0_b64.spatial_model = Some(sm_ep0.clone());
            eval_seeds("1_ep0_direct_b64_sim", &seeds2, &c_ep0_b64);

            let mut c_ep0_b120 = c_ep0_b64.clone();
            c_ep0_b120.beam_width = 120;
            c_ep0_b120.max_per_root = 24;
            eval_seeds("2_ep0_direct_b120_sim", &seeds2, &c_ep0_b120);

            let mut c_wg2_b120 = c_sp_direct.clone();
            c_wg2_b120.beam_width = 120;
            c_wg2_b120.max_per_root = 24;
            eval_seeds("3_wg2_direct_b120_sim", &seeds2, &c_wg2_b120);

            let mut c_ep0_splin_ds3 = c_sp_ewint.clone();
            c_ep0_splin_ds3.spatial_model = Some(sm_ep0);
            c_ep0_splin_ds3.spatial_fast_ew_internal = false;
            eval_seeds("4_ep0_splin_ds3", &seeds2, &c_ep0_splin_ds3);

            let mut c_wg2_splin_ds3 = c_sp_ewint.clone();
            c_wg2_splin_ds3.spatial_model = Some(sm.clone());
            c_wg2_splin_ds3.spatial_fast_ew_internal = false;
            eval_seeds("5_wg2_splin_ds3", &seeds2, &c_wg2_splin_ds3);
        }
        if which == "splin" {
            let mut c_splin_ds3 = c_sp_ewint.clone();
            c_splin_ds3.spatial_model = Some(sm.clone());
            c_splin_ds3.spatial_fast_ew_internal = false;
            c_splin_ds3.spatial_no_pe_internal = false;
            eval_seeds("splin_ds3_pe", &seeds2, &c_splin_ds3);

            let mut c_splin_flat = c_sp_flat.clone();
            c_splin_flat.spatial_model = Some(sm.clone());
            c_splin_flat.spatial_fast_ew_internal = false;
            c_splin_flat.spatial_no_pe_internal = false;
            eval_seeds("splin_flat9_pe", &seeds2, &c_splin_flat);
        }
        if which == "splin_nope" {
            let mut c_splin_ds3 = c_sp_ewint.clone();
            c_splin_ds3.spatial_model = Some(sm.clone());
            c_splin_ds3.spatial_fast_ew_internal = false;
            c_splin_ds3.spatial_no_pe_internal = true;
            eval_seeds("splin_ds3_nope", &seeds2, &c_splin_ds3);

            let mut c_splin_flat = c_sp_flat.clone();
            c_splin_flat.spatial_model = Some(sm);
            c_splin_flat.spatial_fast_ew_internal = false;
            c_splin_flat.spatial_no_pe_internal = true;
            eval_seeds("splin_flat9_nope", &seeds2, &c_splin_flat);
        }
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
        c_fast.rollout_top_k = 0;
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
        c_med.rollout_top_k = 0;
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
            c_sp_ds3.rollout_top_k = 0;
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
