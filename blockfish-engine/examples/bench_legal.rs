use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{cheese_val::FEAT_DIM, BeamConfig, FastAI, LearnedValueModel},
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

    if mode == "dev40" {
        let mut seeds40 = Vec::new();
        seeds40.extend_from_slice(&dev20);
        seeds40.extend_from_slice(&train20);
        println!("=== Legal 40-seed baseline (1..=20 + 101..=120) ===");
        eval_seeds("h2stk03_nog0_b120", &seeds40, &make_cfg(m_h2_nog0));
        return;
    }
}
