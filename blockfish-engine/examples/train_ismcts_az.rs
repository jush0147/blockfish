//! Small-Scale SO-ISMCTS k-Step Anchor-Based Value Iteration (`train_ismcts_az.rs`).
//!
//! Scientific rigor & guardrails:
//! 1. **Explicit Anchor-Based Target**: Value (`SpatialValueModel` FULL-MLP) is trained on
//!    `k`-step bootstrapped returns `G_t^(k) = k + V_anchor(s_{t+k})` (switching to exact
//!    `total_pieces - t` when `t + k >= total_pieces`), explicitly labeled as an anchor-based
//!    target (`k ∈ {8, 12, 16}`).
//! 2. **Topout & High-Stack Risk Constraint + Safety Diagnostics**:
//!    - Topout games are NOT silently discarded without risk feedback: the final 12 states of
//!      any topout trajectory receive a one-sided risk floor `V(s) >= V_anchor(s) + 2.0`.
//!    - High-stack (`max_h >= 11`) states in completed games also receive a one-sided safety
//!      floor `V(s) >= V_anchor(s)` so the model never learns to prefer tall/dangerous stacks.
//!    - Validation always reports `top%`, mean peak height `peak_h`, `h>=12%`, and `cov_nxt%`.
//! 3. **Strict Seed Hygiene & Meaningful Acceptance Gate**:
//!    - Self-play training seeds: `50001..`
//!    - Fresh Dev set (`dev20`): `30001..=30020` (never `20001..=20010`)
//!    - Sealed Final Test set (`test20`): `40001..=40020` (untouched until dev passes!)
//!    - Candidate accepted ONLY if `top == 0` AND `mean_delta <= -1.50` pieces vs Zero-Training BASE.

use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard},
        spatial_val::{extract_spatial_features_ex, DENSE_DIM, SPARSE_DIM, SPATIAL_H1, SPATIAL_H2},
        BagTracker, GameContext, IsMctsConfig, IsMctsSearcher, SpatialValueModel,
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

fn max_board_height(stacker: &Stacker) -> u16 {
    let mut mh = 0u16;
    for ((r, _), _) in stacker.matrix() {
        mh = mh.max(r + 1);
    }
    mh
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

fn avail_mask_from_vis(hold: Option<u8>, vis_q: &[u8], bag_mask: u8) -> u8 {
    let mut mask = 0u8;
    if let Some(h) = hold {
        mask |= 1u8 << h;
    }
    for &p in vis_q {
        mask |= 1u8 << p;
    }
    if vis_q.len() < 3 {
        mask |= bag_mask;
    }
    if mask == 0 {
        0x7F
    } else {
        mask
    }
}

#[derive(Clone)]
struct RawState {
    active_bits: Vec<u16>,
    dense: [f32; DENSE_DIM],
    anchor_v: f32,
    max_h: u16,
}

#[derive(Clone)]
struct ValueSample {
    active_bits: Vec<u16>,
    dense: [f32; DENSE_DIM],
    target_val: f32,
    /// If true, apply one-sided risk floor loss: penalize ONLY when `pred < target_val`.
    is_risk_floor: bool,
}

struct SelfPlayGame {
    pieces: usize,
    topout: bool,
    states: Vec<RawState>,
}

fn play_selfplay_game(
    seed: u64,
    cfg: &IsMctsConfig,
    anchor: &SpatialValueModel,
) -> SelfPlayGame {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut searcher = IsMctsSearcher::new_srs();
    let mut bag_tracker = BagTracker::new();

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut states: Vec<RawState> = Vec::with_capacity(220);
    let mut active_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let mh = max_board_height(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let bag_mask = bag_tracker.observe(snap.hold, &snap.queue);
        let ctx = GameContext {
            garbage_rows: Some(g_rows),
            remaining_garbage: Some(rem_g),
            next_bag_mask: Some(bag_mask),
        };

        let fb = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
        let q_len = snap.queue.len().min(6);
        let mut vis_q = [0u8; 6];
        for i in 0..q_len {
            vis_q[i] = color_to_idx(snap.queue[i]) as u8;
        }
        let hold0 = snap.hold.map(|c| color_to_idx(c) as u8);
        let in_hand0 = hold0.or_else(|| if q_len > 0 { Some(vis_q[0]) } else { None });
        let avail0 = avail_mask_from_vis(hold0, &vis_q[..q_len], bag_mask);
        let n_act = extract_spatial_features_ex(
            &fb,
            rem_g,
            in_hand0,
            avail0,
            true,
            &mut active_buf,
            &mut dense_buf,
        );
        let act_vec = active_buf[..n_act].to_vec();
        let anc_v = anchor.predict_extracted(&act_vec, &dense_buf);
        states.push(RawState {
            active_bits: act_vec,
            dense: dense_buf,
            anchor_v: anc_v,
            max_h: mh,
        });

        let r = searcher.search_with_ctx(&snap, ctx, cfg);
        if r.inputs.is_empty() {
            break;
        }
        let mut inputs = r.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (_lines, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
        if pieces >= 400 {
            break;
        }
    }

    SelfPlayGame {
        pieces,
        topout: ds_total < 100,
        states,
    }
}

/// Build `k`-step anchor-based Value targets + one-sided high-stack/topout risk-floor samples.
fn build_kstep_samples(games: &[SelfPlayGame], k_step: usize) -> (Vec<ValueSample>, usize, usize) {
    let mut out = Vec::new();
    let mut n_kstep = 0usize;
    let mut n_risk = 0usize;

    for g in games {
        let len = g.states.len();
        if g.topout {
            // Topout risk constraint: final 12 states before topout get a one-sided risk floor
            // `V(s) >= V_anchor(s) + 2.0` so failed states are never made optimistically cheap.
            let start = len.saturating_sub(12);
            for s in &g.states[start..] {
                out.push(ValueSample {
                    active_bits: s.active_bits.clone(),
                    dense: s.dense,
                    target_val: s.anchor_v + 2.0,
                    is_risk_floor: true,
                });
                n_risk += 1;
            }
            continue;
        }

        for t in 0..len {
            let s = &g.states[t];
            let target = if t + k_step < len {
                (k_step as f32) + g.states[t + k_step].anchor_v
            } else {
                (g.pieces - t) as f32
            };
            out.push(ValueSample {
                active_bits: s.active_bits.clone(),
                dense: s.dense,
                target_val: target,
                is_risk_floor: false,
            });
            n_kstep += 1;

            // High-stack safety floor on completed games: prevent pushing tall stacks below anchor.
            if s.max_h >= 11 {
                out.push(ValueSample {
                    active_bits: s.active_bits.clone(),
                    dense: s.dense,
                    target_val: s.anchor_v,
                    is_risk_floor: true,
                });
                n_risk += 1;
            }
        }
    }
    (out, n_kstep, n_risk)
}

#[derive(Clone, Debug)]
struct ValSummary {
    mean: f64,
    top: usize,
    n: usize,
    peak_h_mean: f64,
    h12_pct: f64,
    cov_pct: f64,
    wall_s: f64,
    per_seed: Vec<(u64, usize, bool)>,
}

fn eval_val_seeds(seeds: &[u64], cfg: &IsMctsConfig) -> ValSummary {
    let t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<(usize, bool, u16, usize, usize)>> = vec![None; seeds.len()];

    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                let mut searcher = IsMctsSearcher::new_srs();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    let mut game_cfg = BSConfig::default();
                    game_cfg.garbage.total_lines = Some(100);
                    game_cfg.prng_seed = Some(seeds[i]);
                    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
                    let mut bag_tracker = BagTracker::new();
                    let mut ds_total = 0usize;
                    let mut pieces = 0usize;
                    let mut peak_h = 0u16;
                    let mut h12_cnt = 0usize;
                    let mut cov_cnt = 0usize;
                    while ds_total < 100 && stacker.current_piece_type().is_some() {
                        let snap = stacker.snapshot().unwrap();
                        let g_rows = garbage_rows(&stacker);
                        let mh = max_board_height(&stacker);
                        peak_h = peak_h.max(mh);
                        if mh >= 12 {
                            h12_cnt += 1;
                        }
                        let rem_g = 100usize.saturating_sub(ds_total) as u16;
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
                        let mut inputs = r.inputs;
                        assert_eq!(inputs.pop(), Some(Input::HD));
                        stacker.run(inputs);
                        let (_lines, ds) = stacker.hard_drop();
                        pieces += 1;
                        ds_total += ds;
                        if ds > 0 && h1_blocks_over(&stacker) > 0 {
                            cov_cnt += 1;
                        }
                        if pieces >= 400 {
                            break;
                        }
                    }
                    local.push((i, (pieces, ds_total < 100, peak_h, h12_cnt, cov_cnt)));
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

    let mut ok_pcs = Vec::new();
    let mut top = 0usize;
    let mut peak_sum = 0.0f64;
    let mut h12_sum = 0usize;
    let mut tot_moves = 0usize;
    let mut cov_sum = 0.0f64;
    let mut per_seed = Vec::with_capacity(seeds.len());

    for (idx, item) in out.into_iter().enumerate() {
        let (p, is_top, pk, h12, cov) = item.unwrap();
        per_seed.push((seeds[idx], p, is_top));
        peak_sum += pk as f64;
        h12_sum += h12;
        tot_moves += p;
        if is_top {
            top += 1;
        } else {
            ok_pcs.push(p);
            cov_sum += cov as f64;
        }
    }
    let n = seeds.len();
    let mean = if ok_pcs.is_empty() {
        f64::NAN
    } else {
        ok_pcs.iter().sum::<usize>() as f64 / ok_pcs.len() as f64
    };
    ValSummary {
        mean,
        top,
        n,
        peak_h_mean: peak_sum / n.max(1) as f64,
        h12_pct: 100.0 * (h12_sum as f64) / tot_moves.max(1) as f64,
        cov_pct: if ok_pcs.is_empty() {
            0.0
        } else {
            cov_sum / ok_pcs.len() as f64
        },
        wall_s: t0.elapsed().as_secs_f64(),
        per_seed,
    }
}

fn paired_stats(cand: &ValSummary, base: &ValSummary) -> (f64, f64, usize, usize, usize) {
    let mut diffs = Vec::new();
    let mut w = 0usize;
    let mut t = 0usize;
    let mut l = 0usize;
    for (a, b) in cand.per_seed.iter().zip(base.per_seed.iter()) {
        if !a.2 && !b.2 {
            let d = a.1 as f64 - b.1 as f64;
            diffs.push(d);
            if a.1 < b.1 {
                w += 1;
            } else if a.1 == b.1 {
                t += 1;
            } else {
                l += 1;
            }
        }
    }
    if diffs.is_empty() {
        return (f64::NAN, f64::NAN, 0, 0, 0);
    }
    let n = diffs.len() as f64;
    let m = diffs.iter().sum::<f64>() / n;
    let var = if diffs.len() > 1 {
        diffs.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    (m, (var / n).sqrt(), w, t, l)
}

fn train_value_kstep(
    init: &SpatialValueModel,
    anchor: &SpatialValueModel,
    samples: &[ValueSample],
    epochs: usize,
    lr: f32,
    anchor_l2: f32,
) -> (SpatialValueModel, f64, f64) {
    let mut m = init.clone();
    m.fast_ready = false;

    // rem_g-center non-risk samples so global rem_g scale stays locked to anchor
    let mut sum_diff = [0.0f64; 105];
    let mut cnt_rg = [0usize; 105];
    for s in samples {
        if s.is_risk_floor {
            continue;
        }
        let rg = (s.dense[0].round() as usize).min(104);
        let p_anc = anchor.predict_extracted(&s.active_bits, &s.dense);
        sum_diff[rg] += (s.target_val - p_anc) as f64;
        cnt_rg[rg] += 1;
    }
    let mut shift_rg = [0.0f32; 105];
    for rg in 0..105 {
        if cnt_rg[rg] > 0 {
            shift_rg[rg] = (sum_diff[rg] / cnt_rg[rg] as f64) as f32;
        }
    }

    let mut order: Vec<usize> = (0..samples.len()).collect();
    let mut rng: u64 = 0x9E3779B97F4A7C15 ^ (samples.len() as u64);
    let mut init_loss = 0.0f64;
    let mut final_loss = 0.0f64;
    let huber_delta = 2.0f32;

    for ep in 0..epochs {
        for i in (1..order.len()).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((rng >> 32) as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut loss_sum = 0.0f64;
        let mut loss_cnt = 0usize;
        for &si in &order {
            let s = &samples[si];
            let mut lin = m.lin_bias;
            let mut h1_pre = m.b1;
            for &bit in &s.active_bits {
                let idx = bit as usize;
                lin += m.lin_sparse[idx];
                if anchor.fast_sparse_active[idx] {
                    let row = &m.w_sparse[idx];
                    for h in 0..SPATIAL_H1 {
                        h1_pre[h] += row[h];
                    }
                }
            }
            let mut z_vec = [0.0f32; DENSE_DIM];
            let mut zm_vec = [0.0f32; DENSE_DIM];
            for j in 0..DENSE_DIM {
                let z = (s.dense[j] - m.dense_mean[j]) * m.dense_inv_std[j];
                z_vec[j] = z;
                lin += m.lin_dense[j] * z;
                if anchor.fast_dense_active[j] {
                    let zm = z.clamp(-3.0, 3.0);
                    zm_vec[j] = zm;
                    if zm != 0.0 {
                        let row = &m.w_dense[j];
                        for h in 0..SPATIAL_H1 {
                            h1_pre[h] += zm * row[h];
                        }
                    }
                }
            }
            let mut h1_act = [0.0f32; SPATIAL_H1];
            let mut h2_pre = m.b2;
            for i in 0..SPATIAL_H1 {
                let a1 = h1_pre[i].clamp(0.0, 6.0);
                h1_act[i] = a1;
                if a1 > 0.0 {
                    let row = &m.w2[i];
                    for k in 0..SPATIAL_H2 {
                        h2_pre[k] += a1 * row[k];
                    }
                }
            }
            let mut h2_act = [0.0f32; SPATIAL_H2];
            let mut mlp = m.b3;
            for k in 0..SPATIAL_H2 {
                let a2 = h2_pre[k].clamp(0.0, 6.0);
                h2_act[k] = a2;
                mlp += a2 * m.w3[k];
            }
            let denom = 1.0 + mlp.abs() / 5.0;
            let mlp_sat = mlp / denom;
            let pred = lin + mlp_sat;

            let d_pred = if s.is_risk_floor {
                // One-sided safety floor: penalize ONLY if `pred < target_val`
                if pred < s.target_val {
                    let err = pred - s.target_val; // negative
                    err.clamp(-huber_delta, 0.0)
                } else {
                    continue;
                }
            } else {
                let rg = (s.dense[0].round() as usize).min(104);
                let target_cal = s.target_val - shift_rg[rg];
                let err = pred - target_cal;
                let abs_e = err.abs();
                let (loss, dp) = if abs_e <= huber_delta {
                    (0.5 * err * err, err)
                } else {
                    (
                        huber_delta * (abs_e - 0.5 * huber_delta),
                        huber_delta * err.signum(),
                    )
                };
                loss_sum += loss as f64;
                loss_cnt += 1;
                dp
            };

            let d_lin = d_pred;
            let d_mlp = d_pred / (denom * denom);

            for &bit in &s.active_bits {
                let idx = bit as usize;
                m.lin_sparse[idx] -=
                    lr * (d_lin + anchor_l2 * (m.lin_sparse[idx] - anchor.lin_sparse[idx]));
            }
            for j in 1..DENSE_DIM {
                let z = z_vec[j];
                if z != 0.0 {
                    m.lin_dense[j] -=
                        lr * (d_lin * z + anchor_l2 * (m.lin_dense[j] - anchor.lin_dense[j]));
                }
            }

            m.b3 -= lr * (d_mlp + anchor_l2 * (m.b3 - anchor.b3));
            let mut d_h2 = [0.0f32; SPATIAL_H2];
            for k in 0..SPATIAL_H2 {
                let w3_old = m.w3[k];
                if h2_act[k] > 0.0 {
                    m.w3[k] -= lr * (d_mlp * h2_act[k] + anchor_l2 * (w3_old - anchor.w3[k]));
                }
                if h2_pre[k] > 0.0 && h2_pre[k] < 6.0 {
                    d_h2[k] = d_mlp * w3_old;
                }
            }

            let mut d_h1 = [0.0f32; SPATIAL_H1];
            for k in 0..SPATIAL_H2 {
                if d_h2[k] != 0.0 {
                    m.b2[k] -= lr * (d_h2[k] + anchor_l2 * (m.b2[k] - anchor.b2[k]));
                }
            }
            for i in 0..SPATIAL_H1 {
                let a1 = h1_act[i];
                let mut acc = 0.0f32;
                for k in 0..SPATIAL_H2 {
                    let g = d_h2[k];
                    if g != 0.0 {
                        let w2_old = m.w2[i][k];
                        acc += g * w2_old;
                        if a1 > 0.0 {
                            m.w2[i][k] -= lr * (g * a1 + anchor_l2 * (w2_old - anchor.w2[i][k]));
                        }
                    }
                }
                if h1_pre[i] > 0.0 && h1_pre[i] < 6.0 {
                    d_h1[i] = acc;
                }
            }

            for h in 0..SPATIAL_H1 {
                let g = d_h1[h];
                if g != 0.0 {
                    m.b1[h] -= lr * (g + anchor_l2 * (m.b1[h] - anchor.b1[h]));
                }
            }
            for &bit in &s.active_bits {
                let idx = bit as usize;
                if !anchor.fast_sparse_active[idx] {
                    continue;
                }
                for h in 0..SPATIAL_H1 {
                    let g = d_h1[h];
                    if g != 0.0 {
                        m.w_sparse[idx][h] -=
                            lr * (g + anchor_l2 * (m.w_sparse[idx][h] - anchor.w_sparse[idx][h]));
                    }
                }
            }
            for j in 0..DENSE_DIM {
                if !anchor.fast_dense_active[j] {
                    continue;
                }
                let zm = zm_vec[j];
                if zm != 0.0 {
                    for h in 0..SPATIAL_H1 {
                        let g = d_h1[h];
                        if g != 0.0 {
                            m.w_dense[j][h] -= lr
                                * (g * zm
                                    + anchor_l2 * (m.w_dense[j][h] - anchor.w_dense[j][h]));
                        }
                    }
                }
            }
        }
        let avg_l = loss_sum / loss_cnt.max(1) as f64;
        if ep == 0 {
            init_loss = avg_l;
        }
        final_loss = avg_l;
    }

    m.prepare_fast();
    (m, init_loss, final_loss)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base_model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/spatial_bc4.txt");
    let sp_games: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let sp_sims: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(800);
    let dev_n: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20);
    let dev_sims: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1200);

    let anchor_sm =
        Arc::new(SpatialValueModel::load_from_file(base_model_path).expect("load spatial_bc4"));
    // Fresh Dev set (`dev20`: 30001..=30020). Sealed Final Test (`test20`: 40001..=40020) is UNTOUCHED!
    let dev_seeds: Vec<u64> = (30001..30001 + dev_n as u64).collect();

    println!("=== SO-ISMCTS k-Step Anchor-Based Value Iteration (k = 8, 12, 16) ===");
    println!("Target: ANCHOR-BASED k-step return G_t^(k) = k + V_anchor(s_{{t+k}}) + safety risk floor");
    println!(
        "Base: {} | SelfPlay: {} games (50001.., s={}) | Fresh Dev: 30001..={} (n={}, s={}) | Sealed Test: 40001..=40020 (UNTOUCHED)\n",
        base_model_path,
        sp_games,
        sp_sims,
        30001 + dev_n as u64 - 1,
        dev_n,
        dev_sims
    );

    let make_cfg = |sims: usize, val_m: Arc<SpatialValueModel>| IsMctsConfig {
        simulations: sims,
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
        use_endgame_solver: true,
        use_transposition: true,
        spatial_model: Some(val_m),
        policy_prior: None,
        ..Default::default()
    };

    // 1. Zero-Training Anchor BASE on Fresh Dev set
    let base_res = eval_val_seeds(&dev_seeds, &make_cfg(dev_sims, anchor_sm.clone()));
    println!(
        "[Zero-Train BASE] dev_mean={:6.2} top={:2}/{} peak_h={:4.1} h12={:4.1}% cov={:4.1}% | wall={:5.1}s",
        base_res.mean,
        base_res.top,
        base_res.n,
        base_res.peak_h_mean,
        base_res.h12_pct,
        base_res.cov_pct,
        base_res.wall_s
    );

    // 2. Generate one shared pool of Legal p=5 SO-ISMCTS self-play games
    let t_sp = Instant::now();
    let sp_cfg = make_cfg(sp_sims, anchor_sm.clone());
    let sp_seeds: Vec<u64> = (50001..50001 + sp_games as u64).collect();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(sp_games.max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut games_out: Vec<Option<SelfPlayGame>> = (0..sp_games).map(|_| None).collect();

    std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for _ in 0..workers {
            handles.push(sc.spawn(|| {
                let mut local = Vec::new();
                loop {
                    let i = next_idx.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= sp_seeds.len() {
                        break;
                    }
                    local.push((i, play_selfplay_game(sp_seeds[i], &sp_cfg, &anchor_sm)));
                }
                local
            }));
        }
        for h in handles {
            for (i, g) in h.join().unwrap() {
                games_out[i] = Some(g);
            }
        }
    });
    let games: Vec<SelfPlayGame> = games_out.into_iter().map(|x| x.unwrap()).collect();
    let sp_wall = t_sp.elapsed().as_secs_f64();
    let sp_ok: Vec<&SelfPlayGame> = games.iter().filter(|g| !g.topout).collect();
    let sp_mean = sp_ok.iter().map(|g| g.pieces).sum::<usize>() as f64 / sp_ok.len().max(1) as f64;
    println!(
        "\nSelf-Play Pool: {}/{} ok (top={}) mean={:.2} | wall={:.1}s ({:.0} g/hr)\n",
        sp_ok.len(),
        sp_games,
        sp_games - sp_ok.len(),
        sp_mean,
        sp_wall,
        (sp_games as f64) * 3600.0 / sp_wall.max(1e-6)
    );

    // 3. Controlled comparison of k = 8, 12, 16 (anchor-based k-step targets + risk floor)
    const ACCEPT_DELTA: f64 = -1.50;
    for &k_step in &[8usize, 12, 16] {
        let (samples, n_k, n_r) = build_kstep_samples(&games, k_step);
        let (cand_m, l0, l1) =
            train_value_kstep(&anchor_sm, &anchor_sm, &samples, 3, 2e-5, 0.05);
        let cand_res = eval_val_seeds(&dev_seeds, &make_cfg(dev_sims, Arc::new(cand_m)));
        let (md, se, w, t, l) = paired_stats(&cand_res, &base_res);
        let accepted = cand_res.top == 0 && md <= ACCEPT_DELTA;
        println!(
            "[k={:2} anchor-boot] n_k={} n_risk={} Huber {:.4}->{:.4} | dev_mean={:6.2} top={:2}/{} peak_h={:4.1} h12={:4.1}% cov={:4.1}% | vsBASE={:+.2}±{:.2} 95%CI=[{:+.2},{:+.2}] W/T/L={}/{}/{} | {}",
            k_step,
            n_k,
            n_r,
            l0,
            l1,
            cand_res.mean,
            cand_res.top,
            cand_res.n,
            cand_res.peak_h_mean,
            cand_res.h12_pct,
            cand_res.cov_pct,
            md,
            se,
            md - 1.96 * se,
            md + 1.96 * se,
            w,
            t,
            l,
            if accepted { "ACCEPTED" } else { "REJECTED" }
        );
    }
}
