//! Small-Scale SO-ISMCTS Self-Play Policy / Value Iteration (`train_ismcts_az.rs`).
//!
//! Strict scientific design:
//! 1. **Decoupled Value & Policy**:
//!    - Value (`SpatialValueModel` FULL-MLP) is trained PURELY on completed-game ground-truth
//!      remaining pieces `z_t = (total_pieces - t) as f32` (Huber loss + L2 anchor to `bc4`).
//!      Topout trajectories are strictly excluded from Value regression and reported separately.
//!    - Policy Prior (separate `SpatialValueModel` linear head `lin_sparse + lin_dense`, plugged
//!      into `IsMctsConfig::policy_prior`) is trained PURELY on Cross-Entropy against SO-ISMCTS
//!      root visit distributions `pi_mcts(a) = N(root, a) / sum N` (+ L2 anchor to `bc4`).
//! 2. **Strict Independent Held-Out Validation & Non-Regression Gate**:
//!    - Self-play seeds (`50001..`) are strictly disjoint from validation seeds (`20001..`).
//!    - Every round independently ablates `Value-Only`, `Policy-Only`, and `Value+Policy` on
//!      actual Legal `p=5` games and ONLY accepts a new model if `top == 0` and held-out mean
//!      strictly improves over the current best (anchored by Frozen `spatial_bc4`).

use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastShapeTable},
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

#[derive(Clone)]
struct ValueSample {
    active_bits: Vec<u16>,
    dense: [f32; DENSE_DIM],
    target_z: f32,
}

#[derive(Clone)]
struct PolicyCandFeat {
    active_bits: Vec<u16>,
    dense: [f32; DENSE_DIM],
    target_prob: f32,
}

#[derive(Clone)]
struct PolicySample {
    cands: Vec<PolicyCandFeat>,
}

struct SelfPlayGame {
    pieces: usize,
    topout: bool,
    val_samples: Vec<ValueSample>,
    pol_samples: Vec<PolicySample>,
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

fn play_selfplay_game(seed: u64, cfg: &IsMctsConfig, fst: &FastShapeTable) -> SelfPlayGame {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut searcher = IsMctsSearcher::new_srs();
    let mut bag_tracker = BagTracker::new();

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut raw_val: Vec<(Vec<u16>, [f32; DENSE_DIM], usize)> = Vec::with_capacity(220);
    let mut pol_samples: Vec<PolicySample> = Vec::with_capacity(200);
    let mut active_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let bag_mask = bag_tracker.observe(snap.hold, &snap.queue);
        let ctx = GameContext {
            garbage_rows: Some(g_rows),
            remaining_garbage: Some(rem_g),
            next_bag_mask: Some(bag_mask),
        };

        // Extract current state features for Value regression (completed games only!)
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
        raw_val.push((active_buf[..n_act].to_vec(), dense_buf, pieces));

        let r = searcher.search_with_ctx(&snap, ctx, cfg);
        if r.inputs.is_empty() {
            break;
        }

        // Record Policy visit distribution sample when MCTS ran (non-endgame, >= 2 candidates)
        if !r.used_endgame_solver && r.root_candidates.len() >= 2 {
            let total_v: u32 = r.root_candidates.iter().map(|x| x.1).sum();
            if total_v >= 32 {
                let cur0 = if q_len > 0 { Some(vis_q[0]) } else { None };
                let mut cands = Vec::with_capacity(r.root_candidates.len());
                for &(fp, vis, _, _) in &r.root_candidates {
                    let mut b2 = fb;
                    let orient = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                    let rem2 = rem_g.saturating_sub(ds as u16);
                    if rem2 == 0 || b2.rows >= 20 {
                        continue;
                    }
                    let (nh, consumed) = if hold0.is_some() {
                        if fp.did_hold {
                            (cur0, usize::from(cur0.is_some()))
                        } else {
                            (hold0, 1.min(q_len))
                        }
                    } else if fp.did_hold {
                        (cur0, 2.min(q_len))
                    } else {
                        (None, 1.min(q_len))
                    };
                    let rem_vis = &vis_q[consumed.min(q_len)..q_len];
                    let ih = nh.or_else(|| rem_vis.first().copied());
                    let av = avail_mask_from_vis(nh, rem_vis, bag_mask);
                    let na = extract_spatial_features_ex(
                        &b2,
                        rem2,
                        ih,
                        av,
                        false, // policy linear head uses no_pe features
                        &mut active_buf,
                        &mut dense_buf,
                    );
                    cands.push(PolicyCandFeat {
                        active_bits: active_buf[..na].to_vec(),
                        dense: dense_buf,
                        target_prob: (vis as f32) / (total_v as f32),
                    });
                }
                if cands.len() >= 2 {
                    let sum_p: f32 = cands.iter().map(|c| c.target_prob).sum();
                    if sum_p > 0.01 {
                        for c in &mut cands {
                            c.target_prob /= sum_p;
                        }
                        pol_samples.push(PolicySample { cands });
                    }
                }
            }
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

    let topout = ds_total < 100;
    let val_samples = if topout {
        // Strictly exclude topout games from Value regression!
        Vec::new()
    } else {
        raw_val
            .into_iter()
            .map(|(active_bits, dense, t)| ValueSample {
                active_bits,
                dense,
                target_z: (pieces - t) as f32,
            })
            .collect()
    };

    SelfPlayGame {
        pieces,
        topout,
        val_samples,
        pol_samples: if topout { Vec::new() } else { pol_samples },
    }
}

fn eval_val_seeds(seeds: &[u64], cfg: &IsMctsConfig) -> (f64, usize, Vec<usize>, f64) {
    let t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<(usize, bool)>> = vec![None; seeds.len()];

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
                    while ds_total < 100 && stacker.current_piece_type().is_some() {
                        let snap = stacker.snapshot().unwrap();
                        let g_rows = garbage_rows(&stacker);
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
                        if pieces >= 400 {
                            break;
                        }
                    }
                    local.push((i, (pieces, ds_total < 100)));
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
    let mut all_pcs = Vec::with_capacity(seeds.len());
    for item in out {
        let (p, is_top) = item.unwrap();
        all_pcs.push(p);
        if is_top {
            top += 1;
        } else {
            ok_pcs.push(p);
        }
    }
    let mean = if ok_pcs.is_empty() {
        f64::NAN
    } else {
        ok_pcs.iter().sum::<usize>() as f64 / ok_pcs.len() as f64
    };
    (mean, top, all_pcs, t0.elapsed().as_secs_f64())
}

/// Train Value (`SpatialValueModel` FULL-MLP) purely on completed-game `z_t` (Huber + anchor L2),
/// preserving `anchor`'s gated MLP masks and `rem_g`-centering targets so clean visited states
/// cannot drift above unvisited high-stack states.
fn train_value_model(
    init: &SpatialValueModel,
    anchor: &SpatialValueModel,
    samples: &[ValueSample],
    epochs: usize,
    lr: f32,
    anchor_l2: f32,
) -> (SpatialValueModel, f64, f64) {
    let mut m = init.clone();
    m.fast_ready = false; // train directly on raw weights, then call `prepare_fast()`
    let n = samples.len().max(1);

    // Compute per-rem_g mean(z_t - V_anchor(s)) so we keep the global rem_g scale locked to anchor
    // while still regressing strictly to true completed-game outcomes `z_t`.
    let mut sum_diff = [0.0f64; 105];
    let mut cnt_rg = [0usize; 105];
    for s in samples {
        let rg = (s.dense[0].round() as usize).min(104);
        let p_anc = anchor.predict_extracted(&s.active_bits, &s.dense);
        sum_diff[rg] += (s.target_z - p_anc) as f64;
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
    let huber_delta = 3.0f32;

    for ep in 0..epochs {
        // Fisher-Yates shuffle
        for i in (1..order.len()).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = ((rng >> 32) as usize) % (i + 1);
            order.swap(i, j);
        }

        let mut loss_sum = 0.0f64;
        for &si in &order {
            let s = &samples[si];
            let rg = (s.dense[0].round() as usize).min(104);
            let target_cal = s.target_z - shift_rg[rg];

            // Forward pass (raw un-folded, exact match to `predict_extracted`)
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

            let err = pred - target_cal;
            let abs_e = err.abs();
            let (loss, d_pred) = if abs_e <= huber_delta {
                (0.5 * err * err, err)
            } else {
                (
                    huber_delta * (abs_e - 0.5 * huber_delta),
                    huber_delta * err.signum(),
                )
            };
            loss_sum += loss as f64;

            // Backprop (SGD with anchor L2 pull toward `anchor`, preserving gated features & rem_g slope)
            let d_lin = d_pred;
            let d_mlp = d_pred / (denom * denom);

            for &bit in &s.active_bits {
                let idx = bit as usize;
                m.lin_sparse[idx] -=
                    lr * (d_lin + anchor_l2 * (m.lin_sparse[idx] - anchor.lin_sparse[idx]));
            }
            // Keep j==0 (rem_garbage slope) anchored; update geometry/stack dense weights
            for j in 1..DENSE_DIM {
                let z = z_vec[j];
                if z != 0.0 {
                    m.lin_dense[j] -=
                        lr * (d_lin * z + anchor_l2 * (m.lin_dense[j] - anchor.lin_dense[j]));
                }
            }

            // MLP layer 3 -> 2 -> 1
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
        let avg_l = loss_sum / n as f64;
        if ep == 0 {
            init_loss = avg_l;
        }
        final_loss = avg_l;
    }

    m.prepare_fast();
    (m, init_loss, final_loss)
}

/// Train separate Policy Prior (`SpatialValueModel` linear head ONLY) via Cross-Entropy
/// against SO-ISMCTS root visit distribution `pi_mcts`.
fn train_policy_prior(
    init: &SpatialValueModel,
    anchor: &SpatialValueModel,
    samples: &[PolicySample],
    epochs: usize,
    lr: f32,
    anchor_l2: f32,
    prior_temp_pieces: f32,
) -> (SpatialValueModel, f64, f64) {
    let mut m = init.clone();
    m.fast_ready = false;
    let n = samples.len().max(1);
    let inv_t = 1.0f32 / prior_temp_pieces.max(0.2);
    let mut init_ce = 0.0f64;
    let mut final_ce = 0.0f64;

    for ep in 0..epochs {
        let mut ce_sum = 0.0f64;
        for s in samples {
            let k = s.cands.len();
            let mut logits = Vec::with_capacity(k);
            let mut max_l = f32::NEG_INFINITY;
            for c in &s.cands {
                let mut lin = m.lin_bias;
                for &bit in &c.active_bits {
                    lin += m.lin_sparse[bit as usize];
                }
                for j in 0..DENSE_DIM {
                    let z = (c.dense[j] - m.dense_mean[j]) * m.dense_inv_std[j];
                    lin += m.lin_dense[j] * z;
                }
                let logit = -lin * inv_t; // lower score = higher probability
                if logit > max_l {
                    max_l = logit;
                }
                logits.push(logit);
            }
            let mut exp_s = 0.0f32;
            let mut probs = Vec::with_capacity(k);
            for &l in &logits {
                let e = (l - max_l).clamp(-30.0, 0.0).exp();
                probs.push(e);
                exp_s += e;
            }
            let inv_e = 1.0 / exp_s.max(1e-12);
            for p in &mut probs {
                *p *= inv_e;
            }
            for i in 0..k {
                let tp = s.cands[i].target_prob;
                if tp > 0.0 {
                    ce_sum -= (tp as f64) * (probs[i].max(1e-9) as f64).ln();
                }
            }

            // d(CE)/d(lin_i) = -(probs[i] - target_prob[i]) * inv_t
            for i in 0..k {
                let g_lin = -(probs[i] - s.cands[i].target_prob) * inv_t;
                if g_lin.abs() < 1e-9 {
                    continue;
                }
                let c = &s.cands[i];
                for &bit in &c.active_bits {
                    let idx = bit as usize;
                    m.lin_sparse[idx] -=
                        lr * (g_lin + anchor_l2 * (m.lin_sparse[idx] - anchor.lin_sparse[idx]));
                }
                for j in 0..DENSE_DIM {
                    let z = (c.dense[j] - m.dense_mean[j]) * m.dense_inv_std[j];
                    if z != 0.0 {
                        m.lin_dense[j] -= lr
                            * (g_lin * z + anchor_l2 * (m.lin_dense[j] - anchor.lin_dense[j]));
                    }
                }
            }
        }
        let avg_ce = ce_sum / n as f64;
        if ep == 0 {
            init_ce = avg_ce;
        }
        final_ce = avg_ce;
    }

    m.prepare_fast();
    (m, init_ce, final_ce)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let base_model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/spatial_bc4.txt");
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let sp_games: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);
    let sp_sims: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(800);
    let val_n: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);
    let val_sims: usize = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(1200);

    let anchor_sm =
        Arc::new(SpatialValueModel::load_from_file(base_model_path).expect("load spatial_bc4"));
    let val_seeds: Vec<u64> = (20001..20001 + val_n as u64).collect();
    let fst = FastShapeTable::new_srs();

    println!("=== SO-ISMCTS Self-Play Policy / Value Iteration ===");
    println!(
        "Base: {} | rounds={} | sp_games={} (sims={}) | val_seeds=20001..={} (sims={})\n",
        base_model_path,
        rounds,
        sp_games,
        sp_sims,
        20001 + val_n as u64 - 1,
        val_sims
    );

    let make_cfg = |sims: usize, val_m: Arc<SpatialValueModel>, pol_m: Option<Arc<SpatialValueModel>>| {
        IsMctsConfig {
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
            policy_prior: pol_m,
            ..Default::default()
        }
    };

    // Baseline held-out validation
    let base_val_cfg = make_cfg(val_sims, anchor_sm.clone(), None);
    let (base_mean, base_top, base_pcs, base_t) = eval_val_seeds(&val_seeds, &base_val_cfg);
    println!(
        "[Anchor BASE] val_mean={:.2} top={}/{} wall={:.1}s seeds={:?}\n",
        base_mean, base_top, val_n, base_t, base_pcs
    );

    let mut best_val = anchor_sm.clone();
    let mut best_pol: Option<Arc<SpatialValueModel>> = None;
    let mut best_mean = base_mean;
    let mut all_val_samples: Vec<ValueSample> = Vec::new();
    let mut all_pol_samples: Vec<PolicySample> = Vec::new();

    for round in 1..=rounds {
        let t_sp = Instant::now();
        let sp_cfg = make_cfg(sp_sims, best_val.clone(), best_pol.clone());
        let seed_start = 50001u64 + (round as u64 - 1) * 1000;
        let sp_seeds: Vec<u64> = (seed_start..seed_start + sp_games as u64).collect();

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
                        local.push((i, play_selfplay_game(sp_seeds[i], &sp_cfg, &fst)));
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

        let sp_wall = t_sp.elapsed().as_secs_f64();
        let mut sp_ok_sum = 0usize;
        let mut sp_ok_n = 0usize;
        let mut sp_top = 0usize;
        let mut round_v = 0usize;
        let mut round_p = 0usize;
        for g in games_out.into_iter().map(|x| x.unwrap()) {
            if g.topout {
                sp_top += 1;
            } else {
                sp_ok_n += 1;
                sp_ok_sum += g.pieces;
                round_v += g.val_samples.len();
                round_p += g.pol_samples.len();
                all_val_samples.extend(g.val_samples);
                all_pol_samples.extend(g.pol_samples);
            }
        }
        let sp_mean = sp_ok_sum as f64 / sp_ok_n.max(1) as f64;
        println!(
            "--- Round {} Self-Play: {}/{} ok (top={}) mean={:.2} | +{} val, +{} pol (buf: {}/{}) | wall={:.1}s ({:.0} g/hr) ---",
            round,
            sp_ok_n,
            sp_games,
            sp_top,
            sp_mean,
            round_v,
            round_p,
            all_val_samples.len(),
            all_pol_samples.len(),
            sp_wall,
            (sp_games as f64) * 3600.0 / sp_wall.max(1e-6)
        );

        // Train separate Value (on completed-game z_t) & separate Policy Prior (on pi_mcts)
        let (cand_val, v_l0, v_l1) =
            train_value_model(&best_val, &anchor_sm, &all_val_samples, 3, 2e-5, 0.02);
        let init_pol = best_pol.as_deref().unwrap_or(&anchor_sm);
        let (cand_pol, p_ce0, p_ce1) =
            train_policy_prior(init_pol, &anchor_sm, &all_pol_samples, 3, 2e-4, 0.02, 1.2);
        println!(
            "  Train: Value Huber {:.4} -> {:.4} | Policy CE {:.4} -> {:.4}",
            v_l0, v_l1, p_ce0, p_ce1
        );

        let cand_val_arc = Arc::new(cand_val);
        let cand_pol_arc = Arc::new(cand_pol);

        // Strict independent held-out Legal p=5 ablation: Value-Only vs Policy-Only vs Both
        let (m_v, t_v, p_v, w_v) = eval_val_seeds(
            &val_seeds,
            &make_cfg(val_sims, cand_val_arc.clone(), best_pol.clone()),
        );
        let (m_p, t_p, p_p, w_p) = eval_val_seeds(
            &val_seeds,
            &make_cfg(val_sims, best_val.clone(), Some(cand_pol_arc.clone())),
        );
        let (m_vp, t_vp, p_vp, w_vp) = eval_val_seeds(
            &val_seeds,
            &make_cfg(val_sims, cand_val_arc.clone(), Some(cand_pol_arc.clone())),
        );

        println!(
            "  Val [Value-Only]:  mean={:.2} ({:+.2} vs best) top={}/{} wall={:.1}s seeds={:?}",
            m_v,
            m_v - best_mean,
            t_v,
            val_n,
            w_v,
            p_v
        );
        println!(
            "  Val [Policy-Only]: mean={:.2} ({:+.2} vs best) top={}/{} wall={:.1}s seeds={:?}",
            m_p,
            m_p - best_mean,
            t_p,
            val_n,
            w_p,
            p_p
        );
        println!(
            "  Val [Val+Policy]:  mean={:.2} ({:+.2} vs best) top={}/{} wall={:.1}s seeds={:?}",
            m_vp,
            m_vp - best_mean,
            t_vp,
            val_n,
            w_vp,
            p_vp
        );

        // Strict non-regression acceptance gate (must have top == 0 and strictly lower mean!)
        let mut best_cand = (best_mean, 0usize); // 0=keep, 1=val, 2=pol, 3=both
        if t_v == 0 && m_v < best_cand.0 - 0.05 {
            best_cand = (m_v, 1);
        }
        if t_p == 0 && m_p < best_cand.0 - 0.05 {
            best_cand = (m_p, 2);
        }
        if t_vp == 0 && m_vp < best_cand.0 - 0.05 {
            best_cand = (m_vp, 3);
        }
        match best_cand.1 {
            1 => {
                best_val = cand_val_arc;
                best_mean = best_cand.0;
                println!("  >>> ACCEPTED NEW BEST (Value-Only): {:.2}\n", best_mean);
            }
            2 => {
                best_pol = Some(cand_pol_arc);
                best_mean = best_cand.0;
                println!("  >>> ACCEPTED NEW BEST (Policy-Only): {:.2}\n", best_mean);
            }
            3 => {
                best_val = cand_val_arc;
                best_pol = Some(cand_pol_arc);
                best_mean = best_cand.0;
                println!("  >>> ACCEPTED NEW BEST (Val+Policy): {:.2}\n", best_mean);
            }
            _ => {
                println!(
                    "  >>> REJECTED (no strict improvement over best {:.2}; keeping current best)\n",
                    best_mean
                );
            }
        }
    }

    println!(
        "=== DONE: Anchor BASE {:.2} -> Final Best {:.2} ({:+.2}) ===",
        base_mean,
        best_mean,
        best_mean - base_mean
    );
}
