use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        ordered_policy::{
            extract_policy_cand_features, ObservableRootState, OrderedRootPolicy,
            PolicyCandFeatures, PolicyForwardCache, PolicyGradBuf, QueueAblation, POL_ACT_DIM,
            POL_BOARD_DENSE_DIM, POL_COMPAT_DIM, POL_H1, POL_H2, POL_ORDER_DIM, POL_SPARSE_DIM,
        },
        spatial_val::SpatialValueModel,
        BagTracker, BeamConfig, FastAI,
    },
    Input, StackerExt as _,
};
use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[inline]
fn next_rng(rng: &mut u64) -> u64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *rng
}

#[inline]
fn next_usize(rng: &mut u64, bound: usize) -> usize {
    if bound <= 1 {
        return 0;
    }
    ((next_rng(rng) >> 32) as usize) % bound
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

#[inline]
fn hand_piece(held: Option<u8>, q_cursor: usize, q: &[u8]) -> Option<u8> {
    if held.is_some() {
        held
    } else {
        q.get(q_cursor).copied()
    }
}

#[inline]
fn avail_mask_from_q(held: Option<u8>, q_cursor: usize, q: &[u8], bag_mask: u8) -> u8 {
    let start = if held.is_some() { q_cursor } else { q_cursor + 1 };
    if start >= q.len() {
        return bag_mask;
    }
    let end = q.len().min(start + 3);
    let mut m = 0u8;
    for &p in &q[start..end] {
        m |= 1u8 << p;
    }
    if q.len() - start < 2 {
        m |= bag_mask;
    }
    m
}

fn observable_state_hash(
    board: &FastBoard,
    hold: Option<u8>,
    vis_q: &[u8],
    rem_g: u16,
    bag_mask: u8,
    salt: u64,
) -> u64 {
    let mut h: u64 = 0x9E3779B97F4A7C15 ^ salt.wrapping_mul(0xBF58476D1CE4E5B9);
    for r in 0..(board.rows as usize) {
        h = (h.rotate_left(7) ^ (board.data[r] as u64)).wrapping_mul(0x517cc1b727220a95);
    }
    h = (h.rotate_left(5) ^ (board.garbage_rows as u64)).wrapping_mul(0x517cc1b727220a95);
    h = (h.rotate_left(5) ^ (rem_g as u64)).wrapping_mul(0x517cc1b727220a95);
    h = (h.rotate_left(5) ^ (bag_mask as u64)).wrapping_mul(0x517cc1b727220a95);
    h = (h.rotate_left(5) ^ (hold.unwrap_or(7) as u64)).wrapping_mul(0x517cc1b727220a95);
    for &p in vis_q {
        h = (h.rotate_left(5) ^ (p as u64)).wrapping_mul(0x517cc1b727220a95);
    }
    h
}

struct PolicyStateRec {
    state: ObservableRootState,
    cands: Vec<(FastPlace, FastBoard, u8, u8, PolicyCandFeatures, f32)>,
}

fn make_teacher_cfg(
    sm: Arc<SpatialValueModel>,
    worlds: usize,
    width: usize,
    horizon: usize,
    fast_internal: bool,
) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.spatial_model = Some(sm);
    c.rollout_top_k = 0;
    c.beam_width = 80;
    c.max_per_root = 16;
    c.bag_det_extend = true;
    c.endgame_solver = true;
    c.simulate_cheese_spawn = true;
    c.spatial_rerank_only = fast_internal;
    c.spatial_no_pe_internal = fast_internal;
    c.root_pimc_prescreen_worlds = 0;
    c.root_pimc_roots = 32;
    c.root_pimc_fine_roots = 10;
    c.root_pimc_worlds = worlds;
    c.root_pimc_horizon = horizon;
    c.root_pimc_width = width;
    c.root_pimc_margin = 500;
    c.root_pimc_weight_pct = 100;
    c.root_pimc_steady_g4 = false;
    c.root_pimc_cvar_pct = 25;
    c
}

/// Collect pure uncontaminated PIMC-reranked sibling Q-distributions (`ai.last_pimc_reranked()`).
fn collect_pure_pimc_game(
    env_seed: u64,
    bc4: Arc<SpatialValueModel>,
    teacher_worlds: usize,
    teacher_width: usize,
    teacher_horizon: usize,
    fast_internal: bool,
) -> (Vec<PolicyStateRec>, usize, bool) {
    let fst = FastShapeTable::srs();
    let cfg = make_teacher_cfg(
        Arc::clone(&bc4),
        teacher_worlds,
        teacher_width,
        teacher_horizon,
        fast_internal,
    );
    let mut ai = FastAI::new(cfg);
    let mut tracker = BagTracker::new();

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(env_seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut cur_bag_mask = if let Some(snap0) = stacker.snapshot() {
        tracker.observe(snap0.hold, &snap0.queue)
    } else {
        0x7F
    };

    let mut recs: Vec<PolicyStateRec> = Vec::with_capacity(200);
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    while ds_total < 100 && pieces < 320 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100u16.saturating_sub(ds_total as u16).max(g_rows);
        let board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
        let hold = snap.hold.map(|c| color_to_idx(c) as u8);
        let vis_q: Vec<u8> = snap.queue.iter().take(6).map(|&c| color_to_idx(c) as u8).collect();
        if vis_q.is_empty() {
            break;
        }

        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };

        let pure_pimc = ai.last_pimc_reranked().to_vec();
        if pure_pimc.len() >= 3 && rem_g >= 2 {
            let min_sc = pure_pimc.iter().map(|x| x.1).min().unwrap_or(0);
            if min_sc > -5_000_000 && min_sc < 2_000_000 {
                let obs = ObservableRootState {
                    board,
                    hold,
                    vis_q: vis_q.clone(),
                    rem_g,
                    bag_mask: cur_bag_mask,
                };
                let c_nh = vis_q.first().copied();
                let mut raw_cands: Vec<(FastPlace, FastBoard, u8, u8, f32, f32)> =
                    Vec::with_capacity(pure_pimc.len());
                let mut seen_pre: HashSet<(FastBoard, Option<u8>, u8)> = HashSet::new();

                for &(fp, raw_sc) in &pure_pimc {
                    if raw_sc > min_sc + 500 {
                        continue;
                    }
                    let mut b_after = board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b_after.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem1 = rem_g.saturating_sub(ds as u16);
                    let (held1, qc1) = if hold.is_some() {
                        if fp.did_hold { (c_nh, if c_nh.is_some() { 1 } else { 0 }) } else { (hold, 1) }
                    } else if fp.did_hold { (c_nh, 2.min(vis_q.len())) } else { (None, 1.min(vis_q.len())) };
                    if !seen_pre.insert((b_after, held1, qc1 as u8)) {
                        continue;
                    }
                    let ih1 = hand_piece(held1, qc1, &vis_q);
                    let av1 = avail_mask_from_q(held1, qc1, &vis_q, cur_bag_mask);
                    let bc4_v = OrderedRootPolicy::frozen_bc4_score(&bc4, &b_after, rem1, ih1, av1);
                    let q_rel = ((raw_sc - min_sc) as f32) * 0.01;
                    raw_cands.push((fp, b_after, lines, ds, bc4_v, q_rel));
                }

                if raw_cands.len() >= 3 {
                    let min_bc4 = raw_cands.iter().map(|x| x.4).fold(f32::INFINITY, f32::min);
                    let max_q = raw_cands.iter().map(|x| x.5).fold(0.0f32, f32::max);
                    if max_q >= 0.05 {
                        let cands = raw_cands
                            .into_iter()
                            .map(|(fp, b_after, lines, ds, bc4_v, q_rel)| {
                                let feat = extract_policy_cand_features(
                                    &obs,
                                    fp,
                                    &b_after,
                                    lines,
                                    ds,
                                    bc4_v - min_bc4,
                                );
                                (fp, b_after, lines, ds, feat, q_rel)
                            })
                            .collect();
                        recs.push(PolicyStateRec { state: obs, cands });
                    }
                }
            }
        }

        let mut inputs = res.inputs;
        if inputs.last() == Some(&Input::HD) {
            inputs.pop();
        }
        stacker.run(inputs);
        let (_lines, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
        if let Some(snap_a) = stacker.snapshot() {
            cur_bag_mask = tracker.observe(snap_a.hold, &snap_a.queue);
        }
    }

    (recs, pieces, ds_total < 100)
}

struct PolicyAdam {
    step: usize,
    m: PolicyGradBuf,
    v: PolicyGradBuf,
}

impl PolicyAdam {
    fn new() -> Self {
        Self {
            step: 0,
            m: PolicyGradBuf::new(),
            v: PolicyGradBuf::new(),
        }
    }

    #[inline]
    fn upd(p: &mut f32, g: f32, m: &mut f32, v: &mut f32, lr: f32, wd: f32, b1t: f32, b2t: f32) {
        *m = 0.9 * *m + 0.1 * g;
        *v = 0.999 * *v + 0.001 * g * g;
        let mh = *m / b1t;
        let vh = *v / b2t;
        *p -= lr * (mh / (vh.sqrt() + 1e-8) + wd * *p);
    }

    fn apply(&mut self, pol: &mut OrderedRootPolicy, g: &PolicyGradBuf, lr: f32, wd: f32) {
        self.step += 1;
        let b1t = 1.0 - 0.9f32.powi(self.step as i32);
        let b2t = 1.0 - 0.999f32.powi(self.step as i32);

        for h in 0..POL_H1 {
            Self::upd(&mut pol.b_board[h], g.b_board[h], &mut self.m.b_board[h], &mut self.v.b_board[h], lr, 0.0, b1t, b2t);
            Self::upd(&mut pol.b_queue[h], g.b_queue[h], &mut self.m.b_queue[h], &mut self.v.b_queue[h], lr, 0.0, b1t, b2t);
            Self::upd(&mut pol.b_film[h], g.b_film[h], &mut self.m.b_film[h], &mut self.v.b_film[h], lr, 0.0, b1t, b2t);
            Self::upd(&mut pol.w_film[h], g.w_film[h], &mut self.m.w_film[h], &mut self.v.w_film[h], lr, wd, b1t, b2t);
        }
        for i in 0..POL_SPARSE_DIM {
            for h in 0..POL_H1 {
                Self::upd(&mut pol.w_sparse[i][h], g.w_sparse[i][h], &mut self.m.w_sparse[i][h], &mut self.v.w_sparse[i][h], lr, wd, b1t, b2t);
            }
        }
        for j in 0..POL_BOARD_DENSE_DIM {
            for h in 0..POL_H1 {
                Self::upd(&mut pol.w_bdense[j][h], g.w_bdense[j][h], &mut self.m.w_bdense[j][h], &mut self.v.w_bdense[j][h], lr, wd, b1t, b2t);
            }
        }
        for j in 0..POL_ACT_DIM {
            for h in 0..POL_H1 {
                Self::upd(&mut pol.w_act[j][h], g.w_act[j][h], &mut self.m.w_act[j][h], &mut self.v.w_act[j][h], lr, wd, b1t, b2t);
            }
        }
        for j in 0..POL_ORDER_DIM {
            for h in 0..POL_H1 {
                Self::upd(&mut pol.w_order[j][h], g.w_order[j][h], &mut self.m.w_order[j][h], &mut self.v.w_order[j][h], lr, wd, b1t, b2t);
            }
        }
        for j in 0..POL_COMPAT_DIM {
            Self::upd(&mut pol.w_compat_skip[j], g.w_compat_skip[j], &mut self.m.w_compat_skip[j], &mut self.v.w_compat_skip[j], lr, wd, b1t, b2t);
            for h in 0..POL_H1 {
                Self::upd(&mut pol.w_compat[j][h], g.w_compat[j][h], &mut self.m.w_compat[j][h], &mut self.v.w_compat[j][h], lr, wd, b1t, b2t);
            }
        }
        for k in 0..POL_H2 {
            Self::upd(&mut pol.b2[k], g.b2[k], &mut self.m.b2[k], &mut self.v.b2[k], lr, 0.0, b1t, b2t);
            Self::upd(&mut pol.w3[k], g.w3[k], &mut self.m.w3[k], &mut self.v.w3[k], lr, wd, b1t, b2t);
            for h in 0..POL_H1 {
                Self::upd(&mut pol.w2[h][k], g.w2[h][k], &mut self.m.w2[h][k], &mut self.v.w2[h][k], lr, wd, b1t, b2t);
            }
        }
        Self::upd(&mut pol.b3, g.b3, &mut self.m.b3, &mut self.v.b3, lr, 0.0, b1t, b2t);
        Self::upd(&mut pol.w_bc4_skip, g.w_bc4_skip, &mut self.m.w_bc4_skip, &mut self.v.w_bc4_skip, lr, 0.0, b1t, b2t);
    }
}

struct RankMetrics {
    ce: f64,
    top1: f64,
    top3: f64,
    mrr: f64,
    regret: f64,
}

fn eval_holdout(
    pol: &OrderedRootPolicy,
    ablation: QueueAblation,
    states: &[PolicyStateRec],
    tau_tgt: f32,
    swap_q12: bool,
    bc4_only: bool,
) -> RankMetrics {
    let mut ce_sum = 0.0f64;
    let mut top1_cnt = 0usize;
    let mut top3_cnt = 0usize;
    let mut mrr_sum = 0.0f64;
    let mut reg_sum = 0.0f64;
    let mut cache = PolicyForwardCache::default();

    for rec in states {
        let k = rec.cands.len();
        let mut logits = vec![0.0f32; k];
        let mut p_tgt = vec![0.0f32; k];
        let mut zt = 0.0f32;
        for (i, &(fp, ref b_after, lines, ds, ref feat, q_rel)) in rec.cands.iter().enumerate() {
            let et = (-(q_rel / tau_tgt).clamp(-20.0, 20.0)).exp();
            p_tgt[i] = et;
            zt += et;
            logits[i] = if bc4_only {
                -feat.bc4_delta
            } else if swap_q12 {
                // Legal adjacent swap of q[1] and q[2] (preserves 7-bag validity when q[1] != q[2])
                let mut st_sw = rec.state.clone();
                if st_sw.vis_q.len() >= 3 && st_sw.vis_q[1] != st_sw.vis_q[2] {
                    st_sw.vis_q.swap(1, 2);
                }
                let f2 = extract_policy_cand_features(&st_sw, fp, b_after, lines, ds, feat.bc4_delta);
                pol.forward(&f2, ablation, &mut cache)
            } else {
                pol.forward(feat, ablation, &mut cache)
            };
        }

        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut zp = 0.0f32;
        let mut probs = vec![0.0f32; k];
        for i in 0..k {
            let e = (logits[i] - max_l).exp();
            probs[i] = e;
            zp += e;
        }
        for i in 0..k {
            p_tgt[i] /= zt.max(1e-12);
            probs[i] /= zp.max(1e-12);
            ce_sum -= (p_tgt[i] as f64) * (probs[i].max(1e-12).ln() as f64);
        }

        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let best_tgt = rec
            .cands
            .iter()
            .enumerate()
            .min_by(|a, b| a.1 .5.total_cmp(&b.1 .5))
            .unwrap()
            .0;
        let chosen = order[0];
        if chosen == best_tgt {
            top1_cnt += 1;
        }
        let rank_pos = order.iter().position(|&x| x == best_tgt).unwrap_or(k) + 1;
        if rank_pos <= 3 {
            top3_cnt += 1;
        }
        mrr_sum += 1.0 / (rank_pos as f64);
        reg_sum += rec.cands[chosen].5 as f64;
    }

    let n = states.len().max(1) as f64;
    RankMetrics {
        ce: ce_sum / n,
        top1: 100.0 * (top1_cnt as f64) / n,
        top3: 100.0 * (top3_cnt as f64) / n,
        mrr: mrr_sum / n,
        regret: reg_sum / n,
    }
}

fn train_one_ablation(
    name: &str,
    ablation: QueueAblation,
    train_states: &[PolicyStateRec],
    holdout_states: &[PolicyStateRec],
    epochs: usize,
    batch_size: usize,
    tau_tgt: f32,
    lr: f32,
    wd: f32,
) -> (OrderedRootPolicy, RankMetrics) {
    let mut pol = OrderedRootPolicy::new_random(2026);
    let mut best_pol = pol.clone();
    let mut adam = PolicyAdam::new();
    let mut grad = PolicyGradBuf::new();
    let mut best_ho = eval_holdout(&pol, ablation, holdout_states, tau_tgt, false, false);
    let mut order: Vec<usize> = (0..train_states.len()).collect();
    let mut rng = 0x55AA_1234_9988_7766u64;

    for ep in 1..=epochs {
        for i in (1..order.len()).rev() {
            let j = next_usize(&mut rng, i + 1);
            order.swap(i, j);
        }
        let mut idx = 0usize;
        while idx < order.len() {
            let end = (idx + batch_size).min(order.len());
            let bs = end - idx;
            grad.zero();
            for &si in &order[idx..end] {
                let rec = &train_states[si];
                let k = rec.cands.len();
                let mut logits = vec![0.0f32; k];
                let mut caches: Vec<PolicyForwardCache> =
                    (0..k).map(|_| PolicyForwardCache::default()).collect();
                let mut p_tgt = vec![0.0f32; k];
                let mut zt = 0.0f32;
                for ci in 0..k {
                    logits[ci] = pol.forward(&rec.cands[ci].4, ablation, &mut caches[ci]);
                    let et = (-(rec.cands[ci].5 / tau_tgt).clamp(-20.0, 20.0)).exp();
                    p_tgt[ci] = et;
                    zt += et;
                }
                let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut probs = vec![0.0f32; k];
                let mut zp = 0.0f32;
                for ci in 0..k {
                    let e = (logits[ci] - max_l).exp();
                    probs[ci] = e;
                    zp += e;
                }
                for ci in 0..k {
                    p_tgt[ci] /= zt.max(1e-12);
                    probs[ci] /= zp.max(1e-12);
                    let d_li = (probs[ci] - p_tgt[ci]).clamp(-3.0, 3.0);
                    pol.backward(&rec.cands[ci].4, ablation, &caches[ci], d_li, &mut grad);
                }
            }
            grad.scale(1.0 / (bs as f32));
            adam.apply(&mut pol, &grad, lr, wd);
            idx = end;
        }

        let ho = eval_holdout(&pol, ablation, holdout_states, tau_tgt, false, false);
        if ho.regret < best_ho.regret {
            best_ho = ho;
            best_pol = pol.clone();
        }
        if ep == epochs {
            let sw = eval_holdout(&best_pol, ablation, holdout_states, tau_tgt, true, false);
            println!(
                "  [{:<14}] BEST ho_ce={:.4} top1={:5.1}% top3={:5.1}% mrr={:.4} regret={:.3}pc | input_perturb: top1={:5.1}% regret={:.3}pc ce={:.4}",
                name, best_ho.ce, best_ho.top1, best_ho.top3, best_ho.mrr, best_ho.regret, sw.top1, sw.regret, sw.ce
            );
            let _ = std::io::stdout().flush();
        }
    }
    (best_pol, best_ho)
}

#[derive(Clone, Copy)]
struct BNode {
    board: FastBoard,
    held: Option<u8>,
    qc: u8,
    rem_g: u16,
    first_idx: u16,
    score: f32,
}

#[derive(Hash, Eq, PartialEq)]
struct DKey {
    board: FastBoard,
    held: Option<u8>,
    qc: u8,
    rem_g: u16,
}

#[derive(Clone, Copy)]
enum RootPriorMode<'a> {
    None,
    /// Pure Frozen BC-4 1-ply pre-spawn prior (`rv + alpha * bc4_delta`, zero neural network).
    PureBc4SingleStep(f32),
    /// Learned (or random) OrderedRootPolicy prior (`rv - alpha * logit`).
    Policy(&'a OrderedRootPolicy, QueueAblation, f32),
}

/// Legal `p=5` 5-ply Beam Search with FROZEN BC-4 leaf value evaluator + optional Root Prior.
fn play_5ply_beam_game(
    env_seed: u64,
    bc4: &SpatialValueModel,
    prior: RootPriorMode<'_>,
    beam_w: usize,
) -> (usize, bool) {
    let fst = FastShapeTable::srs();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_p0: Vec<FastPlace> = Vec::with_capacity(80);
    let mut tmp_p2: Vec<FastPlace> = Vec::with_capacity(64);
    let mut cur_beam: Vec<BNode> = Vec::with_capacity(beam_w * 2);
    let mut next_beam: Vec<BNode> = Vec::with_capacity(beam_w * 48);
    let mut seen: HashSet<DKey> = HashSet::with_capacity(beam_w * 4);
    let mut cache = PolicyForwardCache::default();

    let mut ai_inputs = FastAI::new(BeamConfig::default());
    let mut tracker = BagTracker::new();

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(env_seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut cur_bag_mask = if let Some(snap0) = stacker.snapshot() {
        tracker.observe(snap0.hold, &snap0.queue)
    } else {
        0x7F
    };

    let mpr = (beam_w / 3).max(4);
    let need_root_meta = !matches!(prior, RootPriorMode::None);
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    while ds_total < 100 && pieces < 360 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100u16.saturating_sub(ds_total as u16).max(g_rows);
        let board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
        let hold = snap.hold.map(|c| color_to_idx(c) as u8);
        let vis_q: Vec<u8> = snap.queue.iter().take(6).map(|&c| color_to_idx(c) as u8).collect();
        if vis_q.is_empty() {
            break;
        }
        let c_nh = vis_q.first().copied();
        let c_h = if hold.is_some() { hold } else { vis_q.get(1).copied() };
        pgen.generate_for_state(&fst, &board, c_nh, c_h, &mut tmp_p0);
        if tmp_p0.is_empty() {
            break;
        }

        let base_seed = observable_state_hash(&board, hold, &vis_q, rem_g, cur_bag_mask, 0xCAFE);
        let max_depth = vis_q.len().min(5);
        cur_beam.clear();
        seen.clear();
        let mut win_fp: Option<FastPlace> = None;

        let mut root_meta: Vec<Option<(FastBoard, u8, u8, f32)>> = vec![None; tmp_p0.len()];
        let mut min_bc4_r = f32::INFINITY;

        for (ridx, &fp0) in tmp_p0.iter().enumerate() {
            let mut b_pre = board;
            let o0 = &fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
            let (l0, ds0, _) = b_pre.place_and_sift(o0, fp0.norm_i, fp0.norm_j);
            let rem1 = rem_g.saturating_sub(ds0 as u16);
            if rem1 == 0 && b_pre.rows < 20 {
                win_fp = Some(fp0);
                break;
            }
            let mut b1 = b_pre;
            if rem1 > 0 {
                let mut r_loc = base_seed ^ 0x1111;
                if !b1.spawn_cheese_rows(rem1, l0, &mut r_loc) {
                    continue;
                }
            }
            let col_h = b1.col_heights();
            if *col_h.iter().max().unwrap() >= 20 || col_h[3..=6].iter().copied().max().unwrap() >= 19 {
                continue;
            }
            let (held1, qc1) = if hold.is_some() {
                if fp0.did_hold { (c_nh, if c_nh.is_some() { 1 } else { 0 }) } else { (hold, 1) }
            } else if fp0.did_hold { (c_nh, 2.min(vis_q.len())) } else { (None, 1.min(vis_q.len())) };
            let ih1 = hand_piece(held1, qc1, &vis_q);
            let av1 = avail_mask_from_q(held1, qc1, &vis_q, cur_bag_mask);
            let sc1 = bc4.evaluate_linear_board_with_bag_no_pe(&b1, rem1, ih1, av1) as f32 * 0.01;
            if need_root_meta {
                let v_pre = OrderedRootPolicy::frozen_bc4_score(bc4, &b_pre, rem1, ih1, av1);
                if v_pre < min_bc4_r {
                    min_bc4_r = v_pre;
                }
                root_meta[ridx] = Some((b_pre, l0, ds0, v_pre));
            }
            let key = DKey { board: b1, held: held1, qc: qc1 as u8, rem_g: rem1 };
            if seen.insert(key) {
                cur_beam.push(BNode {
                    board: b1,
                    held: held1,
                    qc: qc1 as u8,
                    rem_g: rem1,
                    first_idx: ridx as u16,
                    score: sc1,
                });
            }
        }

        let chosen_fp = if let Some(wfp) = win_fp {
            wfp
        } else if cur_beam.is_empty() {
            tmp_p0[0]
        } else {
            cur_beam.sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
            if cur_beam.len() > beam_w {
                cur_beam.truncate(beam_w);
            }
            let mut win_d: Option<FastPlace> = None;
            'dloop: for depth in 2..=max_depth {
                next_beam.clear();
                let d_seed = base_seed ^ ((depth as u64).wrapping_mul(0x9E3779B97F4A7C15));
                for ni in 0..cur_beam.len() {
                    let node = cur_beam[ni];
                    let qc = node.qc as usize;
                    let (c_nh2, c_h2) = if node.held.is_some() {
                        (vis_q.get(qc).copied(), node.held)
                    } else {
                        (vis_q.get(qc).copied(), vis_q.get(qc + 1).copied())
                    };
                    pgen.generate_for_state(&fst, &node.board, c_nh2, c_h2, &mut tmp_p2);
                    for &fp2 in &tmp_p2 {
                        let mut b2 = node.board;
                        let o2 = &fst.shapes[fp2.color_idx as usize].orients[fp2.tf.2 as usize];
                        let (l2, ds2, _) = b2.place_and_sift(o2, fp2.norm_i, fp2.norm_j);
                        let rem2 = node.rem_g.saturating_sub(ds2 as u16);
                        if rem2 == 0 && b2.rows < 20 {
                            win_d = Some(tmp_p0[node.first_idx as usize]);
                            break 'dloop;
                        }
                        if rem2 > 0 {
                            let mut r_loc = d_seed;
                            if !b2.spawn_cheese_rows(rem2, l2, &mut r_loc) {
                                continue;
                            }
                        }
                        let col_h = b2.col_heights();
                        if *col_h.iter().max().unwrap() >= 20 || col_h[3..=6].iter().copied().max().unwrap() >= 19 {
                            continue;
                        }
                        let (held2, qc2) = if node.held.is_some() {
                            if fp2.did_hold { (c_nh2, if c_nh2.is_some() { qc + 1 } else { qc }) } else { (node.held, qc + 1) }
                        } else if fp2.did_hold { (c_nh2, qc + 2) } else { (None, qc + 1) };
                        let ih2 = hand_piece(held2, qc2, &vis_q);
                        let av2 = avail_mask_from_q(held2, qc2, &vis_q, cur_bag_mask);
                        let sc2 = bc4.evaluate_linear_board_with_bag_no_pe(&b2, rem2, ih2, av2) as f32 * 0.01;
                        next_beam.push(BNode {
                            board: b2,
                            held: held2,
                            qc: qc2 as u8,
                            rem_g: rem2,
                            first_idx: node.first_idx,
                            score: sc2,
                        });
                    }
                }
                if next_beam.is_empty() {
                    break;
                }
                next_beam.sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
                cur_beam.clear();
                seen.clear();
                let mut per_root = [0usize; 128];
                let keep_lim = if depth == max_depth { beam_w * 2 } else { beam_w };
                for &n in &next_beam {
                    if cur_beam.len() >= keep_lim {
                        break;
                    }
                    let ridx = n.first_idx as usize;
                    if ridx < 128 && per_root[ridx] >= mpr {
                        continue;
                    }
                    let key = DKey { board: n.board, held: n.held, qc: n.qc, rem_g: n.rem_g };
                    if seen.insert(key) {
                        if ridx < 128 {
                            per_root[ridx] += 1;
                        }
                        cur_beam.push(n);
                    }
                }
            }

            if let Some(wd) = win_d {
                wd
            } else {
                let mut root_v = vec![f32::INFINITY; tmp_p0.len()];
                for &node in &cur_beam {
                    let ih = hand_piece(node.held, node.qc as usize, &vis_q);
                    let av = avail_mask_from_q(node.held, node.qc as usize, &vis_q, cur_bag_mask);
                    let v = OrderedRootPolicy::frozen_bc4_score(bc4, &node.board, node.rem_g, ih, av);
                    let r = node.first_idx as usize;
                    if v < root_v[r] {
                        root_v[r] = v;
                    }
                }

                match prior {
                    RootPriorMode::None => {
                        let mut best_sc = f32::INFINITY;
                        let mut best_fp = tmp_p0[0];
                        for (r, &rv) in root_v.iter().enumerate() {
                            if rv < best_sc {
                                best_sc = rv;
                                best_fp = tmp_p0[r];
                            }
                        }
                        best_fp
                    }
                    RootPriorMode::PureBc4SingleStep(alpha) => {
                        let mut best_sc = f32::INFINITY;
                        let mut best_fp = tmp_p0[0];
                        for (r, &rv) in root_v.iter().enumerate() {
                            if !rv.is_finite() {
                                continue;
                            }
                            let delta = root_meta[r].map_or(0.0, |m| m.3 - min_bc4_r);
                            let comb = rv + alpha * delta;
                            if comb < best_sc {
                                best_sc = comb;
                                best_fp = tmp_p0[r];
                            }
                        }
                        best_fp
                    }
                    RootPriorMode::Policy(p_net, abl, alpha) => {
                        let obs = ObservableRootState {
                            board,
                            hold,
                            vis_q: vis_q.clone(),
                            rem_g,
                            bag_mask: cur_bag_mask,
                        };
                        let mut best_sc = f32::INFINITY;
                        let mut best_fp = tmp_p0[0];
                        for (r, &rv) in root_v.iter().enumerate() {
                            if !rv.is_finite() {
                                continue;
                            }
                            let bonus = if let Some((ref b_pre, l0, ds0, v_pre)) = root_meta[r] {
                                let feat = extract_policy_cand_features(
                                    &obs,
                                    tmp_p0[r],
                                    b_pre,
                                    l0,
                                    ds0,
                                    v_pre - min_bc4_r,
                                );
                                p_net.forward(&feat, abl, &mut cache)
                            } else {
                                0.0
                            };
                            let comb = rv - alpha * bonus;
                            if comb < best_sc {
                                best_sc = comb;
                                best_fp = tmp_p0[r];
                            }
                        }
                        best_fp
                    }
                }
            }
        };

        let mut inputs = match ai_inputs.inputs_for_place(&snap.matrix, chosen_fp) {
            Some(inps) => inps,
            None => break,
        };
        if inputs.last() == Some(&Input::HD) {
            inputs.pop();
        }
        stacker.run(inputs);
        let (_lines, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
        if let Some(snap_a) = stacker.snapshot() {
            cur_bag_mask = tracker.observe(snap_a.hold, &snap_a.queue);
        }
    }
    (pieces, ds_total < 100)
}

fn eval_5ply_set_per_seed(
    bc4: &SpatialValueModel,
    prior: RootPriorMode<'_>,
    n_games: usize,
    beam_w: usize,
    n_threads: usize,
) -> Vec<(u64, usize, bool)> {
    let next_idx = AtomicUsize::new(0);
    let results: Mutex<Vec<(u64, usize, bool)>> = Mutex::new(Vec::with_capacity(n_games));
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n_games {
                    break;
                }
                let seed = 9001u64 + idx as u64;
                let (pcs, top) = play_5ply_beam_game(seed, bc4, prior, beam_w);
                results.lock().unwrap().push((seed, pcs, top));
            });
        }
    });
    let mut r = results.into_inner().unwrap();
    r.sort_by_key(|x| x.0);
    r
}

fn summarize_vs_base(label: &str, res: &[(u64, usize, bool)], base: Option<&[(u64, usize, bool)]>) {
    let n = res.len();
    let mut sum = 0usize;
    let mut ok = 0usize;
    let mut min_p = usize::MAX;
    let mut max_p = 0usize;
    for &(_, p, top) in res {
        if !top {
            sum += p;
            ok += 1;
            min_p = min_p.min(p);
            max_p = max_p.max(p);
        }
    }
    let mean = if ok > 0 { sum as f64 / ok as f64 } else { 999.0 };
    if let Some(b) = base {
        let mut diffs: Vec<f64> = Vec::with_capacity(n);
        let mut wins = 0usize;
        let mut ties = 0usize;
        let mut losses = 0usize;
        for i in 0..n {
            if !res[i].2 && !b[i].2 {
                let d = res[i].1 as f64 - b[i].1 as f64;
                diffs.push(d);
                if d < 0.0 { wins += 1; } else if d > 0.0 { losses += 1; } else { ties += 1; }
            }
        }
        let m_diff = if diffs.is_empty() { 0.0 } else { diffs.iter().sum::<f64>() / diffs.len() as f64 };
        let se = if diffs.len() > 1 {
            let var = diffs.iter().map(|&d| (d - m_diff) * (d - m_diff)).sum::<f64>() / (diffs.len() - 1) as f64;
            (var / diffs.len() as f64).sqrt()
        } else {
            0.0
        };
        println!(
            "  [{:<26}] ok={:2}/{} top={} mean={:6.2} [{}..{}] | vsBASE={:+6.2}±{:.2} W/T/L={}/{}/{}",
            label, ok, n, n - ok, mean, min_p, max_p, m_diff, se, wins, ties, losses
        );
    } else {
        println!(
            "  [{:<26}] ok={:2}/{} top={} mean={:6.2} [{}..{}]",
            label, ok, n, n - ok, mean, min_p, max_p
        );
    }
}

/// True Counterfactual Teacher Test:
/// On real Legal p=5 states where `q[1]` and `q[2]` are VERIFIED to lie inside the SAME 7-bag
/// (`7 - bag_mask.count_ones() <= 3`), re-run the actual `root_pimc` teacher on the swapped snapshot
/// to measure how often ordered `q[1] <-> q[2]` genuinely changes the teacher's best move & Q-value.
fn run_counterfactual_teacher_test(
    bc4: Arc<SpatialValueModel>,
    teacher_worlds: usize,
    teacher_width: usize,
    teacher_horizon: usize,
    fast_internal: bool,
    max_states: usize,
) {
    let cfg = make_teacher_cfg(bc4, teacher_worlds, teacher_width, teacher_horizon, fast_internal);
    let mut ai = FastAI::new(cfg.clone());
    let mut ai_cf = FastAI::new(cfg);
    let mut tracker = BagTracker::new();

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(77_701);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut cur_bag_mask = if let Some(snap0) = stacker.snapshot() {
        tracker.observe(snap0.hold, &snap0.queue)
    } else {
        0x7F
    };

    let mut tested = 0usize;
    let mut flipped = 0usize;
    let mut q_shift_sum = 0.0f64;
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    while ds_total < 100 && pieces < 260 && tested < max_states && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100u16.saturating_sub(ds_total as u16).max(g_rows);
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };

        // Check exact 7-bag boundary: `k_new` pieces at tail of `vis_q[0..6]` belong to the new bag.
        // If `k_new <= 3`, then indices 0, 1, 2 (in particular q[1] and q[2]) are strictly in the SAME 7-bag!
        let k_new = 7u32.saturating_sub(cur_bag_mask.count_ones());
        let orig_pimc = ai.last_pimc_reranked().to_vec();
        if k_new <= 3 && snap.queue.len() >= 6 && snap.queue[1] != snap.queue[2] && orig_pimc.len() >= 3 && rem_g >= 4 {
            let mut snap_sw = snap.clone();
            snap_sw.queue.swap(1, 2);
            // Reset ai_cf bag tracker so it observes the exact same bag state
            ai_cf.reset_bag();
            let _ = ai_cf.best_move(&snap_sw, Some(g_rows), Some(rem_g));
            let sw_pimc = ai_cf.last_pimc_reranked().to_vec();
            if !sw_pimc.is_empty() {
                let best_orig = orig_pimc[0].0;
                let best_sw = sw_pimc[0].0;
                tested += 1;
                let same_move = best_orig.color_idx == best_sw.color_idx
                    && best_orig.norm_i == best_sw.norm_i
                    && best_orig.norm_j == best_sw.norm_j
                    && best_orig.tf.2 == best_sw.tf.2
                    && best_orig.did_hold == best_sw.did_hold;
                if !same_move {
                    flipped += 1;
                }
                // Find orig best move's regret under the counterfactual swapped queue
                let sw_min = sw_pimc[0].1;
                if let Some(&(_, sc_in_sw)) = sw_pimc.iter().find(|&&(fp, _)| {
                    fp.color_idx == best_orig.color_idx
                        && fp.norm_i == best_orig.norm_i
                        && fp.norm_j == best_orig.norm_j
                        && fp.tf.2 == best_orig.tf.2
                        && fp.did_hold == best_orig.did_hold
                }) {
                    let shift = ((sc_in_sw - sw_min).max(0) as f64) * 0.01;
                    if shift < 5.0 {
                        q_shift_sum += shift;
                    }
                }
            }
        }

        let mut inputs = res.inputs;
        if inputs.last() == Some(&Input::HD) {
            inputs.pop();
        }
        stacker.run(inputs);
        let (_lines, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
        if let Some(snap_a) = stacker.snapshot() {
            cur_bag_mask = tracker.observe(snap_a.hold, &snap_a.queue);
        }
    }

    println!(
        "\n--- [0/2] True Same-7-Bag Counterfactual Teacher Test (re-running root_pimc on swap(q1,q2)) ---"
    );
    println!(
        "  verified_same_bag_states={} | teacher_best_move_flipped={}/{} ({:.1}%) | mean_cf_regret={:.3}pc",
        tested,
        flipped,
        tested,
        100.0 * (flipped as f64) / (tested.max(1) as f64),
        q_shift_sum / (tested.max(1) as f64)
    );
    let _ = std::io::stdout().flush();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bc4_path = "models/spatial_bc4.txt".to_string();
    let mut n_games: usize = 15;
    let mut holdout_games: usize = 5;
    let mut teacher_worlds: usize = 4;
    let mut teacher_width: usize = 10;
    let mut teacher_horizon: usize = 8;
    let mut fast_teacher: bool = true;
    let mut epochs: usize = 12;
    let mut batch_size: usize = 32;
    let mut tau_tgt: f32 = 0.25;
    let mut lr: f32 = 1e-3;
    let mut wd: f32 = 3e-4;
    let mut eval_games: usize = 30;
    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bc4" => { i += 1; bc4_path = args[i].clone(); }
            "--games" => { i += 1; n_games = args[i].parse().unwrap(); }
            "--holdout-games" => { i += 1; holdout_games = args[i].parse().unwrap(); }
            "--teacher-worlds" => { i += 1; teacher_worlds = args[i].parse().unwrap(); }
            "--teacher-width" => { i += 1; teacher_width = args[i].parse().unwrap(); }
            "--teacher-horizon" => { i += 1; teacher_horizon = args[i].parse().unwrap(); }
            "--full-teacher" => { fast_teacher = false; }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap(); }
            "--batch-size" => { i += 1; batch_size = args[i].parse().unwrap(); }
            "--tau-tgt" => { i += 1; tau_tgt = args[i].parse().unwrap(); }
            "--lr" => { i += 1; lr = args[i].parse().unwrap(); }
            "--wd" => { i += 1; wd = args[i].parse().unwrap(); }
            "--eval-games" => { i += 1; eval_games = args[i].parse().unwrap(); }
            "--threads" => { i += 1; n_threads = args[i].parse().unwrap(); }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut bc4 = SpatialValueModel::load_from_file(&bc4_path)
        .unwrap_or_else(|e| panic!("load {}: {}", bc4_path, e));
    bc4.prepare_fast();
    let bc4_arc = Arc::new(bc4);

    println!("================================================================");
    println!(" OrderedRootPolicy v2: Matched-Capacity Ablation + 5-Ply Beam Test");
    println!(
        " games={} (holdout={}) teacher=w{}_h{}_b{}_fast={} (NOTE: fast local proxy != fullmlp_w16)",
        n_games, holdout_games, teacher_worlds, teacher_horizon, teacher_width, fast_teacher
    );
    println!(
        " epochs={} bs={} tau={:.2} lr={:.1e} wd={:.1e} eval_5ply={} threads={}",
        epochs, batch_size, tau_tgt, lr, wd, eval_games, n_threads
    );
    println!("================================================================");
    let _ = std::io::stdout().flush();

    let t0 = Instant::now();
    let next_g = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, Vec<PolicyStateRec>, usize, bool)>> =
        Mutex::new(Vec::with_capacity(n_games));
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            let bm = Arc::clone(&bc4_arc);
            let next_g = &next_g;
            let results = &results;
            s.spawn(move || loop {
                let gi = next_g.fetch_add(1, Ordering::Relaxed);
                if gi >= n_games {
                    break;
                }
                let seed = 60_001u64 + gi as u64;
                let (recs, pcs, top) = collect_pure_pimc_game(
                    seed,
                    Arc::clone(&bm),
                    teacher_worlds,
                    teacher_width,
                    teacher_horizon,
                    fast_teacher,
                );
                results.lock().unwrap().push((gi, recs, pcs, top));
            });
        }
    });

    let mut gr = results.into_inner().unwrap();
    gr.sort_by_key(|x| x.0);
    let split = n_games.saturating_sub(holdout_games.min(n_games.saturating_sub(1)));
    let mut train_states: Vec<PolicyStateRec> = Vec::new();
    let mut holdout_states: Vec<PolicyStateRec> = Vec::new();
    let mut sum_p = 0usize;
    let mut ok_g = 0usize;
    for (gi, mut recs, pcs, top) in gr {
        if !top {
            sum_p += pcs;
            ok_g += 1;
        }
        if gi < split {
            train_states.append(&mut recs);
        } else {
            holdout_states.append(&mut recs);
        }
    }
    println!(
        "[DATA] pure_pimc games={}/{} top={} mean={:.2} | train={}g ({} states) | holdout={}g ({} states) ({:.1}s)",
        ok_g,
        n_games,
        n_games - ok_g,
        if ok_g > 0 { sum_p as f64 / ok_g as f64 } else { 999.0 },
        split,
        train_states.len(),
        n_games - split,
        holdout_states.len(),
        t0.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();

    // 0. True Same-7-Bag Counterfactual Teacher Test (actually re-runs root_pimc on swapped queue!)
    run_counterfactual_teacher_test(
        Arc::clone(&bc4_arc),
        teacher_worlds,
        teacher_width,
        teacher_horizon,
        fast_teacher,
        30,
    );

    // 1. Frozen BC-4 prior baseline on independent holdout games
    let dummy_pol = OrderedRootPolicy::new_random(1);
    let bc4_ho = eval_holdout(&dummy_pol, QueueAblation::QueueBlind, &holdout_states, tau_tgt, false, true);
    println!(
        "\n--- [1/2] Matched-Capacity Holdout Ranking Ablation ({} independent games, {} states, FIXED piece idx) ---",
        n_games - split,
        holdout_states.len()
    );
    println!(
        "  [Frozen BC-4   ] BEST ho_ce=  N/A  top1={:5.1}% top3={:5.1}% mrr={:.4} regret={:.3}pc",
        bc4_ho.top1, bc4_ho.top3, bc4_ho.mrr, bc4_ho.regret
    );

    let (pol_blind, _m_blind) = train_one_ablation(
        "QueueBlind",
        QueueAblation::QueueBlind,
        &train_states,
        &holdout_states,
        epochs,
        batch_size,
        tau_tgt,
        lr,
        wd,
    );
    let (_pol_raw, _m_raw) = train_one_ablation(
        "RawOrdered",
        QueueAblation::RawOrdered,
        &train_states,
        &holdout_states,
        epochs,
        batch_size,
        tau_tgt,
        lr,
        wd,
    );
    let (pol_compat, _m_compat) = train_one_ablation(
        "OrderedCompat",
        QueueAblation::OrderedCompat,
        &train_states,
        &holdout_states,
        epochs,
        batch_size,
        tau_tgt,
        lr,
        wd,
    );

    // 2. Full 7-Condition Paired Decomposition of 5-Ply Beam Prior Gain (Frozen BC-4 Leaf)
    println!(
        "\n--- [2/2] 7-Condition Paired Decomposition on Legal p=5 5-Ply Beam=16 ({} seeds, a=0.15) ---",
        eval_games
    );
    let t_ev = Instant::now();
    let alpha = 0.15f32;
    let rand_pol = OrderedRootPolicy::new_random(2026);
    let mut blind_res_only = pol_blind.clone();
    blind_res_only.w_bc4_skip = 0.0;
    let mut compat_res_only = pol_compat.clone();
    compat_res_only.w_bc4_skip = 0.0;

    let r_base = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::None, eval_games, 16, n_threads);
    let r_bc4_1ply = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::PureBc4SingleStep(alpha), eval_games, 16, n_threads);
    let r_rand = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::Policy(&rand_pol, QueueAblation::QueueBlind, alpha), eval_games, 16, n_threads);
    let r_blind_res = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::Policy(&blind_res_only, QueueAblation::QueueBlind, alpha), eval_games, 16, n_threads);
    let r_blind_full = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::Policy(&pol_blind, QueueAblation::QueueBlind, alpha), eval_games, 16, n_threads);
    let r_compat_res = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::Policy(&compat_res_only, QueueAblation::OrderedCompat, alpha), eval_games, 16, n_threads);
    let r_compat_full = eval_5ply_set_per_seed(&bc4_arc, RootPriorMode::Policy(&pol_compat, QueueAblation::OrderedCompat, alpha), eval_games, 16, n_threads);

    summarize_vs_base("1. BASE (Frozen BC-4 5ply)", &r_base, None);
    summarize_vs_base("2. Pure BC-4 1-ply prior", &r_bc4_1ply, Some(&r_base));
    summarize_vs_base("3. Random Init Policy", &r_rand, Some(&r_base));
    summarize_vs_base("4. QueueBlind ResidualOnly", &r_blind_res, Some(&r_base));
    summarize_vs_base("5. QueueBlind Full", &r_blind_full, Some(&r_base));
    summarize_vs_base("6. OrderedCompat ResidOnly", &r_compat_res, Some(&r_base));
    summarize_vs_base("7. OrderedCompat Full", &r_compat_full, Some(&r_base));
    println!(
        "  (Learned w_bc4_skip: QueueBlind={:.3}, OrderedCompat={:.3})",
        pol_blind.w_bc4_skip, pol_compat.w_bc4_skip
    );
    println!("=== DONE FIXED v2 DECOMPOSITION & COUNTERFACTUAL TEST ({:.1}s eval) ===", t_ev.elapsed().as_secs_f64());
}
