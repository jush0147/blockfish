use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        ordered_policy::{
            extract_policy_cand_features, ObservableRootState, OrderedRootPolicy,
            PolicyCandFeatures, PolicyForwardCache, PolicyGradBuf, POL_ACT_DIM,
            POL_BOARD_DENSE_DIM, POL_H1, POL_H2, POL_ORDER_DIM, POL_SPARSE_DIM,
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

struct PolicyStateRec {
    /// Original observable state (kept for ordered-preview sensitivity ablation on holdout!)
    state: ObservableRootState,
    /// Candidate placements + pre-computed features + pure PIMC relative Q (`>= 0.0`, pieces)
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
/// Strictly Legal p=5: ZERO Stage-1 `rest_list` mixing, ZERO future peeking beyond `vis_q[0..6]`.
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

        // ONLY use pure PIMC-reranked candidates (zero Stage-1 rest_list contamination!)
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

#[derive(Clone, Copy)]
enum OrderMode {
    TrueOrdered,
    ReversedNext5,
    ZeroedNext5,
    Bc4PriorOnly,
}

fn eval_holdout_ranking(
    pol: &OrderedRootPolicy,
    states: &[PolicyStateRec],
    tau_tgt: f32,
    mode: OrderMode,
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
            logits[i] = match mode {
                OrderMode::TrueOrdered => pol.forward(feat, &mut cache),
                OrderMode::Bc4PriorOnly => -feat.bc4_delta,
                OrderMode::ZeroedNext5 => {
                    let mut f2 = feat.clone();
                    for s in 8..43 {
                        f2.order_feat[s] = 0.0;
                    }
                    pol.forward(&f2, &mut cache)
                }
                OrderMode::ReversedNext5 => {
                    let mut st_rev = rec.state.clone();
                    if st_rev.vis_q.len() >= 3 {
                        st_rev.vis_q[1..].reverse();
                    }
                    let f2 = extract_policy_cand_features(&st_rev, fp, b_after, lines, ds, feat.bc4_delta);
                    pol.forward(&f2, &mut cache)
                }
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

        // Rank indices by descending logit
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

/// Small-scale Legal `p=5` 1-ply play test: pure Frozen BC-4 vs `OrderedRootPolicy` (zero search!).
fn play_1ply_game(
    env_seed: u64,
    bc4: &SpatialValueModel,
    pol: Option<&OrderedRootPolicy>,
) -> (usize, bool) {
    let fst = FastShapeTable::srs();
    let mut pgen = FastPlaceGen::new();
    let mut places: Vec<FastPlace> = Vec::with_capacity(80);
    let mut ai_inputs = FastAI::new(BeamConfig::default());
    let mut tracker = BagTracker::new();
    let mut cache = PolicyForwardCache::default();

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(env_seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut cur_bag_mask = if let Some(snap0) = stacker.snapshot() {
        tracker.observe(snap0.hold, &snap0.queue)
    } else {
        0x7F
    };

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    while ds_total < 100 && pieces < 400 && stacker.current_piece_type().is_some() {
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
        pgen.generate_for_state(&fst, &board, c_nh, c_h, &mut places);
        if places.is_empty() {
            break;
        }

        let mut cands: Vec<(FastPlace, FastBoard, u8, u8, f32)> = Vec::with_capacity(places.len());
        let mut win_fp: Option<FastPlace> = None;
        for &fp in &places {
            let mut b1 = board;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (l, ds, _) = b1.place_and_sift(o, fp.norm_i, fp.norm_j);
            let rem1 = rem_g.saturating_sub(ds as u16);
            if rem1 == 0 && b1.rows < 20 {
                win_fp = Some(fp);
                break;
            }
            let col_h = b1.col_heights();
            if *col_h.iter().max().unwrap() >= 20 || col_h[3..=6].iter().copied().max().unwrap() >= 19 {
                continue;
            }
            let (held1, qc1) = if hold.is_some() {
                if fp.did_hold { (c_nh, if c_nh.is_some() { 1 } else { 0 }) } else { (hold, 1) }
            } else if fp.did_hold { (c_nh, 2.min(vis_q.len())) } else { (None, 1.min(vis_q.len())) };
            let ih1 = hand_piece(held1, qc1, &vis_q);
            let av1 = avail_mask_from_q(held1, qc1, &vis_q, cur_bag_mask);
            let v = OrderedRootPolicy::frozen_bc4_score(bc4, &b1, rem1, ih1, av1);
            cands.push((fp, b1, l, ds, v));
        }

        let chosen_fp = if let Some(wfp) = win_fp {
            wfp
        } else if cands.is_empty() {
            places[0]
        } else {
            let min_v = cands.iter().map(|x| x.4).fold(f32::INFINITY, f32::min);
            if let Some(p_net) = pol {
                let obs = ObservableRootState {
                    board,
                    hold,
                    vis_q: vis_q.clone(),
                    rem_g,
                    bag_mask: cur_bag_mask,
                };
                let mut best_logit = f32::NEG_INFINITY;
                let mut best_fp = cands[0].0;
                for &(fp, ref b1, l, ds, v) in &cands {
                    let feat = extract_policy_cand_features(&obs, fp, b1, l, ds, v - min_v);
                    let lg = p_net.forward(&feat, &mut cache);
                    if lg > best_logit {
                        best_logit = lg;
                        best_fp = fp;
                    }
                }
                best_fp
            } else {
                cands
                    .iter()
                    .min_by(|a, b| a.4.total_cmp(&b.4))
                    .unwrap()
                    .0
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

fn eval_1ply_set(
    bc4: &SpatialValueModel,
    pol: Option<&OrderedRootPolicy>,
    n_games: usize,
    n_threads: usize,
) -> (f64, usize, usize, usize) {
    let next_idx = AtomicUsize::new(0);
    let results: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::with_capacity(n_games));
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n_games {
                    break;
                }
                let res = play_1ply_game(9001u64 + idx as u64, bc4, pol);
                results.lock().unwrap().push(res);
            });
        }
    });
    let r = results.into_inner().unwrap();
    let mut sum = 0usize;
    let mut ok = 0usize;
    let mut min_p = usize::MAX;
    let mut max_p = 0usize;
    for &(p, top) in &r {
        if !top {
            sum += p;
            ok += 1;
            min_p = min_p.min(p);
            max_p = max_p.max(p);
        }
    }
    let mean = if ok > 0 { sum as f64 / ok as f64 } else { 999.0 };
    (mean, n_games - ok, min_p, max_p)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bc4_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "/tmp/ordered_policy_v1.txt".to_string();
    let mut n_games: usize = 10;
    let mut teacher_worlds: usize = 6;
    let mut teacher_width: usize = 16;
    let mut teacher_horizon: usize = 8;
    let mut fast_teacher: bool = false;
    let mut epochs: usize = 20;
    let mut batch_size: usize = 32;
    let mut tau_tgt: f32 = 0.20;
    let mut lr: f32 = 1.5e-3;
    let mut wd: f32 = 1e-4;
    let mut eval_games: usize = 20;
    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bc4" => { i += 1; bc4_path = args[i].clone(); }
            "--out" => { i += 1; out_path = args[i].clone(); }
            "--games" => { i += 1; n_games = args[i].parse().unwrap(); }
            "--teacher-worlds" => { i += 1; teacher_worlds = args[i].parse().unwrap(); }
            "--teacher-width" => { i += 1; teacher_width = args[i].parse().unwrap(); }
            "--teacher-horizon" => { i += 1; teacher_horizon = args[i].parse().unwrap(); }
            "--fast-teacher" => { fast_teacher = true; }
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
    println!(" OrderedRootPolicy (Policy/Value Separation — Frozen BC-4 Leaf)");
    println!(
        " bc4={} out={} games={} teacher=w{}_h{}_b{}_fast={} epochs={} bs={} tau={:.2} lr={:.1e} threads={}",
        bc4_path, out_path, n_games, teacher_worlds, teacher_horizon, teacher_width, fast_teacher, epochs, batch_size, tau_tgt, lr, n_threads
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
    let n_ho_games = (n_games / 5).max(1);
    let split = n_games.saturating_sub(n_ho_games);
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
        "[DATA] pure_pimc games={}/{} top={} mean={:.2} | train_states={} holdout_states={} ({:.1}s)",
        ok_g,
        n_games,
        n_games - ok_g,
        if ok_g > 0 { sum_p as f64 / ok_g as f64 } else { 999.0 },
        train_states.len(),
        holdout_states.len(),
        t0.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();

    let mut pol = OrderedRootPolicy::new_random(2026);
    let mut best_pol = pol.clone();
    let mut adam = PolicyAdam::new();
    let mut grad = PolicyGradBuf::new();

    let bc4_ho = eval_holdout_ranking(&pol, &holdout_states, tau_tgt, OrderMode::Bc4PriorOnly);
    println!(
        "[BASE Frozen BC-4 1-ply prior] ho_top1={:.1}% ho_top3={:.1}% ho_mrr={:.4} ho_regret={:.3}pc",
        bc4_ho.top1, bc4_ho.top3, bc4_ho.mrr, bc4_ho.regret
    );

    let mut best_ho_top1 = 0.0f64;
    let mut best_ho_regret = f64::INFINITY;
    let mut order: Vec<usize> = (0..train_states.len()).collect();
    let mut rng = 0x55AA_1234_9988_7766u64;

    for ep in 1..=epochs {
        for i in (1..order.len()).rev() {
            let j = next_usize(&mut rng, i + 1);
            order.swap(i, j);
        }
        let mut tr_ce = 0.0f64;
        let mut idx = 0usize;
        while idx < order.len() {
            let end = (idx + batch_size).min(order.len());
            let bs = end - idx;
            grad.zero();
            for &si in &order[idx..end] {
                let rec = &train_states[si];
                let k = rec.cands.len();
                let mut logits = vec![0.0f32; k];
                let mut caches: Vec<PolicyForwardCache> = (0..k).map(|_| PolicyForwardCache::default()).collect();
                let mut p_tgt = vec![0.0f32; k];
                let mut zt = 0.0f32;
                for ci in 0..k {
                    logits[ci] = pol.forward(&rec.cands[ci].4, &mut caches[ci]);
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
                    tr_ce -= (p_tgt[ci] as f64) * (probs[ci].max(1e-12).ln() as f64);
                    // d(CE)/d(logit_i) = probs[i] - p_tgt[i]
                    let d_li = (probs[ci] - p_tgt[ci]).clamp(-3.0, 3.0);
                    pol.backward(&rec.cands[ci].4, &caches[ci], d_li, &mut grad);
                }
            }
            grad.scale(1.0 / (bs as f32));
            adam.apply(&mut pol, &grad, lr, wd);
            idx = end;
        }

        if ep % 2 == 0 || ep == epochs {
            let ho = eval_holdout_ranking(&pol, &holdout_states, tau_tgt, OrderMode::TrueOrdered);
            let is_best = ho.regret < best_ho_regret;
            if is_best {
                best_ho_regret = ho.regret;
                best_ho_top1 = ho.top1;
                best_pol = pol.clone();
                best_pol.save_to_file(&out_path).unwrap();
            }
            println!(
                "[Epoch {:2}/{}] tr_ce={:.4} | ho_ce={:.4} ho_top1={:.1}% ho_top3={:.1}% ho_mrr={:.4} ho_regret={:.3}pc {}",
                ep,
                epochs,
                tr_ce / (order.len().max(1) as f64),
                ho.ce,
                ho.top1,
                ho.top3,
                ho.mrr,
                ho.regret,
                if is_best { "*** BEST ***" } else { "" }
            );
            let _ = std::io::stdout().flush();
        }
    }

    // Mandatory Ordered-Preview Sensitivity Test on independent holdout games
    println!("\n--- Ordered Preview Sensitivity Test (Independent Holdout Games) ---");
    let m_true = eval_holdout_ranking(&best_pol, &holdout_states, tau_tgt, OrderMode::TrueOrdered);
    let m_rev = eval_holdout_ranking(&best_pol, &holdout_states, tau_tgt, OrderMode::ReversedNext5);
    let m_zero = eval_holdout_ranking(&best_pol, &holdout_states, tau_tgt, OrderMode::ZeroedNext5);
    println!("  [Frozen BC-4 prior ] top1={:5.1}% top3={:5.1}% mrr={:.4} regret={:.3}pc", bc4_ho.top1, bc4_ho.top3, bc4_ho.mrr, bc4_ho.regret);
    println!("  [OrderedPolicy TRUE] top1={:5.1}% top3={:5.1}% mrr={:.4} regret={:.3}pc ce={:.4}", m_true.top1, m_true.top3, m_true.mrr, m_true.regret, m_true.ce);
    println!("  [OrderedPolicy REV ] top1={:5.1}% top3={:5.1}% mrr={:.4} regret={:.3}pc ce={:.4}", m_rev.top1, m_rev.top3, m_rev.mrr, m_rev.regret, m_rev.ce);
    println!("  [OrderedPolicy ZERO] top1={:5.1}% top3={:5.1}% mrr={:.4} regret={:.3}pc ce={:.4}", m_zero.top1, m_zero.top3, m_zero.mrr, m_zero.regret, m_zero.ce);

    // Mandatory Small-Scale Legal p=5 1-Ply Practical Play Test (zero search!)
    println!("\n--- Small-Scale Legal p=5 1-Ply Play Test ({} seeds, ZERO search) ---", eval_games);
    let t_ev = Instant::now();
    let (bc4_m, bc4_t, bc4_min, bc4_max) = eval_1ply_set(&bc4_arc, None, eval_games, n_threads);
    let (pol_m, pol_t, pol_min, pol_max) = eval_1ply_set(&bc4_arc, Some(&best_pol), eval_games, n_threads);
    println!(
        "  [Frozen BC-4 1-ply      ] ok={}/{} top={} mean={:.2} [{}..{}]",
        eval_games - bc4_t, eval_games, bc4_t, bc4_m, bc4_min, bc4_max
    );
    println!(
        "  [OrderedRootPolicy 1-ply] ok={}/{} top={} mean={:.2} [{}..{}] (delta={:+.2} pcs, {:.1}s)",
        eval_games - pol_t, eval_games, pol_t, pol_m, pol_min, pol_max, pol_m - bc4_m, t_ev.elapsed().as_secs_f64()
    );
    println!("=== DONE ORDERED POLICY LOCAL VERIFICATION (best ho_top1={:.1}% regret={:.3}pc) ===", best_ho_top1, best_ho_regret);
}
