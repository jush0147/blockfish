use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        spatial_val::{
            extract_spatial_features_ex, SpatialValueModel, DENSE_DIM, SPARSE_DIM, SPATIAL_H1,
            SPATIAL_H2,
        },
        BagTracker, BeamConfig, FastAI,
    },
    StackerExt as _,
};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const TRAIN_H1: usize = 64;
const TRAIN_H2: usize = 32;

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

#[inline]
fn next_rng(rng: &mut u64) -> u64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *rng
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
    let start = if held.is_some() {
        q_cursor
    } else {
        q_cursor + 1
    };
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

fn make_selfplay_cfg(sm: Arc<SpatialValueModel>, beam_w: usize) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.spatial_model = Some(sm);
    c.rollout_top_k = 0;
    c.beam_width = beam_w;
    c.max_per_root = (beam_w / 4).max(4);
    c.bag_det_extend = false;
    c.endgame_solver = false;
    c.simulate_cheese_spawn = true;
    c.spatial_rerank_only = true; // Fast linear internal + Full MLP leaf rerank
    c.spatial_no_pe_internal = true;
    c.root_pimc_worlds = 0;
    c
}

#[derive(Clone)]
struct StateSample {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
    target: f32,
}

struct GameResult {
    pieces: usize,
    topout: bool,
    cov_turns: usize,
    samples: Vec<StateSample>,
    mc_sq_err: f64,
    mc_cnt: usize,
}

fn play_legal_episode(
    seed: u64,
    sm: Arc<SpatialValueModel>,
    beam_w: usize,
    eps_pct: usize,
    lambda: f32,
    sparse_lim: usize,
    collect_samples: bool,
) -> GameResult {
    let fst = FastShapeTable::srs();
    let cfg = make_selfplay_cfg(sm.clone(), beam_w);
    let mut ai = FastAI::new(cfg);
    let mut tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(64);

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut cur_bag_mask = if let Some(snap0) = stacker.snapshot() {
        tracker.observe(snap0.hold, &snap0.queue)
    } else {
        0x7F
    };

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut cov_turns = 0usize;
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1337);

    // Per-step recorded features before taking action t
    let mut step_active: Vec<Vec<u16>> = Vec::with_capacity(240);
    let mut step_dense: Vec<[f32; DENSE_DIM]> = Vec::with_capacity(240);
    let mut step_v_tgt: Vec<f32> = Vec::with_capacity(240);
    let mut step_explore: Vec<bool> = Vec::with_capacity(240);

    let mut active_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

    while ds_total < 100 && pieces < 340 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);

        // Count covered top garbage hole turns (diagnostic)
        if g > 0 {
            let hole_col = (!b0.data[g as usize - 1] & 0x3FF).trailing_zeros() as usize;
            if hole_col < 10 {
                let col_h = b0.col_heights();
                if (col_h[hole_col] as usize) > (g as usize) {
                    cov_turns += 1;
                }
            }
        }

        let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
        let cur_held: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
        let bag_mask = cur_bag_mask;

        let in_hand = hand_piece(cur_held, 0, &vis_q);
        let avail = avail_mask_from_q(cur_held, 0, &vis_q, bag_mask);

        if collect_samples || eps_pct == 0 {
            let n_act = extract_spatial_features_ex(
                &b0,
                rem_g,
                in_hand,
                avail,
                false,
                &mut active_buf,
                &mut dense_buf,
            );
            let mut act_v = Vec::with_capacity(n_act.min(64));
            for &bit in &active_buf[..n_act] {
                if (bit as usize) < sparse_lim {
                    act_v.push(bit);
                }
            }
            let v_pred = sm.predict_extracted(&act_v, &dense_buf);
            step_active.push(act_v);
            step_dense.push(dense_buf);
            step_v_tgt.push(v_pred);
        }

        let do_explore =
            eps_pct > 0 && rem_g > 8 && (((next_rng(&mut rng) >> 33) as usize) % 100) < eps_pct;

        let mut chosen_inputs = None;
        let mut did_explore = false;

        if do_explore {
            let (c_nh, c_h) = if cur_held.is_some() {
                (vis_q.first().copied(), cur_held)
            } else {
                (vis_q.first().copied(), vis_q.get(1).copied())
            };
            pgen.generate_for_state(&fst, &b0, c_nh, c_h, &mut tmp_places);
            let mut scored: Vec<(i64, FastPlace)> = Vec::with_capacity(tmp_places.len());
            for &fp in &tmp_places {
                let mut b = b0;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (_lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem2 = rem_g.saturating_sub(ds as u16);
                let (new_held, nqc) = if cur_held.is_some() {
                    if fp.did_hold {
                        (c_nh, if c_nh.is_some() { 1 } else { 0 })
                    } else {
                        (cur_held, 1)
                    }
                } else if fp.did_hold {
                    (c_nh, 2)
                } else {
                    (None, 1)
                };
                let ih2 = hand_piece(new_held, nqc, &vis_q);
                let av2 = avail_mask_from_q(new_held, nqc, &vis_q, bag_mask);
                let sc = sm.evaluate_board_with_bag(&b, rem2, ih2, av2);
                scored.push((sc, fp));
            }
            scored.sort_by_key(|x| x.0);
            let k = scored.len().min(4);
            if k >= 2 {
                // Pick uniformly among top 2..k (index 1..k) so we actually explore an alternative
                let pick = 1 + (((next_rng(&mut rng) >> 33) as usize) % (k - 1));
                let fp = scored[pick].1;
                if let Some(inps) = ai.inputs_for_place(&snap.matrix, fp) {
                    chosen_inputs = Some(inps);
                    did_explore = true;
                }
            }
        }

        if chosen_inputs.is_none() {
            match ai.best_move(&snap, Some(g), Some(rem_g)) {
                Some(res) => {
                    chosen_inputs = Some(res.inputs);
                }
                None => break,
            }
        }

        if collect_samples || eps_pct == 0 {
            step_explore.push(did_explore);
        }

        let mut inputs = chosen_inputs.unwrap();
        if !inputs.is_empty() {
            inputs.pop(); // remove trailing HardDrop since we call stacker.hard_drop()
        }
        stacker.run(inputs);
        let (_lines, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;

        if let Some(snap_after) = stacker.snapshot() {
            cur_bag_mask = tracker.observe(snap_after.hold, &snap_after.queue);
        }
    }

    let topout = ds_total < 100;
    let rem_g_end = 100usize.saturating_sub(ds_total) as f32;
    let terminal_val: f32 = if topout {
        rem_g_end * 2.5 + 40.0
    } else {
        0.0
    };

    let t_len = step_active.len();
    let mut mc_sq_err = 0.0f64;
    let mut mc_cnt = 0usize;
    let mut samples = Vec::new();

    if t_len > 0 {
        // Compute backward Watkins TD(lambda) targets using frozen V_tgt
        let mut g_lambda = vec![0.0f32; t_len];
        let mut next_g = terminal_val;
        for t in (0..t_len).rev() {
            if step_explore[t] {
                // Exploratory step (including terminal topout): cut trace for predecessor t-1
                g_lambda[t] = step_v_tgt[t];
            } else if t == t_len - 1 {
                g_lambda[t] = 1.0 + terminal_val;
            } else {
                let v_next = step_v_tgt[t + 1];
                g_lambda[t] = 1.0 + (1.0 - lambda) * v_next + lambda * next_g;
            }
            next_g = g_lambda[t];
        }

        for t in 0..t_len {
            let mc_rem = (t_len - t) as f32 + terminal_val;
            if !topout && !step_explore[t] {
                let err = (step_v_tgt[t] - mc_rem) as f64;
                mc_sq_err += err * err;
                mc_cnt += 1;
            }
            if collect_samples && !step_explore[t] {
                samples.push(StateSample {
                    active: std::mem::take(&mut step_active[t]),
                    dense: step_dense[t],
                    target: g_lambda[t],
                });
            }
        }
    }

    GameResult {
        pieces,
        topout,
        cov_turns,
        samples,
        mc_sq_err,
        mc_cnt,
    }
}

struct GradBuf {
    lin_bias: f32,
    lin_sparse: [f32; SPARSE_DIM],
    lin_dense: [f32; DENSE_DIM],
    b1: [f32; SPATIAL_H1],
    w_sparse: Box<[[f32; SPATIAL_H1]; SPARSE_DIM]>,
    w_dense: Box<[[f32; SPATIAL_H1]; DENSE_DIM]>,
    b2: [f32; SPATIAL_H2],
    w2: Box<[[f32; SPATIAL_H2]; SPATIAL_H1]>,
    b3: f32,
    w3: [f32; SPATIAL_H2],
}

impl GradBuf {
    fn new() -> Self {
        Self {
            lin_bias: 0.0,
            lin_sparse: [0.0; SPARSE_DIM],
            lin_dense: [0.0; DENSE_DIM],
            b1: [0.0; SPATIAL_H1],
            w_sparse: Box::new([[0.0; SPATIAL_H1]; SPARSE_DIM]),
            w_dense: Box::new([[0.0; SPATIAL_H1]; DENSE_DIM]),
            b2: [0.0; SPATIAL_H2],
            w2: Box::new([[0.0; SPATIAL_H2]; SPATIAL_H1]),
            b3: 0.0,
            w3: [0.0; SPATIAL_H2],
        }
    }

    fn zero(&mut self, sparse_lim: usize) {
        self.lin_bias = 0.0;
        for i in 0..sparse_lim {
            self.lin_sparse[i] = 0.0;
            for h in 0..TRAIN_H1 {
                self.w_sparse[i][h] = 0.0;
            }
        }
        self.lin_dense.fill(0.0);
        for h in 0..TRAIN_H1 {
            self.b1[h] = 0.0;
            for k in 0..TRAIN_H2 {
                self.w2[h][k] = 0.0;
            }
        }
        for j in 0..DENSE_DIM {
            for h in 0..TRAIN_H1 {
                self.w_dense[j][h] = 0.0;
            }
        }
        for k in 0..TRAIN_H2 {
            self.b2[k] = 0.0;
            self.w3[k] = 0.0;
        }
        self.b3 = 0.0;
    }
}

struct AdamAnchor {
    step: usize,
    m: GradBuf,
    v: GradBuf,
}

impl AdamAnchor {
    fn new() -> Self {
        Self {
            step: 0,
            m: GradBuf::new(),
            v: GradBuf::new(),
        }
    }

    #[inline]
    fn upd(
        p: &mut f32,
        p0: f32,
        g: f32,
        m: &mut f32,
        v: &mut f32,
        lr: f32,
        anchor_wd: f32,
        b1t: f32,
        b2t: f32,
    ) {
        *m = 0.9 * *m + 0.1 * g;
        *v = 0.999 * *v + 0.001 * g * g;
        let mh = *m / b1t;
        let vh = *v / b2t;
        *p -= lr * (mh / (vh.sqrt() + 1e-8) + anchor_wd * (*p - p0));
    }

    fn apply(
        &mut self,
        mdl: &mut SpatialValueModel,
        init: &SpatialValueModel,
        g: &GradBuf,
        lr_lin: f32,
        lr_mlp: f32,
        anchor_wd: f32,
        sparse_lim: usize,
    ) {
        self.step += 1;
        let b1t = 1.0 - 0.9f32.powi(self.step as i32);
        let b2t = 1.0 - 0.999f32.powi(self.step as i32);

        Self::upd(
            &mut mdl.lin_bias,
            init.lin_bias,
            g.lin_bias,
            &mut self.m.lin_bias,
            &mut self.v.lin_bias,
            lr_lin,
            0.0,
            b1t,
            b2t,
        );
        for i in 0..sparse_lim {
            Self::upd(
                &mut mdl.lin_sparse[i],
                init.lin_sparse[i],
                g.lin_sparse[i],
                &mut self.m.lin_sparse[i],
                &mut self.v.lin_sparse[i],
                lr_lin,
                anchor_wd,
                b1t,
                b2t,
            );
        }
        for j in 0..DENSE_DIM {
            if j == 2 {
                mdl.lin_dense[j] = 0.0;
                continue;
            }
            let awd = if j == 0 { 0.0 } else { anchor_wd };
            Self::upd(
                &mut mdl.lin_dense[j],
                init.lin_dense[j],
                g.lin_dense[j],
                &mut self.m.lin_dense[j],
                &mut self.v.lin_dense[j],
                lr_lin,
                awd,
                b1t,
                b2t,
            );
        }

        if lr_mlp > 0.0 {
            for h in 0..TRAIN_H1 {
                Self::upd(
                    &mut mdl.b1[h],
                    init.b1[h],
                    g.b1[h],
                    &mut self.m.b1[h],
                    &mut self.v.b1[h],
                    lr_mlp,
                    anchor_wd,
                    b1t,
                    b2t,
                );
            }
            for i in 0..sparse_lim {
                for h in 0..TRAIN_H1 {
                    Self::upd(
                        &mut mdl.w_sparse[i][h],
                        init.w_sparse[i][h],
                        g.w_sparse[i][h],
                        &mut self.m.w_sparse[i][h],
                        &mut self.v.w_sparse[i][h],
                        lr_mlp,
                        anchor_wd,
                        b1t,
                        b2t,
                    );
                }
            }
            for j in 0..DENSE_DIM {
                if j == 0 || j == 2 {
                    mdl.w_dense[j].fill(0.0);
                    continue;
                }
                for h in 0..TRAIN_H1 {
                    Self::upd(
                        &mut mdl.w_dense[j][h],
                        init.w_dense[j][h],
                        g.w_dense[j][h],
                        &mut self.m.w_dense[j][h],
                        &mut self.v.w_dense[j][h],
                        lr_mlp,
                        anchor_wd,
                        b1t,
                        b2t,
                    );
                }
            }
            for k in 0..TRAIN_H2 {
                Self::upd(
                    &mut mdl.b2[k],
                    init.b2[k],
                    g.b2[k],
                    &mut self.m.b2[k],
                    &mut self.v.b2[k],
                    lr_mlp,
                    anchor_wd,
                    b1t,
                    b2t,
                );
            }
            for h in 0..TRAIN_H1 {
                for k in 0..TRAIN_H2 {
                    Self::upd(
                        &mut mdl.w2[h][k],
                        init.w2[h][k],
                        g.w2[h][k],
                        &mut self.m.w2[h][k],
                        &mut self.v.w2[h][k],
                        lr_mlp,
                        anchor_wd,
                        b1t,
                        b2t,
                    );
                }
            }
            Self::upd(
                &mut mdl.b3,
                init.b3,
                g.b3,
                &mut self.m.b3,
                &mut self.v.b3,
                lr_mlp,
                anchor_wd,
                b1t,
                b2t,
            );
            for k in 0..TRAIN_H2 {
                Self::upd(
                    &mut mdl.w3[k],
                    init.w3[k],
                    g.w3[k],
                    &mut self.m.w3[k],
                    &mut self.v.w3[k],
                    lr_mlp,
                    anchor_wd,
                    b1t,
                    b2t,
                );
            }
        }
    }
}

#[inline]
fn forward_sample(
    mdl: &SpatialValueModel,
    s: &StateSample,
    z_out: &mut [f32; DENSE_DIM],
    a1_out: &mut [f32; SPATIAL_H1],
    a2_out: &mut [f32; SPATIAL_H2],
) -> (f32, f32) {
    let mut lin = mdl.lin_bias;
    for &b in &s.active {
        lin += mdl.lin_sparse[b as usize];
    }
    for j in 0..DENSE_DIM {
        let z = (s.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j];
        z_out[j] = z;
        lin += mdl.lin_dense[j] * z;
    }
    let mut h1 = mdl.b1;
    for &b in &s.active {
        let row = &mdl.w_sparse[b as usize];
        for h in 0..TRAIN_H1 {
            h1[h] += row[h];
        }
    }
    for j in 0..DENSE_DIM {
        if j == 0 || j == 2 {
            continue;
        }
        let zm = z_out[j].clamp(-3.0, 3.0);
        if zm != 0.0 {
            let row = &mdl.w_dense[j];
            for h in 0..TRAIN_H1 {
                h1[h] += zm * row[h];
            }
        }
    }
    let mut h2 = mdl.b2;
    for h in 0..TRAIN_H1 {
        let a1 = h1[h].clamp(0.0, 6.0);
        a1_out[h] = a1;
        if a1 > 0.0 {
            let row = &mdl.w2[h];
            for k in 0..TRAIN_H2 {
                h2[k] += a1 * row[k];
            }
        }
    }
    let mut raw_mlp = mdl.b3;
    for k in 0..TRAIN_H2 {
        let a2 = h2[k].clamp(0.0, 6.0);
        a2_out[k] = a2;
        if a2 > 0.0 {
            raw_mlp += a2 * mdl.w3[k];
        }
    }
    let mlp_sat = raw_mlp / (1.0 + raw_mlp.abs() / 5.0);
    (lin + mlp_sat, raw_mlp)
}

#[inline]
fn backward_sample(
    mdl: &SpatialValueModel,
    s: &StateSample,
    z: &[f32; DENSE_DIM],
    a1: &[f32; SPATIAL_H1],
    a2: &[f32; SPATIAL_H2],
    raw_mlp: f32,
    d_out: f32,
    train_mlp: bool,
    g: &mut GradBuf,
) {
    if d_out == 0.0 {
        return;
    }
    g.lin_bias += d_out;
    for &b in &s.active {
        g.lin_sparse[b as usize] += d_out;
    }
    for j in 0..DENSE_DIM {
        if j != 2 {
            g.lin_dense[j] += d_out * z[j];
        }
    }
    if !train_mlp {
        return;
    }
    let denom = 1.0 + raw_mlp.abs() / 5.0;
    let d_mlp = d_out / (denom * denom);
    g.b3 += d_mlp;
    let mut d_h2 = [0.0f32; SPATIAL_H2];
    for k in 0..TRAIN_H2 {
        if a2[k] > 0.0 {
            g.w3[k] += d_mlp * a2[k];
            if a2[k] < 6.0 {
                let dk = d_mlp * mdl.w3[k];
                d_h2[k] = dk;
                g.b2[k] += dk;
            }
        }
    }
    for h in 0..TRAIN_H1 {
        if a1[h] > 0.0 {
            let mut dh = 0.0f32;
            let row = &mdl.w2[h];
            let grow = &mut g.w2[h];
            for k in 0..TRAIN_H2 {
                let dk = d_h2[k];
                if dk != 0.0 {
                    grow[k] += dk * a1[h];
                    dh += dk * row[k];
                }
            }
            if a1[h] < 6.0 && dh != 0.0 {
                g.b1[h] += dh;
                for &b in &s.active {
                    g.w_sparse[b as usize][h] += dh;
                }
                for j in 0..DENSE_DIM {
                    if j == 0 || j == 2 {
                        continue;
                    }
                    let zm = z[j].clamp(-3.0, 3.0);
                    if zm != 0.0 {
                        g.w_dense[j][h] += dh * zm;
                    }
                }
            }
        }
    }
}

fn run_games_parallel(
    seeds: &[u64],
    sm: Arc<SpatialValueModel>,
    beam_w: usize,
    eps_pct: usize,
    lambda: f32,
    sparse_lim: usize,
    collect_samples: bool,
    threads: usize,
) -> (usize, usize, f64, usize, usize, f64, f64, Vec<StateSample>) {
    let next_idx = AtomicUsize::new(0);
    let ok_pcs: Mutex<Vec<usize>> = Mutex::new(Vec::with_capacity(seeds.len()));
    let topouts = AtomicUsize::new(0);
    let cov_sum = AtomicUsize::new(0);
    let mc_stats: Mutex<(f64, usize)> = Mutex::new((0.0, 0));
    let all_samples: Mutex<Vec<StateSample>> = Mutex::new(Vec::new());

    let n_workers = threads.min(seeds.len().max(1));
    std::thread::scope(|sc| {
        for _ in 0..n_workers {
            sc.spawn(|| {
                let mut local_ok = Vec::new();
                let mut local_top = 0usize;
                let mut local_cov = 0usize;
                let mut local_sq = 0.0f64;
                let mut local_cnt = 0usize;
                let mut local_samples = Vec::new();

                loop {
                    let i = next_idx.fetch_add(1, Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    let res = play_legal_episode(
                        seeds[i],
                        sm.clone(),
                        beam_w,
                        eps_pct,
                        lambda,
                        sparse_lim,
                        collect_samples,
                    );
                    if res.topout {
                        local_top += 1;
                    } else {
                        local_ok.push(res.pieces);
                        local_cov += res.cov_turns;
                    }
                    local_sq += res.mc_sq_err;
                    local_cnt += res.mc_cnt;
                    if collect_samples {
                        local_samples.extend(res.samples);
                    }
                }

                ok_pcs.lock().unwrap().extend(local_ok);
                topouts.fetch_add(local_top, Ordering::Relaxed);
                cov_sum.fetch_add(local_cov, Ordering::Relaxed);
                {
                    let mut ms = mc_stats.lock().unwrap();
                    ms.0 += local_sq;
                    ms.1 += local_cnt;
                }
                if collect_samples {
                    all_samples.lock().unwrap().extend(local_samples);
                }
            });
        }
    });

    let pcs = ok_pcs.into_inner().unwrap();
    let n_ok = pcs.len();
    let n_top = topouts.load(Ordering::Relaxed);
    let mean_pcs = if n_ok > 0 {
        pcs.iter().sum::<usize>() as f64 / n_ok as f64
    } else {
        999.0
    };
    let min_pcs = pcs.iter().copied().min().unwrap_or(999);
    let max_pcs = pcs.iter().copied().max().unwrap_or(999);
    let mean_cov = if n_ok > 0 {
        cov_sum.load(Ordering::Relaxed) as f64 / n_ok as f64
    } else {
        0.0
    };
    let (sq, cnt) = mc_stats.into_inner().unwrap();
    let mc_rmse = if cnt > 0 {
        (sq / cnt as f64).sqrt()
    } else {
        0.0
    };
    let samples = all_samples.into_inner().unwrap();
    (n_ok, n_top, mean_pcs, min_pcs, max_pcs, mean_cov, mc_rmse, samples)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "models/spatial_td_v1.txt".to_string();
    let mut iters: usize = 4;
    let mut games_per_iter: usize = 1500;
    let mut val_games: usize = 40;
    let mut beam_w: usize = 32;
    let mut eps_pct: usize = 3;
    let mut lambda: f32 = 0.85;
    let mut epochs: usize = 3;
    let mut batch_size: usize = 256;
    let mut lr_lin: f32 = 3e-4;
    let mut lr_mlp: f32 = 1.5e-4;
    let mut anchor_wd: f32 = 1e-3;
    let mut huber_delta: f32 = 4.0;
    let mut sparse_lim: usize = 374;
    let mut threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--init" => {
                i += 1;
                init_path = args[i].clone();
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            "--iters" => {
                i += 1;
                iters = args[i].parse().unwrap();
            }
            "--games-per-iter" => {
                i += 1;
                games_per_iter = args[i].parse().unwrap();
            }
            "--val-games" => {
                i += 1;
                val_games = args[i].parse().unwrap();
            }
            "--beam" => {
                i += 1;
                beam_w = args[i].parse().unwrap();
            }
            "--eps" => {
                i += 1;
                eps_pct = args[i].parse().unwrap();
            }
            "--lambda" => {
                i += 1;
                lambda = args[i].parse().unwrap();
            }
            "--epochs" => {
                i += 1;
                epochs = args[i].parse().unwrap();
            }
            "--batch" => {
                i += 1;
                batch_size = args[i].parse().unwrap();
            }
            "--lr-lin" => {
                i += 1;
                lr_lin = args[i].parse().unwrap();
            }
            "--lr-mlp" => {
                i += 1;
                lr_mlp = args[i].parse().unwrap();
            }
            "--anchor-wd" => {
                i += 1;
                anchor_wd = args[i].parse().unwrap();
            }
            "--huber-delta" => {
                i += 1;
                huber_delta = args[i].parse().unwrap();
            }
            "--sparse-lim" => {
                i += 1;
                sparse_lim = args[i].parse().unwrap();
            }
            "--threads" => {
                i += 1;
                threads = args[i].parse().unwrap();
            }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("================================================================");
    println!(" Pure Rust Legal p=5 Iterative Watkins TD(lambda) Self-Play RL");
    println!(
        " init={} out={} iters={} games/iter={} val={} beam={} eps={}%",
        init_path, out_path, iters, games_per_iter, val_games, beam_w, eps_pct
    );
    println!(
        " lambda={:.2} epochs={} batch={} lr_lin={:.1e} lr_mlp={:.1e} anchor_wd={:.1e} huber={:.1} threads={}",
        lambda, epochs, batch_size, lr_lin, lr_mlp, anchor_wd, huber_delta, threads
    );
    println!("================================================================");
    let _ = std::io::stdout().flush();

    let mut mdl = SpatialValueModel::load_from_file(&init_path).expect("load init SpatialValueModel");
    mdl.prepare_fast();
    let init_mdl = mdl.clone();

    let val_seeds: Vec<u64> = (1..=(val_games as u64)).collect();

    // Baseline validation (iter 0)
    let t_val0 = Instant::now();
    let (v_ok0, v_top0, v_mean0, v_min0, v_max0, v_cov0, v_rmse0, _) = run_games_parallel(
        &val_seeds,
        Arc::new(mdl.clone()),
        beam_w,
        0,
        lambda,
        sparse_lim,
        false,
        threads,
    );
    println!(
        "[Iter 0 BASE] val={}/{} top={} mean={:.2} [{}..{}] cov={:.1} val_mc_rmse={:.3} ({:.1}s)",
        v_ok0,
        val_games,
        v_top0,
        v_mean0,
        v_min0,
        v_max0,
        v_cov0,
        v_rmse0,
        t_val0.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();

    let mut best_val_mean = v_mean0;
    let mut best_val_top = v_top0;
    let mut best_mdl = mdl.clone();
    mdl.save_to_file(&out_path).expect("save initial best");

    let mut adam = AdamAnchor::new();
    let mut gbuf = GradBuf::new();
    let mut shuf_rng = 0xCAFE_BABE_1234_5678u64;

    for iter in 1..=iters {
        let t_it = Instant::now();
        let seed_base = 10_000u64 + (iter as u64) * 10_000u64;
        let train_seeds: Vec<u64> = (seed_base..(seed_base + games_per_iter as u64)).collect();

        let sm_frozen = Arc::new(mdl.clone());
        let (tr_ok, tr_top, tr_mean, tr_min, tr_max, tr_cov, tr_rmse, samples) = run_games_parallel(
            &train_seeds,
            sm_frozen,
            beam_w,
            eps_pct,
            lambda,
            sparse_lim,
            true,
            threads,
        );
        let dt_gen = t_it.elapsed().as_secs_f64();
        println!(
            "[Iter {}/{} GEN] ok={}/{} top={} mean={:.2} [{}..{}] cov={:.1} mc_rmse={:.3} states={} ({:.1}s, {:.1} g/s)",
            iter,
            iters,
            tr_ok,
            games_per_iter,
            tr_top,
            tr_mean,
            tr_min,
            tr_max,
            tr_cov,
            tr_rmse,
            samples.len(),
            dt_gen,
            games_per_iter as f64 / dt_gen.max(0.01)
        );
        let _ = std::io::stdout().flush();

        // Cosine LR decay across iterations
        let progress = (iter - 1) as f32 / (iters.max(1) as f32);
        let lr_scale = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()).max(0.15);
        let cur_lr_lin = lr_lin * lr_scale;
        let cur_lr_mlp = lr_mlp * lr_scale;

        let t_tr = Instant::now();
        let mut idxs: Vec<usize> = (0..samples.len()).collect();
        let mut z = [0.0f32; DENSE_DIM];
        let mut a1 = [0.0f32; SPATIAL_H1];
        let mut a2 = [0.0f32; SPATIAL_H2];

        for ep in 1..=epochs {
            // Fisher-Yates shuffle
            for k in (1..idxs.len()).rev() {
                let j = ((next_rng(&mut shuf_rng) >> 32) as usize) % (k + 1);
                idxs.swap(k, j);
            }
            let mut loss_sum = 0.0f64;
            let mut td_sq_sum = 0.0f64;
            let mut n_seen = 0usize;

            for chunk in idxs.chunks(batch_size) {
                gbuf.zero(sparse_lim);
                let inv_bs = 1.0f32 / (chunk.len() as f32);
                for &idx in chunk {
                    let s = &samples[idx];
                    let (pred, raw_mlp) = forward_sample(&mdl, s, &mut z, &mut a1, &mut a2);
                    let diff = pred - s.target;
                    let abs_d = diff.abs();
                    let (h_loss, grad) = if abs_d <= huber_delta {
                        (0.5 * diff * diff, diff)
                    } else {
                        (
                            huber_delta * (abs_d - 0.5 * huber_delta),
                            huber_delta * diff.signum(),
                        )
                    };
                    loss_sum += h_loss as f64;
                    td_sq_sum += (diff * diff) as f64;
                    n_seen += 1;
                    backward_sample(
                        &mdl,
                        s,
                        &z,
                        &a1,
                        &a2,
                        raw_mlp,
                        grad * inv_bs,
                        cur_lr_mlp > 0.0,
                        &mut gbuf,
                    );
                }
                adam.apply(
                    &mut mdl,
                    &init_mdl,
                    &gbuf,
                    cur_lr_lin,
                    cur_lr_mlp,
                    anchor_wd,
                    sparse_lim,
                );
            }
            if ep == epochs {
                println!(
                    "  [Train ep={}/{}] huber={:.3} td_rmse={:.3} lr_lin={:.1e} lr_mlp={:.1e} ({:.1}s)",
                    ep,
                    epochs,
                    loss_sum / n_seen.max(1) as f64,
                    (td_sq_sum / n_seen.max(1) as f64).sqrt(),
                    cur_lr_lin,
                    cur_lr_mlp,
                    t_tr.elapsed().as_secs_f64()
                );
            }
        }

        mdl.prepare_fast();

        // Held-out greedy Legal validation
        let t_v = Instant::now();
        let (v_ok, v_top, v_mean, v_min, v_max, v_cov, v_rmse, _) = run_games_parallel(
            &val_seeds,
            Arc::new(mdl.clone()),
            beam_w,
            0,
            lambda,
            sparse_lim,
            false,
            threads,
        );
        let improved = (v_top < best_val_top) || (v_top == best_val_top && v_mean < best_val_mean);
        if improved {
            best_val_mean = v_mean;
            best_val_top = v_top;
            best_mdl = mdl.clone();
            best_mdl.save_to_file(&out_path).expect("save best model");
        }
        println!(
            "[Iter {}/{} VAL] val={}/{} top={} mean={:.2} [{}..{}] cov={:.1} val_mc_rmse={:.3} ({:.1}s) {}",
            iter,
            iters,
            v_ok,
            val_games,
            v_top,
            v_mean,
            v_min,
            v_max,
            v_cov,
            v_rmse,
            t_v.elapsed().as_secs_f64(),
            if improved { "*** NEW BEST ***" } else { "" }
        );
        let _ = std::io::stdout().flush();
    }

    let last_path = if out_path.ends_with(".txt") {
        format!("{}_last.txt", &out_path[..out_path.len() - 4])
    } else {
        format!("{}_last", out_path)
    };
    mdl.save_to_file(&last_path).expect("save last model");
    println!(
        "=== DONE: base_val={:.2} (top={}) -> best_val={:.2} (top={}) | saved {} and {} ===",
        v_mean0, v_top0, best_val_mean, best_val_top, out_path, last_path
    );
}
