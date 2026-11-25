use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        beam::BagTracker,
        fast::{color_to_idx, idx_to_color, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        spatial_val::{
            extract_spatial_features_ex, SpatialValueModel, DENSE_DIM, SPARSE_DIM, SPATIAL_H1,
            SPATIAL_H2,
        },
        BeamConfig, FastAI,
    },
    StackerExt as _,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
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
    if any_g { max_g } else { 0 }
}

fn spawn_random_cheese(b: &mut FastBoard, rem_g: u16, lines_cleared: u8, mut rng: u64) -> bool {
    let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
    let g = b.garbage_rows as usize;
    let unspawned = (rem_g as usize).saturating_sub(g);
    let num = target_g.saturating_sub(g).min(unspawned);
    if num == 0 {
        return true;
    }
    let old_rows = b.rows as usize;
    if old_rows + num >= 32 {
        return false;
    }
    for r in (0..old_rows).rev() {
        b.data[r + num] = b.data[r];
    }
    let mut prev_h: u16 = if g > 0 {
        let h0 = (!b.data[num] & 0x3FF).trailing_zeros() as u16;
        if h0 < 10 { h0 } else { 0 }
    } else {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 32) as u16) % 10
    };
    for k in (0..num).rev() {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let off = (((rng >> 32) as u16) % 9) + 1;
        prev_h = (prev_h + off) % 10;
        b.data[k] = 0xFC00 | (0x3FF & !(1u16 << prev_h));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
}

fn avail_mask_from_legal(held: Option<u8>, q_slice: &[u8], bag_mask: u8) -> u8 {
    let mut m = bag_mask;
    if let Some(h) = held {
        if h < 7 {
            m |= 1u8 << h;
        }
    }
    for &p in q_slice {
        if p < 7 {
            m |= 1u8 << p;
        }
    }
    if m == 0 { 0x7F } else { m }
}

struct MoveFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
    target_p: f32,
    q_adv: f32,
}

struct Decision {
    moves: Vec<MoveFeat>,
    best_idx: usize,
    rem_pcs: f32,
    is_val: bool,
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
    fn zero(&mut self) {
        self.lin_bias = 0.0;
        self.lin_sparse.fill(0.0);
        self.lin_dense.fill(0.0);
        self.b1.fill(0.0);
        for row in self.w_sparse.iter_mut() { row.fill(0.0); }
        for row in self.w_dense.iter_mut() { row.fill(0.0); }
        self.b2.fill(0.0);
        for row in self.w2.iter_mut() { row.fill(0.0); }
        self.b3 = 0.0;
        self.w3.fill(0.0);
    }
}

struct AdamState {
    t: i32,
    m: GradBuf,
    v: GradBuf,
}

impl AdamState {
    fn new() -> Self {
        Self { t: 0, m: GradBuf::new(), v: GradBuf::new() }
    }
    #[inline]
    fn upd(w: &mut f32, g: f32, m: &mut f32, v: &mut f32, lr: f32, wd: f32, b1t: f32, b2t: f32) {
        *m = 0.9 * *m + 0.1 * g;
        *v = 0.999 * *v + 0.001 * g * g;
        let mh = *m / b1t;
        let vh = *v / b2t;
        *w -= lr * (mh / (vh.sqrt() + 1e-8) + wd * *w);
    }
    fn apply(
        &mut self,
        mdl: &mut SpatialValueModel,
        g: &GradBuf,
        lr_lin: f32,
        lr_mlp: f32,
        wd_lin: f32,
        wd_mlp: f32,
        ah1: usize,
        ah2: usize,
    ) {
        self.t += 1;
        let b1t = 1.0 - 0.9f32.powi(self.t);
        let b2t = 1.0 - 0.999f32.powi(self.t);
        Self::upd(&mut mdl.lin_bias, g.lin_bias, &mut self.m.lin_bias, &mut self.v.lin_bias, lr_lin, 0.0, b1t, b2t);
        for i in 0..SPARSE_DIM {
            Self::upd(&mut mdl.lin_sparse[i], g.lin_sparse[i], &mut self.m.lin_sparse[i], &mut self.v.lin_sparse[i], lr_lin, wd_lin, b1t, b2t);
        }
        for j in 0..DENSE_DIM {
            if j == 2 { continue; }
            let wdj = if j == 0 { 0.0 } else { wd_lin };
            Self::upd(&mut mdl.lin_dense[j], g.lin_dense[j], &mut self.m.lin_dense[j], &mut self.v.lin_dense[j], lr_lin, wdj, b1t, b2t);
        }
        for h in 0..ah1 {
            Self::upd(&mut mdl.b1[h], g.b1[h], &mut self.m.b1[h], &mut self.v.b1[h], lr_mlp, 0.0, b1t, b2t);
        }
        for i in 0..SPARSE_DIM {
            for h in 0..ah1 {
                Self::upd(&mut mdl.w_sparse[i][h], g.w_sparse[i][h], &mut self.m.w_sparse[i][h], &mut self.v.w_sparse[i][h], lr_mlp, wd_mlp, b1t, b2t);
            }
        }
        for j in 0..DENSE_DIM {
            if j == 0 || j == 2 { continue; }
            for h in 0..ah1 {
                Self::upd(&mut mdl.w_dense[j][h], g.w_dense[j][h], &mut self.m.w_dense[j][h], &mut self.v.w_dense[j][h], lr_mlp, wd_mlp, b1t, b2t);
            }
        }
        for k in 0..ah2 {
            Self::upd(&mut mdl.b2[k], g.b2[k], &mut self.m.b2[k], &mut self.v.b2[k], lr_mlp, 0.0, b1t, b2t);
        }
        for h in 0..ah1 {
            for k in 0..ah2 {
                Self::upd(&mut mdl.w2[h][k], g.w2[h][k], &mut self.m.w2[h][k], &mut self.v.w2[h][k], lr_mlp, wd_mlp, b1t, b2t);
            }
        }
        Self::upd(&mut mdl.b3, g.b3, &mut self.m.b3, &mut self.v.b3, lr_mlp, 0.0, b1t, b2t);
        for k in 0..ah2 {
            Self::upd(&mut mdl.w3[k], g.w3[k], &mut self.m.w3[k], &mut self.v.w3[k], lr_mlp, wd_mlp, b1t, b2t);
        }
    }
}

#[inline]
fn forward_move(
    mdl: &SpatialValueModel,
    mf: &MoveFeat,
    ah1: usize,
    ah2: usize,
    drop1: Option<&[f32; SPATIAL_H1]>,
    drop2: Option<&[f32; SPATIAL_H2]>,
    z_out: &mut [f32; DENSE_DIM],
    a1_out: &mut [f32; SPATIAL_H1],
    a2_out: &mut [f32; SPATIAL_H2],
) -> (f32, f32) {
    let mut lin = mdl.lin_bias;
    for &b in &mf.active {
        lin += mdl.lin_sparse[b as usize];
    }
    for j in 0..DENSE_DIM {
        let z = (mf.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j];
        z_out[j] = z;
        lin += mdl.lin_dense[j] * z;
    }
    let mut h1 = mdl.b1;
    for &b in &mf.active {
        let row = &mdl.w_sparse[b as usize];
        for h in 0..ah1 {
            h1[h] += row[h];
        }
    }
    for j in 0..DENSE_DIM {
        if j == 0 || j == 2 { continue; }
        let zm = z_out[j].clamp(-3.0, 3.0);
        if zm != 0.0 {
            let row = &mdl.w_dense[j];
            for h in 0..ah1 {
                h1[h] += zm * row[h];
            }
        }
    }
    let mut h2 = mdl.b2;
    for h in 0..ah1 {
        let d1 = drop1.map_or(1.0, |m| m[h]);
        let a1 = h1[h].clamp(0.0, 6.0) * d1;
        a1_out[h] = a1;
        if a1 > 0.0 {
            let row = &mdl.w2[h];
            for k in 0..ah2 {
                h2[k] += a1 * row[k];
            }
        }
    }
    let mut raw_mlp = mdl.b3;
    for k in 0..ah2 {
        let d2 = drop2.map_or(1.0, |m| m[k]);
        let a2 = h2[k].clamp(0.0, 6.0) * d2;
        a2_out[k] = a2;
        if a2 > 0.0 {
            raw_mlp += a2 * mdl.w3[k];
        }
    }
    let mlp_sat = raw_mlp / (1.0 + raw_mlp.abs() / 5.0);
    (lin + mlp_sat, raw_mlp)
}

#[inline]
fn backward_move(
    mdl: &SpatialValueModel,
    mf: &MoveFeat,
    ah1: usize,
    ah2: usize,
    drop1: Option<&[f32; SPATIAL_H1]>,
    drop2: Option<&[f32; SPATIAL_H2]>,
    z: &[f32; DENSE_DIM],
    a1: &[f32; SPATIAL_H1],
    a2: &[f32; SPATIAL_H2],
    raw_mlp: f32,
    d_out: f32,
    g: &mut GradBuf,
) {
    if d_out == 0.0 { return; }
    g.lin_bias += d_out;
    for &b in &mf.active {
        g.lin_sparse[b as usize] += d_out;
    }
    for j in 0..DENSE_DIM {
        if j != 2 { g.lin_dense[j] += d_out * z[j]; }
    }
    let denom = 1.0 + raw_mlp.abs() / 5.0;
    let d_mlp = d_out / (denom * denom);
    g.b3 += d_mlp;
    let mut d_h2 = [0.0f32; SPATIAL_H2];
    for k in 0..ah2 {
        if a2[k] > 0.0 {
            let d2 = drop2.map_or(1.0, |m| m[k]);
            g.w3[k] += d_mlp * a2[k];
            if d2 > 0.0 && a2[k] < 6.0 * d2 {
                let dk = d_mlp * mdl.w3[k] * d2;
                d_h2[k] = dk;
                g.b2[k] += dk;
            }
        }
    }
    for h in 0..ah1 {
        if a1[h] > 0.0 {
            let d1 = drop1.map_or(1.0, |m| m[h]);
            let mut dh = 0.0f32;
            let row = &mdl.w2[h];
            let grow = &mut g.w2[h];
            for k in 0..ah2 {
                let dk = d_h2[k];
                if dk != 0.0 {
                    grow[k] += dk * a1[h];
                    dh += dk * row[k];
                }
            }
            if d1 > 0.0 && a1[h] < 6.0 * d1 && dh != 0.0 {
                let dh_s = dh * d1;
                g.b1[h] += dh_s;
                for &b in &mf.active {
                    g.w_sparse[b as usize][h] += dh_s;
                }
                for j in 0..DENSE_DIM {
                    if j == 0 || j == 2 { continue; }
                    let zm = z[j].clamp(-3.0, 3.0);
                    if zm != 0.0 { g.w_dense[j][h] += dh_s * zm; }
                }
            }
        }
    }
}

fn generate_legal_exit_game(
    seed: u64,
    teacher_cfg: &BeamConfig,
    tau_q: f32,
    v5_only: bool,
    is_val: bool,
) -> (usize, bool, Vec<Decision>) {
    let fst = FastShapeTable::srs();
    let mut pgen = FastPlaceGen::new();
    let mut places: Vec<FastPlace> = Vec::with_capacity(64);
    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(teacher_cfg.clone());
    let mut bag = BagTracker::new();

    let mut ds_total = 0usize;
    let mut pcs = 0usize;
    let mut topped = false;
    let mut raw_decs: Vec<(usize, Vec<MoveFeat>, usize)> = Vec::with_capacity(200);

    while ds_total < 100 && pcs < 350 {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => { topped = true; break; }
        };
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let bag_mask = bag.observe(snap.hold, &snap.queue);
        let res = match ai.best_move(&snap, Some(g), Some(rem_g)) {
            Some(r) => r,
            None => { topped = true; break; }
        };

        // Record legal decision with soft PIMC Q-targets when not in trivial 1-piece finish
        if rem_g > 2 {
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g);
            let hold0 = snap.hold.map(|c| color_to_idx(c) as u8);
            let q_vec: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
            let cur0 = q_vec.first().copied();
            let alt0 = hold0.or_else(|| q_vec.get(1).copied());
            pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut places);

            let top_cands = ai.last_top_candidates();
            let best_fp = res.best_place;
            let mut best_idx: Option<usize> = None;
            let mut move_qs: Vec<Option<f32>> = vec![None; places.len()];
            let mut min_q = f32::INFINITY;

            for (mi, &fp) in places.iter().enumerate() {
                if fp.color_idx == best_fp.color_idx
                    && fp.did_hold == best_fp.did_hold
                    && fp.norm_i == best_fp.norm_i
                    && fp.norm_j == best_fp.norm_j
                    && fp.tf.2 == best_fp.tf.2
                {
                    best_idx = Some(mi);
                }
                let ch = idx_to_color(fp.color_idx as usize).as_char();
                let ori = fp.tf.2 as u8;
                for &(tc, ti, tj, to, th, tr) in top_cands {
                    if tc == ch && ti == fp.norm_i && tj == fp.norm_j && to == ori && th == fp.did_hold {
                        if tr > -10_000_000 && tr < 4_000_000 {
                            let qv = (tr as f32) / 100.0;
                            move_qs[mi] = Some(qv);
                            if qv < min_q { min_q = qv; }
                        }
                        break;
                    }
                }
            }

            if let Some(bi) = best_idx {
                if places.len() >= 2 {
                    // Build soft probability distribution over ALL legal moves
                    let mut probs = vec![0.0f32; places.len()];
                    if min_q.is_finite() {
                        let mut sum_w = 0.0f32;
                        for mi in 0..places.len() {
                            let qv = move_qs[mi].unwrap_or(min_q + 3.0);
                            let w = (-(qv - min_q) / tau_q).exp();
                            probs[mi] = w;
                            sum_w += w;
                        }
                        let inv_w = 1.0 / sum_w.max(1e-12);
                        for p in probs.iter_mut() { *p *= inv_w; }
                    } else {
                        // Exact endgame win or single candidate: 98% on best_idx, 2% smoothed
                        let u = 0.02 / (places.len() as f32);
                        for mi in 0..places.len() {
                            probs[mi] = if mi == bi { 0.98 + u } else { u };
                        }
                    }

                    let crn = seed
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add((pcs as u64).wrapping_mul(0xBF58476D1CE4E5B9));
                    let mut moves_out = Vec::with_capacity(places.len());
                    let mut new_bi = 0usize;
                    let mut kept_sum = 0.0f32;
                    for (mi, &fp) in places.iter().enumerate() {
                        let mut b = root_board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem2 = rem_g.saturating_sub(ds as u16);
                        if rem2 > 0 {
                            let _ = spawn_random_cheese(&mut b, rem2, lines, crn);
                        }
                        let (held2, qc2) = if hold0.is_some() {
                            if fp.did_hold { (cur0, 1usize) } else { (hold0, 1usize) }
                        } else if fp.did_hold {
                            (cur0, 2usize)
                        } else {
                            (None, 1usize)
                        };
                        let q_rem = if qc2 < q_vec.len() { &q_vec[qc2..] } else { &[] };
                        let in_h = held2.or_else(|| q_rem.first().copied());
                        let av = avail_mask_from_legal(held2, q_rem, bag_mask);
                        let n_act = extract_spatial_features_ex(&b, rem2, in_h, av, false, &mut act_buf, &mut dense_buf);
                        if mi == bi { new_bi = moves_out.len(); }
                        kept_sum += probs[mi];
                        let q_adv = if min_q.is_finite() {
                            move_qs[mi].map_or(f32::NAN, |qv| (qv - min_q).clamp(0.0, 6.0))
                        } else {
                            f32::NAN
                        };
                        let active: Vec<u16> = if v5_only {
                            act_buf[..n_act].iter().copied().filter(|&b| b < 374).collect()
                        } else {
                            act_buf[..n_act].to_vec()
                        };
                        moves_out.push(MoveFeat {
                            active,
                            dense: dense_buf,
                            target_p: probs[mi],
                            q_adv,
                        });
                    }
                    if kept_sum > 0.0 {
                        let inv_s = 1.0 / kept_sum;
                        for mf in moves_out.iter_mut() { mf.target_p *= inv_s; }
                    }
                    raw_decs.push((pcs, moves_out, new_bi));
                }
            }
        }

        let mut inputs = res.inputs;
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        pcs += 1;
    }
    if ds_total < 100 {
        topped = true;
    }
    let final_pcs = if topped { 350 } else { pcs };
    let decs: Vec<Decision> = raw_decs
        .into_iter()
        .map(|(t, moves, best_idx)| Decision {
            moves,
            best_idx,
            rem_pcs: (final_pcs.saturating_sub(t + 1)) as f32,
            is_val,
        })
        .collect();
    (final_pcs, topped, decs)
}

fn eval_exit(
    mdl: &SpatialValueModel,
    decisions: &[Decision],
    idxs: &[usize],
    ah1: usize,
    ah2: usize,
    tau: f32,
) -> (f64, f64, f64, f64) {
    let mut ce_sum = 0.0f64;
    let mut top1 = 0usize;
    let mut top3 = 0usize;
    let mut v_mae = 0.0f64;
    let mut z = [0.0f32; DENSE_DIM];
    let mut a1 = [0.0f32; SPATIAL_H1];
    let mut a2 = [0.0f32; SPATIAL_H2];

    for &di in idxs {
        let dec = &decisions[di];
        let mut logits = Vec::with_capacity(dec.moves.len());
        let mut max_l = f32::NEG_INFINITY;
        let mut star_v = 0.0f32;
        for (mi, mf) in dec.moves.iter().enumerate() {
            let (v, _) = forward_move(mdl, mf, ah1, ah2, None, None, &mut z, &mut a1, &mut a2);
            if mi == dec.best_idx { star_v = v; }
            let l = -v / tau;
            if l > max_l { max_l = l; }
            logits.push(l);
        }
        let sum_e: f64 = logits.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        let ln_z = sum_e.ln();
        let mut soft_ce = 0.0f64;
        for (mi, mf) in dec.moves.iter().enumerate() {
            if mf.target_p > 0.0 {
                let log_p = (logits[mi] - max_l) as f64 - ln_z;
                soft_ce -= (mf.target_p as f64) * log_p;
            }
        }
        ce_sum += soft_ce;
        v_mae += (star_v - dec.rem_pcs).abs() as f64;
        let star_l = logits[dec.best_idx];
        let better_cnt = logits.iter().filter(|&&l| l > star_l).count();
        if better_cnt == 0 { top1 += 1; }
        if better_cnt < 3 { top3 += 1; }
    }
    let n = idxs.len().max(1) as f64;
    (ce_sum / n, (top1 as f64) * 100.0 / n, (top3 as f64) * 100.0 / n, v_mae / n)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "models/spatial_exit1.txt".to_string();
    let mut games: usize = 120;
    let mut worlds: usize = 6;
    let mut horizon: usize = 8;
    let mut width: usize = 20;
    let mut roots: usize = 10;
    let mut cvar_pct: i64 = 25;
    let mut v5_only: bool = false;
    let mut adv_w: f32 = 0.25;
    let mut epochs: usize = 20;
    let mut batch_sz: usize = 32;
    let mut tau: f32 = 0.45;
    let mut tau_q: f32 = 0.35;
    let mut cal_w: f32 = 0.06;
    let mut lr_lin: f32 = 0.00025;
    let mut lr_mlp: f32 = 0.00040;
    let mut wd_lin: f32 = 0.001;
    let mut wd_mlp: f32 = 0.005;
    let mut dropout: f32 = 0.10;
    let mut seed_base: u64 = 6000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--init" => { init_path = args[i + 1].clone(); i += 2; }
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--games" => { games = args[i + 1].parse().unwrap(); i += 2; }
            "--worlds" => { worlds = args[i + 1].parse().unwrap(); i += 2; }
            "--horizon" => { horizon = args[i + 1].parse().unwrap(); i += 2; }
            "--width" => { width = args[i + 1].parse().unwrap(); i += 2; }
            "--roots" => { roots = args[i + 1].parse().unwrap(); i += 2; }
            "--cvar" => { cvar_pct = args[i + 1].parse().unwrap(); i += 2; }
            "--v5-only" => { v5_only = true; i += 1; }
            "--adv-w" => { adv_w = args[i + 1].parse().unwrap(); i += 2; }
            "--epochs" => { epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch_sz = args[i + 1].parse().unwrap(); i += 2; }
            "--tau" => { tau = args[i + 1].parse().unwrap(); i += 2; }
            "--tau-q" => { tau_q = args[i + 1].parse().unwrap(); i += 2; }
            "--cal-w" => { cal_w = args[i + 1].parse().unwrap(); i += 2; }
            "--lr-lin" => { lr_lin = args[i + 1].parse().unwrap(); i += 2; }
            "--lr-mlp" => { lr_mlp = args[i + 1].parse().unwrap(); i += 2; }
            "--wd-lin" => { wd_lin = args[i + 1].parse().unwrap(); i += 2; }
            "--wd-mlp" => { wd_mlp = args[i + 1].parse().unwrap(); i += 2; }
            "--dropout" => { dropout = args[i + 1].parse().unwrap(); i += 2; }
            "--seed-base" => { seed_base = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
    }

    let mut mdl = SpatialValueModel::load_from_file(&init_path).expect("load init model");
    mdl.prepare_fast();
    let ah1 = mdl.active_h1.max(1);
    let ah2 = mdl.active_h2.max(1);
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    println!(
        "=== Legal Expert Iteration (ExIt-2): {} ({}x{}) -> {} ===\n  games={} w={} h={} b={} r={} cvar={} v5_only={} adv_w={} ep={} tau={} tau_q={} cal_w={} lr_lin={} lr_mlp={} drop={} threads={}",
        init_path, ah1, ah2, out_path, games, worlds, horizon, width, roots, cvar_pct, v5_only, adv_w, epochs, tau, tau_q, cal_w, lr_lin, lr_mlp, dropout, n_threads
    );

    let mut teacher_cfg = BeamConfig::default();
    teacher_cfg.beam_width = 64;
    teacher_cfg.max_per_root = 16;
    teacher_cfg.rollout_top_k = 0;
    teacher_cfg.bag_det_extend = true;
    teacher_cfg.endgame_wide = true;
    teacher_cfg.endgame_solver = true;
    teacher_cfg.simulate_cheese_spawn = true;
    teacher_cfg.spatial_model = Some(Arc::new(mdl.clone()));
    teacher_cfg.spatial_rerank_only = false;
    teacher_cfg.spatial_no_pe_internal = true;
    teacher_cfg.root_pimc_prescreen_worlds = 0;
    teacher_cfg.root_pimc_roots = roots;
    teacher_cfg.root_pimc_worlds = worlds;
    teacher_cfg.root_pimc_horizon = horizon;
    teacher_cfg.root_pimc_width = width;
    teacher_cfg.root_pimc_margin = 350;
    teacher_cfg.root_pimc_weight_pct = 100;
    teacher_cfg.root_pimc_cvar_pct = cvar_pct;
    teacher_cfg.root_pimc_steady_g4 = false;

    let t0 = Instant::now();
    let done_cnt = AtomicUsize::new(0);
    let seeds: Vec<u64> = (0..games).map(|k| seed_base + k as u64).collect();
    let val_start = ((games * 85) / 100).max(1).min(games.saturating_sub(1));
    let chunk_sz = (games + n_threads - 1) / n_threads;
    let raw_games: Vec<(usize, bool, Vec<Decision>)> = std::thread::scope(|sc| {
        let mut handles = Vec::new();
        for (ci, chunk) in seeds.chunks(chunk_sz).enumerate() {
            let base_idx = ci * chunk_sz;
            let cfg_ref = &teacher_cfg;
            let done_ref = &done_cnt;
            handles.push(sc.spawn(move || {
                let mut out = Vec::new();
                for (li, &s) in chunk.iter().enumerate() {
                    let is_val = (base_idx + li) >= val_start;
                    let res = generate_legal_exit_game(s, cfg_ref, tau_q, v5_only, is_val);
                    let d = done_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 8 == 0 || d == games {
                        println!("  [Self-Play {:3}/{}] last_pcs={} dt={:.1}s", d, games, res.0, t0.elapsed().as_secs_f64());
                    }
                    out.push(res);
                }
                out
            }));
        }
        let mut all = Vec::new();
        for h in handles { all.extend(h.join().unwrap()); }
        all
    });

    let mut decisions: Vec<Decision> = Vec::new();
    let mut sum_pcs = 0usize;
    let mut tops = 0usize;
    for (pcs, top, decs) in raw_games {
        sum_pcs += pcs;
        if top { tops += 1; }
        decisions.extend(decs);
    }
    let train_idxs: Vec<usize> = (0..decisions.len()).filter(|&i| !decisions[i].is_val).collect();
    let val_idxs: Vec<usize> = (0..decisions.len()).filter(|&i| decisions[i].is_val).collect();
    let total_moves: usize = decisions.iter().map(|d| d.moves.len()).sum();
    println!(
        "Phase 1 Legal Self-Play: {} games (mean={:.2} top={}) -> {} decisions ({} train, {} val), {} moves in {:.1}s",
        games, sum_pcs as f64 / games.max(1) as f64, tops, decisions.len(), train_idxs.len(), val_idxs.len(), total_moves, t0.elapsed().as_secs_f64()
    );

    let (init_ce, init_t1, init_t3, init_mae) = eval_exit(&mdl, &decisions, &val_idxs, ah1, ah2, tau);
    println!("  [INIT ep  0/{:2}] val_soft_ce={:.4} top1={:.1}% top3={:.1}% legal_v_mae={:.2}", epochs, init_ce, init_t1, init_t3, init_mae);

    let mut adam = AdamState::new();
    let mut grad = GradBuf::new();
    let mut order = train_idxs.clone();
    let mut rng_s = 0x517CC1B727220A95u64;
    let mut drop_rng = 0xA0761D6478BD642Fu64;
    let keep_p = (1.0 - dropout).clamp(0.2, 1.0);
    let inv_keep = 1.0 / keep_p;
    let drop_thresh = (keep_p * (u32::MAX as f32)) as u32;
    let mut drop1_buf = [1.0f32; SPATIAL_H1];
    let mut drop2_buf = [1.0f32; SPATIAL_H2];

    let mut z_cache: Vec<[f32; DENSE_DIM]> = Vec::new();
    let mut a1_cache: Vec<[f32; SPATIAL_H1]> = Vec::new();
    let mut a2_cache: Vec<[f32; SPATIAL_H2]> = Vec::new();
    let mut raw_mlp_cache: Vec<f32> = Vec::new();
    let mut v_cache: Vec<f32> = Vec::new();
    let mut logits_buf: Vec<f32> = Vec::new();

    let mut best_ce = init_ce;
    let mut best_mdl = mdl.clone();

    for ep in 1..=epochs {
        for k in (1..order.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((rng_s >> 32) as usize) % (k + 1);
            order.swap(k, j);
        }
        let prog = (ep - 1) as f32 / (epochs.max(1) as f32);
        let cos_m = 0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
        let lrl = lr_lin * cos_m;
        let lrm = lr_mlp * cos_m;

        for chunk in order.chunks(batch_sz) {
            grad.zero();
            let inv_b = 1.0 / (chunk.len() as f32);
            for &di in chunk {
                let dec = &decisions[di];
                let nm = dec.moves.len();
                z_cache.resize(nm, [0.0; DENSE_DIM]);
                a1_cache.resize(nm, [0.0; SPATIAL_H1]);
                a2_cache.resize(nm, [0.0; SPATIAL_H2]);
                raw_mlp_cache.resize(nm, 0.0);
                v_cache.resize(nm, 0.0);
                logits_buf.resize(nm, 0.0);

                let (d1_opt, d2_opt) = if dropout > 0.0 {
                    for h in 0..ah1 {
                        drop_rng = drop_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        drop1_buf[h] = if (drop_rng >> 32) as u32 <= drop_thresh { inv_keep } else { 0.0 };
                    }
                    for k in 0..ah2 {
                        drop_rng = drop_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        drop2_buf[k] = if (drop_rng >> 32) as u32 <= drop_thresh { inv_keep } else { 0.0 };
                    }
                    (Some(&drop1_buf), Some(&drop2_buf))
                } else {
                    (None, None)
                };

                let mut max_l = f32::NEG_INFINITY;
                let mut star_v = 0.0f32;
                for m in 0..nm {
                    let (v, rm) = forward_move(&mdl, &dec.moves[m], ah1, ah2, d1_opt, d2_opt, &mut z_cache[m], &mut a1_cache[m], &mut a2_cache[m]);
                    raw_mlp_cache[m] = rm;
                    v_cache[m] = v;
                    if m == dec.best_idx { star_v = v; }
                    let l = -v / tau;
                    if l > max_l { max_l = l; }
                    logits_buf[m] = l;
                }
                let mut sum_e = 0.0f32;
                for m in 0..nm {
                    logits_buf[m] = (logits_buf[m] - max_l).exp();
                    sum_e += logits_buf[m];
                }
                let inv_e = 1.0 / sum_e.max(1e-12);
                let mut d_v_buf: Vec<f32> = vec![0.0; nm];
                for m in 0..nm {
                    let p_m = logits_buf[m] * inv_e;
                    let y_m = dec.moves[m].target_p;
                    let mut d_v = -(p_m - y_m) / tau;
                    if m == dec.best_idx {
                        let err = star_v - dec.rem_pcs;
                        let huber_g = if err.abs() <= 3.0 { err } else { 3.0 * err.signum() };
                        d_v += cal_w * huber_g;
                    }
                    d_v_buf[m] = d_v;
                }
                if adv_w > 0.0 {
                    let mut n_other = 0usize;
                    for m in 0..nm {
                        if m != dec.best_idx && dec.moves[m].q_adv.is_finite() {
                            n_other += 1;
                        }
                    }
                    if n_other > 0 {
                        let w_each = adv_w / (n_other as f32);
                        for m in 0..nm {
                            if m != dec.best_idx && dec.moves[m].q_adv.is_finite() {
                                let err = (v_cache[m] - star_v) - dec.moves[m].q_adv;
                                let hg = if err.abs() <= 2.0 { err } else { 2.0 * err.signum() };
                                let g_adv = w_each * hg;
                                d_v_buf[m] += g_adv;
                                d_v_buf[dec.best_idx] -= g_adv;
                            }
                        }
                    }
                }
                for m in 0..nm {
                    backward_move(&mdl, &dec.moves[m], ah1, ah2, d1_opt, d2_opt, &z_cache[m], &a1_cache[m], &a2_cache[m], raw_mlp_cache[m], d_v_buf[m] * inv_b, &mut grad);
                }
            }
            adam.apply(&mut mdl, &grad, lrl, lrm, wd_lin, wd_mlp, ah1, ah2);
        }

        let (v_ce, v_t1, v_t3, v_mae) = eval_exit(&mdl, &decisions, &val_idxs, ah1, ah2, tau);
        if v_ce < best_ce {
            best_ce = v_ce;
            best_mdl = mdl.clone();
        }
        if ep == 1 || ep % 2 == 0 || ep == epochs {
            println!(
                "  [ExIt ep {:2}/{:2}] lr={:.5} | val_soft_ce={:.4} top1={:.1}% top3={:.1}% legal_v_mae={:.2}",
                ep, epochs, lrm, v_ce, v_t1, v_t3, v_mae
            );
        }
    }

    best_mdl.prepare_fast();
    best_mdl.save_to_file(&out_path).expect("save exit model");
    println!(
        "Saved Legal ExIt model (val_soft_ce={:.4}, rem_g={:.3}, ah={}x{}) to {}",
        best_ce,
        best_mdl.lin_dense[0] * best_mdl.dense_inv_std[0],
        best_mdl.active_h1,
        best_mdl.active_h2,
        out_path
    );
}
