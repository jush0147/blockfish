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
    Input, StackerExt as _,
};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const TRAIN_H1: usize = 64;
const TRAIN_H2: usize = 32;

#[inline]
fn next_rng(rng: &mut u64) -> u64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *rng
}

#[inline]
fn next_f32(rng: &mut u64) -> f32 {
    ((next_rng(rng) >> 40) as f32) * (1.0 / ((1u32 << 24) as f32))
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

#[derive(Clone)]
struct CandFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
    ref_v: f32,
}

struct StepRec {
    rem_g: usize,
    chosen_idx: usize,
    cands: Vec<CandFeat>,
}

struct EpisodeRec {
    pieces: usize,
    topout: bool,
    pen_pieces: f32,
    steps: Vec<StepRec>,
}

/// Play a 100% Legal p=5 game using Boltzmann afterstate policy `softmax(-V(s') / tau)`.
/// When `tau <= 0.0`, plays deterministically greedy (`argmin V(s')`).
/// Note: `sample_rng_seed` is strictly an independent exploration seed, never derived from env seed.
fn play_grpo_episode(
    env_seed: u64,
    explore_seed: u64,
    mdl: &SpatialValueModel,
    ref_mdl: &SpatialValueModel,
    tau: f32,
    top_m: usize,
    sparse_lim: usize,
    record_steps: bool,
) -> EpisodeRec {
    let fst = FastShapeTable::srs();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(80);
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

    let mut rng = explore_seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1337);
    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut steps: Vec<StepRec> = if record_steps {
        Vec::with_capacity(220)
    } else {
        Vec::new()
    };

    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

    while ds_total < 100 && pieces < 320 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);
        let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
        let hold0: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
        let cur0 = vis_q.first().copied();
        let (c_nh, c_h) = if hold0.is_some() {
            (cur0, hold0)
        } else {
            (cur0, vis_q.get(1).copied())
        };

        pgen.generate_for_state(&fst, &b0, c_nh, c_h, &mut tmp_places);
        if tmp_places.is_empty() {
            break;
        }

        // Shared per-step CRN spawn RNG derived strictly from observable state (no hidden env seed)
        let mut spawn_rng_base: u64 = 0x517CC1B727220A95
            ^ ((rem_g as u64) << 32)
            ^ ((cur_bag_mask as u64) << 16)
            ^ (hold0.unwrap_or(7) as u64)
            ^ (pieces as u64).wrapping_mul(0x9E3779B97F4A7C15);
        for r in 0..(b0.rows as usize).min(20) {
            spawn_rng_base = spawn_rng_base
                .wrapping_mul(6364136223846793005)
                .wrapping_add(b0.data[r] as u64);
        }

        // Evaluate all legal 1-ply moves, then expand top-14 to 2-ply Legal p=5 lookahead!
        let mut p1_scored: Vec<(f32, usize, FastPlace, FastBoard, Option<u8>, usize, u16)> =
            Vec::with_capacity(tmp_places.len());
        let mut win_fp: Option<FastPlace> = None;

        for (idx, &fp) in tmp_places.iter().enumerate() {
            let mut b = b0;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            let rem2 = rem_g.saturating_sub(ds as u16);
            if rem2 == 0 && b.rows < 20 {
                win_fp = Some(fp);
                break;
            }
            if rem2 > 0 {
                let mut r_loc = spawn_rng_base;
                if !b.spawn_cheese_rows(rem2, lines, &mut r_loc) {
                    continue;
                }
            }
            let col_h = b.col_heights();
            let max_h = *col_h.iter().max().unwrap();
            let center_h = col_h[3..=6].iter().copied().max().unwrap();
            if max_h >= 20 || center_h >= 19 {
                continue;
            }
            let (new_held, nqc) = if hold0.is_some() {
                if fp.did_hold {
                    (cur0, 1usize)
                } else {
                    (hold0, 1usize)
                }
            } else if fp.did_hold {
                (cur0, 2usize)
            } else {
                (None, 1usize)
            };
            let ih2 = hand_piece(new_held, nqc, &vis_q);
            let av2 = avail_mask_from_q(new_held, nqc, &vis_q, cur_bag_mask);
            let v1 = mdl.evaluate_linear_board_with_bag_no_pe(&b, rem2, ih2, av2) as f32 * 0.01;
            p1_scored.push((v1, idx, fp, b, new_held, nqc, rem2));
        }

        let mut scored: Vec<(f32, usize, FastPlace, Vec<u16>, [f32; DENSE_DIM])> =
            Vec::with_capacity(16);
        let mut tmp_p2: Vec<FastPlace> = Vec::with_capacity(64);
        let spawn_rng_d2 = spawn_rng_base.wrapping_add(0x517CC1B727220A95);

        if win_fp.is_none() && !p1_scored.is_empty() {
            p1_scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let n_expand = p1_scored.len().min(14);
            'root_loop: for &( _, ridx, fp0, b1, held1, qc1, rem1 ) in &p1_scored[..n_expand] {
                let (c_nh2, c_h2) = if held1.is_some() {
                    (vis_q.get(qc1).copied(), held1)
                } else {
                    (vis_q.get(qc1).copied(), vis_q.get(qc1 + 1).copied())
                };
                pgen.generate_for_state(&fst, &b1, c_nh2, c_h2, &mut tmp_p2);
                let mut best_v2 = f32::INFINITY;
                let mut best_act = Vec::new();
                let mut best_dense = [0.0f32; DENSE_DIM];

                for &fp2 in &tmp_p2 {
                    let mut b2 = b1;
                    let o2 = &fst.shapes[fp2.color_idx as usize].orients[fp2.tf.2 as usize];
                    let (l2, ds2, _) = b2.place_and_sift(o2, fp2.norm_i, fp2.norm_j);
                    let rem2 = rem1.saturating_sub(ds2 as u16);
                    if rem2 == 0 && b2.rows < 20 {
                        win_fp = Some(fp0);
                        break 'root_loop;
                    }
                    if rem2 > 0 {
                        let mut r_loc = spawn_rng_d2;
                        if !b2.spawn_cheese_rows(rem2, l2, &mut r_loc) {
                            continue;
                        }
                    }
                    let col_h = b2.col_heights();
                    let max_h = *col_h.iter().max().unwrap();
                    let center_h = col_h[3..=6].iter().copied().max().unwrap();
                    if max_h >= 20 || center_h >= 19 {
                        continue;
                    }
                    let (held2, qc2) = if held1.is_some() {
                        if fp2.did_hold {
                            (c_nh2, if c_nh2.is_some() { qc1 + 1 } else { qc1 })
                        } else {
                            (held1, qc1 + 1)
                        }
                    } else if fp2.did_hold {
                        (c_nh2, qc1 + 2)
                    } else {
                        (None, qc1 + 1)
                    };
                    let ih2 = hand_piece(held2, qc2, &vis_q);
                    let av2 = avail_mask_from_q(held2, qc2, &vis_q, cur_bag_mask);
                    let n_act = extract_spatial_features_ex(
                        &b2,
                        rem2,
                        ih2,
                        av2,
                        false,
                        &mut act_buf,
                        &mut dense_buf,
                    );
                    let mut act_v = Vec::with_capacity(n_act.min(64));
                    for &bit in &act_buf[..n_act] {
                        if (bit as usize) < sparse_lim {
                            act_v.push(bit);
                        }
                    }
                    let v2 = mdl.predict_extracted(&act_v, &dense_buf);
                    if v2 < best_v2 {
                        best_v2 = v2;
                        best_act = act_v;
                        best_dense = dense_buf;
                    }
                }
                if best_v2.is_finite() {
                    scored.push((best_v2, ridx, fp0, best_act, best_dense));
                }
            }
        }

        let chosen_fp = if let Some(wfp) = win_fp {
            wfp
        } else if scored.is_empty() {
            tmp_places[0]
        } else {
            // Sort by lowest predicted remaining pieces
            scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
            let k = scored.len().min(top_m.max(2));
            let pick = if tau <= 0.0 || k == 1 {
                0usize
            } else {
                let v_min = scored[0].0;
                let mut probs = [0.0f32; 32];
                let mut z_sum = 0.0f32;
                for c in 0..k {
                    let logit = -((scored[c].0 - v_min) / tau).clamp(-20.0, 20.0);
                    let e = logit.exp();
                    probs[c] = e;
                    z_sum += e;
                }
                let u = next_f32(&mut rng) * z_sum;
                let mut acc = 0.0f32;
                let mut chosen = 0usize;
                for c in 0..k {
                    acc += probs[c];
                    if u <= acc {
                        chosen = c;
                        break;
                    }
                }
                chosen
            };

            if record_steps && k >= 2 {
                let mut cands = Vec::with_capacity(k);
                for c in 0..k {
                    let rv = ref_mdl.predict_extracted(&scored[c].3, &scored[c].4);
                    cands.push(CandFeat {
                        active: std::mem::take(&mut scored[c].3),
                        dense: scored[c].4,
                        ref_v: rv,
                    });
                }
                steps.push(StepRec {
                    rem_g: rem_g as usize,
                    chosen_idx: pick,
                    cands,
                });
            }
            scored[pick].2
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

    let topout = ds_total < 100;
    let rem_end = 100usize.saturating_sub(ds_total) as f32;
    let pen_pieces = if topout {
        (pieces as f32) + rem_end * 2.5 + 40.0
    } else {
        pieces as f32
    };

    EpisodeRec {
        pieces,
        topout,
        pen_pieces,
        steps,
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
            Self::upd(
                &mut mdl.lin_dense[j],
                init.lin_dense[j],
                g.lin_dense[j],
                &mut self.m.lin_dense[j],
                &mut self.v.lin_dense[j],
                lr_lin,
                anchor_wd,
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
fn forward_cand(
    mdl: &SpatialValueModel,
    c: &CandFeat,
    z_out: &mut [f32; DENSE_DIM],
    a1_out: &mut [f32; SPATIAL_H1],
    a2_out: &mut [f32; SPATIAL_H2],
) -> (f32, f32) {
    let mut lin = mdl.lin_bias;
    for &b in &c.active {
        lin += mdl.lin_sparse[b as usize];
    }
    for j in 0..DENSE_DIM {
        let z = (c.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j];
        z_out[j] = z;
        lin += mdl.lin_dense[j] * z;
    }
    let mut h1 = mdl.b1;
    for &b in &c.active {
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
fn backward_cand(
    mdl: &SpatialValueModel,
    c: &CandFeat,
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
    for &b in &c.active {
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
                for &b in &c.active {
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

struct GrpoStep {
    adv: f32,
    chosen_idx: usize,
    cands: Vec<CandFeat>,
}

fn eval_greedy_val(
    seeds: &[u64],
    mdl: &Arc<SpatialValueModel>,
    sparse_lim: usize,
    threads: usize,
) -> (usize, usize, f64, usize, usize) {
    let next_idx = AtomicUsize::new(0);
    let ok_pcs: Mutex<Vec<usize>> = Mutex::new(Vec::with_capacity(seeds.len()));
    let tops = AtomicUsize::new(0);
    let n_workers = threads.min(seeds.len().max(1));

    std::thread::scope(|sc| {
        let next_ref = &next_idx;
        let ok_ref = &ok_pcs;
        let top_ref = &tops;
        for _ in 0..n_workers {
            let m_c = mdl.clone();
            sc.spawn(move || {
                let mut local_ok = Vec::new();
                let mut local_top = 0usize;
                loop {
                    let i = next_ref.fetch_add(1, Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    let ep = play_grpo_episode(
                        seeds[i],
                        1,
                        &m_c,
                        &m_c,
                        0.0, // deterministic greedy
                        12,
                        sparse_lim,
                        false,
                    );
                    if ep.topout {
                        local_top += 1;
                    } else {
                        local_ok.push(ep.pieces);
                    }
                }
                ok_ref.lock().unwrap().extend(local_ok);
                top_ref.fetch_add(local_top, Ordering::Relaxed);
            });
        }
    });

    let pcs = ok_pcs.into_inner().unwrap();
    let n_ok = pcs.len();
    let n_top = tops.load(Ordering::Relaxed);
    let mean = if n_ok > 0 {
        pcs.iter().sum::<usize>() as f64 / n_ok as f64
    } else {
        999.0
    };
    let min_p = pcs.iter().copied().min().unwrap_or(999);
    let max_p = pcs.iter().copied().max().unwrap_or(999);
    (n_ok, n_top, mean, min_p, max_p)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "/tmp/spatial_grpo_v1.txt".to_string();
    let mut iters: usize = 30;
    let mut seeds_per_iter: usize = 48;
    let mut group_size: usize = 8;
    let mut val_games: usize = 80;
    let mut tau: f32 = 0.25;
    let mut top_m: usize = 10;
    let mut kl_beta: f32 = 0.08;
    let mut lr_lin: f32 = 4e-4;
    let mut lr_mlp: f32 = 2e-4;
    let mut anchor_wd: f32 = 5e-4;
    let mut batch_steps: usize = 512;
    let mut sparse_lim: usize = 374;
    let mut threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--init" => { init_path = args[i + 1].clone(); i += 2; }
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            "--seeds-per-iter" => { seeds_per_iter = args[i + 1].parse().unwrap(); i += 2; }
            "--group-size" => { group_size = args[i + 1].parse().unwrap(); i += 2; }
            "--val-games" => { val_games = args[i + 1].parse().unwrap(); i += 2; }
            "--tau" => { tau = args[i + 1].parse().unwrap(); i += 2; }
            "--top-m" => { top_m = args[i + 1].parse().unwrap(); i += 2; }
            "--kl-beta" => { kl_beta = args[i + 1].parse().unwrap(); i += 2; }
            "--lr-lin" => { lr_lin = args[i + 1].parse().unwrap(); i += 2; }
            "--lr-mlp" => { lr_mlp = args[i + 1].parse().unwrap(); i += 2; }
            "--anchor-wd" => { anchor_wd = args[i + 1].parse().unwrap(); i += 2; }
            "--batch-steps" => { batch_steps = args[i + 1].parse().unwrap(); i += 2; }
            "--sparse-lim" => { sparse_lim = args[i + 1].parse().unwrap(); i += 2; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
    }

    println!("================================================================");
    println!(" Pure Rust Legal p=5 CRN-GRPO (Group Relative Policy Optimization)");
    println!(
        " init={} out={} iters={} seeds/iter={} G={} ({} games/iter) val={}",
        init_path,
        out_path,
        iters,
        seeds_per_iter,
        group_size,
        seeds_per_iter * group_size,
        val_games
    );
    println!(
        " tau={:.2} top_m={} kl_beta={:.3} lr_lin={:.1e} lr_mlp={:.1e} anchor_wd={:.1e} threads={}",
        tau, top_m, kl_beta, lr_lin, lr_mlp, anchor_wd, threads
    );
    println!("================================================================");
    let _ = std::io::stdout().flush();

    let mut mdl = SpatialValueModel::load_from_file(&init_path).expect("load init model");
    mdl.prepare_fast();
    let ref_mdl = Arc::new(mdl.clone());

    let val_seeds: Vec<u64> = (1..=(val_games as u64)).collect();
    let t0 = Instant::now();
    let (v_ok0, v_top0, v_mean0, v_min0, v_max0) =
        eval_greedy_val(&val_seeds, &Arc::new(mdl.clone()), sparse_lim, threads);
    println!(
        "[Iter  0 BASE] val={}/{} top={} mean={:.2} [{}..{}] ({:.1}s)",
        v_ok0,
        val_games,
        v_top0,
        v_mean0,
        v_min0,
        v_max0,
        t0.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();

    let mut best_val = v_mean0;
    let mut best_top = v_top0;
    mdl.save_to_file(&out_path).expect("save initial best");

    let mut adam = AdamAnchor::new();
    let mut gbuf = GradBuf::new();
    let mut shuf_rng = 0xD1B54A32D192ED03u64;

    for iter in 1..=iters {
        let t_it = Instant::now();
        let seed_base = 20_000u64 + (iter as u64) * 1_000u64;
        let group_seeds: Vec<u64> = (seed_base..(seed_base + seeds_per_iter as u64)).collect();
        let cur_arc = Arc::new(mdl.clone());
        let next_s = AtomicUsize::new(0);
        let collected: Mutex<Vec<GrpoStep>> = Mutex::new(Vec::new());
        let tr_ok_pcs: Mutex<Vec<usize>> = Mutex::new(Vec::new());
        let tr_tops = AtomicUsize::new(0);

        let n_workers = threads.min(seeds_per_iter.max(1));
        std::thread::scope(|sc| {
            let next_ref = &next_s;
            let col_ref = &collected;
            let ok_ref = &tr_ok_pcs;
            let top_ref = &tr_tops;
            let gs_ref = &group_seeds;
            for _ in 0..n_workers {
                let m_c = cur_arc.clone();
                let r_c = ref_mdl.clone();
                sc.spawn(move || {
                    let mut local_steps = Vec::new();
                    let mut local_ok = Vec::new();
                    let mut local_top = 0usize;

                    loop {
                        let si = next_ref.fetch_add(1, Ordering::Relaxed);
                        if si >= gs_ref.len() {
                            break;
                        }
                        let env_seed = gs_ref[si];
                        // Play G CRN episodes on the exact same env_seed with independent exploration seeds
                        let mut eps: Vec<EpisodeRec> = Vec::with_capacity(group_size);
                        for g_idx in 0..group_size {
                            let exp_seed = ((iter as u64) << 32)
                                ^ ((si as u64) << 12)
                                ^ ((g_idx as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15);
                            let ep = play_grpo_episode(
                                env_seed,
                                exp_seed,
                                &m_c,
                                &r_c,
                                tau,
                                top_m,
                                sparse_lim,
                                true,
                            );
                            if ep.topout {
                                local_top += 1;
                            } else {
                                local_ok.push(ep.pieces);
                            }
                            eps.push(ep);
                        }

                        // Step-aligned CRN remaining-pieces baseline per rem_g bucket (1..=100)
                        // Eliminates both seed difficulty AND early-game noise!
                        let mut sum_rem = [0.0f32; 101];
                        let mut sq_rem = [0.0f32; 101];
                        let mut cnt_rem = [0u32; 101];
                        for ep in &eps {
                            for (t_idx, st) in ep.steps.iter().enumerate() {
                                let rg = st.rem_g.min(100);
                                let rem_p = (ep.pen_pieces - t_idx as f32).max(0.0);
                                sum_rem[rg] += rem_p;
                                sq_rem[rg] += rem_p * rem_p;
                                cnt_rem[rg] += 1;
                            }
                        }

                        // Episode-level fallback baseline
                        let g_mean = eps.iter().map(|e| e.pen_pieces).sum::<f32>() / (eps.len() as f32);
                        let g_var = eps
                            .iter()
                            .map(|e| (e.pen_pieces - g_mean) * (e.pen_pieces - g_mean))
                            .sum::<f32>()
                            / (eps.len() as f32);
                        let g_std = g_var.sqrt() + 1.0;

                        for mut ep in eps {
                            let pen = ep.pen_pieces;
                            for (t_idx, st) in ep.steps.drain(..).enumerate() {
                                let rg = st.rem_g.min(100);
                                let rem_p = (pen - t_idx as f32).max(0.0);
                                let adv = if cnt_rem[rg] >= 3 {
                                    let m = sum_rem[rg] / (cnt_rem[rg] as f32);
                                    let v = (sq_rem[rg] / (cnt_rem[rg] as f32) - m * m).max(0.0);
                                    ((m - rem_p) / (v.sqrt() + 1.0)).clamp(-2.5, 2.5)
                                } else {
                                    ((g_mean - pen) / g_std).clamp(-2.5, 2.5)
                                };
                                if adv.abs() > 1e-4 {
                                    local_steps.push(GrpoStep {
                                        adv,
                                        chosen_idx: st.chosen_idx,
                                        cands: st.cands,
                                    });
                                }
                            }
                        }
                    }

                    col_ref.lock().unwrap().extend(local_steps);
                    ok_ref.lock().unwrap().extend(local_ok);
                    top_ref.fetch_add(local_top, Ordering::Relaxed);
                });
            }
        });

        let steps_all = collected.into_inner().unwrap();
        let tr_pcs = tr_ok_pcs.into_inner().unwrap();
        let tr_top = tr_tops.load(Ordering::Relaxed);
        let tr_mean = if !tr_pcs.is_empty() {
            tr_pcs.iter().sum::<usize>() as f64 / tr_pcs.len() as f64
        } else {
            999.0
        };
        let dt_gen = t_it.elapsed().as_secs_f64();

        // Cosine LR schedule
        let prog = (iter - 1) as f32 / (iters.max(1) as f32);
        let lr_scale = 0.5 * (1.0 + (std::f32::consts::PI * prog).cos()).max(0.15);
        let cur_lr_lin = lr_lin * lr_scale;
        let cur_lr_mlp = lr_mlp * lr_scale;

        // 1 on-policy GRPO pass over all collected steps in shuffled mini-batches
        let t_tr = Instant::now();
        let mut idxs: Vec<usize> = (0..steps_all.len()).collect();
        for k in (1..idxs.len()).rev() {
            let j = ((next_rng(&mut shuf_rng) >> 32) as usize) % (k + 1);
            idxs.swap(k, j);
        }

        let mut z_bufs = vec![[0.0f32; DENSE_DIM]; 32];
        let mut a1_bufs = vec![[0.0f32; SPATIAL_H1]; 32];
        let mut a2_bufs = vec![[0.0f32; SPATIAL_H2]; 32];
        let mut preds = [0.0f32; 32];
        let mut raw_mlps = [0.0f32; 32];
        let mut p_cur = [0.0f32; 32];
        let mut p_ref = [0.0f32; 32];

        let mut kl_sum = 0.0f64;
        let mut n_upd = 0usize;

        for chunk in idxs.chunks(batch_steps) {
            gbuf.zero(sparse_lim);
            let inv_bs = 1.0f32 / (chunk.len() as f32);
            for &si in chunk {
                let st = &steps_all[si];
                let k = st.cands.len().min(32);
                let mut min_v = f32::INFINITY;
                let mut min_rv = f32::INFINITY;
                for c in 0..k {
                    let (v, rm) = forward_cand(
                        &mdl,
                        &st.cands[c],
                        &mut z_bufs[c],
                        &mut a1_bufs[c],
                        &mut a2_bufs[c],
                    );
                    preds[c] = v;
                    raw_mlps[c] = rm;
                    if v < min_v {
                        min_v = v;
                    }
                    if st.cands[c].ref_v < min_rv {
                        min_rv = st.cands[c].ref_v;
                    }
                }
                let mut z_cur = 0.0f32;
                let mut z_ref = 0.0f32;
                for c in 0..k {
                    let ec = (-((preds[c] - min_v) / tau).clamp(-20.0, 20.0)).exp();
                    let er = (-((st.cands[c].ref_v - min_rv) / tau).clamp(-20.0, 20.0)).exp();
                    p_cur[c] = ec;
                    p_ref[c] = er;
                    z_cur += ec;
                    z_ref += er;
                }
                let inv_zc = 1.0 / z_cur.max(1e-12);
                let inv_zr = 1.0 / z_ref.max(1e-12);
                for c in 0..k {
                    p_cur[c] *= inv_zc;
                    p_ref[c] *= inv_zr;
                    if p_ref[c] > 1e-7 && p_cur[c] > 1e-7 {
                        kl_sum += (p_ref[c] * (p_ref[c] / p_cur[c]).ln()) as f64;
                    }
                }
                n_upd += 1;

                let inv_tau = 1.0 / tau;
                for c in 0..k {
                    let ind = if c == st.chosen_idx { 1.0f32 } else { 0.0f32 };
                    let d_pg = st.adv * (ind - p_cur[c]) * inv_tau;
                    let d_kl = kl_beta * (p_ref[c] - p_cur[c]) * inv_tau;
                    let d_out = (d_pg + d_kl) * inv_bs;
                    backward_cand(
                        &mdl,
                        &st.cands[c],
                        &z_bufs[c],
                        &a1_bufs[c],
                        &a2_bufs[c],
                        raw_mlps[c],
                        d_out,
                        cur_lr_mlp > 0.0,
                        &mut gbuf,
                    );
                }
            }
            adam.apply(
                &mut mdl,
                &ref_mdl,
                &gbuf,
                cur_lr_lin,
                cur_lr_mlp,
                anchor_wd,
                sparse_lim,
            );
        }

        mdl.prepare_fast();
        let dt_tr = t_tr.elapsed().as_secs_f64();
        let mean_kl = kl_sum / (n_upd.max(1) as f64);

        if iter % 2 == 0 || iter == iters {
            let (v_ok, v_top, v_mean, v_min, v_max) =
                eval_greedy_val(&val_seeds, &Arc::new(mdl.clone()), sparse_lim, threads);
            let improved = (v_top < best_top) || (v_top == best_top && v_mean < best_val);
            if improved {
                best_val = v_mean;
                best_top = v_top;
                mdl.save_to_file(&out_path).expect("save best grpo model");
            }
            println!(
                "[Iter {:2}/{}] tr_ok={}/{} tr_mean={:.2} top={} steps={} kl={:.4} ({:.1}s+{:.1}s) | VAL={}/{} top={} mean={:.2} [{}..{}] {}",
                iter,
                iters,
                tr_pcs.len(),
                seeds_per_iter * group_size,
                tr_mean,
                tr_top,
                steps_all.len(),
                mean_kl,
                dt_gen,
                dt_tr,
                v_ok,
                val_games,
                v_top,
                v_mean,
                v_min,
                v_max,
                if improved { "*** NEW BEST ***" } else { "" }
            );
        } else {
            println!(
                "[Iter {:2}/{}] tr_ok={}/{} tr_mean={:.2} top={} steps={} kl={:.4} ({:.1}s+{:.1}s)",
                iter,
                iters,
                tr_pcs.len(),
                seeds_per_iter * group_size,
                tr_mean,
                tr_top,
                steps_all.len(),
                mean_kl,
                dt_gen,
                dt_tr
            );
        }
        let _ = std::io::stdout().flush();
    }

    let last_path = if out_path.ends_with(".txt") {
        format!("{}_last.txt", &out_path[..out_path.len() - 4])
    } else {
        format!("{}_last", out_path)
    };
    mdl.save_to_file(&last_path).expect("save last grpo model");
    println!(
        "=== DONE CRN-GRPO: base_val={:.2} (top={}) -> best_val={:.2} (top={}) | saved {} ===",
        v_mean0, v_top0, best_val, best_top, out_path
    );
}
