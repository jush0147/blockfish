use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        beam::FxHasher,
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        spatial_val::SpatialValueModel,
        BagTracker, BeamConfig, FastAI,
    },
    Color, Input, StackerExt as _,
};
use std::collections::HashSet;
use std::convert::TryInto;
use std::hash::BuildHasherDefault;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Instant;

type FxBuild = BuildHasherDefault<FxHasher>;

const MOVE_BYTES: usize = 44;

struct ResNetClient {
    stream: UnixStream,
    req_buf: Vec<u8>,
    resp_buf: Vec<u8>,
}

impl ResNetClient {
    fn connect(sock_path: &str) -> Self {
        let stream = UnixStream::connect(sock_path)
            .unwrap_or_else(|e| panic!("Failed to connect to {}: {}", sock_path, e));
        Self {
            stream,
            req_buf: Vec::with_capacity(1 << 20),
            resp_buf: Vec::with_capacity(1 << 18),
        }
    }

    /// Evaluate N packed 44-byte MoveRecs on GPU; returns Vec<(logit, val)>.
    fn eval_batch(&mut self, packed: &[u8]) -> Vec<(f32, f32)> {
        let n = packed.len() / MOVE_BYTES;
        if n == 0 {
            return Vec::new();
        }
        self.req_buf.clear();
        self.req_buf.extend_from_slice(&(n as u32).to_le_bytes());
        self.req_buf.extend_from_slice(packed);
        self.stream.write_all(&self.req_buf).expect("socket write failed");

        self.resp_buf.resize(n * 8, 0);
        self.stream.read_exact(&mut self.resp_buf).expect("socket read failed");

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let off = i * 8;
            let logit = f32::from_le_bytes(self.resp_buf[off..off + 4].try_into().unwrap());
            let val = f32::from_le_bytes(self.resp_buf[off + 4..off + 8].try_into().unwrap());
            out.push((logit, val));
        }
        out
    }
}

#[inline]
fn pack_move_rec(
    out: &mut Vec<u8>,
    b: &FastBoard,
    rem_g: u16,
    held: Option<u8>,
    next5: [u8; 5],
    bag_mask: u8,
    lines: u8,
    ds_d: u8,
) {
    let mut rec = [0u8; MOVE_BYTES];
    for r in 0..16 {
        let le = (b.data[r] & 0x3FF).to_le_bytes();
        rec[2 * r] = le[0];
        rec[2 * r + 1] = le[1];
    }
    rec[32] = b.rows;
    rec[33] = b.garbage_rows;
    rec[34] = rem_g.min(255) as u8;
    rec[35] = held.unwrap_or(7);
    rec[36..41].copy_from_slice(&next5);
    rec[41] = bag_mask;
    rec[42] = lines;
    rec[43] = ds_d;
    out.extend_from_slice(&rec);
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

fn next_rng(rng: &mut u64) -> u64 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *rng
}

fn spawn_random_cheese(b: &mut FastBoard, rem_g: u16, lines_cleared: u8, rng: &mut u64) -> bool {
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
    let mut prev_hole: usize = if g > 0 {
        let m = (!b.data[0]) & 0x3FF;
        if m != 0 { m.trailing_zeros() as usize } else { 10 }
    } else {
        10
    };
    for r in (0..old_rows).rev() {
        b.data[r + num] = b.data[r];
    }
    for k in 0..num {
        let r = next_rng(rng);
        let hole = if prev_hole < 10 {
            let mut h = ((r >> 33) as usize) % 9;
            if h >= prev_hole {
                h += 1;
            }
            h
        } else {
            ((r >> 33) as usize) % 10
        };
        prev_hole = hole;
        b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
}

/// Sample a legal future queue extending `vis_q` using `after_vis_bag_mask` (strictly BagTracker-only).
/// Also returns `bag_masks[idx]` = remaining 7-bag mask at cursor `idx`!
fn sample_world_queue(vis_q: &[u8], after_vis_mask: u8, total_len: usize, rng: &mut u64) -> (Vec<u8>, Vec<u8>) {
    let mut q = Vec::with_capacity(total_len);
    q.extend_from_slice(vis_q);

    let mut mask = if after_vis_mask == 0 { 0x7F } else { after_vis_mask };
    while q.len() < total_len {
        let mut avail = [0u8; 7];
        let mut cnt = 0usize;
        for p in 0..7u8 {
            if (mask & (1u8 << p)) != 0 {
                avail[cnt] = p;
                cnt += 1;
            }
        }
        // Fisher-Yates shuffle
        for i in (1..cnt).rev() {
            let j = ((next_rng(rng) >> 33) as usize) % (i + 1);
            avail.swap(i, j);
        }
        for i in 0..cnt {
            if q.len() < total_len {
                q.push(avail[i]);
            }
        }
        mask = 0x7F;
    }

    // Reconstruct bag_masks backwards from `vis_q.len()` where mask is `after_vis_mask`
    let mut bag_masks = vec![0x7Fu8; q.len() + 1];
    // Forward from vis_q.len()
    let mut fm = if after_vis_mask == 0 { 0x7F } else { after_vis_mask };
    for idx in vis_q.len()..q.len() {
        bag_masks[idx] = fm;
        fm &= !(1u8 << q[idx]);
        if fm == 0 {
            fm = 0x7F;
        }
    }
    bag_masks[q.len()] = fm;

    // Backwards for visible prefix 0..vis_q.len()
    let mut bm = if after_vis_mask == 0x7F { 0u8 } else { after_vis_mask };
    for idx in (0..vis_q.len()).rev() {
        let bit = 1u8 << q[idx];
        if (bm & bit) != 0 {
            // Bag boundary crossed going backwards
            bm = bit;
        } else {
            bm |= bit;
        }
        bag_masks[idx] = if bm == 0 { 0x7F } else { bm };
    }

    (q, bag_masks)
}

fn next5_at(q: &[u8], qc: usize) -> [u8; 5] {
    let mut out = [7u8; 5];
    for k in 0..5 {
        if let Some(&p) = q.get(qc + k) {
            out[k] = p;
        }
    }
    out
}

fn avail_mask_from_q(held: Option<u8>, q: &[u8], nqc: usize) -> u8 {
    let mut m = 0u8;
    if let Some(h) = held {
        m |= 1u8 << h;
    }
    let end = (nqc + 6).min(q.len());
    for &p in &q[nqc.min(q.len())..end] {
        m |= 1u8 << p;
    }
    if m == 0 { 0x7F } else { m }
}

fn run_greedy_game(
    seed: u64,
    client: &mut Option<ResNetClient>,
    fst: &FastShapeTable,
    hybrid_sm: Option<&SpatialValueModel>,
    delta_w: f32,
) -> (usize, bool) {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(BeamConfig::default());
    let mut tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places = Vec::with_capacity(64);
    let mut packed = Vec::with_capacity(64 * MOVE_BYTES);

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);

    while ds_total < 100 && pieces < 350 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => return (pieces, true),
        };
        let after_vis_mask = tracker.observe(snap.hold, &snap.queue);
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);

        let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
        let cur_held: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
        let (wq, bmasks) = sample_world_queue(&vis_q, after_vis_mask, 10, &mut rng);

        let (c_nh, c_h) = if cur_held.is_some() {
            (wq.get(0).copied(), cur_held)
        } else {
            (wq.get(0).copied(), wq.get(1).copied())
        };
        pgen.generate_for_state(fst, &b0, c_nh, c_h, &mut tmp_places);
        if tmp_places.is_empty() {
            return (pieces, true);
        }

        packed.clear();
        let mut cands: Vec<FastPlace> = Vec::with_capacity(tmp_places.len());
        let mut base_vs: Vec<f32> = Vec::with_capacity(tmp_places.len());
        let mut win_fp: Option<FastPlace> = None;
        let step_rng = next_rng(&mut rng);

        for &fp in &tmp_places {
            let mut b = b0;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            let nrem = rem_g.saturating_sub(ds_d as u16);
            if nrem == 0 && b.rows < 20 {
                win_fp = Some(fp);
                break;
            }
            let mut r_loc = step_rng;
            if !spawn_random_cheese(&mut b, nrem, lines, &mut r_loc) || b.rows >= 20 {
                continue;
            }
            let (nh, nqc) = if cur_held.is_some() {
                if fp.did_hold { (c_nh, 1) } else { (cur_held, 1) }
            } else if fp.did_hold {
                (c_nh, 2)
            } else {
                (None, 1)
            };
            if let Some(sm) = hybrid_sm {
                let in_h = nh.or_else(|| wq.get(nqc).copied());
                let av = avail_mask_from_q(nh, &wq, nqc);
                let bv = ((sm.evaluate_board_with_bag(&b, nrem, in_h, av) as f32) / 100.0).clamp(-50.0, 500.0);
                base_vs.push(bv);
            }
            if hybrid_sm.is_none() || delta_w != 0.0 {
                pack_move_rec(
                    &mut packed,
                    &b,
                    nrem,
                    nh,
                    next5_at(&wq, nqc),
                    bmasks[nqc.min(bmasks.len() - 1)],
                    lines,
                    ds_d,
                );
            }
            cands.push(fp);
        }

        let chosen = if let Some(wfp) = win_fp {
            wfp
        } else if cands.is_empty() {
            tmp_places[0]
        } else if hybrid_sm.is_some() {
            let best_i = if delta_w == 0.0 {
                base_vs
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            } else {
                let evals = client.as_mut().unwrap().eval_batch(&packed);
                base_vs
                    .iter()
                    .zip(evals.iter())
                    .enumerate()
                    .min_by(|a, b| {
                        let sa = a.1 .0 + delta_w * a.1 .1 .1;
                        let sb = b.1 .0 + delta_w * b.1 .1 .1;
                        sa.total_cmp(&sb)
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            };
            cands[best_i]
        } else {
            let scores = client.as_mut().unwrap().eval_batch(&packed);
            let best_i = scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);
            cands[best_i]
        };

        let mut inputs = match ai.inputs_for_place(&snap.matrix, chosen) {
            Some(inp) => inp,
            None => return (pieces, true),
        };
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        pieces += 1;
    }
    (pieces, ds_total < 100)
}

#[derive(Clone, Copy)]
struct BeamState {
    board: FastBoard,
    held: Option<u8>,
    qc: u8,
    rem_g: u16,
    root_idx: u8,
    score: f32,
    won: bool,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct DedupKey {
    board: FastBoard,
    held: Option<u8>,
    qc: u8,
    root_idx: u8,
}

/// 100% Legal Lockstep-Batched GPU root_pimc:
/// Steps all W worlds in lockstep at depth d = 0..H so every ply is ONE single GPU batch!
fn run_pimc_game(
    seed: u64,
    client: &mut Option<ResNetClient>,
    fst: &FastShapeTable,
    worlds: usize,
    horizon: usize,
    beam_w: usize,
    max_per_root: usize,
    cvar_pct: usize,
    val_w: f32,
    logit_w: f32,
    hybrid_sm: Option<&SpatialValueModel>,
    delta_w: f32,
) -> (usize, bool) {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(BeamConfig::default());
    let mut tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places = Vec::with_capacity(64);

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut rng = seed.wrapping_mul(0xD1B54A32D192ED03).wrapping_add(7);
    let mut packed: Vec<u8> = Vec::with_capacity(worlds * beam_w * 48 * MOVE_BYTES);
    let mut seen: HashSet<DedupKey, FxBuild> =
        HashSet::with_capacity_and_hasher(beam_w * 32, FxBuild::default());

    while ds_total < 100 && pieces < 350 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => return (pieces, true),
        };
        let after_vis_mask = tracker.observe(snap.hold, &snap.queue);
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);

        let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
        let cur_held: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);

        let (c_nh0, c_h0) = if cur_held.is_some() {
            (vis_q.get(0).copied(), cur_held)
        } else {
            (vis_q.get(0).copied(), vis_q.get(1).copied())
        };
        pgen.generate_for_state(fst, &b0, c_nh0, c_h0, &mut tmp_places);
        if tmp_places.is_empty() {
            return (pieces, true);
        }
        let root_places = tmp_places.clone();
        let n_roots = root_places.len();

        // Check immediate 1-ply terminal win
        let mut instant_win: Option<FastPlace> = None;
        for &fp in &root_places {
            let mut b = b0;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            if rem_g.saturating_sub(ds_d as u16) == 0 && b.rows < 20 {
                instant_win = Some(fp);
                break;
            }
        }

        let chosen = if let Some(wfp) = instant_win {
            wfp
        } else {
            // Sample W independent legal worlds
            let mut w_queues: Vec<Vec<u8>> = Vec::with_capacity(worlds);
            let mut w_bmasks: Vec<Vec<u8>> = Vec::with_capacity(worlds);
            let mut w_rngs: Vec<u64> = Vec::with_capacity(worlds);
            for _ in 0..worlds {
                let (wq, bm) = sample_world_queue(&vis_q, after_vis_mask, horizon + 8, &mut rng);
                w_queues.push(wq);
                w_bmasks.push(bm);
                w_rngs.push(next_rng(&mut rng));
            }

            let mut beams: Vec<Vec<BeamState>> = vec![Vec::new(); worlds];
            let mut best_win_score: Vec<Vec<f32>> = vec![vec![f32::INFINITY; n_roots]; worlds];
            let need_gpu = hybrid_sm.is_none() || delta_w != 0.0;

            // Depth 1: expand all root_places in all W worlds in lockstep
            packed.clear();
            let mut cand_meta: Vec<(usize, BeamState)> = Vec::with_capacity(worlds * n_roots);

            for w in 0..worlds {
                let step_seed = next_rng(&mut w_rngs[w]);
                for (ri, &fp) in root_places.iter().enumerate() {
                    let mut b = b0;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let nrem = rem_g.saturating_sub(ds_d as u16);
                    if nrem == 0 && b.rows < 20 {
                        best_win_score[w][ri] = best_win_score[w][ri].min(-1000.0 + 1.0);
                        continue;
                    }
                    let mut r_loc = step_seed;
                    if !spawn_random_cheese(&mut b, nrem, lines, &mut r_loc) || b.rows >= 20 {
                        continue;
                    }
                    let (nh, nqc) = if cur_held.is_some() {
                        if fp.did_hold { (c_nh0, 1usize) } else { (cur_held, 1usize) }
                    } else if fp.did_hold {
                        (c_nh0, 2usize)
                    } else {
                        (None, 1usize)
                    };
                    let bv = if let Some(sm) = hybrid_sm {
                        let in_h = nh.or_else(|| w_queues[w].get(nqc).copied());
                        let av = avail_mask_from_q(nh, &w_queues[w], nqc);
                        ((sm.evaluate_board_with_bag(&b, nrem, in_h, av) as f32) / 100.0).clamp(-50.0, 500.0)
                    } else {
                        0.0
                    };
                    if need_gpu {
                        pack_move_rec(
                            &mut packed,
                            &b,
                            nrem,
                            nh,
                            next5_at(&w_queues[w], nqc),
                            w_bmasks[w][nqc.min(w_bmasks[w].len() - 1)],
                            lines,
                            ds_d,
                        );
                    }
                    cand_meta.push((
                        w,
                        BeamState {
                            board: b,
                            held: nh,
                            qc: nqc as u8,
                            rem_g: nrem,
                            root_idx: ri as u8,
                            score: bv,
                            won: false,
                        },
                    ));
                }
            }

            if need_gpu {
                let evals = client.as_mut().unwrap().eval_batch(&packed);
                for ((w, mut st), (logit, val)) in cand_meta.into_iter().zip(evals.into_iter()) {
                    if hybrid_sm.is_some() {
                        st.score = 1.0 + st.score + delta_w * val;
                    } else {
                        st.score = 1.0 + val_w * val - logit_w * logit;
                    }
                    beams[w].push(st);
                }
            } else {
                for (w, mut st) in cand_meta {
                    st.score = 1.0 + st.score;
                    beams[w].push(st);
                }
            }

            // Prune depth 1 beam per world
            for w in 0..worlds {
                beams[w].sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
                if beams[w].len() > beam_w {
                    beams[w].truncate(beam_w);
                }
            }

            // Depths 2..=horizon in lockstep across all W worlds
            for depth in 2..=horizon {
                packed.clear();
                let mut next_meta: Vec<(usize, BeamState)> =
                    Vec::with_capacity(worlds * beam_w * 40);

                for w in 0..worlds {
                    let step_seed = next_rng(&mut w_rngs[w]);
                    for &parent in &beams[w] {
                        if parent.won {
                            continue;
                        }
                        let qc = parent.qc as usize;
                        let (c_nh, c_h) = if parent.held.is_some() {
                            (w_queues[w].get(qc).copied(), parent.held)
                        } else {
                            (w_queues[w].get(qc).copied(), w_queues[w].get(qc + 1).copied())
                        };
                        if c_nh.is_none() && c_h.is_none() {
                            continue;
                        }
                        pgen.generate_for_state(fst, &parent.board, c_nh, c_h, &mut tmp_places);
                        for &fp in &tmp_places {
                            let mut b = parent.board;
                            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                            let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                            let nrem = parent.rem_g.saturating_sub(ds_d as u16);
                            if nrem == 0 && b.rows < 20 {
                                let ws = -1000.0 + depth as f32;
                                let ri = parent.root_idx as usize;
                                if ws < best_win_score[w][ri] {
                                    best_win_score[w][ri] = ws;
                                }
                                continue;
                            }
                            let mut r_loc = step_seed;
                            if !spawn_random_cheese(&mut b, nrem, lines, &mut r_loc) || b.rows >= 20 {
                                continue;
                            }
                            let (nh, nqc) = if parent.held.is_some() {
                                if fp.did_hold { (c_nh, qc + 1) } else { (parent.held, qc + 1) }
                            } else if fp.did_hold {
                                (c_nh, qc + 2)
                            } else {
                                (None, qc + 1)
                            };
                            let bv = if let Some(sm) = hybrid_sm {
                                let in_h = nh.or_else(|| w_queues[w].get(nqc).copied());
                                let av = avail_mask_from_q(nh, &w_queues[w], nqc);
                                ((sm.evaluate_board_with_bag(&b, nrem, in_h, av) as f32) / 100.0).clamp(-50.0, 500.0)
                            } else {
                                0.0
                            };
                            if need_gpu {
                                pack_move_rec(
                                    &mut packed,
                                    &b,
                                    nrem,
                                    nh,
                                    next5_at(&w_queues[w], nqc),
                                    w_bmasks[w][nqc.min(w_bmasks[w].len() - 1)],
                                    lines,
                                    ds_d,
                                );
                            }
                            next_meta.push((
                                w,
                                BeamState {
                                    board: b,
                                    held: nh,
                                    qc: nqc as u8,
                                    rem_g: nrem,
                                    root_idx: parent.root_idx,
                                    score: bv,
                                    won: false,
                                },
                            ));
                        }
                    }
                }

                if next_meta.is_empty() {
                    break;
                }
                let mut per_w: Vec<Vec<BeamState>> = vec![Vec::new(); worlds];
                if need_gpu {
                    let evals = client.as_mut().unwrap().eval_batch(&packed);
                    for ((w, mut st), (logit, val)) in next_meta.into_iter().zip(evals.into_iter()) {
                        if hybrid_sm.is_some() {
                            st.score = (depth as f32) + st.score + delta_w * val;
                        } else {
                            st.score = (depth as f32) + val_w * val - logit_w * logit;
                        }
                        per_w[w].push(st);
                    }
                } else {
                    for (w, mut st) in next_meta {
                        st.score = (depth as f32) + st.score;
                        per_w[w].push(st);
                    }
                }

                for w in 0..worlds {
                    per_w[w].sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
                    seen.clear();
                    let mut root_cnt = vec![0usize; n_roots];
                    beams[w].clear();
                    for st in per_w[w].drain(..) {
                        if beams[w].len() >= beam_w {
                            break;
                        }
                        let ri = st.root_idx as usize;
                        if root_cnt[ri] >= max_per_root {
                            continue;
                        }
                        let dk = DedupKey {
                            board: st.board,
                            held: st.held,
                            qc: st.qc,
                            root_idx: st.root_idx,
                        };
                        if seen.insert(dk) {
                            root_cnt[ri] += 1;
                            beams[w].push(st);
                        }
                    }
                }
            }

            // Aggregate root scores across W worlds (mean or CVaR tail)
            let mut best_ri = 0usize;
            let mut best_agg = f32::INFINITY;
            let mut w_scores = Vec::with_capacity(worlds);

            for ri in 0..n_roots {
                w_scores.clear();
                for w in 0..worlds {
                    let mut sc = best_win_score[w][ri];
                    for st in &beams[w] {
                        if st.root_idx as usize == ri && st.score < sc {
                            sc = st.score;
                        }
                    }
                    if sc.is_finite() {
                        w_scores.push(sc);
                    }
                }
                if w_scores.is_empty() {
                    continue;
                }
                // Penalize roots missing from some worlds
                let missing = worlds - w_scores.len();
                for _ in 0..missing {
                    w_scores.push(500.0);
                }
                w_scores.sort_unstable_by(|a, b| a.total_cmp(b));
                let agg = if cvar_pct > 0 && cvar_pct < 100 {
                    let tail_n = ((worlds * cvar_pct + 99) / 100).max(1);
                    let start = worlds - tail_n;
                    let sum: f32 = w_scores[start..].iter().sum();
                    sum / (tail_n as f32)
                } else {
                    let sum: f32 = w_scores.iter().sum();
                    sum / (worlds as f32)
                };
                if agg < best_agg {
                    best_agg = agg;
                    best_ri = ri;
                }
            }
            root_places[best_ri]
        };

        let mut inputs = match ai.inputs_for_place(&snap.matrix, chosen) {
            Some(inp) => inp,
            None => return (pieces, true),
        };
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        pieces += 1;
    }
    (pieces, ds_total < 100)
}

fn extended_queue(stacker: &Stacker, n: usize) -> Vec<Color> {
    let mut st = stacker.clone();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let snap = match st.snapshot() {
            Some(s) => s,
            None => break,
        };
        if let Some(&c) = snap.queue.first() {
            out.push(c);
        } else {
            break;
        }
        st.hard_drop();
    }
    out
}

fn collect_exact_seed(s: u64) -> Option<(Vec<u16>, Vec<u8>)> {
    let hole_at_row = |st: &Stacker, target_r: u16| -> u16 {
        let mut present = [false; 10];
        for ((r, c), _) in st.matrix() {
            if r == target_r {
                present[c as usize] = true;
            }
        }
        (0..10u16).find(|&c| !present[c as usize]).unwrap()
    };
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(s);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut holes = Vec::with_capacity(100);
    let g0 = garbage_rows(&stacker);
    for r in (0..g0).rev() {
        holes.push(hole_at_row(&stacker, r));
    }
    let mut pieces = Vec::with_capacity(340);
    pieces.push(color_to_idx(stacker.current_piece_type().unwrap().try_into().unwrap()) as u8);
    for &ch in stacker.next() {
        pieces.push(color_to_idx(ch.try_into().unwrap()) as u8);
    }
    let mut c = BeamConfig::default();
    c.beam_width = 24;
    c.max_per_root = 6;
    c.rollout_top_k = 0;
    let mut ai = FastAI::new(c);
    let mut ds_total = 0usize;
    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot()?;
        let g_before = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = ai.best_move(&snap, Some(g_before), Some(rem_g))?;
        let had_hold = stacker.held().is_some();
        let did_hold = res.inputs.first() == Some(&Input::Hold);
        let mut inputs = res.inputs;
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        let g_after = garbage_rows(&stacker);
        let new_g = g_after.saturating_sub(g_before.saturating_sub(ds as u16));
        for r in (0..new_g).rev() {
            holes.push(hole_at_row(&stacker, r));
        }
        let nx = stacker.next();
        if did_hold && !had_hold && nx.len() >= 2 {
            pieces.push(color_to_idx(nx[nx.len() - 2].try_into().unwrap()) as u8);
            pieces.push(color_to_idx(nx[nx.len() - 1].try_into().unwrap()) as u8);
        } else if !nx.is_empty() {
            pieces.push(color_to_idx(nx[nx.len() - 1].try_into().unwrap()) as u8);
        }
    }
    if holes.len() != 100 {
        return None;
    }
    let skip_n = stacker.snapshot().map_or(0, |sn| sn.queue.len());
    if stacker.current_piece_type().is_some() {
        for &col in extended_queue(&stacker, 260).iter().skip(skip_n) {
            pieces.push(color_to_idx(col) as u8);
        }
    }
    Some((holes, pieces))
}

fn spawn_exact_cheese(b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]) -> bool {
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
    let spawned_so_far = 100usize - unspawned;
    if spawned_so_far + num > holes.len() {
        return false;
    }
    for r in (0..old_rows).rev() {
        b.data[r + num] = b.data[r];
    }
    for k in 0..num {
        let hole = holes[spawned_so_far + k];
        b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
}

fn bag_mask_at(qc: usize, pieces: &[u8]) -> u8 {
    let bag_end = ((qc / 7) + 1) * 7;
    let mut m = 0u8;
    for idx in qc..bag_end.min(pieces.len()) {
        m |= 1u8 << pieces[idx];
    }
    if m == 0 { 0x7F } else { m }
}

/// Diagnostic-only Oracle 20-preview + exact holes beam search to measure evaluator ceiling.
fn run_or_p20_game(
    seed: u64,
    client: &mut ResNetClient,
    fst: &FastShapeTable,
    horizon: usize,
    beam_w: usize,
    max_per_root: usize,
    val_w: f32,
    logit_w: f32,
) -> (usize, bool) {
    let (holes, ed_pieces) = match collect_exact_seed(seed) {
        Some(x) => x,
        None => return (0, true),
    };
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let snap0 = stacker.snapshot().unwrap();
    let mut cur_b = FastBoard::from_basic_matrix(&snap0.matrix, garbage_rows(&stacker));
    let mut cur_held: Option<u8> = None;
    let mut cur_qc: usize = 0;
    let mut cur_rem_g: u16 = 100;
    let mut pieces = 0usize;

    let mut pgen = FastPlaceGen::new();
    let mut tmp_places = Vec::with_capacity(64);
    let mut packed = Vec::with_capacity(beam_w * 48 * MOVE_BYTES);
    let mut seen: HashSet<DedupKey, FxBuild> =
        HashSet::with_capacity_and_hasher(beam_w * 32, FxBuild::default());

    while cur_rem_g > 0 && pieces < 350 {
        let (c_nh0, c_h0) = if cur_held.is_some() {
            (ed_pieces.get(cur_qc).copied(), cur_held)
        } else {
            (ed_pieces.get(cur_qc).copied(), ed_pieces.get(cur_qc + 1).copied())
        };
        pgen.generate_for_state(fst, &cur_b, c_nh0, c_h0, &mut tmp_places);
        if tmp_places.is_empty() {
            return (pieces, true);
        }
        let root_places = tmp_places.clone();
        let n_roots = root_places.len();

        let mut best_win: Option<(f32, usize)> = None;
        packed.clear();
        let mut beam: Vec<BeamState> = Vec::with_capacity(beam_w);
        let mut cand_meta: Vec<BeamState> = Vec::with_capacity(n_roots);

        for (ri, &fp) in root_places.iter().enumerate() {
            let mut b = cur_b;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            let nrem = cur_rem_g.saturating_sub(ds_d as u16);
            if nrem == 0 && b.rows < 20 {
                best_win = Some((-1000.0 + 1.0, ri));
                break;
            }
            if !spawn_exact_cheese(&mut b, nrem, lines, &holes) || b.rows >= 20 {
                continue;
            }
            let (nh, nqc) = if cur_held.is_some() {
                if fp.did_hold { (c_nh0, cur_qc + 1) } else { (cur_held, cur_qc + 1) }
            } else if fp.did_hold {
                (c_nh0, cur_qc + 2)
            } else {
                (None, cur_qc + 1)
            };
            pack_move_rec(
                &mut packed,
                &b,
                nrem,
                nh,
                next5_at(&ed_pieces, nqc),
                bag_mask_at(nqc, &ed_pieces),
                lines,
                ds_d,
            );
            cand_meta.push(BeamState {
                board: b,
                held: nh,
                qc: (nqc - cur_qc) as u8,
                rem_g: nrem,
                root_idx: ri as u8,
                score: 0.0,
                won: false,
            });
        }

        let chosen_ri = if let Some((_, ri)) = best_win {
            ri
        } else {
            let evals = client.eval_batch(&packed);
            for (mut st, (logit, val)) in cand_meta.into_iter().zip(evals.into_iter()) {
                st.score = 1.0 + val_w * val - logit_w * logit;
                beam.push(st);
            }
            beam.sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
            if beam.len() > beam_w {
                beam.truncate(beam_w);
            }

            for depth in 2..=horizon {
                packed.clear();
                let mut next_meta: Vec<BeamState> = Vec::with_capacity(beam_w * 40);
                for &parent in &beam {
                    let qc = cur_qc + parent.qc as usize;
                    let (c_nh, c_h) = if parent.held.is_some() {
                        (ed_pieces.get(qc).copied(), parent.held)
                    } else {
                        (ed_pieces.get(qc).copied(), ed_pieces.get(qc + 1).copied())
                    };
                    if c_nh.is_none() && c_h.is_none() {
                        continue;
                    }
                    pgen.generate_for_state(fst, &parent.board, c_nh, c_h, &mut tmp_places);
                    for &fp in &tmp_places {
                        let mut b = parent.board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let nrem = parent.rem_g.saturating_sub(ds_d as u16);
                        if nrem == 0 && b.rows < 20 {
                            let ws = -1000.0 + depth as f32;
                            if best_win.map_or(true, |(bw, _)| ws < bw) {
                                best_win = Some((ws, parent.root_idx as usize));
                            }
                            continue;
                        }
                        if !spawn_exact_cheese(&mut b, nrem, lines, &holes) || b.rows >= 20 {
                            continue;
                        }
                        let (nh, nqc) = if parent.held.is_some() {
                            if fp.did_hold { (c_nh, qc + 1) } else { (parent.held, qc + 1) }
                        } else if fp.did_hold {
                            (c_nh, qc + 2)
                        } else {
                            (None, qc + 1)
                        };
                        pack_move_rec(
                            &mut packed,
                            &b,
                            nrem,
                            nh,
                            next5_at(&ed_pieces, nqc),
                            bag_mask_at(nqc, &ed_pieces),
                            lines,
                            ds_d,
                        );
                        next_meta.push(BeamState {
                            board: b,
                            held: nh,
                            qc: (nqc - cur_qc) as u8,
                            rem_g: nrem,
                            root_idx: parent.root_idx,
                            score: 0.0,
                            won: false,
                        });
                    }
                }
                if next_meta.is_empty() {
                    break;
                }
                let evals = client.eval_batch(&packed);
                let mut scored: Vec<BeamState> = next_meta
                    .into_iter()
                    .zip(evals.into_iter())
                    .map(|(mut st, (logit, val))| {
                        st.score = (depth as f32) + val_w * val - logit_w * logit;
                        st
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
                seen.clear();
                let mut root_cnt = vec![0usize; n_roots];
                beam.clear();
                for st in scored {
                    if beam.len() >= beam_w {
                        break;
                    }
                    let ri = st.root_idx as usize;
                    if root_cnt[ri] >= max_per_root {
                        continue;
                    }
                    let dk = DedupKey {
                        board: st.board,
                        held: st.held,
                        qc: st.qc,
                        root_idx: st.root_idx,
                    };
                    if seen.insert(dk) {
                        root_cnt[ri] += 1;
                        beam.push(st);
                    }
                }
            }

            if let Some((_, ri)) = best_win {
                ri
            } else {
                beam.first().map(|st| st.root_idx as usize).unwrap_or(0)
            }
        };

        let chosen = root_places[chosen_ri];
        let o = &fst.shapes[chosen.color_idx as usize].orients[chosen.tf.2 as usize];
        let (lines, ds_d, _) = cur_b.place_and_sift(o, chosen.norm_i, chosen.norm_j);
        cur_rem_g = cur_rem_g.saturating_sub(ds_d as u16);
        pieces += 1;
        if cur_rem_g == 0 {
            break;
        }
        if !spawn_exact_cheese(&mut cur_b, cur_rem_g, lines, &holes) || cur_b.rows >= 20 {
            return (pieces, true);
        }
        let (nh, nqc) = if cur_held.is_some() {
            if chosen.did_hold { (c_nh0, cur_qc + 1) } else { (cur_held, cur_qc + 1) }
        } else if chosen.did_hold {
            (c_nh0, cur_qc + 2)
        } else {
            (None, cur_qc + 1)
        };
        cur_held = nh;
        cur_qc = nqc;
    }
    (pieces, cur_rem_g > 0)
}

/// Full-strength Per-Root Leaf-Batched GPU root_pimc:
/// 1. Endgame solver via FastAI when rem_g <= 5.
/// 2. Fast depth-4 spatial_bc4 beam prescreen to select top K roots (preserving multi-ply setups).
/// 3. For each (world in 0..W, root in 0..K), run dedicated H-ply beam (width B) with spatial_bc4.
/// 4. Batch ALL W * K * L top leaves into ONE SINGLE GPU call for ResNet residual delta_v!
fn run_pimc_v2_game(
    seed: u64,
    client: &mut Option<ResNetClient>,
    fst: &FastShapeTable,
    worlds: usize,
    horizon: usize,
    beam_w: usize,
    top_roots: usize,
    leaf_top: usize,
    cvar_pct: usize,
    sm: &SpatialValueModel,
    delta_w: f32,
) -> (usize, bool) {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut eg_cfg = BeamConfig::default();
    eg_cfg.beam_width = 80;
    eg_cfg.max_per_root = 16;
    eg_cfg.endgame_solver = true;
    eg_cfg.bag_det_extend = true;
    eg_cfg.simulate_cheese_spawn = true;
    let mut ai = FastAI::new(eg_cfg);

    let mut tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_places = Vec::with_capacity(64);
    let mut seen: HashSet<DedupKey, FxBuild> =
        HashSet::with_capacity_and_hasher(beam_w * 16, FxBuild::default());
    let mut packed: Vec<u8> = Vec::with_capacity(worlds * top_roots * leaf_top * MOVE_BYTES);

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    let mut rng = seed.wrapping_mul(0xD1B54A32D192ED03).wrapping_add(17);

    while ds_total < 100 && pieces < 350 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => return (pieces, true),
        };
        let after_vis_mask = tracker.observe(snap.hold, &snap.queue);
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;

        // Exact endgame solver for rem_g <= 5
        if rem_g <= 5 {
            if let Some(res) = ai.best_move(&snap, Some(g), Some(rem_g)) {
                let mut inputs = res.inputs;
                inputs.pop();
                stacker.run(inputs);
                let (_, ds) = stacker.hard_drop();
                ds_total += ds;
                pieces += 1;
                continue;
            }
        }

        let b0 = FastBoard::from_basic_matrix(&snap.matrix, g);
        let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
        let cur_held: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
        let (c_nh0, c_h0) = if cur_held.is_some() {
            (vis_q.get(0).copied(), cur_held)
        } else {
            (vis_q.get(0).copied(), vis_q.get(1).copied())
        };
        pgen.generate_for_state(fst, &b0, c_nh0, c_h0, &mut tmp_places);
        if tmp_places.is_empty() {
            return (pieces, true);
        }
        let root_places = tmp_places.clone();
        let n_roots = root_places.len();

        // Immediate 1-ply terminal win check
        let mut instant_win: Option<FastPlace> = None;
        for &fp in &root_places {
            let mut b = b0;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            if rem_g.saturating_sub(ds_d as u16) == 0 && b.rows < 20 {
                instant_win = Some(fp);
                break;
            }
        }

        let chosen = if let Some(wfp) = instant_win {
            wfp
        } else {
            // Step 1: Fast depth-4 spatial_bc4 prescreen to select top K roots (with CRN holes)
            let mut root_best_pre = vec![f32::INFINITY; n_roots];
            {
                let mut pre_rng = next_rng(&mut rng);
                let (pq, _) = sample_world_queue(&vis_q, after_vis_mask, 8, &mut pre_rng);
                let mut pbeam: Vec<BeamState> = Vec::with_capacity(64);
                let s1 = next_rng(&mut pre_rng);
                for (ri, &fp) in root_places.iter().enumerate() {
                    let mut b = b0;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let nrem = rem_g.saturating_sub(ds_d as u16);
                    let mut rl = s1;
                    if !spawn_random_cheese(&mut b, nrem, lines, &mut rl) || b.rows >= 20 {
                        continue;
                    }
                    let (nh, nqc) = if cur_held.is_some() {
                        if fp.did_hold { (c_nh0, 1usize) } else { (cur_held, 1usize) }
                    } else if fp.did_hold {
                        (c_nh0, 2usize)
                    } else {
                        (None, 1usize)
                    };
                    let in_h = nh.or_else(|| pq.get(nqc).copied());
                    let av = avail_mask_from_q(nh, &pq, nqc);
                    let sc = 1.0 + ((sm.evaluate_linear_board_with_bag_no_pe(&b, nrem, in_h, av) as f32) / 100.0).clamp(-50.0, 500.0);
                    if sc < root_best_pre[ri] {
                        root_best_pre[ri] = sc;
                    }
                    pbeam.push(BeamState {
                        board: b,
                        held: nh,
                        qc: nqc as u8,
                        rem_g: nrem,
                        root_idx: ri as u8,
                        score: sc,
                        won: false,
                    });
                }
                pbeam.sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
                pbeam.truncate(64);
                for d in 2..=4usize {
                    let sd = next_rng(&mut pre_rng);
                    let mut nxt: Vec<BeamState> = Vec::with_capacity(64 * 32);
                    for &par in &pbeam {
                        let qc = par.qc as usize;
                        let (c_nh, c_h) = if par.held.is_some() {
                            (pq.get(qc).copied(), par.held)
                        } else {
                            (pq.get(qc).copied(), pq.get(qc + 1).copied())
                        };
                        if c_nh.is_none() && c_h.is_none() { continue; }
                        pgen.generate_for_state(fst, &par.board, c_nh, c_h, &mut tmp_places);
                        for &fp in &tmp_places {
                            let mut b = par.board;
                            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                            let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                            let nrem = par.rem_g.saturating_sub(ds_d as u16);
                            if nrem == 0 && b.rows < 20 {
                                let ws = -1000.0 + d as f32;
                                if ws < root_best_pre[par.root_idx as usize] {
                                    root_best_pre[par.root_idx as usize] = ws;
                                }
                                continue;
                            }
                            let mut rl = sd;
                            if !spawn_random_cheese(&mut b, nrem, lines, &mut rl) || b.rows >= 20 {
                                continue;
                            }
                            let (nh, nqc) = if par.held.is_some() {
                                if fp.did_hold { (c_nh, qc + 1) } else { (par.held, qc + 1) }
                            } else if fp.did_hold {
                                (c_nh, qc + 2)
                            } else {
                                (None, qc + 1)
                            };
                            let in_h = nh.or_else(|| pq.get(nqc).copied());
                            let av = avail_mask_from_q(nh, &pq, nqc);
                            let sc = (d as f32) + ((sm.evaluate_linear_board_with_bag_no_pe(&b, nrem, in_h, av) as f32) / 100.0).clamp(-50.0, 500.0);
                            if d == 4 && sc < root_best_pre[par.root_idx as usize] {
                                root_best_pre[par.root_idx as usize] = sc;
                            }
                            nxt.push(BeamState {
                                board: b,
                                held: nh,
                                qc: nqc as u8,
                                rem_g: nrem,
                                root_idx: par.root_idx,
                                score: sc,
                                won: false,
                            });
                        }
                    }
                    nxt.sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
                    seen.clear();
                    let mut rc = vec![0usize; n_roots];
                    pbeam.clear();
                    for st in nxt {
                        if pbeam.len() >= 64 { break; }
                        let ri = st.root_idx as usize;
                        if rc[ri] >= 8 { continue; }
                        if seen.insert(DedupKey { board: st.board, held: st.held, qc: st.qc, root_idx: st.root_idx }) {
                            rc[ri] += 1;
                            pbeam.push(st);
                        }
                    }
                }
            }

            let mut root_order: Vec<usize> = (0..n_roots).filter(|&ri| root_best_pre[ri].is_finite()).collect();
            root_order.sort_unstable_by(|&a, &b| root_best_pre[a].total_cmp(&root_best_pre[b]));
            if root_order.is_empty() {
                root_order.push(0);
            }
            root_order.truncate(top_roots.min(root_order.len()));
            let k_roots = root_order.len();

            // Pre-generate deterministic world queues & step seeds so parallel threads are 100% deterministic
            let mut world_inputs: Vec<(Vec<u8>, Vec<u8>, Vec<u64>)> = Vec::with_capacity(worlds);
            for _ in 0..worlds {
                let (wq, bmasks) = sample_world_queue(&vis_q, after_vis_mask, horizon + 8, &mut rng);
                let mut step_seeds = Vec::with_capacity(horizon);
                for _ in 0..horizon {
                    step_seeds.push(next_rng(&mut rng));
                }
                world_inputs.push((wq, bmasks, step_seeds));
            }

            // Step 2: Parallelize all W worlds across CPU cores!
            // Internal depths use fast linear_no_pe; leaf depth uses full MLP + batched GPU ResNet delta_v.
            let mut per_world_out: Vec<(Vec<f32>, Vec<u8>, Vec<(usize, f32)>)> =
                vec![(vec![f32::INFINITY; k_roots], Vec::new(), Vec::new()); worlds];

            let root_places_ref = &root_places;
            let root_order_ref = &root_order;
            let world_inputs_ref = &world_inputs;

            std::thread::scope(|scope| {
                for (w, out_slot) in per_world_out.iter_mut().enumerate() {
                    scope.spawn(move || {
                        let (wq, bmasks, step_seeds) = &world_inputs_ref[w];
                        let mut pgen_w = FastPlaceGen::new();
                        let mut tmp_w = Vec::with_capacity(64);
                        let mut seen_w: HashSet<DedupKey, FxBuild> =
                            HashSet::with_capacity_and_hasher(beam_w * 16, FxBuild::default());
                        let mut cur_beam: Vec<(BeamState, u8, u8)> = Vec::with_capacity(beam_w);
                        let mut nxt_beam: Vec<(BeamState, u8, u8)> = Vec::with_capacity(beam_w * 40);

                        for (ki, &ri) in root_order_ref.iter().enumerate() {
                            let fp0 = root_places_ref[ri];
                            let mut b1 = b0;
                            let o0 = &fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
                            let (lines1, ds1, _) = b1.place_and_sift(o0, fp0.norm_i, fp0.norm_j);
                            let nrem1 = rem_g.saturating_sub(ds1 as u16);
                            if nrem1 == 0 && b1.rows < 20 {
                                out_slot.0[ki] = -1000.0 + 1.0;
                                continue;
                            }
                            let mut rl1 = step_seeds[0];
                            if !spawn_random_cheese(&mut b1, nrem1, lines1, &mut rl1) || b1.rows >= 20 {
                                continue;
                            }
                            let (nh1, nqc1) = if cur_held.is_some() {
                                if fp0.did_hold { (c_nh0, 1usize) } else { (cur_held, 1usize) }
                            } else if fp0.did_hold {
                                (c_nh0, 2usize)
                            } else {
                                (None, 1usize)
                            };
                            let in_h1 = nh1.or_else(|| wq.get(nqc1).copied());
                            let av1 = avail_mask_from_q(nh1, wq, nqc1);
                            let sc1 = 1.0 + ((sm.evaluate_linear_board_with_bag_no_pe(&b1, nrem1, in_h1, av1) as f32) / 100.0).clamp(-50.0, 500.0);

                            cur_beam.clear();
                            cur_beam.push((
                                BeamState {
                                    board: b1,
                                    held: nh1,
                                    qc: nqc1 as u8,
                                    rem_g: nrem1,
                                    root_idx: ki as u8,
                                    score: sc1,
                                    won: false,
                                },
                                lines1,
                                ds1,
                            ));

                            let mut best_win_w = f32::INFINITY;
                            for d in 2..=horizon {
                                let sd = step_seeds[d - 1];
                                let is_leaf = d == horizon;
                                nxt_beam.clear();
                                for &(par, _, _) in &cur_beam {
                                    let qc = par.qc as usize;
                                    let (c_nh, c_h) = if par.held.is_some() {
                                        (wq.get(qc).copied(), par.held)
                                    } else {
                                        (wq.get(qc).copied(), wq.get(qc + 1).copied())
                                    };
                                    if c_nh.is_none() && c_h.is_none() { continue; }
                                    pgen_w.generate_for_state(fst, &par.board, c_nh, c_h, &mut tmp_w);
                                    for &fp in &tmp_w {
                                        let mut b = par.board;
                                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                                        let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                                        let nrem = par.rem_g.saturating_sub(ds_d as u16);
                                        if nrem == 0 && b.rows < 20 {
                                            let ws = -1000.0 + d as f32;
                                            if ws < best_win_w { best_win_w = ws; }
                                            continue;
                                        }
                                        let mut rl = sd;
                                        if !spawn_random_cheese(&mut b, nrem, lines, &mut rl) || b.rows >= 20 {
                                            continue;
                                        }
                                        let (nh, nqc) = if par.held.is_some() {
                                            if fp.did_hold { (c_nh, qc + 1) } else { (par.held, qc + 1) }
                                        } else if fp.did_hold {
                                            (c_nh, qc + 2)
                                        } else {
                                            (None, qc + 1)
                                        };
                                        let in_h = nh.or_else(|| wq.get(nqc).copied());
                                        let av = avail_mask_from_q(nh, wq, nqc);
                                        let raw_ev = if is_leaf {
                                            sm.evaluate_board_with_bag(&b, nrem, in_h, av)
                                        } else {
                                            sm.evaluate_linear_board_with_bag_no_pe(&b, nrem, in_h, av)
                                        };
                                        let sc = (d as f32) + ((raw_ev as f32) / 100.0).clamp(-50.0, 500.0);
                                        nxt_beam.push((
                                            BeamState {
                                                board: b,
                                                held: nh,
                                                qc: nqc as u8,
                                                rem_g: nrem,
                                                root_idx: ki as u8,
                                                score: sc,
                                                won: false,
                                            },
                                            lines,
                                            ds_d,
                                        ));
                                    }
                                }
                                if nxt_beam.is_empty() {
                                    cur_beam.clear();
                                    break;
                                }
                                nxt_beam.sort_unstable_by(|a, b| a.0.score.total_cmp(&b.0.score));
                                seen_w.clear();
                                cur_beam.clear();
                                for item in nxt_beam.drain(..) {
                                    if cur_beam.len() >= beam_w { break; }
                                    let st = item.0;
                                    if seen_w.insert(DedupKey { board: st.board, held: st.held, qc: st.qc, root_idx: 0 }) {
                                        cur_beam.push(item);
                                    }
                                }
                            }

                            if best_win_w.is_finite() {
                                out_slot.0[ki] = best_win_w;
                            } else if delta_w == 0.0 {
                                if let Some((st, _, _)) = cur_beam.first() {
                                    out_slot.0[ki] = st.score;
                                }
                            } else {
                                for &(st, ln, dd) in cur_beam.iter().take(leaf_top) {
                                    let nqc = st.qc as usize;
                                    pack_move_rec(
                                        &mut out_slot.1,
                                        &st.board,
                                        st.rem_g,
                                        st.held,
                                        next5_at(wq, nqc),
                                        bmasks[nqc.min(bmasks.len() - 1)],
                                        ln,
                                        dd,
                                    );
                                    out_slot.2.push((ki, st.score));
                                }
                            }
                        }
                    });
                }
            });

            // Merge per-world outputs into ONE single GPU batch
            let mut w_root_best: Vec<Vec<f32>> = Vec::with_capacity(worlds);
            packed.clear();
            let mut leaf_meta: Vec<(usize, usize, f32)> = Vec::with_capacity(worlds * k_roots * leaf_top);
            for (w, (rb, pbuf, metas)) in per_world_out.into_iter().enumerate() {
                w_root_best.push(rb);
                packed.extend_from_slice(&pbuf);
                for (ki, sc) in metas {
                    leaf_meta.push((w, ki, sc));
                }
            }

            // Step 3: ONE single GPU batch call for all W * K * L leaves!
            if delta_w != 0.0 && !leaf_meta.is_empty() {
                let evals = client.as_mut().unwrap().eval_batch(&packed);
                for ((w, ki, base_sc), (_, delta_v)) in leaf_meta.into_iter().zip(evals.into_iter()) {
                    let sc = base_sc + delta_w * delta_v;
                    if sc < w_root_best[w][ki] {
                        w_root_best[w][ki] = sc;
                    }
                }
            }

            // Step 4: CVaR aggregation across W worlds
            let mut best_ki = 0usize;
            let mut best_agg = f32::INFINITY;
            let mut ws_buf = Vec::with_capacity(worlds);
            for ki in 0..k_roots {
                ws_buf.clear();
                for w in 0..worlds {
                    let sc = w_root_best[w][ki];
                    ws_buf.push(if sc.is_finite() { sc } else { 500.0 });
                }
                ws_buf.sort_unstable_by(|a, b| a.total_cmp(b));
                let agg = if cvar_pct > 0 && cvar_pct < 100 {
                    let tail_n = ((worlds * cvar_pct + 99) / 100).max(1);
                    let start = worlds - tail_n;
                    ws_buf[start..].iter().sum::<f32>() / (tail_n as f32)
                } else {
                    ws_buf.iter().sum::<f32>() / (worlds as f32)
                };
                if agg < best_agg {
                    best_agg = agg;
                    best_ki = ki;
                }
            }
            root_places[root_order[best_ki]]
        };

        let mut inputs = match ai.inputs_for_place(&snap.matrix, chosen) {
            Some(inp) => inp,
            None => return (pieces, true),
        };
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        pieces += 1;
    }
    (pieces, ds_total < 100)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = String::from("greedy");
    let mut sock = String::from("/tmp/blockfish_resnet.sock");
    let mut seed_start: u64 = 101;
    let mut n_seeds: usize = 10;
    let mut worlds: usize = 16;
    let mut horizon: usize = 6;
    let mut beam_w: usize = 16;
    let mut max_per_root: usize = 4;
    let mut top_roots: usize = 10;
    let mut leaf_top: usize = 4;
    let mut cvar_pct: usize = 25;
    let mut val_w: f32 = 1.0;
    let mut logit_w: f32 = 0.0;
    let mut hybrid_model_path: Option<String> = None;
    let mut delta_w: f32 = 1.0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" => { i += 1; mode = args[i].clone(); }
            "--sock" => { i += 1; sock = args[i].clone(); }
            "--seed-start" => { i += 1; seed_start = args[i].parse().unwrap(); }
            "--n-seeds" => { i += 1; n_seeds = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; worlds = args[i].parse().unwrap(); }
            "--horizon" => { i += 1; horizon = args[i].parse().unwrap(); }
            "--beam" => { i += 1; beam_w = args[i].parse().unwrap(); }
            "--max-per-root" => { i += 1; max_per_root = args[i].parse().unwrap(); }
            "--top-roots" => { i += 1; top_roots = args[i].parse().unwrap(); }
            "--leaf-top" => { i += 1; leaf_top = args[i].parse().unwrap(); }
            "--cvar" => { i += 1; cvar_pct = args[i].parse().unwrap(); }
            "--val-w" => { i += 1; val_w = args[i].parse().unwrap(); }
            "--logit-w" => { i += 1; logit_w = args[i].parse().unwrap(); }
            "--hybrid-model" => { i += 1; hybrid_model_path = Some(args[i].clone()); }
            "--delta-w" => { i += 1; delta_w = args[i].parse().unwrap(); }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let fst = FastShapeTable::srs();
    let hybrid_sm = hybrid_model_path.as_deref().map(|p| {
        let mut sm = SpatialValueModel::load_from_file(p).expect("load hybrid spatial model");
        sm.prepare_fast();
        sm
    });
    let need_client = hybrid_sm.is_none() || delta_w != 0.0 || mode == "or_p20";
    let mut client = if need_client {
        Some(ResNetClient::connect(&sock))
    } else {
        None
    };
    let t0 = Instant::now();

    println!(
        "=== eval_resnet mode={} seeds={}..={} (n={}) w={} h={} b={} k={} l={} cvar={} hybrid={} delta_w={:.2} ===",
        mode, seed_start, seed_start + n_seeds as u64 - 1, n_seeds,
        worlds, horizon, beam_w, top_roots, leaf_top, cvar_pct,
        hybrid_sm.is_some(), delta_w
    );

    let mut ok_pcs = Vec::with_capacity(n_seeds);
    let mut topouts = 0usize;
    for k in 0..n_seeds {
        let s = seed_start + k as u64;
        let ts = Instant::now();
        let (pcs, top) = match mode.as_str() {
            "greedy" => run_greedy_game(s, &mut client, &fst, hybrid_sm.as_ref(), delta_w),
            "or_p20" => run_or_p20_game(
                s, client.as_mut().unwrap(), &fst, horizon, beam_w, max_per_root, val_w, logit_w
            ),
            "pimc" => run_pimc_game(
                s, &mut client, &fst, worlds, horizon, beam_w, max_per_root, cvar_pct, val_w, logit_w,
                hybrid_sm.as_ref(), delta_w
            ),
            "pimc_v2" => run_pimc_v2_game(
                s, &mut client, &fst, worlds, horizon, beam_w, top_roots, leaf_top, cvar_pct,
                hybrid_sm.as_ref().expect("--hybrid-model required for pimc_v2"), delta_w
            ),
            other => panic!("Unknown mode: {}", other),
        };
        if top {
            topouts += 1;
        } else {
            ok_pcs.push(pcs);
        }
        println!("  [seed {:3}] pcs={:3} top={} | {:.1}s", s, pcs, top as u8, ts.elapsed().as_secs_f64());
    }

    let ok_n = ok_pcs.len();
    let mean = if ok_n > 0 {
        ok_pcs.iter().sum::<usize>() as f64 / ok_n as f64
    } else {
        0.0
    };
    let min_p = ok_pcs.iter().copied().min().unwrap_or(0);
    let max_p = ok_pcs.iter().copied().max().unwrap_or(0);
    let le166 = ok_pcs.iter().filter(|&&p| p <= 166).count();
    let le161 = ok_pcs.iter().filter(|&&p| p <= 161).count();
    println!(
        "SUMMARY mode={} n={} ok={}/{} mean={:.2} min={} max={} <=166:{} <=161:{} top={} | {:.1}s",
        mode, n_seeds, ok_n, n_seeds, mean, min_p, max_p, le166, le161, topouts, t0.elapsed().as_secs_f64()
    );
}
