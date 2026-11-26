use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        beam::FxHasher,
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
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

fn run_greedy_game(
    seed: u64,
    client: &mut ResNetClient,
    fst: &FastShapeTable,
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
            cands.push(fp);
        }

        let chosen = if let Some(wfp) = win_fp {
            wfp
        } else if cands.is_empty() {
            tmp_places[0]
        } else {
            let scores = client.eval_batch(&packed);
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
    client: &mut ResNetClient,
    fst: &FastShapeTable,
    worlds: usize,
    horizon: usize,
    beam_w: usize,
    max_per_root: usize,
    cvar_pct: usize,
    val_w: f32,
    logit_w: f32,
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

            // World beams: beam[w] holds active states for world w
            let mut beams: Vec<Vec<BeamState>> = vec![Vec::new(); worlds];
            // Best terminal win score per (world, root_idx)
            let mut best_win_score: Vec<Vec<f32>> = vec![vec![f32::INFINITY; n_roots]; worlds];

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
                    cand_meta.push((
                        w,
                        BeamState {
                            board: b,
                            held: nh,
                            qc: nqc as u8,
                            rem_g: nrem,
                            root_idx: ri as u8,
                            score: 0.0,
                            won: false,
                        },
                    ));
                }
            }

            let evals = client.eval_batch(&packed);
            for ((w, mut st), (logit, val)) in cand_meta.into_iter().zip(evals.into_iter()) {
                st.score = 1.0 + val_w * val - logit_w * logit;
                beams[w].push(st);
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
                            next_meta.push((
                                w,
                                BeamState {
                                    board: b,
                                    held: nh,
                                    qc: nqc as u8,
                                    rem_g: nrem,
                                    root_idx: parent.root_idx,
                                    score: 0.0,
                                    won: false,
                                },
                            ));
                        }
                    }
                }

                if next_meta.is_empty() {
                    break;
                }
                let evals = client.eval_batch(&packed);
                let mut per_w: Vec<Vec<BeamState>> = vec![Vec::new(); worlds];
                for ((w, mut st), (logit, val)) in next_meta.into_iter().zip(evals.into_iter()) {
                    st.score = (depth as f32) + val_w * val - logit_w * logit;
                    per_w[w].push(st);
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
    let mut cvar_pct: usize = 25;
    let mut val_w: f32 = 1.0;
    let mut logit_w: f32 = 0.0;

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
            "--cvar" => { i += 1; cvar_pct = args[i].parse().unwrap(); }
            "--val-w" => { i += 1; val_w = args[i].parse().unwrap(); }
            "--logit-w" => { i += 1; logit_w = args[i].parse().unwrap(); }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let fst = FastShapeTable::srs();
    let mut client = ResNetClient::connect(&sock);
    let t0 = Instant::now();

    println!(
        "=== eval_resnet mode={} seeds={}..={} (n={}) w={} h={} b={} mpr={} cvar={} val_w={:.2} logit_w={:.2} ===",
        mode, seed_start, seed_start + n_seeds as u64 - 1, n_seeds,
        worlds, horizon, beam_w, max_per_root, cvar_pct, val_w, logit_w
    );

    let mut ok_pcs = Vec::with_capacity(n_seeds);
    let mut topouts = 0usize;
    for k in 0..n_seeds {
        let s = seed_start + k as u64;
        let ts = Instant::now();
        let (pcs, top) = match mode.as_str() {
            "greedy" => run_greedy_game(s, &mut client, &fst),
            "or_p20" => run_or_p20_game(s, &mut client, &fst, horizon, beam_w, max_per_root, val_w, logit_w),
            "pimc" => run_pimc_game(
                s, &mut client, &fst, worlds, horizon, beam_w, max_per_root, cvar_pct, val_w, logit_w
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
