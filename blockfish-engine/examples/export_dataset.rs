use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        beam::FxHasher,
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        fast_eval::{eval_fast_with_bag, EvalWeights},
        BeamConfig, FastAI,
    },
    Color, Input, Orientation, StackerExt as _,
};
use std::collections::HashSet;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

type FxBuild = BuildHasherDefault<FxHasher>;

struct ExactSeedData {
    holes: Vec<u16>,
    pieces: Vec<u8>,
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

fn extended_queue(stacker: &Stacker, n: usize) -> Vec<Color> {
    use std::convert::TryInto;
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
    if out.len() < n {
        if let Some(c) = st.current_piece_type() {
            if let Ok(col) = c.try_into() {
                out.push(col);
            }
        }
    }
    out
}

fn collect_exact_seed(s: u64) -> Option<ExactSeedData> {
    use std::convert::TryInto;
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
    Some(ExactSeedData { holes, pieces })
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

/// Zero-leakage Jstris real-hole spawn using CRN per decision step.
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
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let hole = if prev_hole < 10 {
            let mut h = ((rng >> 33) as usize) % 9;
            if h >= prev_hole {
                h += 1;
            }
            h
        } else {
            ((rng >> 33) as usize) % 10
        };
        prev_hole = hole;
        b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << hole));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
}

fn ew_cons22() -> EvalWeights {
    let mut w = EvalWeights::default();
    w.garbage_row = 165;
    w.piece_estimate = 0;
    w.stack_cell = -22;
    w.upper_cell = 3;
    w.stack_h_sq = 0;
    w.stack_over2_sq = 3;
    w.max_stack_h_excess = 14;
    w.covered_hole = 55;
    w.hole_row = 20;
    w.blocks_over_hole = 14;
    w.h1_block = 32;
    w.h1_contig_cover = 140;
    w.h1_empty_blocking = 8;
    w.h1_height = 10;
    w.h2_block = 26;
    w.h2_contig_cover = 95;
    w.h2_extra_empty = 6;
    w.h3_block = 22;
    w.h3_extra_empty = 3;
    w.h4_block = 8;
    w.h1_narrow_1 = 16;
    w.h1_narrow_2plus = 26;
    w.h1_fit_scarcity = 8;
    w.h1_unfit_hand = 28;
    w.zero_o_spot_pen = 0;
    w.bag_no_i_chute_pen = 18;
    w.open_h1_bonus = 24;
    w.combo2_bonus = 22;
    w.h1_h2_adj = 4;
    w.bumpiness = 11;
    w.sharp_diff = 5;
    w.i_dep = 35;
    w.two_well = 10;
    w.i_dep_multi = 50;
    w.i_dep_chute = 12;
    w.row_transition = 3;
    w.near_full_9_bonus = 0;
    w.near_full_8_bonus = 0;
    w.row_g_aligned_bonus = 0;
    w.endgame_cell_taper = 47;
    w.endgame_cell_taper_rem = 5;
    w.hold_bonus = [-10, 4, -8, 4, 4, -2, -2];
    w
}

fn avail_mask_from_q(held: Option<u8>, qc: usize, pieces: &[u8]) -> u8 {
    let mut m = 0u8;
    if let Some(h) = held {
        m |= 1u8 << h;
    }
    let end = (qc + 6).min(pieces.len());
    for &p in &pieces[qc.min(pieces.len())..end] {
        m |= 1u8 << p;
    }
    if m == 0 { 0x7F } else { m }
}

fn bag_mask_at(qc: usize, pieces: &[u8]) -> u8 {
    let bag_end = ((qc / 7) + 1) * 7;
    let mut m = 0u8;
    for idx in qc..bag_end.min(pieces.len()) {
        m |= 1u8 << pieces[idx];
    }
    if m == 0 { 0x7F } else { m }
}

/// Fixed 44-byte afterstate record for fast zero-copy loading in PyTorch.
#[derive(Clone, Copy)]
struct MoveRec {
    rows: [u16; 16],
    board_rows: u8,
    garbage_rows: u8,
    rem_g: u8,
    held: u8,
    next5: [u8; 5],
    bag_mask: u8,
    lines_cleared: u8,
    ds_cleared: u8,
}

impl MoveRec {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut buf = [0u8; 44];
        for r in 0..16 {
            let le = self.rows[r].to_le_bytes();
            buf[2 * r] = le[0];
            buf[2 * r + 1] = le[1];
        }
        buf[32] = self.board_rows;
        buf[33] = self.garbage_rows;
        buf[34] = self.rem_g;
        buf[35] = self.held;
        buf[36..41].copy_from_slice(&self.next5);
        buf[41] = self.bag_mask;
        buf[42] = self.lines_cleared;
        buf[43] = self.ds_cleared;
        w.write_all(&buf)
    }
}

struct DecisionRec {
    seed: u32,
    step: u16,
    rem_pcs: u16,
    target_idx: u8,
    flags: u8,
    moves: Vec<MoveRec>,
}

impl DecisionRec {
    fn write_to<W: Write>(&self, w: &mut W) -> std::io::Result<()> {
        let mut hdr = [0u8; 12];
        hdr[0..4].copy_from_slice(&self.seed.to_le_bytes());
        hdr[4..6].copy_from_slice(&self.step.to_le_bytes());
        hdr[6..8].copy_from_slice(&self.rem_pcs.to_le_bytes());
        hdr[8] = self.moves.len() as u8;
        hdr[9] = self.target_idx;
        hdr[10] = self.flags;
        hdr[11] = 0;
        w.write_all(&hdr)?;
        for m in &self.moves {
            m.write_to(w)?;
        }
        Ok(())
    }
}

fn solve_and_export_game(seed: u64, wg_beam: usize, is_val: bool) -> Option<(usize, Vec<DecisionRec>)> {
    let ed = collect_exact_seed(seed)?;
    let fst = FastShapeTable::srs();
    let teacher_w = ew_cons22();

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let snap0 = stacker.snapshot()?;
    let g0 = garbage_rows(&stacker);
    let b0 = FastBoard::from_basic_matrix(&snap0.matrix, g0);

    let mut pgen = FastPlaceGen::new();
    let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(64);
    let q_len_ed = ed.pieces.len();

    #[derive(Clone, Copy)]
    struct ArenaNode {
        parent: u32,
        fp: FastPlace,
    }
    #[derive(Clone, Copy)]
    struct DsNode {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: u16,
        pieces_used: u16,
        rem_g: u16,
        score: i64,
        parent_aidx: u32,
        fp: FastPlace,
        has_fp: bool,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct DsKey {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: u16,
    }

    let dummy_fp = FastPlace {
        color_idx: 0,
        did_hold: false,
        norm_i: 0,
        norm_j: 0,
        tf: (0, 0, Orientation::R0),
    };
    let mut arena: Vec<ArenaNode> = Vec::with_capacity(wg_beam * 32);
    let mut buckets: Vec<Vec<DsNode>> = vec![Vec::new(); 101];
    let init_sc = eval_fast_with_bag(&b0, 100, ed.pieces.first().copied(), 0x7F, &teacher_w);
    buckets[0].push(DsNode {
        board: b0,
        held: None,
        q_cursor: 0,
        pieces_used: 0,
        rem_g: 100,
        score: init_sc,
        parent_aidx: u32::MAX,
        fp: dummy_fp,
        has_fp: false,
    });

    let mut best_win: Option<(u16, u8, u32)> = None;
    let mut seen_ds: HashSet<DsKey, FxBuild> =
        HashSet::with_capacity_and_hasher(wg_beam * 8, FxBuild::default());
    let mut cur_sub: Vec<DsNode> = Vec::with_capacity(wg_beam);
    let mut next_sub: Vec<DsNode> = Vec::with_capacity(wg_beam * 16);

    for ds_cur in 0..100usize {
        let mut bucket = std::mem::take(&mut buckets[ds_cur]);
        if bucket.is_empty() {
            continue;
        }
        bucket.sort_unstable_by_key(|n| (n.score, n.pieces_used));
        seen_ds.clear();
        cur_sub.clear();

        let min_np = bucket.iter().map(|n| n.pieces_used).min().unwrap_or(0);
        let mut np_cnt = [0usize; 16];
        let per_np_cap = (wg_beam / 4).max(32);
        let strat_cap = (wg_beam * 3) / 4;
        let mut used = vec![false; bucket.len()];
        for (bi, n_ref) in bucket.iter().enumerate() {
            if cur_sub.len() >= strat_cap {
                break;
            }
            let tier = (n_ref.pieces_used.saturating_sub(min_np) as usize).min(15);
            if np_cnt[tier] >= per_np_cap {
                continue;
            }
            let mut n = *n_ref;
            if seen_ds.insert(DsKey { board: n.board, held: n.held, q_cursor: n.q_cursor }) {
                used[bi] = true;
                np_cnt[tier] += 1;
                if n.has_fp {
                    let aidx = arena.len() as u32;
                    arena.push(ArenaNode { parent: n.parent_aidx, fp: n.fp });
                    n.parent_aidx = aidx;
                    n.has_fp = false;
                }
                cur_sub.push(n);
            }
        }
        for (bi, mut n) in bucket.into_iter().enumerate() {
            if cur_sub.len() >= wg_beam {
                break;
            }
            if used[bi] {
                continue;
            }
            if seen_ds.insert(DsKey { board: n.board, held: n.held, q_cursor: n.q_cursor }) {
                if n.has_fp {
                    let aidx = arena.len() as u32;
                    arena.push(ArenaNode { parent: n.parent_aidx, fp: n.fp });
                    n.parent_aidx = aidx;
                    n.has_fp = false;
                }
                cur_sub.push(n);
            }
        }

        let mut best_next_ds_sc = i64::MAX;
        for z_sub in 0..=10usize {
            if cur_sub.is_empty() {
                break;
            }
            if z_sub > 0 && best_next_ds_sc != i64::MAX && cur_sub[0].score > best_next_ds_sc + 1200 {
                break;
            }
            next_sub.clear();
            for &node in &cur_sub {
                if let Some((wp, _, _)) = best_win {
                    if node.pieces_used + node.rem_g >= wp {
                        continue;
                    }
                }
                let qc = node.q_cursor as usize;
                let (c_nh, c_h) = if node.held.is_some() {
                    (ed.pieces.get(qc).copied(), node.held)
                } else {
                    (ed.pieces.get(qc).copied(), ed.pieces.get(qc + 1).copied())
                };
                if c_nh.is_none() && c_h.is_none() {
                    continue;
                }
                pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                for &fp in &tmp_places {
                    let mut b = node.board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem_g = node.rem_g.saturating_sub(ds_d as u16);
                    let np = node.pieces_used + 1;
                    if rem_g == 0 && b.rows < 20 {
                        let res_cells: u16 = (0..b.rows as usize)
                            .map(|r| (b.data[r] & 0x3FF).count_ones() as u16)
                            .sum();
                        let better = match best_win {
                            None => true,
                            Some((wp, wr, _)) => (np, res_cells.min(255) as u8) < (wp, wr),
                        };
                        if better {
                            let aidx = arena.len() as u32;
                            arena.push(ArenaNode { parent: node.parent_aidx, fp });
                            best_win = Some((np, res_cells.min(255) as u8, aidx));
                        }
                        continue;
                    }
                    if !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
                        continue;
                    }
                    let (held, nqc) = if node.held.is_some() {
                        if fp.did_hold {
                            (c_nh, (qc + 1).min(q_len_ed))
                        } else {
                            (node.held, (qc + 1).min(q_len_ed))
                        }
                    } else if fp.did_hold {
                        (c_nh, (qc + 2).min(q_len_ed))
                    } else {
                        (None, (qc + 1).min(q_len_ed))
                    };
                    let in_h = held.or_else(|| ed.pieces.get(nqc).copied());
                    let av = avail_mask_from_q(held, nqc, &ed.pieces);
                    let sc = (np as i64) * 100 + eval_fast_with_bag(&b, rem_g, in_h, av, &teacher_w);
                    if ds_d > 0 {
                        let nds = (ds_cur + ds_d as usize).min(100);
                        if sc < best_next_ds_sc {
                            best_next_ds_sc = sc;
                        }
                        buckets[nds].push(DsNode {
                            board: b,
                            held,
                            q_cursor: nqc as u16,
                            pieces_used: np,
                            rem_g,
                            score: sc,
                            parent_aidx: node.parent_aidx,
                            fp,
                            has_fp: true,
                        });
                    } else {
                        next_sub.push(DsNode {
                            board: b,
                            held,
                            q_cursor: nqc as u16,
                            pieces_used: np,
                            rem_g,
                            score: sc,
                            parent_aidx: node.parent_aidx,
                            fp,
                            has_fp: true,
                        });
                    }
                }
            }
            if next_sub.is_empty() {
                break;
            }
            next_sub.sort_unstable_by_key(|n| (n.score, n.pieces_used));
            cur_sub.clear();
            for mut ch in next_sub.iter().copied() {
                if cur_sub.len() >= wg_beam {
                    break;
                }
                if seen_ds.insert(DsKey { board: ch.board, held: ch.held, q_cursor: ch.q_cursor }) {
                    if ch.has_fp {
                        let aidx = arena.len() as u32;
                        arena.push(ArenaNode { parent: ch.parent_aidx, fp: ch.fp });
                        ch.parent_aidx = aidx;
                        ch.has_fp = false;
                    }
                    cur_sub.push(ch);
                }
            }
        }
    }

    let (_, _, mut aidx) = best_win?;
    let mut win_moves = Vec::new();
    while aidx != u32::MAX {
        let an = arena[aidx as usize];
        win_moves.push(an.fp);
        aidx = an.parent;
    }
    win_moves.reverse();
    let total_pcs = win_moves.len();

    let mut decisions = Vec::with_capacity(total_pcs);
    let mut cur_b = b0;
    let mut cur_held: Option<u8> = None;
    let mut cur_qc: usize = 0;
    let mut cur_rem_g: u16 = 100;

    for (step, &fp_star) in win_moves.iter().enumerate() {
        let (c_nh, c_h) = if cur_held.is_some() {
            (ed.pieces.get(cur_qc).copied(), cur_held)
        } else {
            (ed.pieces.get(cur_qc).copied(), ed.pieces.get(cur_qc + 1).copied())
        };
        pgen.generate_for_state(&fst, &cur_b, c_nh, c_h, &mut tmp_places);

        let crn = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add((step as u64).wrapping_mul(0xBF58476D1CE4E5B9));

        let mut moves = Vec::with_capacity(tmp_places.len());
        let mut target_idx: Option<usize> = None;
        for &fp in &tmp_places {
            let is_star = fp.color_idx == fp_star.color_idx
                && fp.did_hold == fp_star.did_hold
                && fp.norm_i == fp_star.norm_i
                && fp.norm_j == fp_star.norm_j
                && fp.tf.2 == fp_star.tf.2;
            let mut b = cur_b;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            let rem_g = cur_rem_g.saturating_sub(ds_d as u16);
            if rem_g == 0 && b.rows < 20 {
                if is_star {
                    target_idx = None;
                    break;
                }
                continue;
            }
            let spawn_ok = spawn_random_cheese(&mut b, rem_g, lines, crn);
            if !spawn_ok || b.rows >= 20 {
                if !is_star {
                    continue;
                }
            }
            let (held, nqc) = if cur_held.is_some() {
                if fp.did_hold {
                    (c_nh, (cur_qc + 1).min(q_len_ed))
                } else {
                    (cur_held, (cur_qc + 1).min(q_len_ed))
                }
            } else if fp.did_hold {
                (c_nh, (cur_qc + 2).min(q_len_ed))
            } else {
                (None, (cur_qc + 1).min(q_len_ed))
            };
            let mut rows = [0u16; 16];
            for r in 0..16 {
                rows[r] = b.data[r] & 0x3FF;
            }
            let mut next5 = [7u8; 5];
            for k in 0..5 {
                if let Some(&p) = ed.pieces.get(nqc + k) {
                    next5[k] = p;
                }
            }
            let rec = MoveRec {
                rows,
                board_rows: b.rows,
                garbage_rows: b.garbage_rows,
                rem_g: rem_g.min(255) as u8,
                held: held.unwrap_or(7),
                next5,
                bag_mask: bag_mask_at(nqc, &ed.pieces),
                lines_cleared: lines,
                ds_cleared: ds_d,
            };
            if is_star {
                target_idx = Some(moves.len());
            }
            if moves.len() < 255 {
                moves.push(rec);
            }
        }

        if let Some(ti) = target_idx {
            if moves.len() >= 2 && ti < moves.len() {
                decisions.push(DecisionRec {
                    seed: seed as u32,
                    step: step as u16,
                    rem_pcs: (total_pcs.saturating_sub(step + 1)) as u16,
                    target_idx: ti as u8,
                    flags: if is_val { 1 } else { 0 },
                    moves,
                });
            }
        }

        // Advance along exact expert trajectory
        let o_star = &fst.shapes[fp_star.color_idx as usize].orients[fp_star.tf.2 as usize];
        let (lines, ds_d, _) = cur_b.place_and_sift(o_star, fp_star.norm_i, fp_star.norm_j);
        cur_rem_g = cur_rem_g.saturating_sub(ds_d as u16);
        if cur_rem_g == 0 {
            break;
        }
        if !spawn_exact_cheese(&mut cur_b, cur_rem_g, lines, &ed.holes) {
            break;
        }
        let (nh, nqc) = if cur_held.is_some() {
            if fp_star.did_hold {
                (c_nh, (cur_qc + 1).min(q_len_ed))
            } else {
                (cur_held, (cur_qc + 1).min(q_len_ed))
            }
        } else if fp_star.did_hold {
            (c_nh, (cur_qc + 2).min(q_len_ed))
        } else {
            (None, (cur_qc + 1).min(q_len_ed))
        };
        cur_held = nh;
        cur_qc = nqc;
    }

    Some((total_pcs, decisions))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut train_games: usize = 600;
    let mut val_games: usize = 40;
    let mut train_seed_start: u64 = 1001;
    let mut val_seed_start: u64 = 201;
    let mut wg_beam: usize = 1400;
    let mut threads: usize = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let mut out_path = String::from("results/wg_ds_v1.bin");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--train-games" => {
                i += 1;
                train_games = args[i].parse().unwrap();
            }
            "--val-games" => {
                i += 1;
                val_games = args[i].parse().unwrap();
            }
            "--train-seed-start" => {
                i += 1;
                train_seed_start = args[i].parse().unwrap();
            }
            "--val-seed-start" => {
                i += 1;
                val_seed_start = args[i].parse().unwrap();
            }
            "--wg-beam" => {
                i += 1;
                wg_beam = args[i].parse().unwrap();
            }
            "--threads" => {
                i += 1;
                threads = args[i].parse().unwrap();
            }
            "--out" => {
                i += 1;
                out_path = args[i].clone();
            }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let mut jobs: Vec<(u64, bool)> = Vec::with_capacity(train_games + val_games);
    for k in 0..val_games {
        jobs.push((val_seed_start + k as u64, true));
    }
    for k in 0..train_games {
        jobs.push((train_seed_start + k as u64, false));
    }

    let total_jobs = jobs.len();
    println!(
        "=== export_dataset (BFDS0001) ===\n  train={} (seeds {}..{}), val={} (seeds {}..{}), wg_beam={}, threads={}\n  out={}",
        train_games,
        train_seed_start,
        train_seed_start + train_games as u64,
        val_games,
        val_seed_start,
        val_seed_start + val_games as u64,
        wg_beam,
        threads,
        out_path
    );

    let t0 = Instant::now();
    let next_idx = AtomicUsize::new(0);
    let done_cnt = AtomicUsize::new(0);
    let results: Mutex<Vec<(u64, usize, Vec<DecisionRec>)>> = Mutex::new(Vec::with_capacity(total_jobs));

    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| {
                loop {
                    let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                    if idx >= total_jobs {
                        break;
                    }
                    let (seed, is_val) = jobs[idx];
                    if let Some((pcs, decs)) = solve_and_export_game(seed, wg_beam, is_val) {
                        results.lock().unwrap().push((seed, pcs, decs));
                    }
                    let d = done_cnt.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 20 == 0 || d == total_jobs {
                        let elapsed = t0.elapsed().as_secs_f64();
                        let rate = d as f64 / elapsed.max(0.001);
                        let eta = (total_jobs - d) as f64 / rate.max(1e-6);
                        println!(
                            "  [{:4}/{}] elapsed={:6.1}s rate={:.2} games/s ETA={:6.1}s",
                            d, total_jobs, elapsed, rate, eta
                        );
                    }
                }
            });
        }
    });

    let mut all = results.into_inner().unwrap();
    all.sort_unstable_by_key(|(seed, _, _)| *seed);

    let n_games = all.len();
    let sum_pcs: usize = all.iter().map(|(_, p, _)| *p).sum();
    let n_decisions: usize = all.iter().map(|(_, _, d)| d.len()).sum();
    let n_moves: usize = all.iter().flat_map(|(_, _, d)| d.iter()).map(|dr| dr.moves.len()).sum();
    let n_val_decs: usize = all
        .iter()
        .flat_map(|(_, _, d)| d.iter())
        .filter(|dr| (dr.flags & 1) != 0)
        .count();

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    let f = File::create(&out_path).expect("failed to create output file");
    let mut w = BufWriter::with_capacity(1 << 20, f);
    w.write_all(b"BFDS0001").unwrap();
    w.write_all(&(n_games as u32).to_le_bytes()).unwrap();
    w.write_all(&(n_decisions as u32).to_le_bytes()).unwrap();
    for (_, _, decs) in &all {
        for dr in decs {
            dr.write_to(&mut w).unwrap();
        }
    }
    w.flush().unwrap();
    let file_bytes = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "DONE in {:.1}s | games={} mean_pcs={:.2} | decisions={} (val={}) | moves={} ({:.1}/dec) | size={:.2} MB",
        t0.elapsed().as_secs_f64(),
        n_games,
        sum_pcs as f64 / n_games.max(1) as f64,
        n_decisions,
        n_val_decs,
        n_moves,
        n_moves as f64 / n_decisions.max(1) as f64,
        file_bytes as f64 / (1024.0 * 1024.0)
    );
}
