use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        beam::FxHasher,
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        fast_eval::{eval_fast_with_bag, EvalWeights},
        spatial_val::{
            extract_spatial_features_ex, SpatialValueModel, DENSE_DIM, SPARSE_DIM, SPATIAL_H1,
            SPATIAL_H2,
        },
        BeamConfig, FastAI,
    },
    Color, Input, Orientation, StackerExt as _,
};
use std::collections::HashSet;
use std::hash::BuildHasherDefault;
use std::sync::Arc;
use std::time::Instant;

type FxBuild = BuildHasherDefault<FxHasher>;

struct ExactSeedData {
    seed: u64,
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
    Some(ExactSeedData { seed: s, holes, pieces })
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

fn spawn_opaque_cheese(b: &mut FastBoard, rem_g: u16, lines_cleared: u8) -> bool {
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
    for k in 0..num {
        b.data[k] = 0xFC00 | 0x3FF; // opaque garbage row (matches beam.rs simulate_cheese_spawn)
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

struct MoveFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
}

struct Decision {
    moves: Vec<MoveFeat>,
    target_idx: usize,
    rem_pcs: f32,
    is_val: bool,
}

fn solve_and_extract_decisions(seed: u64, wg_beam: usize, is_val: bool) -> Option<(usize, Vec<Decision>)> {
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

    // Replay winning trajectory and extract ALL legal candidate afterstates at every step
    let mut decisions = Vec::with_capacity(total_pcs);
    let mut cur_b = b0;
    let mut cur_held: Option<u8> = None;
    let mut cur_qc: usize = 0;
    let mut cur_rem_g: u16 = 100;
    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

    for (step, &fp_star) in win_moves.iter().enumerate() {
        let (c_nh, c_h) = if cur_held.is_some() {
            (ed.pieces.get(cur_qc).copied(), cur_held)
        } else {
            (ed.pieces.get(cur_qc).copied(), ed.pieces.get(cur_qc + 1).copied())
        };
        pgen.generate_for_state(&fst, &cur_b, c_nh, c_h, &mut tmp_places);

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
                // Terminal winning move: skip decision (endgame solver handles rem_g==0 trivially)
                if is_star {
                    target_idx = None;
                    break;
                }
                continue;
            }
            if !spawn_opaque_cheese(&mut b, rem_g, lines) || b.rows >= 20 {
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
            let in_h = held.or_else(|| ed.pieces.get(nqc).copied());
            let av = avail_mask_from_q(held, nqc, &ed.pieces);
            // compute_pe = false so trained model sets needs_pe = false (2.67x faster even with MLP!)
            let n_act = extract_spatial_features_ex(
                &b,
                rem_g,
                in_h,
                av,
                false,
                &mut act_buf,
                &mut dense_buf,
            );
            if is_star {
                target_idx = Some(moves.len());
            }
            moves.push(MoveFeat {
                active: act_buf[..n_act].to_vec(),
                dense: dense_buf,
            });
        }

        if let Some(ti) = target_idx {
            if moves.len() >= 2 {
                decisions.push(Decision {
                    moves,
                    target_idx: ti,
                    rem_pcs: (total_pcs - step - 1) as f32,
                    is_val,
                });
            }
        }

        // Advance true board with exact garbage holes
        let o_star = &fst.shapes[fp_star.color_idx as usize].orients[fp_star.tf.2 as usize];
        let (lines_s, ds_s, _) = cur_b.place_and_sift(o_star, fp_star.norm_i, fp_star.norm_j);
        cur_rem_g = cur_rem_g.saturating_sub(ds_s as u16);
        if cur_rem_g == 0 {
            break;
        }
        if !spawn_exact_cheese(&mut cur_b, cur_rem_g, lines_s, &ed.holes) {
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

fn has_h1_cover(b: &FastBoard) -> bool {
    let g = b.garbage_rows as usize;
    if g == 0 || (b.rows as usize) <= g {
        return false;
    }
    let hole_mask = (!b.data[g - 1]) & 0x3FF;
    if hole_mask == 0 {
        return false;
    }
    for r in g..(b.rows as usize) {
        if (b.data[r] & hole_mask) != 0 {
            return true;
        }
    }
    false
}

fn solve_local_wg_ds_first_move(
    fst: &FastShapeTable,
    pgen: &mut FastPlaceGen,
    tmp_places: &mut Vec<FastPlace>,
    b0: &FastBoard,
    held0: Option<u8>,
    qc0: usize,
    rem_g0: u16,
    ed: &ExactSeedData,
    teacher_w: &EvalWeights,
    ds_target_in: usize,
    beam_w: usize,
) -> Option<(FastPlace, f32)> {
    #[derive(Clone, Copy)]
    struct LocalDsNode {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: u16,
        pieces_used: u16,
        rem_g: u16,
        score: i64,
        first_fp: FastPlace,
        has_first: bool,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct LocalKey {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: u16,
    }

    let target_ds = ds_target_in.min(rem_g0 as usize).max(1);
    let q_len_ed = ed.pieces.len();
    let dummy_fp = FastPlace {
        color_idx: 0,
        did_hold: false,
        norm_i: 0,
        norm_j: 0,
        tf: (0, 0, Orientation::R0),
    };
    let mut buckets: Vec<Vec<LocalDsNode>> = vec![Vec::new(); target_ds + 1];
    let init_sc = eval_fast_with_bag(
        b0,
        rem_g0,
        held0.or_else(|| ed.pieces.get(qc0).copied()),
        avail_mask_from_q(held0, qc0, &ed.pieces),
        teacher_w,
    );
    buckets[0].push(LocalDsNode {
        board: *b0,
        held: held0,
        q_cursor: qc0 as u16,
        pieces_used: 0,
        rem_g: rem_g0,
        score: init_sc,
        first_fp: dummy_fp,
        has_first: false,
    });

    let mut seen: HashSet<LocalKey, FxBuild> =
        HashSet::with_capacity_and_hasher(beam_w * 6, FxBuild::default());
    let mut cur_sub: Vec<LocalDsNode> = Vec::with_capacity(beam_w);
    let mut next_sub: Vec<LocalDsNode> = Vec::with_capacity(beam_w * 16);
    let mut best_win: Option<(u16, FastPlace)> = None;

    for ds_cur in 0..target_ds {
        let mut bucket = std::mem::take(&mut buckets[ds_cur]);
        if bucket.is_empty() {
            continue;
        }
        bucket.sort_unstable_by_key(|n| (n.score, n.pieces_used));
        seen.clear();
        cur_sub.clear();
        let min_np = bucket.iter().map(|n| n.pieces_used).min().unwrap_or(0);
        let mut np_cnt = [0usize; 12];
        let per_np_cap = (beam_w / 3).max(16);
        let strat_cap = (beam_w * 3) / 4;
        for &n in &bucket {
            if cur_sub.len() >= strat_cap {
                break;
            }
            let tier = (n.pieces_used.saturating_sub(min_np) as usize).min(11);
            if np_cnt[tier] >= per_np_cap {
                continue;
            }
            if seen.insert(LocalKey { board: n.board, held: n.held, q_cursor: n.q_cursor }) {
                np_cnt[tier] += 1;
                cur_sub.push(n);
            }
        }
        for n in bucket {
            if cur_sub.len() >= beam_w {
                break;
            }
            if seen.insert(LocalKey { board: n.board, held: n.held, q_cursor: n.q_cursor }) {
                cur_sub.push(n);
            }
        }

        let mut best_next_ds_sc = i64::MAX;
        for z_sub in 0..=6usize {
            if cur_sub.is_empty() {
                break;
            }
            if z_sub > 0 && best_next_ds_sc != i64::MAX && cur_sub[0].score > best_next_ds_sc + 900 {
                break;
            }
            next_sub.clear();
            for &node in &cur_sub {
                let qc = node.q_cursor as usize;
                let (c_nh, c_h) = if node.held.is_some() {
                    (ed.pieces.get(qc).copied(), node.held)
                } else {
                    (ed.pieces.get(qc).copied(), ed.pieces.get(qc + 1).copied())
                };
                if c_nh.is_none() && c_h.is_none() {
                    continue;
                }
                pgen.generate_for_state(fst, &node.board, c_nh, c_h, tmp_places);
                for &fp in tmp_places.iter() {
                    let mut b = node.board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds_d, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem_g = node.rem_g.saturating_sub(ds_d as u16);
                    let np = node.pieces_used + 1;
                    let ffp = if node.has_first { node.first_fp } else { fp };
                    if rem_g == 0 && b.rows < 20 {
                        if best_win.map_or(true, |(wp, _)| np < wp) {
                            best_win = Some((np, ffp));
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
                    let sc = (np as i64) * 100 + eval_fast_with_bag(&b, rem_g, in_h, av, teacher_w);
                    if ds_d > 0 {
                        let nds = (ds_cur + ds_d as usize).min(target_ds);
                        if sc < best_next_ds_sc {
                            best_next_ds_sc = sc;
                        }
                        buckets[nds].push(LocalDsNode {
                            board: b,
                            held,
                            q_cursor: nqc as u16,
                            pieces_used: np,
                            rem_g,
                            score: sc,
                            first_fp: ffp,
                            has_first: true,
                        });
                    } else {
                        next_sub.push(LocalDsNode {
                            board: b,
                            held,
                            q_cursor: nqc as u16,
                            pieces_used: np,
                            rem_g,
                            score: sc,
                            first_fp: ffp,
                            has_first: true,
                        });
                    }
                }
            }
            if next_sub.is_empty() {
                break;
            }
            next_sub.sort_unstable_by_key(|n| (n.score, n.pieces_used));
            cur_sub.clear();
            for ch in next_sub.iter().copied() {
                if cur_sub.len() >= beam_w {
                    break;
                }
                if seen.insert(LocalKey { board: ch.board, held: ch.held, q_cursor: ch.q_cursor }) {
                    cur_sub.push(ch);
                }
            }
        }
    }

    if let Some((wp, ffp)) = best_win {
        return Some((ffp, (wp.saturating_sub(1)) as f32));
    }
    if !buckets[target_ds].is_empty() {
        buckets[target_ds].sort_unstable_by_key(|n| (n.score, n.pieces_used));
        let best = buckets[target_ds][0];
        if best.has_first {
            let est_rem = (best.pieces_used as f32 - 1.0).max(0.0) + (best.rem_g as f32) * 1.67;
            return Some((best.first_fp, est_rem));
        }
    }
    None
}

fn generate_dagger_decisions(
    seed: u64,
    dagger_mdl: &Arc<SpatialValueModel>,
    is_val: bool,
) -> Option<(usize, Vec<Decision>)> {
    let ed = collect_exact_seed(seed)?;
    let fst = FastShapeTable::srs();
    let teacher_w = ew_cons22();

    let mut cfg = BeamConfig::default();
    cfg.beam_width = 64;
    cfg.max_per_root = 16;
    cfg.rollout_top_k = 0;
    cfg.bag_det_extend = true;
    cfg.endgame_wide = true;
    cfg.endgame_solver = true;
    cfg.simulate_cheese_spawn = true;
    cfg.spatial_model = Some(dagger_mdl.clone());
    cfg.spatial_rerank_only = true;
    cfg.spatial_no_pe_internal = true;

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg);

    let mut pgen = FastPlaceGen::new();
    let mut tmp_places: Vec<FastPlace> = Vec::with_capacity(64);
    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];
    let q_len_ed = ed.pieces.len();

    let mut decisions = Vec::new();
    let mut ds_total = 0usize;
    let mut step = 0usize;
    let mut cur_held: Option<u8> = None;
    let mut cur_qc: usize = 0;

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let cur_b = FastBoard::from_basic_matrix(&snap.matrix, g_rows);

        // Query local WG_ds oracle on every 3rd step OR whenever h1 is covered (messy Legal state!)
        if rem_g > 6 && (step % 3 == 0 || has_h1_cover(&cur_b)) {
            if let Some((fp_star, est_rem_pcs)) = solve_local_wg_ds_first_move(
                &fst,
                &mut pgen,
                &mut tmp_places,
                &cur_b,
                cur_held,
                cur_qc,
                rem_g,
                &ed,
                &teacher_w,
                10,
                220,
            ) {
                let (c_nh, c_h) = if cur_held.is_some() {
                    (ed.pieces.get(cur_qc).copied(), cur_held)
                } else {
                    (ed.pieces.get(cur_qc).copied(), ed.pieces.get(cur_qc + 1).copied())
                };
                pgen.generate_for_state(&fst, &cur_b, c_nh, c_h, &mut tmp_places);
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
                    let rg_after = rem_g.saturating_sub(ds_d as u16);
                    if rg_after == 0 && b.rows < 20 {
                        if is_star {
                            target_idx = None;
                            break;
                        }
                        continue;
                    }
                    if !spawn_opaque_cheese(&mut b, rg_after, lines) || b.rows >= 20 {
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
                    let in_h = held.or_else(|| ed.pieces.get(nqc).copied());
                    let av = avail_mask_from_q(held, nqc, &ed.pieces);
                    let n_act = extract_spatial_features_ex(
                        &b,
                        rg_after,
                        in_h,
                        av,
                        false,
                        &mut act_buf,
                        &mut dense_buf,
                    );
                    if is_star {
                        target_idx = Some(moves.len());
                    }
                    moves.push(MoveFeat {
                        active: act_buf[..n_act].to_vec(),
                        dense: dense_buf,
                    });
                }
                if let Some(ti) = target_idx {
                    if moves.len() >= 2 {
                        decisions.push(Decision {
                            moves,
                            target_idx: ti,
                            rem_pcs: est_rem_pcs,
                            is_val,
                        });
                    }
                }
            }
        }

        // Advance Legal student game
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };
        let did_hold = res.inputs.first() == Some(&Input::Hold);
        let mut inputs = res.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        step += 1;
        let c_nh = ed.pieces.get(cur_qc).copied();
        let (nh, nqc) = if cur_held.is_some() {
            if did_hold {
                (c_nh, (cur_qc + 1).min(q_len_ed))
            } else {
                (cur_held, (cur_qc + 1).min(q_len_ed))
            }
        } else if did_hold {
            (c_nh, (cur_qc + 2).min(q_len_ed))
        } else {
            (None, (cur_qc + 1).min(q_len_ed))
        };
        cur_held = nh;
        cur_qc = nqc;
        if step >= 340 {
            break;
        }
    }
    Some((step, decisions))
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
        for r in self.w_sparse.iter_mut() { r.fill(0.0); }
        for r in self.w_dense.iter_mut() { r.fill(0.0); }
        self.b2.fill(0.0);
        for r in self.w2.iter_mut() { r.fill(0.0); }
        self.b3 = 0.0;
        self.w3.fill(0.0);
    }
}

struct AdamState {
    step: usize,
    m: GradBuf,
    v: GradBuf,
}

impl AdamState {
    fn new() -> Self {
        Self { step: 0, m: GradBuf::new(), v: GradBuf::new() }
    }
    #[inline]
    fn upd(p: &mut f32, g: f32, m: &mut f32, v: &mut f32, lr: f32, wd: f32, b1t: f32, b2t: f32) {
        *m = 0.9 * *m + 0.1 * g;
        *v = 0.999 * *v + 0.001 * g * g;
        let mh = *m / b1t;
        let vh = *v / b2t;
        *p -= lr * (mh / (vh.sqrt() + 1e-8) + wd * *p);
    }
    fn apply(&mut self, mdl: &mut SpatialValueModel, g: &GradBuf, lr_lin: f32, lr_mlp: f32, wd_lin: f32, wd_mlp: f32, train_mlp: bool) {
        self.step += 1;
        let b1t = 1.0 - 0.9f32.powi(self.step as i32);
        let b2t = 1.0 - 0.999f32.powi(self.step as i32);
        Self::upd(&mut mdl.lin_bias, g.lin_bias, &mut self.m.lin_bias, &mut self.v.lin_bias, lr_lin, 0.0, b1t, b2t);
        for i in 0..SPARSE_DIM {
            Self::upd(&mut mdl.lin_sparse[i], g.lin_sparse[i], &mut self.m.lin_sparse[i], &mut self.v.lin_sparse[i], lr_lin, wd_lin, b1t, b2t);
        }
        for j in 0..DENSE_DIM {
            if j == 2 {
                mdl.lin_dense[j] = 0.0; // keep pe_net = 0 so needs_pe = false
                continue;
            }
            Self::upd(&mut mdl.lin_dense[j], g.lin_dense[j], &mut self.m.lin_dense[j], &mut self.v.lin_dense[j], lr_lin, if j == 0 { 0.0 } else { wd_lin }, b1t, b2t);
        }
        if train_mlp {
            for h in 0..SPATIAL_H1 {
                Self::upd(&mut mdl.b1[h], g.b1[h], &mut self.m.b1[h], &mut self.v.b1[h], lr_mlp, 0.0, b1t, b2t);
            }
            for i in 0..SPARSE_DIM {
                for h in 0..SPATIAL_H1 {
                    Self::upd(&mut mdl.w_sparse[i][h], g.w_sparse[i][h], &mut self.m.w_sparse[i][h], &mut self.v.w_sparse[i][h], lr_mlp, wd_mlp, b1t, b2t);
                }
            }
            for j in 0..DENSE_DIM {
                if j == 0 || j == 2 {
                    mdl.w_dense[j].fill(0.0);
                    continue;
                }
                for h in 0..SPATIAL_H1 {
                    Self::upd(&mut mdl.w_dense[j][h], g.w_dense[j][h], &mut self.m.w_dense[j][h], &mut self.v.w_dense[j][h], lr_mlp, wd_mlp, b1t, b2t);
                }
            }
            for k in 0..SPATIAL_H2 {
                Self::upd(&mut mdl.b2[k], g.b2[k], &mut self.m.b2[k], &mut self.v.b2[k], lr_mlp, 0.0, b1t, b2t);
            }
            for h in 0..SPATIAL_H1 {
                for k in 0..SPATIAL_H2 {
                    Self::upd(&mut mdl.w2[h][k], g.w2[h][k], &mut self.m.w2[h][k], &mut self.v.w2[h][k], lr_mlp, wd_mlp, b1t, b2t);
                }
            }
            Self::upd(&mut mdl.b3, g.b3, &mut self.m.b3, &mut self.v.b3, lr_mlp, 0.0, b1t, b2t);
            for k in 0..SPATIAL_H2 {
                Self::upd(&mut mdl.w3[k], g.w3[k], &mut self.m.w3[k], &mut self.v.w3[k], lr_mlp, wd_mlp, b1t, b2t);
            }
        }
    }
}

#[inline]
fn forward_move(
    mdl: &SpatialValueModel,
    mf: &MoveFeat,
    use_mlp: bool,
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
    if !use_mlp {
        return (lin, 0.0);
    }
    let mut h1 = mdl.b1;
    for &b in &mf.active {
        let row = &mdl.w_sparse[b as usize];
        for h in 0..SPATIAL_H1 {
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
            for h in 0..SPATIAL_H1 {
                h1[h] += zm * row[h];
            }
        }
    }
    let mut h2 = mdl.b2;
    for h in 0..SPATIAL_H1 {
        let d1 = drop1.map_or(1.0, |m| m[h]);
        let a1 = h1[h].clamp(0.0, 6.0) * d1;
        a1_out[h] = a1;
        if a1 > 0.0 {
            let row = &mdl.w2[h];
            for k in 0..SPATIAL_H2 {
                h2[k] += a1 * row[k];
            }
        }
    }
    let mut raw_mlp = mdl.b3;
    for k in 0..SPATIAL_H2 {
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
    use_mlp: bool,
    drop1: Option<&[f32; SPATIAL_H1]>,
    drop2: Option<&[f32; SPATIAL_H2]>,
    z: &[f32; DENSE_DIM],
    a1: &[f32; SPATIAL_H1],
    a2: &[f32; SPATIAL_H2],
    raw_mlp: f32,
    d_out: f32,
    g: &mut GradBuf,
) {
    if d_out == 0.0 {
        return;
    }
    g.lin_bias += d_out;
    for &b in &mf.active {
        g.lin_sparse[b as usize] += d_out;
    }
    for j in 0..DENSE_DIM {
        if j != 2 {
            g.lin_dense[j] += d_out * z[j];
        }
    }
    if !use_mlp {
        return;
    }
    let denom = 1.0 + raw_mlp.abs() / 5.0;
    let d_mlp = d_out / (denom * denom);
    g.b3 += d_mlp;
    let mut d_h2 = [0.0f32; SPATIAL_H2];
    for k in 0..SPATIAL_H2 {
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
    for h in 0..SPATIAL_H1 {
        if a1[h] > 0.0 {
            let d1 = drop1.map_or(1.0, |m| m[h]);
            let mut dh = 0.0f32;
            let row = &mdl.w2[h];
            let grow = &mut g.w2[h];
            for k in 0..SPATIAL_H2 {
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
                    if j == 0 || j == 2 {
                        continue;
                    }
                    let zm = z[j].clamp(-3.0, 3.0);
                    if zm != 0.0 {
                        g.w_dense[j][h] += dh_s * zm;
                    }
                }
            }
        }
    }
}

fn eval_decisions(mdl: &SpatialValueModel, decisions: &[Decision], idxs: &[usize], use_mlp: bool, tau: f32) -> (f64, f64, f64, f64) {
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
            let (v, _) = forward_move(mdl, mf, use_mlp, None, None, &mut z, &mut a1, &mut a2);
            if mi == dec.target_idx {
                star_v = v;
            }
            let l = -v / tau;
            if l > max_l { max_l = l; }
            logits.push(l);
        }
        let sum_e: f64 = logits.iter().map(|&l| ((l - max_l) as f64).exp()).sum();
        let log_p = (logits[dec.target_idx] - max_l) as f64 - sum_e.ln();
        ce_sum -= log_p;
        v_mae += (star_v - dec.rem_pcs).abs() as f64;
        let star_l = logits[dec.target_idx];
        let better_cnt = logits.iter().filter(|&&l| l > star_l).count();
        if better_cnt == 0 { top1 += 1; }
        if better_cnt < 3 { top3 += 1; }
    }
    let n = idxs.len().max(1) as f64;
    (ce_sum / n, (top1 as f64) * 100.0 / n, (top3 as f64) * 100.0 / n, v_mae / n)
}

fn play_legal_seed(seed: u64, cfg: &BeamConfig) -> (usize, bool) {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());
    let mut ds_total = 0usize;
    let mut moves = 0usize;
    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() { Some(s) => s, None => break };
        assert!(snap.queue.len() <= 6);
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) { Some(r) => r, None => break };
        let mut inputs = res.inputs;
        assert_eq!(inputs.pop(), Some(Input::HD));
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        ds_total += ds;
        moves += 1;
        if moves >= 380 { break; }
    }
    (if ds_total < 100 { 400 } else { moves }, ds_total < 100)
}

fn eval_legal_set(label: &str, seeds: &[u64], mdl: &Arc<SpatialValueModel>, beam_w: usize, mpr: usize, with_pimc: bool) {
    let t0 = Instant::now();
    let mut cfg = BeamConfig::default();
    cfg.beam_width = beam_w;
    cfg.max_per_root = mpr;
    cfg.rollout_top_k = 0;
    cfg.bag_det_extend = true;
    cfg.endgame_wide = true;
    cfg.endgame_solver = true;
    cfg.simulate_cheese_spawn = true;
    cfg.spatial_model = Some(mdl.clone());
    cfg.spatial_rerank_only = false;
    cfg.spatial_no_pe_internal = true;
    if with_pimc {
        cfg.spatial_rerank_only = false;
        cfg.root_pimc_worlds = 8;
        cfg.root_pimc_horizon = 8;
        cfg.root_pimc_roots = 8;
        cfg.root_pimc_width = 24;
        cfg.root_pimc_margin = 350;
        cfg.root_pimc_weight_pct = 100;
        cfg.root_pimc_steady_g4 = false;
    }
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).min(seeds.len().max(1));
    let chunk_sz = (seeds.len() + n_threads - 1) / n_threads;
    let res: Vec<(usize, bool)> = std::thread::scope(|sc| {
        let hs: Vec<_> = seeds.chunks(chunk_sz).map(|sl| {
            let c_ref = &cfg;
            sc.spawn(move || sl.iter().map(|&s| play_legal_seed(s, c_ref)).collect::<Vec<_>>())
        }).collect();
        let mut out = Vec::new();
        for h in hs { out.extend(h.join().unwrap()); }
        out
    });
    let mean = res.iter().map(|&(p, _)| p as f64).sum::<f64>() / (res.len().max(1) as f64);
    let tops = res.iter().filter(|&&(_, t)| t).count();
    let pcs: Vec<usize> = res.iter().map(|&(p, _)| p).collect();
    println!("  {:28} mean={:.2} top={} pcs={:?} dt={:.1}s", label, mean, tops, pcs, t0.elapsed().as_secs_f64());
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_path = "models/spatial_bc1.txt".to_string();
    let mut games: usize = 80;
    let mut wg_beam: usize = 1400;
    let mut dagger_init: Option<String> = None;
    let mut dagger_games: usize = 0;
    let mut lin_epochs: usize = 14;
    let mut mlp_epochs: usize = 16;
    let mut batch_sz: usize = 32;
    let mut tau: f32 = 0.50;
    let mut cal_w: f32 = 0.025;
    let mut lr_lin: f32 = 0.008;
    let mut lr_mlp: f32 = 0.0008;
    let mut wd: f32 = 0.002;
    let mut wd_mlp: f32 = 0.015;
    let mut dropout: f32 = 0.15;
    let mut label_smooth: f32 = 0.02;
    let mut seed_base: u64 = 3000;
    let mut eval_legal: usize = 5;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--games" => { games = args[i + 1].parse().unwrap(); i += 2; }
            "--wg-beam" => { wg_beam = args[i + 1].parse().unwrap(); i += 2; }
            "--dagger-init" => { dagger_init = Some(args[i + 1].clone()); i += 2; }
            "--dagger-games" => { dagger_games = args[i + 1].parse().unwrap(); i += 2; }
            "--lin-epochs" => { lin_epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--mlp-epochs" => { mlp_epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch_sz = args[i + 1].parse().unwrap(); i += 2; }
            "--tau" => { tau = args[i + 1].parse().unwrap(); i += 2; }
            "--cal-w" => { cal_w = args[i + 1].parse().unwrap(); i += 2; }
            "--lr-lin" => { lr_lin = args[i + 1].parse().unwrap(); i += 2; }
            "--lr-mlp" => { lr_mlp = args[i + 1].parse().unwrap(); i += 2; }
            "--wd" => { wd = args[i + 1].parse().unwrap(); i += 2; }
            "--wd-mlp" => { wd_mlp = args[i + 1].parse().unwrap(); i += 2; }
            "--dropout" => { dropout = args[i + 1].parse().unwrap(); i += 2; }
            "--label-smooth" => { label_smooth = args[i + 1].parse().unwrap(); i += 2; }
            "--seed-base" => { seed_base = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-legal" => { eval_legal = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
    }

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
    println!(
        "=== Full-Action-Space Afterstate BC + DAgger (WG_ds -> SpatialValueModel {}x{}) ===\n  out={} games={} wg_beam={} dagger_games={} lin_ep={} mlp_ep={} batch={} tau={} cal_w={} lr_lin={} lr_mlp={} wd={} wd_mlp={} drop={} ls={} threads={}",
        SPATIAL_H1, SPATIAL_H2, out_path, games, wg_beam, dagger_games, lin_epochs, mlp_epochs, batch_sz, tau, cal_w, lr_lin, lr_mlp, wd, wd_mlp, dropout, label_smooth, n_threads
    );

    let t0 = Instant::now();
    let seeds: Vec<u64> = (0..games).map(|k| seed_base + k as u64).collect();
    let val_start = ((games * 85) / 100).max(1).min(games.saturating_sub(1));
    let chunk_sz = (games + n_threads - 1) / n_threads;
    let raw_games: Vec<(usize, Vec<Decision>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = seeds.chunks(chunk_sz).enumerate().map(|(ci, sl)| {
            let base_i = ci * chunk_sz;
            sc.spawn(move || {
                sl.iter().enumerate().filter_map(|(k, &s)| {
                    solve_and_extract_decisions(s, wg_beam, base_i + k >= val_start)
                }).collect::<Vec<_>>()
            })
        }).collect();
        let mut out = Vec::new();
        for h in hs { out.extend(h.join().unwrap()); }
        out
    });

    let mean_pcs = raw_games.iter().map(|(p, _)| *p as f64).sum::<f64>() / (raw_games.len().max(1) as f64);
    let mut decisions = Vec::new();
    let mut total_moves = 0usize;
    for (_, ds) in raw_games {
        for d in ds {
            total_moves += d.moves.len();
            decisions.push(d);
        }
    }
    let clean_decs = decisions.len();

    if dagger_games > 0 {
        let d_path = dagger_init.as_deref().unwrap_or("models/spatial_bc1_lin.txt");
        let d_mdl = Arc::new(SpatialValueModel::load_from_file(d_path).expect("load dagger init"));
        let d_seeds: Vec<u64> = (0..dagger_games).map(|k| seed_base + 5000 + k as u64).collect();
        let d_val_start = ((dagger_games * 85) / 100).max(1).min(dagger_games.saturating_sub(1));
        let d_chunk = (dagger_games + n_threads - 1) / n_threads;
        let d_raw: Vec<(usize, Vec<Decision>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = d_seeds.chunks(d_chunk).enumerate().map(|(ci, sl)| {
                let base_i = ci * d_chunk;
                let dm = d_mdl.clone();
                sc.spawn(move || {
                    sl.iter().enumerate().filter_map(|(k, &s)| {
                        generate_dagger_decisions(s, &dm, base_i + k >= d_val_start)
                    }).collect::<Vec<_>>()
                })
            }).collect();
            let mut out = Vec::new();
            for h in hs { out.extend(h.join().unwrap()); }
            out
        });
        let d_mean_pcs = d_raw.iter().map(|(p, _)| *p as f64).sum::<f64>() / (d_raw.len().max(1) as f64);
        let mut d_decs = 0usize;
        for (_, ds) in d_raw {
            for d in ds {
                total_moves += d.moves.len();
                d_decs += 1;
                decisions.push(d);
            }
        }
        println!(
            "Phase 1B (BC-DAgger): {} Legal games (mean={:.2} pcs) -> {} recovery decisions",
            dagger_games, d_mean_pcs, d_decs
        );
    }

    let train_idxs: Vec<usize> = (0..decisions.len()).filter(|&i| !decisions[i].is_val).collect();
    let val_idxs: Vec<usize> = (0..decisions.len()).filter(|&i| decisions[i].is_val).collect();
    println!(
        "Phase 1 Total: {} clean WG_ds (mean={:.2}) + {} DAgger -> {} decisions ({} clean, {} train, {} val), {} legal moves in {:.1}s",
        games, mean_pcs, dagger_games, decisions.len(), clean_decs, train_idxs.len(), val_idxs.len(), total_moves, t0.elapsed().as_secs_f64()
    );

    // Compute dense normalization stats across all moves in training decisions
    let mut sum_d = [0.0f64; DENSE_DIM];
    let mut sum_d2 = [0.0f64; DENSE_DIM];
    let mut cnt_m = 0usize;
    let mut sum_y = 0.0f64;
    for &di in &train_idxs {
        sum_y += decisions[di].rem_pcs as f64;
        for mf in &decisions[di].moves {
            cnt_m += 1;
            for j in 0..DENSE_DIM {
                let v = mf.dense[j] as f64;
                sum_d[j] += v;
                sum_d2[j] += v * v;
            }
        }
    }
    let mut mdl = SpatialValueModel::default();
    mdl.lin_bias = (sum_y / (train_idxs.len().max(1) as f64)) as f32;
    for j in 0..DENSE_DIM {
        let mean = sum_d[j] / (cnt_m.max(1) as f64);
        let var = (sum_d2[j] / (cnt_m.max(1) as f64) - mean * mean).max(0.0);
        let std = var.sqrt();
        mdl.dense_mean[j] = mean as f32;
        mdl.dense_inv_std[j] = if std > 1e-4 { (1.0 / std) as f32 } else { 0.0 };
    }
    if mdl.dense_inv_std[0] > 0.0 {
        mdl.lin_dense[0] = 1.65 / mdl.dense_inv_std[0];
    }

    // Stage A: Pure-Linear Full-Action-Space Softmax BC
    let mut adam = AdamState::new();
    let mut grad = GradBuf::new();
    let mut order = train_idxs.clone();
    let mut rng_s = 0x517CC1B727220A95u64;
    let mut best_lin_ce = f64::INFINITY;
    let mut best_lin_mdl = mdl.clone();
    let mut z_cache: Vec<[f32; DENSE_DIM]> = Vec::new();
    let mut a1_cache: Vec<[f32; SPATIAL_H1]> = Vec::new();
    let mut a2_cache: Vec<[f32; SPATIAL_H2]> = Vec::new();
    let mut raw_mlp_cache: Vec<f32> = Vec::new();
    let mut logits_buf: Vec<f32> = Vec::new();

    for ep in 1..=lin_epochs {
        for k in (1..order.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((rng_s >> 32) as usize) % (k + 1);
            order.swap(k, j);
        }
        let prog = (ep - 1) as f32 / (lin_epochs.max(1) as f32);
        let lr_cur = lr_lin * (0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos()));

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
                logits_buf.resize(nm, 0.0);

                let mut max_l = f32::NEG_INFINITY;
                let mut star_v = 0.0f32;
                for m in 0..nm {
                    let (v, rm) = forward_move(&mdl, &dec.moves[m], false, None, None, &mut z_cache[m], &mut a1_cache[m], &mut a2_cache[m]);
                    raw_mlp_cache[m] = rm;
                    if m == dec.target_idx { star_v = v; }
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
                let ls_u = label_smooth / (nm.max(1) as f32);
                for m in 0..nm {
                    let p_m = logits_buf[m] * inv_e;
                    let y_m = if m == dec.target_idx { 1.0f32 - label_smooth + ls_u } else { ls_u };
                    // d(CE)/d(v_m) = -(p_m - y_m) / tau
                    let mut d_v = -(p_m - y_m) / tau;
                    if m == dec.target_idx {
                        let err = star_v - dec.rem_pcs;
                        let huber_g = if err.abs() <= 2.0 { err } else { 2.0 * err.signum() };
                        d_v += cal_w * huber_g;
                    }
                    backward_move(&mdl, &dec.moves[m], false, None, None, &z_cache[m], &a1_cache[m], &a2_cache[m], 0.0, d_v * inv_b, &mut grad);
                }
            }
            adam.apply(&mut mdl, &grad, lr_cur, 0.0, wd, 0.0, false);
        }
        let (v_ce, v_top1, v_top3, v_mae) = eval_decisions(&mdl, &decisions, &val_idxs, false, tau);
        if v_ce < best_lin_ce {
            best_lin_ce = v_ce;
            best_lin_mdl = mdl.clone();
        }
        if ep == 1 || ep % 3 == 0 || ep == lin_epochs {
            println!("  [LIN ep {:2}/{}] lr={:.5} | val_ce={:.4} top1={:.1}% top3={:.1}% v_mae={:.2}", ep, lin_epochs, lr_cur, v_ce, v_top1, v_top3, v_mae);
        }
    }

    best_lin_mdl.prepare_fast();
    let lin_path = if out_path.ends_with(".txt") {
        format!("{}_lin.txt", &out_path[..out_path.len() - 4])
    } else {
        format!("{}_lin.txt", out_path)
    };
    best_lin_mdl.save_to_file(&lin_path).expect("save lin BC");
    println!("Saved pure-linear BC model (val_ce={:.4}, rem_g={:.3}) to {}", best_lin_ce, best_lin_mdl.lin_dense[0] * best_lin_mdl.dense_inv_std[0], lin_path);

    // Stage B: Full-MLP BC initialized from best pure-linear BC
    mdl = best_lin_mdl.clone();
    let mut init_rng = 0xD6E8FEB86659FD93u64;
    let mut rand_w = |scale: f32| -> f32 {
        init_rng = init_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (((init_rng >> 32) as f32) / (u32::MAX as f32)) * 2.0 - 1.0;
        u * scale
    };
    let tr_h1 = 64usize.min(SPATIAL_H1);
    let tr_h2 = 32usize.min(SPATIAL_H2);
    for i in 0..SPARSE_DIM {
        for h in 0..tr_h1 { mdl.w_sparse[i][h] = rand_w(0.025); }
    }
    for j in 0..DENSE_DIM {
        if j == 0 || j == 2 { continue; }
        for h in 0..tr_h1 { mdl.w_dense[j][h] = rand_w(0.035); }
    }
    for h in 0..tr_h1 {
        mdl.b1[h] = 0.05;
        for k in 0..tr_h2 { mdl.w2[h][k] = rand_w(0.045); }
    }
    for k in 0..tr_h2 {
        mdl.b2[k] = 0.05;
        mdl.w3[k] = rand_w(0.018);
    }
    mdl.b3 = 0.0;

    let mut adam_mlp = AdamState::new();
    let mut best_mlp_ce = best_lin_ce;
    let mut best_mlp_mdl = best_lin_mdl.clone();
    let mut drop_rng = 0xA0761D6478BD642Fu64;
    let keep_p = (1.0 - dropout).clamp(0.2, 1.0);
    let inv_keep = 1.0 / keep_p;
    let drop_thresh = (keep_p * (u32::MAX as f32)) as u32;
    let mut drop1_buf = [1.0f32; SPATIAL_H1];
    let mut drop2_buf = [1.0f32; SPATIAL_H2];

    for ep in 1..=mlp_epochs {
        for k in (1..order.len()).rev() {
            rng_s = rng_s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = ((rng_s >> 32) as usize) % (k + 1);
            order.swap(k, j);
        }
        let prog = (ep - 1) as f32 / (mlp_epochs.max(1) as f32);
        let cos_m = 0.1 + 0.9 * 0.5 * (1.0 + (std::f32::consts::PI * prog).cos());
        let lrm = lr_mlp * cos_m;
        let lrl = lr_mlp * 0.25 * cos_m;

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
                logits_buf.resize(nm, 0.0);

                let (d1_opt, d2_opt) = if dropout > 0.0 {
                    for h in 0..SPATIAL_H1 {
                        drop_rng = drop_rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        drop1_buf[h] = if (drop_rng >> 32) as u32 <= drop_thresh { inv_keep } else { 0.0 };
                    }
                    for k in 0..SPATIAL_H2 {
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
                    let (v, rm) = forward_move(&mdl, &dec.moves[m], true, d1_opt, d2_opt, &mut z_cache[m], &mut a1_cache[m], &mut a2_cache[m]);
                    raw_mlp_cache[m] = rm;
                    if m == dec.target_idx { star_v = v; }
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
                let ls_u = label_smooth / (nm.max(1) as f32);
                for m in 0..nm {
                    let p_m = logits_buf[m] * inv_e;
                    let y_m = if m == dec.target_idx { 1.0f32 - label_smooth + ls_u } else { ls_u };
                    let mut d_v = -(p_m - y_m) / tau;
                    if m == dec.target_idx {
                        let err = star_v - dec.rem_pcs;
                        let huber_g = if err.abs() <= 2.0 { err } else { 2.0 * err.signum() };
                        d_v += cal_w * huber_g;
                    }
                    backward_move(&mdl, &dec.moves[m], true, d1_opt, d2_opt, &z_cache[m], &a1_cache[m], &a2_cache[m], raw_mlp_cache[m], d_v * inv_b, &mut grad);
                }
            }
            adam_mlp.apply(&mut mdl, &grad, lrl, lrm, wd, wd_mlp, true);
        }
        let (v_ce, v_top1, v_top3, v_mae) = eval_decisions(&mdl, &decisions, &val_idxs, true, tau);
        if v_ce < best_mlp_ce {
            best_mlp_ce = v_ce;
            best_mlp_mdl = mdl.clone();
        }
        if ep == 1 || ep % 3 == 0 || ep == mlp_epochs {
            println!("  [MLP ep {:2}/{}] lr={:.5} | val_ce={:.4} top1={:.1}% top3={:.1}% v_mae={:.2}", ep, mlp_epochs, lrm, v_ce, v_top1, v_top3, v_mae);
        }
    }

    best_mlp_mdl.prepare_fast();
    best_mlp_mdl.save_to_file(&out_path).expect("save mlp BC");
    println!("Saved full-MLP BC model (val_ce={:.4}, needs_pe={}) to {}", best_mlp_ce, best_mlp_mdl.needs_pe, out_path);

    if eval_legal > 0 {
        println!("\n=== Post-BC Held-Out Legal Verification (seeds 101..={}) ===", 100 + eval_legal);
        let seeds_ev: Vec<u64> = (101..(101 + eval_legal as u64)).collect();
        let lin_arc = Arc::new(best_lin_mdl);
        let mlp_arc = Arc::new(best_mlp_mdl);
        eval_legal_set("LEGAL_b120_bc_lin", &seeds_ev, &lin_arc, 120, 24, false);
        eval_legal_set("LEGAL_b120_bc_mlp", &seeds_ev, &mlp_arc, 120, 24, false);
        eval_legal_set("LEGAL_pimc_w8_bc_lin", &seeds_ev, &lin_arc, 64, 16, true);
        eval_legal_set("LEGAL_pimc_w8_bc_mlp", &seeds_ev, &mlp_arc, 64, 16, true);
    }
}
