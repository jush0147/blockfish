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
use std::collections::HashSet;
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

/// Deterministic observable-state hash (strictly Legal p=5: ZERO hidden seed!).
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

/// Spawn garbage rows from a pre-generated Legal CRN hole stream (`min_height=4, max_height=9`).
/// Returns `false` on topout (`rows >= 20`).
#[inline]
fn spawn_from_hole_stream(
    b: &mut FastBoard,
    rem_g: u16,
    lines_cleared: u8,
    holes: &[u16],
    hole_cursor: &mut usize,
) -> bool {
    let target_g = if lines_cleared > 0 { 4usize } else { 9usize }.min(rem_g as usize);
    let cur_g = b.garbage_rows as usize;
    if target_g <= cur_g {
        return (b.rows as usize) < 20;
    }
    let num_new = target_g - cur_g;
    if (b.rows as usize) + num_new >= 20 {
        return false;
    }
    // Shift existing rows up by `num_new`
    let old_rows = b.rows as usize;
    for r in (0..old_rows).rev() {
        b.data[r + num_new] = b.data[r];
    }
    for k in 0..num_new {
        let hc = holes[(*hole_cursor) % holes.len()];
        *hole_cursor += 1;
        // Row at bottom: `num_new - 1 - k` so earliest spawned row ends up highest among new rows
        b.data[num_new - 1 - k] = 0x3FF ^ (1u16 << hc);
    }
    b.rows = (old_rows + num_new) as u8;
    b.garbage_rows = (cur_g + num_new) as u8;
    true
}

/// Generate a Legal `p=5` CRN future world (`world_q`, `world_holes`) strictly from observable state.
struct CrnWorld {
    q: Vec<u8>,
    holes: Vec<u16>,
    init_bag: u8,
}

fn generate_crn_world(
    board: &FastBoard,
    vis_q: &[u8],
    bag_mask: u8,
    rng: &mut u64,
    q_len: usize,
    holes_len: usize,
) -> CrnWorld {
    let mut q = Vec::with_capacity(q_len);
    for &p in vis_q {
        q.push(p);
    }
    let mut mask = if bag_mask == 0 { 0x7F } else { bag_mask & 0x7F };
    while q.len() < q_len {
        let mut pool = [0u8; 7];
        let mut cnt = 0usize;
        for p in 0..7u8 {
            if (mask >> p) & 1 != 0 {
                pool[cnt] = p;
                cnt += 1;
            }
        }
        // Fisher-Yates shuffle
        for i in (1..cnt).rev() {
            let j = next_usize(rng, i + 1);
            pool.swap(i, j);
        }
        for &p in &pool[..cnt] {
            if q.len() < q_len {
                q.push(p);
            }
        }
        mask = 0x7F;
    }

    // Legal Jstris garbage hole stream: consecutive holes never equal (`col_new != col_prev`)
    let mut last_hole: u16 = if board.garbage_rows > 0 {
        let h = (!board.data[0] & 0x3FF).trailing_zeros() as u16;
        if h < 10 { h } else { 10 }
    } else {
        10
    };
    let mut holes = Vec::with_capacity(holes_len);
    for _ in 0..holes_len {
        let col = if last_hole < 10 {
            let r = next_usize(rng, 9) as u16;
            if r >= last_hole { r + 1 } else { r }
        } else {
            next_usize(rng, 10) as u16
        };
        holes.push(col);
        last_hole = col;
    }

    CrnWorld {
        q,
        holes,
        init_bag: bag_mask,
    }
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

struct BeamScratch {
    pgen: FastPlaceGen,
    tmp_p0: Vec<FastPlace>,
    tmp_p2: Vec<FastPlace>,
    cur_beam: Vec<BNode>,
    next_beam: Vec<BNode>,
    seen: HashSet<DKey>,
    act_buf: [u16; SPARSE_DIM],
    dense_buf: [f32; DENSE_DIM],
}

impl BeamScratch {
    fn new(beam_w: usize) -> Self {
        Self {
            pgen: FastPlaceGen::new(),
            tmp_p0: Vec::with_capacity(80),
            tmp_p2: Vec::with_capacity(64),
            cur_beam: Vec::with_capacity(beam_w.max(16)),
            next_beam: Vec::with_capacity(beam_w.max(16) * 48),
            seen: HashSet::with_capacity(beam_w.max(16) * 4),
            act_buf: [0u16; SPARSE_DIM],
            dense_buf: [0.0f32; DENSE_DIM],
        }
    }
}

/// Fast 100% Legal `p=5` 5-ply beam search returning `(best_fp, root_scores_sorted)`.
/// Uses ONLY `(board, hold, vis_q[0..<=6], rem_g, bag_mask)`.
/// Any simulated garbage spawn inside the 5-ply beam is seeded strictly from observable state.
fn legal_p5_beam_rank(
    fst: &FastShapeTable,
    sc: &mut BeamScratch,
    board: &FastBoard,
    hold: Option<u8>,
    vis_q: &[u8],
    rem_g: u16,
    bag_mask: u8,
    mdl: &SpatialValueModel,
    beam_w: usize,
    sparse_lim: usize,
) -> (Option<FastPlace>, Vec<(f32, FastPlace)>) {
    let c_nh = vis_q.first().copied();
    let c_h = if hold.is_some() {
        hold
    } else {
        vis_q.get(1).copied()
    };
    sc.pgen.generate_for_state(fst, board, c_nh, c_h, &mut sc.tmp_p0);
    if sc.tmp_p0.is_empty() {
        return (None, Vec::new());
    }

    let root_places = sc.tmp_p0.clone();
    let base_spawn_seed = observable_state_hash(board, hold, vis_q, rem_g, bag_mask, 0xCAFE);
    let max_depth = vis_q.len().min(5);

    sc.cur_beam.clear();
    sc.seen.clear();
    let mut root_best: Vec<f32> = vec![f32::INFINITY; root_places.len()];
    let mut win_fp: Option<FastPlace> = None;

    for (ridx, &fp0) in root_places.iter().enumerate() {
        let mut b1 = *board;
        let o0 = &fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
        let (l0, ds0, _) = b1.place_and_sift(o0, fp0.norm_i, fp0.norm_j);
        let rem1 = rem_g.saturating_sub(ds0 as u16);
        if rem1 == 0 && b1.rows < 20 {
            win_fp = Some(fp0);
            root_best[ridx] = -1000.0;
            break;
        }
        if rem1 > 0 {
            let mut r_loc = base_spawn_seed ^ 0x1111;
            if !b1.spawn_cheese_rows(rem1, l0, &mut r_loc) {
                continue;
            }
        }
        let col_h = b1.col_heights();
        if *col_h.iter().max().unwrap() >= 20 || col_h[3..=6].iter().copied().max().unwrap() >= 19 {
            continue;
        }
        let (held1, qc1) = if hold.is_some() {
            if fp0.did_hold {
                (c_nh, if c_nh.is_some() { 1 } else { 0 })
            } else {
                (hold, 1)
            }
        } else if fp0.did_hold {
            (c_nh, 2)
        } else {
            (None, 1)
        };
        let ih1 = hand_piece(held1, qc1, vis_q);
        let av1 = avail_mask_from_q(held1, qc1, vis_q, bag_mask);
        let sc1 = mdl.evaluate_linear_board_with_bag_no_pe(&b1, rem1, ih1, av1) as f32 * 0.01;
        root_best[ridx] = sc1;

        let key = DKey {
            board: b1,
            held: held1,
            qc: qc1 as u8,
            rem_g: rem1,
        };
        if sc.seen.insert(key) {
            sc.cur_beam.push(BNode {
                board: b1,
                held: held1,
                qc: qc1 as u8,
                rem_g: rem1,
                first_idx: ridx as u16,
                score: sc1,
            });
        }
    }

    if let Some(wfp) = win_fp {
        let mut out = Vec::with_capacity(1);
        out.push((-1000.0, wfp));
        return (Some(wfp), out);
    }

    if sc.cur_beam.is_empty() {
        return (Some(root_places[0]), vec![(0.0, root_places[0])]);
    }

    // Keep top beam_w at depth 1
    sc.cur_beam
        .sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
    if sc.cur_beam.len() > beam_w {
        sc.cur_beam.truncate(beam_w);
    }

    let mpr = (beam_w / 3).max(4);

    // Expand depths 2..=max_depth using fast linear scoring (10x faster than full MLP)
    for depth in 2..=max_depth {
        if sc.cur_beam.is_empty() {
            break;
        }
        sc.next_beam.clear();
        let depth_seed = base_spawn_seed ^ ((depth as u64).wrapping_mul(0x9E3779B97F4A7C15));

        for ni in 0..sc.cur_beam.len() {
            let node = sc.cur_beam[ni];
            let qc = node.qc as usize;
            let (c_nh2, c_h2) = if node.held.is_some() {
                (vis_q.get(qc).copied(), node.held)
            } else {
                (vis_q.get(qc).copied(), vis_q.get(qc + 1).copied())
            };
            sc.pgen
                .generate_for_state(fst, &node.board, c_nh2, c_h2, &mut sc.tmp_p2);
            for &fp2 in &sc.tmp_p2 {
                let mut b2 = node.board;
                let o2 = &fst.shapes[fp2.color_idx as usize].orients[fp2.tf.2 as usize];
                let (l2, ds2, _) = b2.place_and_sift(o2, fp2.norm_i, fp2.norm_j);
                let rem2 = node.rem_g.saturating_sub(ds2 as u16);
                if rem2 == 0 && b2.rows < 20 {
                    let s_win = -1000.0 + (depth as f32) * 10.0;
                    if s_win < root_best[node.first_idx as usize] {
                        root_best[node.first_idx as usize] = s_win;
                    }
                    continue;
                }
                if rem2 > 0 {
                    let mut r_loc = depth_seed;
                    if !b2.spawn_cheese_rows(rem2, l2, &mut r_loc) {
                        continue;
                    }
                }
                let col_h = b2.col_heights();
                if *col_h.iter().max().unwrap() >= 20
                    || col_h[3..=6].iter().copied().max().unwrap() >= 19
                {
                    continue;
                }
                let (held2, qc2) = if node.held.is_some() {
                    if fp2.did_hold {
                        (c_nh2, if c_nh2.is_some() { qc + 1 } else { qc })
                    } else {
                        (node.held, qc + 1)
                    }
                } else if fp2.did_hold {
                    (c_nh2, qc + 2)
                } else {
                    (None, qc + 1)
                };
                let ih2 = hand_piece(held2, qc2, vis_q);
                let av2 = avail_mask_from_q(held2, qc2, vis_q, bag_mask);
                let sc2 = mdl.evaluate_linear_board_with_bag_no_pe(&b2, rem2, ih2, av2) as f32 * 0.01;
                sc.next_beam.push(BNode {
                    board: b2,
                    held: held2,
                    qc: qc2 as u8,
                    rem_g: rem2,
                    first_idx: node.first_idx,
                    score: sc2,
                });
            }
        }
        if sc.next_beam.is_empty() {
            break;
        }
        sc.next_beam
            .sort_unstable_by(|a, b| a.score.total_cmp(&b.score));
        sc.cur_beam.clear();
        sc.seen.clear();
        let mut per_root = [0usize; 128];
        let keep_lim = if depth == max_depth { beam_w * 2 } else { beam_w };
        for i in 0..sc.next_beam.len() {
            if sc.cur_beam.len() >= keep_lim {
                break;
            }
            let n = sc.next_beam[i];
            let ridx = n.first_idx as usize;
            if ridx < 128 && per_root[ridx] >= mpr {
                continue;
            }
            let key = DKey {
                board: n.board,
                held: n.held,
                qc: n.qc,
                rem_g: n.rem_g,
            };
            if sc.seen.insert(key) {
                if ridx < 128 {
                    per_root[ridx] += 1;
                }
                sc.cur_beam.push(n);
            }
        }
    }

    // Full-MLP leaf rerank on final beam leaves
    let mut seen_root = vec![false; root_places.len()];
    for i in 0..sc.cur_beam.len() {
        let node = sc.cur_beam[i];
        let ih = hand_piece(node.held, node.qc as usize, vis_q);
        let av = avail_mask_from_q(node.held, node.qc as usize, vis_q, bag_mask);
        let n_act = extract_spatial_features_ex(
            &node.board,
            node.rem_g,
            ih,
            av,
            false,
            &mut sc.act_buf,
            &mut sc.dense_buf,
        );
        let v_mlp = predict_with_lim(mdl, &sc.act_buf[..n_act], &sc.dense_buf, sparse_lim);
        let r = node.first_idx as usize;
        if !seen_root[r] || v_mlp < root_best[r] {
            root_best[r] = v_mlp;
            seen_root[r] = true;
        }
    }

    let mut ranked: Vec<(f32, FastPlace)> = root_places
        .iter()
        .enumerate()
        .filter_map(|(i, &fp)| {
            if root_best[i].is_finite() {
                // Penalize roots pruned before final depth slightly so surviving roots rank above them
                let s = if seen_root[i] {
                    root_best[i]
                } else {
                    root_best[i] + 5.0
                };
                Some((s, fp))
            } else {
                None
            }
        })
        .collect();
    ranked.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let best = ranked.first().map(|x| x.1).or_else(|| Some(root_places[0]));
    (best, ranked)
}

#[inline]
fn predict_with_lim(
    mdl: &SpatialValueModel,
    act: &[u16],
    dense: &[f32; DENSE_DIM],
    sparse_lim: usize,
) -> f32 {
    let mut lin = mdl.lin_bias;
    for &b in act {
        if (b as usize) < sparse_lim {
            lin += mdl.lin_sparse[b as usize];
        }
    }
    let mut z_arr = [0.0f32; DENSE_DIM];
    for j in 0..DENSE_DIM {
        let z = (dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j];
        z_arr[j] = z;
        lin += mdl.lin_dense[j] * z;
    }
    let mut h1 = mdl.b1;
    for &b in act {
        if (b as usize) < sparse_lim {
            let row = &mdl.w_sparse[b as usize];
            for h in 0..TRAIN_H1 {
                h1[h] += row[h];
            }
        }
    }
    for j in 0..DENSE_DIM {
        if j == 0 || j == 2 {
            continue;
        }
        let zm = z_arr[j].clamp(-3.0, 3.0);
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
        if a2 > 0.0 {
            raw_mlp += a2 * mdl.w3[k];
        }
    }
    let mlp_sat = raw_mlp / (1.0 + raw_mlp.abs() / 5.0);
    lin + mlp_sat
}

/// Strictly Legal `p=5` CRN rollout from a post-action afterstate `(b, held, qc, rem_g)`
/// inside `world`, stopping at exact `goal_rem` (or topout / `max_steps`).
/// Returns total pieces from the branching state (including the 1st branching move)
/// plus terminal `ref_mdl` value if `goal_rem > 0`.
fn rollout_legal_p5_to_goal(
    fst: &FastShapeTable,
    sc: &mut BeamScratch,
    mut b: FastBoard,
    mut held: Option<u8>,
    mut qc: usize,
    mut rem_g: u16,
    goal_rem: u16,
    mut hole_cursor: usize,
    world: &CrnWorld,
    roll_mdl: &SpatialValueModel,
    term_mdl: &SpatialValueModel,
    roll_beam: usize,
    sparse_lim: usize,
    max_steps: usize,
) -> (f32, bool) {
    if rem_g == 0 {
        return (1.0, false);
    }
    // Track legal bag_mask incrementally as new pieces enter the 6-piece visible window
    let mut bag_mask = world.init_bag;
    let mut vis_end = 6usize.min(world.q.len());
    // If initial action consumed 1 or 2 pieces from queue, advance `vis_end` accordingly
    let mut steps = 1usize; // already placed the branching root move

    let consume_into_bag = |mask: &mut u8, p: u8| {
        let bit = 1u8 << p;
        if (*mask & bit) == 0 {
            *mask = 0x7F;
        }
        *mask &= !bit;
        if *mask == 0 {
            *mask = 0x7F;
        }
    };

    // Sync visible window end to `qc + 6`
    let target_vis = (qc + 6).min(world.q.len());
    while vis_end < target_vis {
        consume_into_bag(&mut bag_mask, world.q[vis_end]);
        vis_end += 1;
    }

    while rem_g > goal_rem && steps < max_steps {
        let q_end = (qc + 6).min(world.q.len());
        if qc >= q_end {
            break;
        }
        let vis_q = &world.q[qc..q_end];
        let (best_fp, _) = legal_p5_beam_rank(
            fst,
            sc,
            &b,
            held,
            vis_q,
            rem_g,
            bag_mask,
            roll_mdl,
            roll_beam,
            sparse_lim,
        );
        let fp = match best_fp {
            Some(f) => f,
            None => {
                let rem_left = rem_g.saturating_sub(goal_rem) as f32;
                return (steps as f32 + rem_left * 2.5 + 35.0, true);
            }
        };
        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
        steps += 1;
        rem_g = rem_g.saturating_sub(ds as u16);
        if rem_g == 0 && b.rows < 20 {
            return (steps as f32, false);
        }
        if rem_g > 0 && !spawn_from_hole_stream(&mut b, rem_g, lines, &world.holes, &mut hole_cursor)
        {
            let rem_left = rem_g.saturating_sub(goal_rem) as f32;
            return (steps as f32 + rem_left * 2.5 + 35.0, true);
        }
        let col_h = b.col_heights();
        if *col_h.iter().max().unwrap() >= 20 || col_h[3..=6].iter().copied().max().unwrap() >= 19 {
            let rem_left = rem_g.saturating_sub(goal_rem) as f32;
            return (steps as f32 + rem_left * 2.5 + 35.0, true);
        }
        let c_nh = world.q.get(qc).copied();
        let (nh, nqc) = if held.is_some() {
            if fp.did_hold {
                (c_nh, if c_nh.is_some() { qc + 1 } else { qc })
            } else {
                (held, qc + 1)
            }
        } else if fp.did_hold {
            (c_nh, qc + 2)
        } else {
            (None, qc + 1)
        };
        held = nh;
        qc = nqc;
        let target_vis = (qc + 6).min(world.q.len());
        while vis_end < target_vis {
            consume_into_bag(&mut bag_mask, world.q[vis_end]);
            vis_end += 1;
        }
    }

    if rem_g == 0 {
        return (steps as f32, false);
    }

    // Evaluate terminal residual at `goal_rem` (exact same `goal_rem` across all surviving siblings!)
    let q_end = (qc + 6).min(world.q.len());
    let vis_q = if qc < q_end { &world.q[qc..q_end] } else { &[] };
    let ih = hand_piece(held, 0, vis_q);
    let av = avail_mask_from_q(held, 0, vis_q, bag_mask);
    let n_act = extract_spatial_features_ex(
        &b,
        rem_g,
        ih,
        av,
        false,
        &mut sc.act_buf,
        &mut sc.dense_buf,
    );
    let v_term = predict_with_lim(term_mdl, &sc.act_buf[..n_act], &sc.dense_buf, sparse_lim);
    (steps as f32 + v_term, false)
}

#[derive(Clone)]
struct AfterFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
    is_win: bool,
}

struct SiblingRec {
    /// Post-spawn afterstate features across the W CRN worlds
    world_feats: Vec<AfterFeat>,
    /// Exact CRN mean Q-value across W worlds
    q_crn: f32,
}

struct StateSample {
    siblings: Vec<SiblingRec>,
    /// Model's pre-rollout rank-0 index (0 by construction) vs CRN best
    crn_gain: f32,
    crn_disagree: bool,
}

/// Collect exact same-state sibling Q-Advantage samples from one Legal `p=5` outer game.
fn collect_game_samples(
    env_seed: u64,
    roll_mdl: &SpatialValueModel,
    term_mdl: &SpatialValueModel,
    outer_beam: usize,
    roll_beam: usize,
    states_per_game: usize,
    k_sibs: usize,
    w_worlds: usize,
    target_ds: u16,
    sparse_lim: usize,
) -> (Vec<StateSample>, usize, bool) {
    let fst = FastShapeTable::srs();
    let mut sc = BeamScratch::new(outer_beam.max(roll_beam));
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

    // Sample target `rem_g` checkpoints spread across the 100L race
    let mut sample_rem_targets: Vec<u16> = Vec::with_capacity(states_per_game);
    let mut s_rng = env_seed.wrapping_mul(0xD1B54A32D192ED03).wrapping_add(97);
    let stride = 88usize / states_per_game.max(1);
    for idx in 0..states_per_game {
        let lo = 6 + idx * stride;
        let off = next_usize(&mut s_rng, stride.max(1));
        sample_rem_targets.push((96usize.saturating_sub(lo + off)).max(5) as u16);
    }
    sample_rem_targets.sort_unstable_by(|a, b| b.cmp(a)); // descending rem_g
    let mut next_target_idx = 0usize;

    let mut samples: Vec<StateSample> = Vec::with_capacity(states_per_game);
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    while ds_total < 100 && pieces < 360 {
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

        let (best_fp_opt, ranked) = legal_p5_beam_rank(
            &fst,
            &mut sc,
            &board,
            hold,
            &vis_q,
            rem_g,
            cur_bag_mask,
            roll_mdl,
            outer_beam,
            sparse_lim,
        );
        let chosen_fp = match best_fp_opt {
            Some(f) => f,
            None => break,
        };

        // Check if we should branch exact CRN sibling rollouts at this state
        if next_target_idx < sample_rem_targets.len()
            && rem_g <= sample_rem_targets[next_target_idx]
            && ranked.len() >= 3
            && rem_g >= 4
        {
            next_target_idx += 1;

            // Select up to `k_sibs` distinct sibling actions:
            // - Top `(k_sibs - 2)` by current 5-ply beam
            // - Plus up to 2 diverse actions (best ds=1 not yet included, best clean ds=0 not yet included)
            let mut chosen_sibs: Vec<FastPlace> = Vec::with_capacity(k_sibs);
            let mut seen_pre: HashSet<(FastBoard, Option<u8>, u8)> = HashSet::new();
            let c_nh = vis_q.first().copied();

            let try_add = |fp: FastPlace,
                           list: &mut Vec<FastPlace>,
                           seen_pre: &mut HashSet<(FastBoard, Option<u8>, u8)>|
             -> bool {
                let mut b_tmp = board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (_l, _ds, _) = b_tmp.place_and_sift(o, fp.norm_i, fp.norm_j);
                let (h1, q1) = if hold.is_some() {
                    if fp.did_hold {
                        (c_nh, if c_nh.is_some() { 1u8 } else { 0u8 })
                    } else {
                        (hold, 1u8)
                    }
                } else if fp.did_hold {
                    (c_nh, 2u8)
                } else {
                    (None, 1u8)
                };
                if seen_pre.insert((b_tmp, h1, q1)) {
                    list.push(fp);
                    true
                } else {
                    false
                }
            };

            let top_keep = k_sibs.saturating_sub(2).max(2);
            for &(_s, fp) in &ranked {
                if chosen_sibs.len() >= top_keep {
                    break;
                }
                try_add(fp, &mut chosen_sibs, &mut seen_pre);
            }

            // Diverse candidate 1: highest-ranked ds=1 not yet in chosen_sibs
            if chosen_sibs.len() < k_sibs {
                for &(_s, fp) in &ranked {
                    let mut b_tmp = board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (_l, ds, _) = b_tmp.place_and_sift(o, fp.norm_i, fp.norm_j);
                    if ds > 0 && try_add(fp, &mut chosen_sibs, &mut seen_pre) {
                        break;
                    }
                }
            }
            // Diverse candidate 2: highest-ranked ds=0 (or clean skim) not yet in chosen_sibs
            if chosen_sibs.len() < k_sibs {
                for &(_s, fp) in &ranked {
                    let mut b_tmp = board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (_l, ds, _) = b_tmp.place_and_sift(o, fp.norm_i, fp.norm_j);
                    if ds == 0 && try_add(fp, &mut chosen_sibs, &mut seen_pre) {
                        break;
                    }
                }
            }
            // Fill remaining slots from ranked
            for &(_s, fp) in &ranked {
                if chosen_sibs.len() >= k_sibs {
                    break;
                }
                try_add(fp, &mut chosen_sibs, &mut seen_pre);
            }

            if chosen_sibs.len() >= 3 {
                // Generate W CRN worlds strictly from observable state (ZERO hidden env seed!)
                let mut crn_rng = observable_state_hash(
                    &board,
                    hold,
                    &vis_q,
                    rem_g,
                    cur_bag_mask,
                    (next_target_idx as u64) ^ 0x51B11A6,
                );
                let mut worlds: Vec<CrnWorld> = Vec::with_capacity(w_worlds);
                for _ in 0..w_worlds {
                    worlds.push(generate_crn_world(
                        &board,
                        &vis_q,
                        cur_bag_mask,
                        &mut crn_rng,
                        80,
                        48,
                    ));
                }

                let goal_rem = rem_g.saturating_sub(target_ds);
                let mut sib_recs: Vec<SiblingRec> = Vec::with_capacity(chosen_sibs.len());

                for &fp0 in &chosen_sibs {
                    let mut q_sum = 0.0f32;
                    let mut w_feats: Vec<AfterFeat> = Vec::with_capacity(w_worlds);
                    let mut valid_worlds = 0usize;

                    for world in &worlds {
                        let mut b1 = board;
                        let o0 = &fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
                        let (l0, ds0, _) = b1.place_and_sift(o0, fp0.norm_i, fp0.norm_j);
                        let rem1 = rem_g.saturating_sub(ds0 as u16);

                        if rem1 == 0 && b1.rows < 20 {
                            q_sum += 1.0;
                            w_feats.push(AfterFeat {
                                active: Vec::new(),
                                dense: [0.0; DENSE_DIM],
                                is_win: true,
                            });
                            valid_worlds += 1;
                            continue;
                        }

                        let mut h_cur = 0usize;
                        let alive = spawn_from_hole_stream(
                            &mut b1,
                            rem1,
                            l0,
                            &world.holes,
                            &mut h_cur,
                        );
                        if !alive {
                            let rem_left = rem1.saturating_sub(goal_rem) as f32;
                            q_sum += 1.0 + rem_left * 2.5 + 35.0;
                            continue;
                        }
                        let (held1, qc1) = if hold.is_some() {
                            if fp0.did_hold {
                                (c_nh, if c_nh.is_some() { 1 } else { 0 })
                            } else {
                                (hold, 1)
                            }
                        } else if fp0.did_hold {
                            (c_nh, 2)
                        } else {
                            (None, 1)
                        };

                        // Extract post-spawn afterstate features for exact model training
                        let q_end1 = (qc1 + 6).min(world.q.len());
                        let vis1 = &world.q[qc1..q_end1];
                        let ih1 = hand_piece(held1, 0, vis1);
                        let av1 = avail_mask_from_q(held1, 0, vis1, world.init_bag);
                        let n_act = extract_spatial_features_ex(
                            &b1,
                            rem1,
                            ih1,
                            av1,
                            false,
                            &mut sc.act_buf,
                            &mut sc.dense_buf,
                        );
                        let mut act_v = Vec::with_capacity(n_act.min(64));
                        for &bit in &sc.act_buf[..n_act] {
                            if (bit as usize) < sparse_lim {
                                act_v.push(bit);
                            }
                        }
                        w_feats.push(AfterFeat {
                            active: act_v,
                            dense: sc.dense_buf,
                            is_win: false,
                        });
                        valid_worlds += 1;

                        let (ret, _top) = rollout_legal_p5_to_goal(
                            &fst,
                            &mut sc,
                            b1,
                            held1,
                            qc1,
                            rem1,
                            goal_rem,
                            h_cur,
                            world,
                            roll_mdl,
                            term_mdl,
                            roll_beam,
                            sparse_lim,
                            55,
                        );
                        q_sum += ret;
                    }

                    if valid_worlds > 0 {
                        sib_recs.push(SiblingRec {
                            world_feats: w_feats,
                            q_crn: q_sum / (w_worlds as f32),
                        });
                    }
                }

                if sib_recs.len() >= 3 {
                    let mut best_k = 0usize;
                    let mut q_min = f32::INFINITY;
                    let mut q_max = f32::NEG_INFINITY;
                    for (k, sr) in sib_recs.iter().enumerate() {
                        if sr.q_crn < q_min {
                            q_min = sr.q_crn;
                            best_k = k;
                        }
                        if sr.q_crn > q_max {
                            q_max = sr.q_crn;
                        }
                    }
                    if q_max - q_min >= 0.05 {
                        let crn_gain = (sib_recs[0].q_crn - q_min).max(0.0);
                        samples.push(StateSample {
                            siblings: sib_recs,
                            crn_gain,
                            crn_disagree: best_k != 0,
                        });
                    }
                }
            }
        }

        // Execute chosen move in outer game
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

    (samples, pieces, ds_total < 100)
}

/// Evaluate a model on fixed Legal `p=5` validation games using 5-ply `beam_w`.
fn eval_val_game(env_seed: u64, mdl: &SpatialValueModel, beam_w: usize, sparse_lim: usize) -> (usize, bool) {
    let fst = FastShapeTable::srs();
    let mut sc = BeamScratch::new(beam_w);
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

    let mut ds_total = 0usize;
    let mut pieces = 0usize;
    while ds_total < 100 && pieces < 360 {
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
        let (best_fp, _) = legal_p5_beam_rank(
            &fst,
            &mut sc,
            &board,
            hold,
            &vis_q,
            rem_g,
            cur_bag_mask,
            mdl,
            beam_w,
            sparse_lim,
        );
        let fp = match best_fp {
            Some(f) => f,
            None => break,
        };
        let mut inputs = match ai_inputs.inputs_for_place(&snap.matrix, fp) {
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

fn evaluate_val_set(
    mdl: &SpatialValueModel,
    n_games: usize,
    beam_w: usize,
    sparse_lim: usize,
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
                let seed = 8001u64 + idx as u64;
                let res = eval_val_game(seed, mdl, beam_w, sparse_lim);
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

struct GradBuf {
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

    fn scale(&mut self, s: f32, sparse_lim: usize) {
        for i in 0..sparse_lim {
            self.lin_sparse[i] *= s;
            for h in 0..TRAIN_H1 {
                self.w_sparse[i][h] *= s;
            }
        }
        for j in 0..DENSE_DIM {
            self.lin_dense[j] *= s;
            for h in 0..TRAIN_H1 {
                self.w_dense[j][h] *= s;
            }
        }
        for h in 0..TRAIN_H1 {
            self.b1[h] *= s;
            for k in 0..TRAIN_H2 {
                self.w2[h][k] *= s;
            }
        }
        for k in 0..TRAIN_H2 {
            self.b2[k] *= s;
            self.w3[k] *= s;
        }
        self.b3 *= s;
    }
}

struct AdamState {
    step: usize,
    m: GradBuf,
    v: GradBuf,
}

impl AdamState {
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
fn forward_after(
    mdl: &SpatialValueModel,
    f: &AfterFeat,
    z_out: &mut [f32; DENSE_DIM],
    a1_out: &mut [f32; SPATIAL_H1],
    a2_out: &mut [f32; SPATIAL_H2],
) -> (f32, f32) {
    if f.is_win {
        return (0.0, 0.0);
    }
    let mut lin = mdl.lin_bias;
    for &b in &f.active {
        lin += mdl.lin_sparse[b as usize];
    }
    for j in 0..DENSE_DIM {
        let z = (f.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j];
        z_out[j] = z;
        lin += mdl.lin_dense[j] * z;
    }
    let mut h1 = mdl.b1;
    for &b in &f.active {
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
fn backward_after(
    mdl: &SpatialValueModel,
    f: &AfterFeat,
    z: &[f32; DENSE_DIM],
    a1: &[f32; SPATIAL_H1],
    a2: &[f32; SPATIAL_H2],
    raw_mlp: f32,
    d_out: f32,
    train_mlp: bool,
    g: &mut GradBuf,
) {
    if f.is_win || d_out == 0.0 {
        return;
    }
    for &b in &f.active {
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
    let d_raw = d_out / (denom * denom);
    g.b3 += d_raw;

    let mut d_h2 = [0.0f32; TRAIN_H2];
    for k in 0..TRAIN_H2 {
        g.w3[k] += d_raw * a2[k];
        if a2[k] > 0.0 && a2[k] < 6.0 {
            d_h2[k] = d_raw * mdl.w3[k];
        }
    }
    let mut d_h1 = [0.0f32; TRAIN_H1];
    for k in 0..TRAIN_H2 {
        g.b2[k] += d_h2[k];
    }
    for h in 0..TRAIN_H1 {
        let a1h = a1[h];
        let mut dh = 0.0f32;
        for k in 0..TRAIN_H2 {
            let dk = d_h2[k];
            if dk != 0.0 {
                if a1h > 0.0 {
                    g.w2[h][k] += a1h * dk;
                }
                dh += mdl.w2[h][k] * dk;
            }
        }
        if a1h > 0.0 && a1h < 6.0 {
            d_h1[h] = dh;
        }
    }
    for h in 0..TRAIN_H1 {
        g.b1[h] += d_h1[h];
    }
    for &b in &f.active {
        let row = &mut g.w_sparse[b as usize];
        for h in 0..TRAIN_H1 {
            row[h] += d_h1[h];
        }
    }
    for j in 0..DENSE_DIM {
        if j == 0 || j == 2 {
            continue;
        }
        let zm = z[j].clamp(-3.0, 3.0);
        if zm != 0.0 {
            let row = &mut g.w_dense[j];
            for h in 0..TRAIN_H1 {
                row[h] += zm * d_h1[h];
            }
        }
    }
}

/// Pure Relative Sibling Q-Advantage Loss (Listwise Softmax CE + Centered Relative Huber MSE).
/// ZERO absolute scalar regression!
fn accumulate_state_loss(
    mdl: &SpatialValueModel,
    sample: &StateSample,
    tau_tgt: f32,
    tau_pred: f32,
    w_ce: f32,
    w_adv: f32,
    huber_delta: f32,
    train_mlp: bool,
    g: &mut GradBuf,
) -> (f32, f32, bool) {
    let k = sample.siblings.len();
    if k < 2 {
        return (0.0, 0.0, false);
    }

    let mut q_hat = vec![0.0f32; k];
    let mut q_tgt = vec![0.0f32; k];

    // Cache forward passes per (sibling, world)
    struct WCache {
        z: [f32; DENSE_DIM],
        a1: [f32; SPATIAL_H1],
        a2: [f32; SPATIAL_H2],
        raw_mlp: f32,
    }
    let mut caches: Vec<Vec<WCache>> = Vec::with_capacity(k);

    for i in 0..k {
        let sr = &sample.siblings[i];
        q_tgt[i] = sr.q_crn;
        let nw = sr.world_feats.len();
        let mut wc = Vec::with_capacity(nw);
        let mut v_sum = 0.0f32;
        for wf in &sr.world_feats {
            let mut z = [0.0f32; DENSE_DIM];
            let mut a1 = [0.0f32; SPATIAL_H1];
            let mut a2 = [0.0f32; SPATIAL_H2];
            let (v, rm) = forward_after(mdl, wf, &mut z, &mut a1, &mut a2);
            v_sum += 1.0 + v;
            wc.push(WCache { z, a1, a2, raw_mlp: rm });
        }
        q_hat[i] = v_sum / (nw.max(1) as f32);
        caches.push(wc);
    }

    // 1. Target & predicted softmax distributions
    let q_tgt_min = q_tgt.iter().copied().fold(f32::INFINITY, f32::min);
    let q_hat_min = q_hat.iter().copied().fold(f32::INFINITY, f32::min);
    let mut p_tgt = vec![0.0f32; k];
    let mut p_hat = vec![0.0f32; k];
    let mut zt = 0.0f32;
    let mut zh = 0.0f32;
    for i in 0..k {
        let et = (-((q_tgt[i] - q_tgt_min) / tau_tgt).clamp(-20.0, 20.0)).exp();
        let eh = (-((q_hat[i] - q_hat_min) / tau_pred).clamp(-20.0, 20.0)).exp();
        p_tgt[i] = et;
        p_hat[i] = eh;
        zt += et;
        zh += eh;
    }
    let mut ce_loss = 0.0f32;
    for i in 0..k {
        p_tgt[i] /= zt.max(1e-12);
        p_hat[i] /= zh.max(1e-12);
        ce_loss -= p_tgt[i] * p_hat[i].max(1e-12).ln();
    }

    // 2. Centered relative Q-Advantage Huber loss (absolute baseline cancels identically!)
    let mean_tgt: f32 = q_tgt.iter().sum::<f32>() / (k as f32);
    let mean_hat: f32 = q_hat.iter().sum::<f32>() / (k as f32);
    let mut adv_loss = 0.0f32;
    let mut d_delta = vec![0.0f32; k];
    for i in 0..k {
        let err = (q_hat[i] - mean_hat) - (q_tgt[i] - mean_tgt);
        let ae = err.abs();
        if ae <= huber_delta {
            adv_loss += 0.5 * err * err / (k as f32);
            d_delta[i] = err / (k as f32);
        } else {
            adv_loss += huber_delta * (ae - 0.5 * huber_delta) / (k as f32);
            d_delta[i] = huber_delta * err.signum() / (k as f32);
        }
    }
    let mean_dd: f32 = d_delta.iter().sum::<f32>() / (k as f32);

    // 3. Exact gradient dL / d q_hat[i]
    //    d(L_CE) / d(q_hat[i]) = (p_tgt[i] - p_hat[i]) / tau_pred
    //    d(L_adv) / d(q_hat[i]) = d_delta[i] - mean(d_delta)
    for i in 0..k {
        let g_ce = w_ce * (p_tgt[i] - p_hat[i]) / tau_pred;
        let g_adv = w_adv * (d_delta[i] - mean_dd);
        let d_qi = (g_ce + g_adv).clamp(-5.0, 5.0);
        let nw = sample.siblings[i].world_feats.len().max(1) as f32;
        let d_w = d_qi / nw;
        for (wi, wf) in sample.siblings[i].world_feats.iter().enumerate() {
            let wc = &caches[i][wi];
            backward_after(mdl, wf, &wc.z, &wc.a1, &wc.a2, wc.raw_mlp, d_w, train_mlp, g);
        }
    }

    let best_tgt = q_tgt
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    let best_hat = q_hat
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    (ce_loss, adv_loss, best_tgt == best_hat)
}

fn self_check() {
    // 1. Verify DKey distinguishes qc
    let b = FastBoard::empty();
    let k1 = DKey { board: b, held: None, qc: 1, rem_g: 50 };
    let k2 = DKey { board: b, held: None, qc: 2, rem_g: 50 };
    assert!(k1 != k2, "DKey must distinguish qc");

    // 2. Verify observable_state_hash is deterministic and independent of any hidden seed
    let h1 = observable_state_hash(&b, Some(1), &[0, 2, 3, 4, 5, 6], 50, 0x7F, 42);
    let h2 = observable_state_hash(&b, Some(1), &[0, 2, 3, 4, 5, 6], 50, 0x7F, 42);
    assert_eq!(h1, h2);

    // 3. Verify spawn_from_hole_stream respects min_height=4, max_height=9 & hole stream
    let mut b_sp = FastBoard::empty();
    let mut hc = 0usize;
    let holes = [3u16, 7, 1, 8, 2, 5, 0, 9, 4];
    assert!(spawn_from_hole_stream(&mut b_sp, 50, 0, &holes, &mut hc));
    assert_eq!(b_sp.garbage_rows, 9);
    assert_eq!(hc, 9);
}

fn main() {
    self_check();

    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "/tmp/spatial_sibq_v1.txt".to_string();
    let mut rounds: usize = 4;
    let mut games_per_round: usize = 32;
    let mut states_per_game: usize = 6;
    let mut k_sibs: usize = 8;
    let mut w_worlds: usize = 6;
    let mut target_ds: u16 = 14;
    let mut outer_beam: usize = 16;
    let mut roll_beam: usize = 10;
    let mut val_games: usize = 60;
    let mut sparse_lim: usize = 1591;
    let mut epochs: usize = 6;
    let mut batch_size: usize = 24;
    let mut tau_tgt: f32 = 0.35;
    let mut tau_pred: f32 = 0.35;
    let mut w_ce: f32 = 1.0;
    let mut w_adv: f32 = 0.5;
    let mut lr_lin: f32 = 3e-4;
    let mut lr_mlp: f32 = 1.5e-4;
    let mut anchor_wd: f32 = 2e-4;
    let mut n_threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--init" => { i += 1; init_path = args[i].clone(); }
            "--out" => { i += 1; out_path = args[i].clone(); }
            "--rounds" => { i += 1; rounds = args[i].parse().unwrap(); }
            "--games-per-round" => { i += 1; games_per_round = args[i].parse().unwrap(); }
            "--states-per-game" => { i += 1; states_per_game = args[i].parse().unwrap(); }
            "--k-sibs" => { i += 1; k_sibs = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; w_worlds = args[i].parse().unwrap(); }
            "--target-ds" => { i += 1; target_ds = args[i].parse().unwrap(); }
            "--outer-beam" => { i += 1; outer_beam = args[i].parse().unwrap(); }
            "--roll-beam" => { i += 1; roll_beam = args[i].parse().unwrap(); }
            "--val-games" => { i += 1; val_games = args[i].parse().unwrap(); }
            "--sparse-lim" => { i += 1; sparse_lim = args[i].parse().unwrap(); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap(); }
            "--batch-size" => { i += 1; batch_size = args[i].parse().unwrap(); }
            "--tau-tgt" => { i += 1; tau_tgt = args[i].parse().unwrap(); }
            "--tau-pred" => { i += 1; tau_pred = args[i].parse().unwrap(); }
            "--w-ce" => { i += 1; w_ce = args[i].parse().unwrap(); }
            "--w-adv" => { i += 1; w_adv = args[i].parse().unwrap(); }
            "--lr-lin" => { i += 1; lr_lin = args[i].parse().unwrap(); }
            "--lr-mlp" => { i += 1; lr_mlp = args[i].parse().unwrap(); }
            "--anchor-wd" => { i += 1; anchor_wd = args[i].parse().unwrap(); }
            "--threads" => { i += 1; n_threads = args[i].parse().unwrap(); }
            other => {
                eprintln!("Unknown arg: {}", other);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    let init_mdl = SpatialValueModel::load_from_file(&init_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", init_path, e));
    let mut mdl = init_mdl.clone();
    let mut best_mdl = init_mdl.clone();
    let mut adam = AdamState::new();
    let mut grad = GradBuf::new();

    println!("================================================================");
    println!(" Pure Rust True Same-State Sibling-Action CRN Q-Advantage PI");
    println!(
        " init={} out={} rounds={} games/rd={} states/g={} K={} W={} dG={}",
        init_path, out_path, rounds, games_per_round, states_per_game, k_sibs, w_worlds, target_ds
    );
    println!(
        " outer_beam={} roll_beam={} val={} sparse_lim={} epochs={} bs={} tau={:.2}/{:.2} w_ce={:.2} w_adv={:.2}",
        outer_beam, roll_beam, val_games, sparse_lim, epochs, batch_size, tau_tgt, tau_pred, w_ce, w_adv
    );
    println!(
        " lr_lin={:.1e} lr_mlp={:.1e} anchor_wd={:.1e} threads={}",
        lr_lin, lr_mlp, anchor_wd, n_threads
    );
    println!("================================================================");
    let _ = std::io::stdout().flush();

    let t0 = Instant::now();
    let (base_val, base_top, base_min, base_max) =
        evaluate_val_set(&mdl, val_games, outer_beam, sparse_lim, n_threads);
    println!(
        "[Round 0 BASE] val={}/{} top={} mean={:.2} [{}..{}] ({:.1}s)",
        val_games - base_top,
        val_games,
        base_top,
        base_val,
        base_min,
        base_max,
        t0.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();
    let mut best_val = base_val;
    let mut best_top = base_top;

    // Replay buffer across rounds (keeps recent 2 rounds for stability + policy improvement)
    let mut all_rounds: Vec<Vec<StateSample>> = Vec::new();
    let mut shuffle_rng = 0x1234_5678_9ABC_DEF0u64;

    for rd in 1..=rounds {
        let t_gen = Instant::now();
        let roll_snap = Arc::new(mdl.clone());
        let term_snap = Arc::new(mdl.clone());
        let next_g = AtomicUsize::new(0);
        let collected: Mutex<Vec<StateSample>> =
            Mutex::new(Vec::with_capacity(games_per_round * states_per_game));
        let outer_stats: Mutex<Vec<(usize, bool)>> = Mutex::new(Vec::with_capacity(games_per_round));

        std::thread::scope(|s| {
            for _ in 0..n_threads {
                let rm = Arc::clone(&roll_snap);
                let tm = Arc::clone(&term_snap);
                let next_g = &next_g;
                let collected = &collected;
                let outer_stats = &outer_stats;
                s.spawn(move || loop {
                    let gi = next_g.fetch_add(1, Ordering::Relaxed);
                    if gi >= games_per_round {
                        break;
                    }
                    let env_seed = 40_000u64 + (rd as u64) * 1000 + (gi as u64);
                    let (mut samps, pcs, top) = collect_game_samples(
                        env_seed,
                        &rm,
                        &tm,
                        outer_beam,
                        roll_beam,
                        states_per_game,
                        k_sibs,
                        w_worlds,
                        target_ds,
                        sparse_lim,
                    );
                    collected.lock().unwrap().append(&mut samps);
                    outer_stats.lock().unwrap().push((pcs, top));
                });
            }
        });

        let rd_samples = collected.into_inner().unwrap();
        let o_stats = outer_stats.into_inner().unwrap();
        let n_samps = rd_samples.len();
        let disagree_cnt = rd_samples.iter().filter(|x| x.crn_disagree).count();
        let mean_gain = if n_samps > 0 {
            rd_samples.iter().map(|x| x.crn_gain as f64).sum::<f64>() / (n_samps as f64)
        } else {
            0.0
        };
        let mut o_sum = 0usize;
        let mut o_ok = 0usize;
        for &(p, top) in &o_stats {
            if !top {
                o_sum += p;
                o_ok += 1;
            }
        }
        let o_mean = if o_ok > 0 { o_sum as f64 / o_ok as f64 } else { 999.0 };
        let gen_s = t_gen.elapsed().as_secs_f64();

        all_rounds.push(rd_samples);
        if all_rounds.len() > 2 {
            all_rounds.remove(0);
        }

        // Flatten indices across active buffer
        let mut flat: Vec<(usize, usize)> = Vec::new();
        for (ri, vec_s) in all_rounds.iter().enumerate() {
            for si in 0..vec_s.len() {
                flat.push((ri, si));
            }
        }

        let t_tr = Instant::now();
        let mut last_ce = 0.0f64;
        let mut last_adv = 0.0f64;
        let mut last_acc = 0.0f64;

        for _ep in 0..epochs {
            // Shuffle
            for i in (1..flat.len()).rev() {
                let j = next_usize(&mut shuffle_rng, i + 1);
                flat.swap(i, j);
            }
            let mut ce_sum = 0.0f64;
            let mut adv_sum = 0.0f64;
            let mut acc_cnt = 0usize;

            let mut idx = 0usize;
            while idx < flat.len() {
                let end = (idx + batch_size).min(flat.len());
                let bs = end - idx;
                grad.zero(sparse_lim);
                for &(ri, si) in &flat[idx..end] {
                    let (ce, adv, ok) = accumulate_state_loss(
                        &mdl,
                        &all_rounds[ri][si],
                        tau_tgt,
                        tau_pred,
                        w_ce,
                        w_adv,
                        2.0,
                        lr_mlp > 0.0,
                        &mut grad,
                    );
                    ce_sum += ce as f64;
                    adv_sum += adv as f64;
                    if ok {
                        acc_cnt += 1;
                    }
                }
                grad.scale(1.0 / (bs as f32), sparse_lim);
                adam.apply(&mut mdl, &init_mdl, &grad, lr_lin, lr_mlp, anchor_wd, sparse_lim);
                idx = end;
            }
            let denom = flat.len().max(1) as f64;
            last_ce = ce_sum / denom;
            last_adv = adv_sum / denom;
            last_acc = 100.0 * (acc_cnt as f64) / denom;
        }
        let tr_s = t_tr.elapsed().as_secs_f64();

        let (v_mean, v_top, v_min, v_max) =
            evaluate_val_set(&mdl, val_games, outer_beam, sparse_lim, n_threads);
        let is_best = v_top < best_top || (v_top == best_top && v_mean < best_val);
        if is_best {
            best_val = v_mean;
            best_top = v_top;
            best_mdl = mdl.clone();
            best_mdl.save_to_file(&out_path).unwrap();
        }
        println!(
            "[Round {:2}/{}] outer={:.2}({}/{}) states={} crn_disagree={:.1}% crn_gain={:.3}pc | ce={:.4} adv={:.4} top1={:.1}% ({:.1}s+{:.1}s) | VAL={}/{} top={} mean={:.2} [{}..{}] {}",
            rd,
            rounds,
            o_mean,
            o_ok,
            games_per_round,
            n_samps,
            100.0 * (disagree_cnt as f64) / (n_samps.max(1) as f64),
            mean_gain,
            last_ce,
            last_adv,
            last_acc,
            gen_s,
            tr_s,
            val_games - v_top,
            val_games,
            v_top,
            v_mean,
            v_min,
            v_max,
            if is_best { "*** NEW BEST ***" } else { "" }
        );
        let _ = std::io::stdout().flush();
    }

    best_mdl.save_to_file(&out_path).unwrap();
    let last_path = out_path.replace(".txt", "_last.txt");
    mdl.save_to_file(&last_path).unwrap();
    println!(
        "=== DONE SIBLING-Q PI: base_val={:.2} (top={}) -> best_val={:.2} (top={}) | saved {} ===",
        base_val, base_top, best_val, best_top, out_path
    );
}
