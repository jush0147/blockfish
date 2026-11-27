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
    Color, Input, StackerExt as _,
};
use std::collections::HashSet;
use std::convert::TryFrom;
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

fn make_fullmlp_teacher_cfg(sm_arc: Arc<SpatialValueModel>, worlds: usize, width: usize) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.spatial_model = Some(sm_arc);
    c.rollout_top_k = 0;
    c.beam_width = 80;
    c.max_per_root = 16;
    c.bag_det_extend = true;
    c.endgame_solver = true;
    c.simulate_cheese_spawn = true;
    c.spatial_rerank_only = false; // REAL FULL MLP
    c.spatial_no_pe_internal = false;
    c.root_pimc_prescreen_worlds = 0;
    c.root_pimc_roots = 32;
    c.root_pimc_fine_roots = 10;
    c.root_pimc_worlds = worlds;
    c.root_pimc_horizon = 8;
    c.root_pimc_width = width;
    c.root_pimc_margin = 500;
    c.root_pimc_weight_pct = 100;
    c.root_pimc_steady_g4 = false;
    c.root_pimc_cvar_pct = 25;
    c
}

#[derive(Clone)]
struct AfterFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
    is_win: bool,
}

struct SiblingRec {
    world_feats: Vec<AfterFeat>,
    /// Relative Q-advantage in pieces from strong Legal p=5 `fullmlp_w16` teacher (`>= 0.0`).
    q_rel: f32,
}

struct StateSample {
    siblings: Vec<SiblingRec>,
}

/// Play one full Legal `p=5` `fullmlp_w16` game and collect exact same-state sibling Q-distributions.
fn collect_teacher_game(
    env_seed: u64,
    teacher_sm: Arc<SpatialValueModel>,
    teacher_worlds: usize,
    teacher_width: usize,
    spawn_worlds: usize,
    sparse_lim: usize,
) -> (Vec<StateSample>, usize, bool) {
    let fst = FastShapeTable::srs();
    let cfg = make_fullmlp_teacher_cfg(teacher_sm, teacher_worlds, teacher_width);
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

    let mut samples: Vec<StateSample> = Vec::with_capacity(200);
    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

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
        let top_cands = ai.last_top_candidates().to_vec();

        // Extract exact same-state sibling Q-distribution from `fullmlp_w16`
        if top_cands.len() >= 2 && rem_g >= 2 {
            let min_rating = top_cands.iter().map(|x| x.5).min().unwrap_or(0);
            // Skip immediate-win sentinel turns where ratings are near -15_000_000
            if min_rating > -5_000_000 && min_rating < 2_000_000 {
                let c_nh = vis_q.first().copied();
                let mut sibs: Vec<SiblingRec> = Vec::with_capacity(top_cands.len());
                let mut seen_pre: HashSet<(FastBoard, Option<u8>, u8)> = HashSet::new();

                for &(c_char, norm_i, norm_j, rot, did_hold, rating) in &top_cands {
                    // Exclude extreme topout/unreranked outliers (> 5.0 pieces worse than best)
                    if rating > min_rating + 500 {
                        continue;
                    }
                    let c_idx = match Color::try_from(c_char) {
                        Ok(col) => color_to_idx(col),
                        Err(_) => continue,
                    };
                    let mut b_pre = board;
                    let orient = &fst.shapes[c_idx].orients[rot as usize];
                    let (l0, ds0, _) = b_pre.place_and_sift(orient, norm_i, norm_j);
                    let rem1 = rem_g.saturating_sub(ds0 as u16);

                    let (held1, qc1) = if hold.is_some() {
                        if did_hold {
                            (c_nh, if c_nh.is_some() { 1usize } else { 0usize })
                        } else {
                            (hold, 1usize)
                        }
                    } else if did_hold {
                        (c_nh, 2usize)
                    } else {
                        (None, 1usize)
                    };

                    if !seen_pre.insert((b_pre, held1, qc1 as u8)) {
                        continue;
                    }

                    let mut w_feats: Vec<AfterFeat> = Vec::with_capacity(spawn_worlds);
                    let mut all_alive = true;

                    for w in 0..spawn_worlds {
                        if rem1 == 0 && b_pre.rows < 20 {
                            w_feats.push(AfterFeat {
                                active: Vec::new(),
                                dense: [0.0; DENSE_DIM],
                                is_win: true,
                            });
                            continue;
                        }
                        let mut b1 = b_pre;
                        let mut r_loc = observable_state_hash(
                            &board,
                            hold,
                            &vis_q,
                            rem_g,
                            cur_bag_mask,
                            (w as u64).wrapping_add(1),
                        );
                        if rem1 > 0 && !b1.spawn_cheese_rows(rem1, l0, &mut r_loc) {
                            all_alive = false;
                            break;
                        }
                        let col_h = b1.col_heights();
                        if *col_h.iter().max().unwrap() >= 20 {
                            all_alive = false;
                            break;
                        }
                        let ih1 = hand_piece(held1, qc1, &vis_q);
                        let av1 = avail_mask_from_q(held1, qc1, &vis_q, cur_bag_mask);
                        let n_act = extract_spatial_features_ex(
                            &b1,
                            rem1,
                            ih1,
                            av1,
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
                        w_feats.push(AfterFeat {
                            active: act_v,
                            dense: dense_buf,
                            is_win: false,
                        });
                    }

                    if all_alive {
                        debug_assert_eq!(w_feats.len(), spawn_worlds);
                        sibs.push(SiblingRec {
                            world_feats: w_feats,
                            q_rel: ((rating - min_rating) as f32) * 0.01,
                        });
                    }
                }

                if sibs.len() >= 2 {
                    let q_max = sibs.iter().map(|s| s.q_rel).fold(0.0f32, f32::max);
                    if q_max >= 0.05 {
                        samples.push(StateSample { siblings: sibs });
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

    (samples, pieces, ds_total < 100)
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

/// Fast 5-ply Legal `p=5` beam evaluator for 60-seed progress tracking.
fn eval_val_game(env_seed: u64, mdl: &SpatialValueModel, beam_w: usize, sparse_lim: usize) -> (usize, bool) {
    let fst = FastShapeTable::srs();
    let mut pgen = FastPlaceGen::new();
    let mut tmp_p0: Vec<FastPlace> = Vec::with_capacity(80);
    let mut tmp_p2: Vec<FastPlace> = Vec::with_capacity(64);
    let mut cur_beam: Vec<BNode> = Vec::with_capacity(beam_w * 2);
    let mut next_beam: Vec<BNode> = Vec::with_capacity(beam_w * 48);
    let mut seen: HashSet<DKey> = HashSet::with_capacity(beam_w * 4);
    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];

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

        for (ridx, &fp0) in tmp_p0.iter().enumerate() {
            let mut b1 = board;
            let o0 = &fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
            let (l0, ds0, _) = b1.place_and_sift(o0, fp0.norm_i, fp0.norm_j);
            let rem1 = rem_g.saturating_sub(ds0 as u16);
            if rem1 == 0 && b1.rows < 20 {
                win_fp = Some(fp0);
                break;
            }
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
            } else if fp0.did_hold { (c_nh, 2) } else { (None, 1) };
            let ih1 = hand_piece(held1, qc1, &vis_q);
            let av1 = avail_mask_from_q(held1, qc1, &vis_q, cur_bag_mask);
            let sc1 = mdl.evaluate_linear_board_with_bag_no_pe(&b1, rem1, ih1, av1) as f32 * 0.01;
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
                        let sc2 = mdl.evaluate_linear_board_with_bag_no_pe(&b2, rem2, ih2, av2) as f32 * 0.01;
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
                let mut best_sc = f32::INFINITY;
                let mut best_fp = tmp_p0[0];
                for &node in &cur_beam {
                    let ih = hand_piece(node.held, node.qc as usize, &vis_q);
                    let av = avail_mask_from_q(node.held, node.qc as usize, &vis_q, cur_bag_mask);
                    let n_act = extract_spatial_features_ex(
                        &node.board,
                        node.rem_g,
                        ih,
                        av,
                        false,
                        &mut act_buf,
                        &mut dense_buf,
                    );
                    let v = predict_with_lim(mdl, &act_buf[..n_act], &dense_buf, sparse_lim);
                    if v < best_sc {
                        best_sc = v;
                        best_fp = tmp_p0[node.first_idx as usize];
                    }
                }
                best_fp
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

fn eval_or_grad_state(
    mdl: &SpatialValueModel,
    sample: &StateSample,
    tau_tgt: f32,
    tau_pred: f32,
    w_ce: f32,
    w_adv: f32,
    huber_delta: f32,
    train_mlp: bool,
    g: Option<&mut GradBuf>,
) -> (f32, f32, bool) {
    let k = sample.siblings.len();
    if k < 2 {
        return (0.0, 0.0, false);
    }

    let mut q_hat = vec![0.0f32; k];
    let mut q_tgt = vec![0.0f32; k];

    struct WCache {
        z: [f32; DENSE_DIM],
        a1: [f32; SPATIAL_H1],
        a2: [f32; SPATIAL_H2],
        raw_mlp: f32,
    }
    let mut caches: Vec<Vec<WCache>> = Vec::with_capacity(k);

    for i in 0..k {
        let sr = &sample.siblings[i];
        q_tgt[i] = sr.q_rel;
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

    if let Some(grad) = g {
        let mean_dd: f32 = d_delta.iter().sum::<f32>() / (k as f32);
        for i in 0..k {
            let g_ce = w_ce * (p_tgt[i] - p_hat[i]) / tau_pred;
            let g_adv = w_adv * (d_delta[i] - mean_dd);
            let d_qi = (g_ce + g_adv).clamp(-5.0, 5.0);
            let nw = sample.siblings[i].world_feats.len().max(1) as f32;
            let d_w = d_qi / nw;
            for (wi, wf) in sample.siblings[i].world_feats.iter().enumerate() {
                let wc = &caches[i][wi];
                backward_after(mdl, wf, &wc.z, &wc.a1, &wc.a2, wc.raw_mlp, d_w, train_mlp, grad);
            }
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut init_path = "models/spatial_bc4.txt".to_string();
    let mut out_path = "/tmp/spatial_exit_w16_v1.txt".to_string();
    let mut n_games: usize = 48;
    let mut teacher_worlds: usize = 16;
    let mut teacher_width: usize = 28;
    let mut spawn_worlds: usize = 4;
    let mut val_games: usize = 60;
    let mut val_beam: usize = 16;
    let mut sparse_lim: usize = 1591;
    let mut epochs: usize = 12;
    let mut batch_size: usize = 32;
    let mut tau_tgt: f32 = 0.25;
    let mut tau_pred: f32 = 0.25;
    let mut w_ce: f32 = 1.0;
    let mut w_adv: f32 = 0.25;
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
            "--games" => { i += 1; n_games = args[i].parse().unwrap(); }
            "--teacher-worlds" => { i += 1; teacher_worlds = args[i].parse().unwrap(); }
            "--teacher-width" => { i += 1; teacher_width = args[i].parse().unwrap(); }
            "--spawn-worlds" => { i += 1; spawn_worlds = args[i].parse().unwrap(); }
            "--val-games" => { i += 1; val_games = args[i].parse().unwrap(); }
            "--val-beam" => { i += 1; val_beam = args[i].parse().unwrap(); }
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

    let mut init_mdl = SpatialValueModel::load_from_file(&init_path)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", init_path, e));
    init_mdl.prepare_fast();
    let teacher_arc = Arc::new(init_mdl.clone());

    println!("================================================================");
    println!(" Pure Rust Strong-Teacher Legal p=5 Expert Iteration (fullmlp_w16)");
    println!(
        " init={} out={} games={} teacher=w{}_b{} spawn_W={} sparse_lim={}",
        init_path, out_path, n_games, teacher_worlds, teacher_width, spawn_worlds, sparse_lim
    );
    println!(
        " epochs={} bs={} tau={:.2}/{:.2} w_ce={:.2} w_adv={:.2} lr={:.1e}/{:.1e} wd={:.1e} threads={}",
        epochs, batch_size, tau_tgt, tau_pred, w_ce, w_adv, lr_lin, lr_mlp, anchor_wd, n_threads
    );
    println!("================================================================");
    let _ = std::io::stdout().flush();

    // 1. Collect strong Legal p=5 fullmlp_w16 games in parallel
    let t_gen = Instant::now();
    let next_g = AtomicUsize::new(0);
    let game_results: Mutex<Vec<(usize, Vec<StateSample>, usize, bool)>> =
        Mutex::new(Vec::with_capacity(n_games));

    std::thread::scope(|s| {
        for _ in 0..n_threads {
            let tm = Arc::clone(&teacher_arc);
            let next_g = &next_g;
            let game_results = &game_results;
            s.spawn(move || loop {
                let gi = next_g.fetch_add(1, Ordering::Relaxed);
                if gi >= n_games {
                    break;
                }
                let env_seed = 50_001u64 + gi as u64;
                let (samps, pcs, top) = collect_teacher_game(
                    env_seed,
                    Arc::clone(&tm),
                    teacher_worlds,
                    teacher_width,
                    spawn_worlds,
                    sparse_lim,
                );
                game_results.lock().unwrap().push((gi, samps, pcs, top));
            });
        }
    });

    let mut gr = game_results.into_inner().unwrap();
    gr.sort_by_key(|x| x.0);
    let mut sum_pcs = 0usize;
    let mut ok_games = 0usize;
    let mut train_samples: Vec<StateSample> = Vec::new();
    let mut holdout_samples: Vec<StateSample> = Vec::new();
    let n_holdout_games = (n_games / 10).max(1);
    let split_idx = n_games.saturating_sub(n_holdout_games);

    for (gi, mut samps, pcs, top) in gr {
        if !top {
            sum_pcs += pcs;
            ok_games += 1;
        }
        if gi < split_idx {
            train_samples.append(&mut samps);
        } else {
            holdout_samples.append(&mut samps);
        }
    }
    let teacher_mean = if ok_games > 0 {
        sum_pcs as f64 / ok_games as f64
    } else {
        999.0
    };
    println!(
        "[DATA] teacher_games={}/{} top={} mean={:.2} | train_states={} holdout_states={} ({:.1}s)",
        ok_games,
        n_games,
        n_games - ok_games,
        teacher_mean,
        train_samples.len(),
        holdout_samples.len(),
        t_gen.elapsed().as_secs_f64()
    );
    let _ = std::io::stdout().flush();

    let mut mdl = init_mdl.clone();
    let mut best_mdl = init_mdl.clone();
    let mut adam = AdamState::new();
    let mut grad = GradBuf::new();

    // Baseline 60-seed 5-ply beam=16 val + holdout state metrics
    let t0 = Instant::now();
    let (base_val, base_top, base_min, base_max) =
        evaluate_val_set(&mdl, val_games, val_beam, sparse_lim, n_threads);
    let (mut ho_ce, mut ho_adv, mut ho_acc) = (0.0f64, 0.0f64, 0usize);
    for s in &holdout_samples {
        let (ce, adv, ok) = eval_or_grad_state(&mdl, s, tau_tgt, tau_pred, w_ce, w_adv, 1.0, false, None);
        ho_ce += ce as f64;
        ho_adv += adv as f64;
        if ok { ho_acc += 1; }
    }
    let ho_n = holdout_samples.len().max(1) as f64;
    println!(
        "[Epoch  0 BASE] ho_ce={:.4} ho_adv={:.4} ho_top1={:.1}% | VAL={}/{} top={} mean={:.2} [{}..{}] ({:.1}s)",
        ho_ce / ho_n,
        ho_adv / ho_n,
        100.0 * (ho_acc as f64) / ho_n,
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
    let mut order: Vec<usize> = (0..train_samples.len()).collect();
    let mut rng = 0xCAFE_BABE_1234_5678u64;

    for ep in 1..=epochs {
        let t_ep = Instant::now();
        for i in (1..order.len()).rev() {
            let j = next_usize(&mut rng, i + 1);
            order.swap(i, j);
        }
        let mut tr_ce = 0.0f64;
        let mut tr_acc = 0usize;

        let mut idx = 0usize;
        while idx < order.len() {
            let end = (idx + batch_size).min(order.len());
            let bs = end - idx;
            grad.zero(sparse_lim);
            for &si in &order[idx..end] {
                let (ce, _adv, ok) = eval_or_grad_state(
                    &mdl,
                    &train_samples[si],
                    tau_tgt,
                    tau_pred,
                    w_ce,
                    w_adv,
                    1.0,
                    lr_mlp > 0.0,
                    Some(&mut grad),
                );
                tr_ce += ce as f64;
                if ok { tr_acc += 1; }
            }
            grad.scale(1.0 / (bs as f32), sparse_lim);
            adam.apply(&mut mdl, &init_mdl, &grad, lr_lin, lr_mlp, anchor_wd, sparse_lim);
            idx = end;
        }
        let tr_n = order.len().max(1) as f64;

        let (mut h_ce, mut h_adv, mut h_acc) = (0.0f64, 0.0f64, 0usize);
        for s in &holdout_samples {
            let (ce, adv, ok) = eval_or_grad_state(&mdl, s, tau_tgt, tau_pred, w_ce, w_adv, 1.0, false, None);
            h_ce += ce as f64;
            h_adv += adv as f64;
            if ok { h_acc += 1; }
        }

        if ep % 2 == 0 || ep == epochs {
            let (v_mean, v_top, v_min, v_max) =
                evaluate_val_set(&mdl, val_games, val_beam, sparse_lim, n_threads);
            let is_best = v_top < best_top || (v_top == best_top && v_mean < best_val);
            if is_best {
                best_val = v_mean;
                best_top = v_top;
                best_mdl = mdl.clone();
                best_mdl.save_to_file(&out_path).unwrap();
            }
            println!(
                "[Epoch {:2}/{}] tr_ce={:.4} tr_top1={:.1}% | ho_ce={:.4} ho_adv={:.4} ho_top1={:.1}% ({:.1}s) | VAL={}/{} top={} mean={:.2} [{}..{}] {}",
                ep,
                epochs,
                tr_ce / tr_n,
                100.0 * (tr_acc as f64) / tr_n,
                h_ce / ho_n,
                h_adv / ho_n,
                100.0 * (h_acc as f64) / ho_n,
                t_ep.elapsed().as_secs_f64(),
                val_games - v_top,
                val_games,
                v_top,
                v_mean,
                v_min,
                v_max,
                if is_best { "*** NEW BEST ***" } else { "" }
            );
        } else {
            println!(
                "[Epoch {:2}/{}] tr_ce={:.4} tr_top1={:.1}% | ho_ce={:.4} ho_adv={:.4} ho_top1={:.1}% ({:.1}s)",
                ep,
                epochs,
                tr_ce / tr_n,
                100.0 * (tr_acc as f64) / tr_n,
                h_ce / ho_n,
                h_adv / ho_n,
                100.0 * (h_acc as f64) / ho_n,
                t_ep.elapsed().as_secs_f64(),
            );
        }
        let _ = std::io::stdout().flush();
    }

    best_mdl.save_to_file(&out_path).unwrap();
    let last_path = out_path.replace(".txt", "_last.txt");
    mdl.save_to_file(&last_path).unwrap();
    println!(
        "=== DONE EXIT-W16: base_val={:.2} (top={}) -> best_val={:.2} (top={}) | saved {} ===",
        base_val, base_top, best_val, best_top, out_path
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teacher_collection_world_alignment_and_loss() {
        let mut mdl = SpatialValueModel::load_from_file("../models/spatial_bc4.txt")
            .or_else(|_| SpatialValueModel::load_from_file("models/spatial_bc4.txt"))
            .expect("load spatial_bc4.txt");
        mdl.prepare_fast();
        // Fast 2-world smoke check on 1 game verifying 1-to-1 spawn_worlds alignment & finite loss
        let (samps, pcs, top) =
            collect_teacher_game(101, Arc::new(mdl.clone()), 1, 4, 4, 1591);
        assert!(!top && pcs > 100 && !samps.is_empty());
        let mut g = GradBuf::new();
        for s in samps.iter().take(20) {
            assert!(s.siblings.len() >= 2);
            for sib in &s.siblings {
                assert_eq!(sib.world_feats.len(), 4);
                assert!(sib.q_rel.is_finite() && sib.q_rel >= 0.0);
            }
            let (ce, adv, _) =
                eval_or_grad_state(&mdl, s, 0.25, 0.25, 1.0, 0.25, 1.0, true, Some(&mut g));
            assert!(ce.is_finite() && adv.is_finite());
        }
    }
}
