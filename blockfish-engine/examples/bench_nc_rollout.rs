use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        fast::{color_to_idx, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        spatial_val::SpatialValueModel,
        BagTracker, BeamConfig, FastAI,
    },
    Input, StackerExt as _,
};
use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

#[inline]
fn next_rng(rng: &mut u64) -> u64 {
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *rng
}

#[inline]
fn draw_from_bag(mask: &mut u8, rng: &mut u64) -> u8 {
    if *mask == 0 {
        *mask = 0x7F;
    }
    let cnt = mask.count_ones();
    let mut pick = ((next_rng(rng) >> 32) as u32) % cnt;
    for p in 0..7u8 {
        if (*mask & (1u8 << p)) != 0 {
            if pick == 0 {
                *mask &= !(1u8 << p);
                if *mask == 0 {
                    *mask = 0x7F;
                }
                return p;
            }
            pick -= 1;
        }
    }
    0
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

#[inline]
fn spawn_cheese_from_stream(
    b: &mut FastBoard,
    rem_g: u16,
    lines_cleared: u8,
    unspawned_0: usize,
    holes: &[u16],
) -> bool {
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
    let spawned_offset = unspawned_0.saturating_sub(unspawned);
    if spawned_offset + num > holes.len() {
        return false;
    }
    for r in (0..old_rows).rev() {
        b.data[r + num] = b.data[r];
    }
    for k in 0..num {
        let hole = holes[spawned_offset + k];
        let dst_r = num - 1 - k;
        b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
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

fn h1_info(stacker: &Stacker) -> (Option<usize>, usize, usize, usize) {
    let g = garbage_rows(stacker) as usize;
    if g == 0 {
        return (None, 0, 0, 0);
    }
    let mut grid = [[false; 10]; 24];
    let mut col_h = [0usize; 10];
    for ((r, c), _) in stacker.matrix() {
        let ru = r as usize;
        let cu = c as usize;
        if ru < 24 && cu < 10 {
            grid[ru][cu] = true;
            col_h[cu] = col_h[cu].max(ru + 1);
        }
    }
    let mut h1 = None;
    for c in 0..10 {
        if !grid[g - 1][c] {
            h1 = Some(c);
            break;
        }
    }
    let h1c = match h1 {
        Some(c) => c,
        None => return (None, 0, 0, 0),
    };
    let mut blocks_over = 0usize;
    for r in g..col_h[h1c].min(24) {
        if grid[r][h1c] {
            blocks_over += 1;
        }
    }
    let left_h = if h1c > 0 { col_h[h1c - 1].saturating_sub(g) } else { 99 };
    let right_h = if h1c + 1 < 10 { col_h[h1c + 1].saturating_sub(g) } else { 99 };
    let chute_min = left_h.min(right_h);
    let mut cells_ab = 0usize;
    for r in g..24 {
        for c in 0..10 {
            if grid[r][c] {
                cells_ab += 1;
            }
        }
    }
    (Some(h1c), blocks_over, chute_min, cells_ab)
}

#[derive(Clone, Copy)]
struct RNode {
    board: FastBoard,
    held: Option<u8>,
    qc: u8,
    rem_g: u16,
    first_idx: u16,
    score: i64,
}

#[derive(Hash, Eq, PartialEq)]
struct DKey {
    board: FastBoard,
    held: Option<u8>,
    qc: u8,
    rem_g: u16,
}

struct RolloutBuffers {
    pgen: FastPlaceGen,
    root_places: Vec<FastPlace>,
    tmp_places: Vec<FastPlace>,
    cur: Vec<RNode>,
    next: Vec<RNode>,
    seen: HashSet<DKey>,
}

impl RolloutBuffers {
    fn new() -> Self {
        Self {
            pgen: FastPlaceGen::new(),
            root_places: Vec::with_capacity(96),
            tmp_places: Vec::with_capacity(96),
            cur: Vec::with_capacity(128),
            next: Vec::with_capacity(1024),
            seen: HashSet::with_capacity(1024),
        }
    }
}

/// Fast 100% Legal p=5 beam search (uses ONLY `vis_q` <= 6 pieces and `bag_mask`,
/// NEVER peeks at future world holes or pieces beyond p=5!).
/// Returns the chosen legal `FastPlace` for the current state, or `None` if topped out.
fn legal_p5_beam_move(
    fst: &FastShapeTable,
    sm: &SpatialValueModel,
    buf: &mut RolloutBuffers,
    root_board: &FastBoard,
    hold0: Option<u8>,
    vis_q: &[u8],
    rem_g0: u16,
    bag_mask: u8,
    beam_w: usize,
    use_mlp_internal: bool,
    sim_rng: &mut u64,
) -> Option<FastPlace> {
    let q_len = vis_q.len().min(6);
    if q_len == 0 {
        return None;
    }
    let cur0 = Some(vis_q[0]);
    let (c_nh0, c_h0) = if hold0.is_some() {
        (cur0, hold0)
    } else {
        (cur0, vis_q.get(1).copied())
    };
    buf.pgen
        .generate_for_state(fst, root_board, c_nh0, c_h0, &mut buf.root_places);
    if buf.root_places.is_empty() {
        return None;
    }

    let max_depth = if hold0.is_some() { q_len } else { q_len.saturating_sub(1) }.clamp(1, 5);
    buf.cur.clear();
    buf.seen.clear();

    let mut best_goal: Option<(usize, u16, i64)> = None;
    let spawn_seed_base = next_rng(sim_rng);

    for (idx, &fp) in buf.root_places.iter().enumerate() {
        let mut b = *root_board;
        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
        let rem_g = rem_g0.saturating_sub(ds as u16);
        if rem_g == 0 && b.rows < 20 {
            let g_sc = -30_000_000 + 100 + (b.rows as i64) * 5;
            if best_goal.map_or(true, |(_, _, r)| g_sc < r) {
                best_goal = Some((1, idx as u16, g_sc));
            }
        }
        if rem_g > 0 {
            let mut r_loc = spawn_seed_base;
            if !b.spawn_cheese_rows(rem_g, lines, &mut r_loc) {
                continue;
            }
        }
        let (held, qc) = if hold0.is_some() {
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
        let ih = hand_piece(held, qc, &vis_q[..q_len]);
        let av = avail_mask_from_q(held, qc, &vis_q[..q_len], bag_mask);
        let sc = if use_mlp_internal || max_depth == 1 {
            sm.evaluate_board_with_bag(&b, rem_g, ih, av)
        } else {
            sm.evaluate_linear_board_with_bag_no_pe(&b, rem_g, ih, av)
        };
        let key = DKey { board: b, held, qc: qc as u8, rem_g };
        if buf.seen.insert(key) {
            buf.cur.push(RNode {
                board: b,
                held,
                qc: qc as u8,
                rem_g,
                first_idx: idx as u16,
                score: sc,
            });
        }
    }

    if let Some((_, fidx, _)) = best_goal {
        return Some(buf.root_places[fidx as usize]);
    }
    if buf.cur.is_empty() {
        return Some(buf.root_places[0]);
    }

    let mpr = (beam_w / 3).max(3);
    if buf.cur.len() > beam_w {
        buf.cur.sort_unstable_by_key(|n| n.score);
        buf.cur.truncate(beam_w);
    }

    for depth in 2..=max_depth {
        buf.next.clear();
        let depth_seed = spawn_seed_base.wrapping_add((depth as u64).wrapping_mul(0x517CC1B727220A95));
        for ni in 0..buf.cur.len() {
            let node = buf.cur[ni];
            if node.rem_g == 0 {
                continue;
            }
            let qc = node.qc as usize;
            let (c_nh, c_h) = if node.held.is_some() {
                (if qc < q_len { Some(vis_q[qc]) } else { None }, node.held)
            } else {
                (
                    if qc < q_len { Some(vis_q[qc]) } else { None },
                    if qc + 1 < q_len { Some(vis_q[qc + 1]) } else { None },
                )
            };
            buf.pgen
                .generate_for_state(fst, &node.board, c_nh, c_h, &mut buf.tmp_places);
            for &fp in &buf.tmp_places {
                let mut b = node.board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = node.rem_g.saturating_sub(ds as u16);
                if rem_g == 0 && b.rows < 20 {
                    let g_sc = -30_000_000 + (depth as i64) * 100 + (b.rows as i64) * 5;
                    if best_goal.map_or(true, |(_, _, r)| g_sc < r) {
                        best_goal = Some((depth, node.first_idx, g_sc));
                    }
                }
                if rem_g > 0 {
                    let mut r_loc = depth_seed;
                    if !b.spawn_cheese_rows(rem_g, lines, &mut r_loc) {
                        continue;
                    }
                }
                let (held, nqc) = if node.held.is_some() {
                    if fp.did_hold {
                        (c_nh, if c_nh.is_some() { qc + 1 } else { qc })
                    } else {
                        (node.held, qc + 1)
                    }
                } else if fp.did_hold {
                    (c_nh, qc + 2)
                } else {
                    (None, qc + 1)
                };
                let ih = hand_piece(held, nqc, &vis_q[..q_len]);
                let av = avail_mask_from_q(held, nqc, &vis_q[..q_len], bag_mask);
                let sc = if use_mlp_internal {
                    sm.evaluate_board_with_bag(&b, rem_g, ih, av)
                } else {
                    sm.evaluate_linear_board_with_bag_no_pe(&b, rem_g, ih, av)
                };
                buf.next.push(RNode {
                    board: b,
                    held,
                    qc: nqc as u8,
                    rem_g,
                    first_idx: node.first_idx,
                    score: sc,
                });
            }
        }
        if let Some((_, fidx, _)) = best_goal {
            return Some(buf.root_places[fidx as usize]);
        }
        if buf.next.is_empty() {
            break;
        }
        buf.next.sort_unstable_by_key(|n| n.score);
        buf.cur.clear();
        buf.seen.clear();
        let mut per_root = [0usize; 96];
        for &n in &buf.next {
            if buf.cur.len() >= beam_w {
                break;
            }
            let ridx = n.first_idx as usize;
            if ridx < 96 && per_root[ridx] >= mpr {
                continue;
            }
            let key = DKey {
                board: n.board,
                held: n.held,
                qc: n.qc,
                rem_g: n.rem_g,
            };
            if buf.seen.insert(key) {
                if ridx < 96 {
                    per_root[ridx] += 1;
                }
                buf.cur.push(n);
            }
        }
    }

    // Pick best leaf's first_idx
    buf.cur
        .iter()
        .min_by_key(|n| n.score)
        .map(|n| buf.root_places[n.first_idx as usize])
}

/// Stage-1 Legal p=5 Full-MLP beam to shortlist top-K diverse root candidates
fn shortlist_root_candidates(
    fst: &FastShapeTable,
    sm: &SpatialValueModel,
    buf: &mut RolloutBuffers,
    root_board: &FastBoard,
    hold0: Option<u8>,
    vis_q: &[u8],
    rem_g0: u16,
    bag_mask: u8,
    top_k: usize,
    sim_rng: &mut u64,
) -> Vec<(FastPlace, i64)> {
    let q_len = vis_q.len().min(6);
    let cur0 = Some(vis_q[0]);
    let (c_nh0, c_h0) = if hold0.is_some() {
        (cur0, hold0)
    } else {
        (cur0, vis_q.get(1).copied())
    };
    buf.pgen
        .generate_for_state(fst, root_board, c_nh0, c_h0, &mut buf.root_places);
    if buf.root_places.is_empty() {
        return Vec::new();
    }
    let n_roots = buf.root_places.len();
    let max_depth = if hold0.is_some() { q_len } else { q_len.saturating_sub(1) }.clamp(1, 5);
    let beam_w = 80usize;
    let mpr = 16usize;

    buf.cur.clear();
    buf.seen.clear();
    let mut root_best = vec![i64::MAX; n_roots];
    let spawn_seed_base = next_rng(sim_rng);

    for (idx, &fp) in buf.root_places.iter().enumerate() {
        let mut b = *root_board;
        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
        let rem_g = rem_g0.saturating_sub(ds as u16);
        if rem_g == 0 && b.rows < 20 {
            let g_sc = -30_000_000 + 100 + (b.rows as i64) * 5;
            return vec![(fp, g_sc)];
        }
        if rem_g > 0 {
            let mut r_loc = spawn_seed_base;
            if !b.spawn_cheese_rows(rem_g, lines, &mut r_loc) {
                continue;
            }
        }
        let (held, qc) = if hold0.is_some() {
            if fp.did_hold { (cur0, 1usize) } else { (hold0, 1usize) }
        } else if fp.did_hold {
            (cur0, 2usize)
        } else {
            (None, 1usize)
        };
        let ih = hand_piece(held, qc, &vis_q[..q_len]);
        let av = avail_mask_from_q(held, qc, &vis_q[..q_len], bag_mask);
        let sc = if max_depth == 1 {
            sm.evaluate_board_with_bag(&b, rem_g, ih, av)
        } else {
            sm.evaluate_linear_board_with_bag_no_pe(&b, rem_g, ih, av)
        };
        root_best[idx] = root_best[idx].min(sc);
        let key = DKey { board: b, held, qc: qc as u8, rem_g };
        if buf.seen.insert(key) {
            buf.cur.push(RNode {
                board: b,
                held,
                qc: qc as u8,
                rem_g,
                first_idx: idx as u16,
                score: sc,
            });
        }
    }

    if buf.cur.len() > beam_w {
        buf.cur.sort_unstable_by_key(|n| n.score);
        buf.cur.truncate(beam_w);
    }

    for depth in 2..=max_depth {
        buf.next.clear();
        let depth_seed = spawn_seed_base.wrapping_add((depth as u64).wrapping_mul(0x517CC1B727220A95));
        for ni in 0..buf.cur.len() {
            let node = buf.cur[ni];
            if node.rem_g == 0 {
                continue;
            }
            let qc = node.qc as usize;
            let (c_nh, c_h) = if node.held.is_some() {
                (if qc < q_len { Some(vis_q[qc]) } else { None }, node.held)
            } else {
                (
                    if qc < q_len { Some(vis_q[qc]) } else { None },
                    if qc + 1 < q_len { Some(vis_q[qc + 1]) } else { None },
                )
            };
            buf.pgen
                .generate_for_state(fst, &node.board, c_nh, c_h, &mut buf.tmp_places);
            for &fp in &buf.tmp_places {
                let mut b = node.board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = node.rem_g.saturating_sub(ds as u16);
                if rem_g == 0 && b.rows < 20 {
                    let g_sc = -30_000_000 + (depth as i64) * 100 + (b.rows as i64) * 5;
                    root_best[node.first_idx as usize] = root_best[node.first_idx as usize].min(g_sc);
                }
                if rem_g > 0 {
                    let mut r_loc = depth_seed;
                    if !b.spawn_cheese_rows(rem_g, lines, &mut r_loc) {
                        continue;
                    }
                }
                let (held, nqc) = if node.held.is_some() {
                    if fp.did_hold {
                        (c_nh, if c_nh.is_some() { qc + 1 } else { qc })
                    } else {
                        (node.held, qc + 1)
                    }
                } else if fp.did_hold {
                    (c_nh, qc + 2)
                } else {
                    (None, qc + 1)
                };
                let ih = hand_piece(held, nqc, &vis_q[..q_len]);
                let av = avail_mask_from_q(held, nqc, &vis_q[..q_len], bag_mask);
                let sc = sm.evaluate_linear_board_with_bag_no_pe(&b, rem_g, ih, av);
                buf.next.push(RNode {
                    board: b,
                    held,
                    qc: nqc as u8,
                    rem_g,
                    first_idx: node.first_idx,
                    score: sc,
                });
            }
        }
        if buf.next.is_empty() {
            break;
        }
        buf.next.sort_unstable_by_key(|n| n.score);
        if depth == max_depth {
            root_best.fill(i64::MAX);
            for &n in &buf.next {
                let ri = n.first_idx as usize;
                if n.score < root_best[ri] {
                    root_best[ri] = n.score;
                }
            }
            break;
        }
        buf.cur.clear();
        buf.seen.clear();
        let mut per_root = vec![0usize; n_roots];
        for &n in &buf.next {
            if buf.cur.len() >= beam_w {
                break;
            }
            let ridx = n.first_idx as usize;
            if per_root[ridx] >= mpr {
                continue;
            }
            let key = DKey {
                board: n.board,
                held: n.held,
                qc: n.qc,
                rem_g: n.rem_g,
            };
            if buf.seen.insert(key) {
                per_root[ridx] += 1;
                buf.cur.push(n);
            }
        }
    }

    let mut ranked: Vec<(FastPlace, i64)> = buf
        .root_places
        .iter()
        .enumerate()
        .filter(|&(i, _)| root_best[i] < i64::MAX)
        .map(|(i, &fp)| (fp, root_best[i]))
        .collect();
    ranked.sort_unstable_by_key(|x| x.1);
    ranked.truncate(top_k);
    ranked
}

#[derive(Clone, Copy)]
struct NcConfig {
    top_k: usize,
    worlds: usize,
    horizon: usize,
    rollout_beam: usize,
    use_mlp_internal: bool,
    cvar_pct: i64,
}

struct GameStats {
    seed: u64,
    pieces: usize,
    topout: bool,
    nds: usize,
    zds: usize,
    zds_h1_cov: usize,
    zds_open_o: usize,
    zds_open_nar: usize,
    zds_open_wide: usize,
    zds_end: usize,
    ds1: usize,
    ds_cov_next: usize,
}

fn play_nc_rollout_seed(seed: u64, sm: Arc<SpatialValueModel>, nc: NcConfig) -> GameStats {
    let fst = FastShapeTable::srs();
    let mut endgame_cfg = BeamConfig::default();
    endgame_cfg.spatial_model = Some(sm.clone());
    endgame_cfg.beam_width = 120;
    endgame_cfg.max_per_root = 24;
    endgame_cfg.rollout_top_k = 0;
    endgame_cfg.bag_det_extend = true;
    endgame_cfg.endgame_solver = true;
    endgame_cfg.simulate_cheese_spawn = true;
    endgame_cfg.spatial_rerank_only = true;
    endgame_cfg.spatial_no_pe_internal = true;
    let mut ai = FastAI::new(endgame_cfg);
    let mut tracker = BagTracker::new();

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);

    let mut cur_bag_mask = if let Some(snap0) = stacker.snapshot() {
        tracker.observe(snap0.hold, &snap0.queue)
    } else {
        0x7F
    };

    let mut stage1_buf = RolloutBuffers::new();
    let mut roll_buf = RolloutBuffers::new();
    let mut st = GameStats {
        seed,
        pieces: 0,
        topout: false,
        nds: 0,
        zds: 0,
        zds_h1_cov: 0,
        zds_open_o: 0,
        zds_open_nar: 0,
        zds_open_wide: 0,
        zds_end: 0,
        ds1: 0,
        ds_cov_next: 0,
    };

    let mut ds_total = 0usize;
    let mut step_idx = 0u64;

    while ds_total < 100 && st.pieces < 340 && stacker.current_piece_type().is_some() {
        let snap = match stacker.snapshot() {
            Some(s) => s,
            None => break,
        };
        assert!(snap.queue.len() <= 6, "illegal preview length");
        let g = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let (_, blocks_over, chute_min, _) = h1_info(&stacker);
        let cur_c = snap.queue[0].as_char();
        let hold_c = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|c| c.as_char());

        // Use exact endgame solver when all remaining garbage is visible (rem_g <= 9 && rem_g == g)
        let chosen_inputs = if rem_g <= 9 && rem_g == g {
            match ai.best_move(&snap, Some(g), Some(rem_g)) {
                Some(r) => r.inputs,
                None => break,
            }
        } else {
            let root_board = FastBoard::from_basic_matrix(&snap.matrix, g);
            let vis_q: Vec<u8> = snap.queue.iter().map(|&c| color_to_idx(c) as u8).collect();
            let hold0: Option<u8> = snap.hold.map(|c| color_to_idx(c) as u8);
            let mut crn_rng: u64 = 0x9E3779B97F4A7C15
                ^ ((rem_g as u64) << 32)
                ^ ((cur_bag_mask as u64) << 16)
                ^ (hold0.unwrap_or(7) as u64)
                ^ step_idx.wrapping_mul(0xBF58476D1CE4E5B9);
            for r in 0..(root_board.rows as usize).min(20) {
                crn_rng = crn_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(root_board.data[r] as u64);
            }
            for &p in &vis_q {
                crn_rng = crn_rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(p as u64 + 1);
            }

            let cands = shortlist_root_candidates(
                &fst,
                &sm,
                &mut stage1_buf,
                &root_board,
                hold0,
                &vis_q,
                rem_g,
                cur_bag_mask,
                nc.top_k,
                &mut crn_rng,
            );
            if cands.is_empty() {
                break;
            }

            let best_fp = if cands.len() == 1 || cands[0].1 <= -10_000_000 {
                cands[0].0
            } else {
                // Sample W CRN future worlds (7-bag piece sequences + Jstris consecutive holes)
                let need_q = vis_q.len() + nc.horizon + 8;
                let need_h = nc.horizon * 2 + 12;
                let unspawned_0 = (rem_g as usize).saturating_sub(g as usize);
                let bottom_hole: Option<u16> = if g > 0 {
                    let h = (!root_board.data[0] & 0x3FF).trailing_zeros() as u16;
                    if h < 10 { Some(h) } else { None }
                } else {
                    None
                };

                let mut world_qs: Vec<Vec<u8>> = Vec::with_capacity(nc.worlds);
                let mut world_bms: Vec<Vec<u8>> = Vec::with_capacity(nc.worlds);
                let mut world_holes: Vec<Vec<u16>> = Vec::with_capacity(nc.worlds);

                for w in 0..nc.worlds {
                    let mut wrng = crn_rng ^ ((w as u64) + 1).wrapping_mul(0x94D049BB133111EB);
                    let mut wq = Vec::with_capacity(need_q);
                    let mut wbm = Vec::with_capacity(need_q);
                    wq.extend_from_slice(&vis_q);
                    // Track bag_mask AFTER each index in wq (so bag_mask after 6 visible pieces at qc is wbm[qc + 5])
                    let mut bm = cur_bag_mask;
                    for _ in 0..vis_q.len() {
                        wbm.push(bm);
                    }
                    while wq.len() < need_q {
                        let p = draw_from_bag(&mut bm, &mut wrng);
                        wq.push(p);
                        wbm.push(bm);
                    }
                    let mut wh = Vec::with_capacity(need_h);
                    let mut prev_h = bottom_hole;
                    for _ in 0..need_h {
                        let h = match prev_h {
                            Some(ph) => {
                                let off = 1 + (((next_rng(&mut wrng) >> 32) as u16) % 9);
                                (ph + off) % 10
                            }
                            None => ((next_rng(&mut wrng) >> 32) as u16) % 10,
                        };
                        wh.push(h);
                        prev_h = Some(h);
                    }
                    world_qs.push(wq);
                    world_bms.push(wbm);
                    world_holes.push(wh);
                }

                // Evaluate each candidate root move via Zero-Strategy-Fusion Legal p=5 rollouts!
                let mut best_cand_fp = cands[0].0;
                let mut best_cand_sc = i64::MAX;
                let mut w_scores = vec![0i64; nc.worlds];

                for &(fp0, _) in &cands {
                    let mut sum_w = 0i64;
                    for w in 0..nc.worlds {
                        let mut b = root_board;
                        let o0 = &fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
                        let (lines0, ds0, _) = b.place_and_sift(o0, fp0.norm_i, fp0.norm_j);
                        let mut rem = rem_g.saturating_sub(ds0 as u16);
                        let mut held = if hold0.is_some() {
                            if fp0.did_hold { Some(vis_q[0]) } else { hold0 }
                        } else if fp0.did_hold {
                            Some(vis_q[0])
                        } else {
                            None
                        };
                        let mut qc = if hold0.is_some() {
                            1usize
                        } else if fp0.did_hold {
                            2usize
                        } else {
                            1usize
                        };

                        let w_sc = if rem == 0 && b.rows < 20 {
                            100 + (b.rows as i64) * 5
                        } else if !spawn_cheese_from_stream(
                            &mut b,
                            rem,
                            lines0,
                            unspawned_0,
                            &world_holes[w],
                        ) {
                            500_000 + (rem as i64) * 500
                        } else {
                            let mut roll_rng = crn_rng
                                ^ ((w as u64) + 101).wrapping_mul(0xD6E8FEB86659FD93);
                            let mut done_sc: Option<i64> = None;
                            let mut steps_done = 1usize;

                            for _step in 1..nc.horizon {
                                let wq_end = (qc + 6).min(world_qs[w].len());
                                let vis_slice = &world_qs[w][qc..wq_end];
                                let bm_after_vis = world_bms[w][wq_end.saturating_sub(1)];

                                let fp_step = match legal_p5_beam_move(
                                    &fst,
                                    &sm,
                                    &mut roll_buf,
                                    &b,
                                    held,
                                    vis_slice,
                                    rem,
                                    bm_after_vis,
                                    nc.rollout_beam,
                                    nc.use_mlp_internal,
                                    &mut roll_rng,
                                ) {
                                    Some(f) => f,
                                    None => {
                                        done_sc = Some(500_000 + (rem as i64) * 500);
                                        break;
                                    }
                                };

                                let os = &fst.shapes[fp_step.color_idx as usize].orients
                                    [fp_step.tf.2 as usize];
                                let (ls, dss, _) =
                                    b.place_and_sift(os, fp_step.norm_i, fp_step.norm_j);
                                rem = rem.saturating_sub(dss as u16);
                                steps_done += 1;

                                let c_cur = vis_slice.first().copied();
                                let (new_held, d_qc) = if held.is_some() {
                                    if fp_step.did_hold {
                                        (c_cur, 1usize)
                                    } else {
                                        (held, 1usize)
                                    }
                                } else if fp_step.did_hold {
                                    (c_cur, 2usize)
                                } else {
                                    (None, 1usize)
                                };
                                held = new_held;
                                qc += d_qc;

                                if rem == 0 && b.rows < 20 {
                                    done_sc = Some((steps_done as i64) * 100 + (b.rows as i64) * 5);
                                    break;
                                }
                                if !spawn_cheese_from_stream(
                                    &mut b,
                                    rem,
                                    ls,
                                    unspawned_0,
                                    &world_holes[w],
                                ) {
                                    done_sc = Some(500_000 + (rem as i64) * 500);
                                    break;
                                }
                            }

                            match done_sc {
                                Some(s) => s,
                                None => {
                                    let wq_end = (qc + 6).min(world_qs[w].len());
                                    let vis_slice = &world_qs[w][qc..wq_end];
                                    let bm_after_vis = world_bms[w][wq_end.saturating_sub(1)];
                                    let ih = hand_piece(held, 0, vis_slice);
                                    let av = avail_mask_from_q(held, 0, vis_slice, bm_after_vis);
                                    (steps_done as i64) * 100
                                        + sm.evaluate_board_with_bag(&b, rem, ih, av)
                                }
                            }
                        };

                        w_scores[w] = w_sc;
                        sum_w += w_sc;
                    }

                    let mean_sc = sum_w / (nc.worlds as i64);
                    let agg_sc = if nc.cvar_pct > 0 && nc.worlds >= 4 {
                        let tail_k = (nc.worlds + 3) / 4;
                        w_scores.sort_unstable_by(|a, b| b.cmp(a));
                        let tail_sc = w_scores[..tail_k].iter().sum::<i64>() / (tail_k as i64);
                        ((100 - nc.cvar_pct) * mean_sc + nc.cvar_pct * tail_sc) / 100
                    } else {
                        mean_sc
                    };

                    if agg_sc < best_cand_sc {
                        best_cand_sc = agg_sc;
                        best_cand_fp = fp0;
                    }
                }
                best_cand_fp
            };

            match ai.inputs_for_place(&snap.matrix, best_fp) {
                Some(inps) => inps,
                None => match ai.best_move(&snap, Some(g), Some(rem_g)) {
                    Some(r) => r.inputs,
                    None => break,
                },
            }
        };

        let mut inputs = chosen_inputs;
        let used_hold = inputs.contains(&Input::Hold);
        let placed_c = if used_hold {
            hold_c.unwrap_or(cur_c)
        } else {
            cur_c
        };
        if inputs.last() == Some(&Input::HD) {
            inputs.pop();
        }
        stacker.run(inputs);
        let (lines, ds) = stacker.hard_drop();
        st.pieces += 1;
        step_idx += 1;
        ds_total += ds;
        st.nds += lines.saturating_sub(ds);
        if std::env::var_os("VERBOSE").is_some() && st.pieces % 40 == 0 {
            eprintln!("  [seed {}] pcs={} ds={}", seed, st.pieces, ds_total);
        }

        if ds > 0 {
            st.ds1 += 1;
            let (_, new_bo, _, _) = h1_info(&stacker);
            if new_bo > 0 {
                st.ds_cov_next += 1;
            }
        } else {
            st.zds += 1;
            if rem_g <= 7 && rem_g == g {
                st.zds_end += 1;
            } else if blocks_over > 0 {
                st.zds_h1_cov += 1;
            } else {
                let is_o = placed_c == 'O';
                let has_i = cur_c == 'I' || hold_c == Some('I');
                if is_o {
                    st.zds_open_o += 1;
                } else if chute_min >= 1 && !has_i {
                    st.zds_open_nar += 1;
                } else {
                    st.zds_open_wide += 1;
                }
            }
        }

        if let Some(snap_a) = stacker.snapshot() {
            cur_bag_mask = tracker.observe(snap_a.hold, &snap_a.queue);
        }
    }

    st.topout = ds_total < 100;
    st
}

fn eval_nc(label: &str, seeds: &[u64], sm: Arc<SpatialValueModel>, nc: NcConfig) {
    let t0 = Instant::now();
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let next_idx = AtomicUsize::new(0);
    let mut results: Vec<Option<GameStats>> = (0..seeds.len()).map(|_| None).collect();

    std::thread::scope(|sc| {
        let mut hs = Vec::new();
        let next_ref = &next_idx;
        for _ in 0..workers {
            let sm_c = sm.clone();
            hs.push(sc.spawn(move || {
                let mut local = Vec::new();
                loop {
                    let i = next_ref.fetch_add(1, Ordering::Relaxed);
                    if i >= seeds.len() {
                        break;
                    }
                    let st = play_nc_rollout_seed(seeds[i], sm_c.clone(), nc);
                    println!(
                        "  [seed {:3}] pcs={:3} nds={:2} 0ds={:2} (cov={:2} O={:2} nar={:2} wide={:2} end={:2}) cov_nxt={:4.1}% | {:.1}s",
                        st.seed,
                        if st.topout { 999 } else { st.pieces },
                        st.nds,
                        st.zds,
                        st.zds_h1_cov,
                        st.zds_open_o,
                        st.zds_open_nar,
                        st.zds_open_wide,
                        st.zds_end,
                        100.0 * st.ds_cov_next as f64 / st.ds1.max(1) as f64,
                        t0.elapsed().as_secs_f64()
                    );
                    let _ = std::io::stdout().flush();
                    local.push((i, st));
                }
                local
            }));
        }
        for h in hs {
            for (i, st) in h.join().unwrap() {
                results[i] = Some(st);
            }
        }
    });

    let stats: Vec<GameStats> = results.into_iter().map(|x| x.unwrap()).collect();
    let dt = t0.elapsed().as_secs_f64();
    let tops = stats.iter().filter(|s| s.topout).count();
    let ok: Vec<&GameStats> = stats.iter().filter(|s| !s.topout).collect();
    let n_ok = ok.len().max(1) as f64;
    let mean = if ok.is_empty() {
        999.0
    } else {
        ok.iter().map(|s| s.pieces as f64).sum::<f64>() / n_ok
    };
    let min_p = ok.iter().map(|s| s.pieces).min().unwrap_or(999);
    let max_p = ok.iter().map(|s| s.pieces).max().unwrap_or(999);
    let le166 = ok.iter().filter(|s| s.pieces <= 166).count();
    let le161 = ok.iter().filter(|s| s.pieces <= 161).count();
    let m_cov = ok.iter().map(|s| s.zds_h1_cov as f64).sum::<f64>() / n_ok;
    let m_o = ok.iter().map(|s| s.zds_open_o as f64).sum::<f64>() / n_ok;
    let m_nar = ok.iter().map(|s| s.zds_open_nar as f64).sum::<f64>() / n_ok;
    let m_wide = ok.iter().map(|s| s.zds_open_wide as f64).sum::<f64>() / n_ok;
    let m_end = ok.iter().map(|s| s.zds_end as f64).sum::<f64>() / n_ok;
    let m_cnxt = ok
        .iter()
        .map(|s| 100.0 * s.ds_cov_next as f64 / s.ds1.max(1) as f64)
        .sum::<f64>()
        / n_ok;

    println!(
        "{:<28} n={:2} mean={:6.2} min={:3} max={:3} <=166:{:2} <=161:{:2} top={:2} | cov={:.1} O={:.1} nar={:.1} wide={:.1} end={:.1} cov_nxt={:.1}% | {:.1}s",
        label,
        stats.len(),
        mean,
        min_p,
        max_p,
        le166,
        le161,
        tops,
        m_cov,
        m_o,
        m_nar,
        m_wide,
        m_end,
        m_cnxt,
        dt
    );
    let seed_str: Vec<String> = stats
        .iter()
        .map(|s| format!("{}:{}", s.seed, if s.topout { 999 } else { s.pieces }))
        .collect();
    println!("  seeds: [{}]", seed_str.join(", "));
    let _ = std::io::stdout().flush();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = "models/spatial_bc4.txt".to_string();
    let mut n_seeds: u64 = 5;
    let mut seed_start: u64 = 101;
    let mut top_k: usize = 8;
    let mut worlds: usize = 8;
    let mut horizon: usize = 10;
    let mut rollout_beam: usize = 12;
    let mut use_mlp_internal = false;
    let mut cvar_pct: i64 = 25;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args[i + 1].clone(); i += 2; }
            "--n-seeds" => { n_seeds = args[i + 1].parse().unwrap(); i += 2; }
            "--seed-start" => { seed_start = args[i + 1].parse().unwrap(); i += 2; }
            "--top-k" => { top_k = args[i + 1].parse().unwrap(); i += 2; }
            "--worlds" => { worlds = args[i + 1].parse().unwrap(); i += 2; }
            "--horizon" => { horizon = args[i + 1].parse().unwrap(); i += 2; }
            "--rollout-beam" => { rollout_beam = args[i + 1].parse().unwrap(); i += 2; }
            "--mlp-internal" => { use_mlp_internal = true; i += 1; }
            "--cvar" => { cvar_pct = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("Unknown arg: {}", other); std::process::exit(1); }
        }
    }

    let mut sm_val = SpatialValueModel::load_from_file(&model_path).expect("load spatial model");
    sm_val.prepare_fast();
    let sm = Arc::new(sm_val);

    let seeds: Vec<u64> = (seed_start..(seed_start + n_seeds)).collect();
    let nc = NcConfig {
        top_k,
        worlds,
        horizon,
        rollout_beam,
        use_mlp_internal,
        cvar_pct,
    };
    let label = format!(
        "nc_k{}_w{}_h{}_b{}{}_c{}",
        top_k,
        worlds,
        horizon,
        rollout_beam,
        if use_mlp_internal { "m" } else { "l" },
        cvar_pct
    );
    eval_nc(&label, &seeds, sm, nc);
}
