use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        beam::{BagTracker, FxHasher},
        cheese_val::{FEAT_DIM, MLP_HIDDEN},
        fast::{color_to_idx, FastBoard, FastPlaceGen, FastShapeTable},
        spatial_val::{extract_spatial_features, DENSE_DIM, SPARSE_DIM, SPATIAL_H1, SPATIAL_H2},
        BeamConfig, FastAI, LearnedValueModel, SpatialValueModel,
    },
    StackerExt as _,
};
use std::collections::HashSet;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::{BufRead, BufReader};
use std::sync::Arc;
use std::time::Instant;

type FxBuild = BuildHasherDefault<FxHasher>;

fn load_learned_model(path: &str, nog0: bool, no_o_bias: bool) -> LearnedValueModel {
    let f = File::open(path).expect("open learned model");
    let mut lines = BufReader::new(f).lines().map(|l| l.unwrap());
    let mut m = LearnedValueModel::default();
    let l1: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    m.linear_bias = l1[0];
    m.mlp_weight = l1[1];
    let l2: Vec<f32> = lines
        .next()
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    let n_f = (l2.len() / 3).min(FEAT_DIM);
    for j in 0..n_f {
        m.mean[j] = l2[j * 3];
        m.inv_std[j] = l2[j * 3 + 1];
        m.linear_w[j] = l2[j * 3 + 2];
    }
    m.b2 = lines.next().unwrap().trim().parse().unwrap();
    for h in 0..MLP_HIDDEN {
        let lh: Vec<f32> = lines
            .next()
            .unwrap()
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        m.b1[h] = lh[0];
        m.w2[h] = lh[1];
        let n_w = lh.len().saturating_sub(2).min(FEAT_DIM);
        for j in 0..n_w {
            m.w1[h][j] = lh[2 + j];
        }
    }
    if nog0 {
        m.linear_w[2] = 0.0;
    }
    if no_o_bias {
        m.linear_w[63] = 0.0;
    }
    m.prepare_fast();
    m
}

#[derive(Clone)]
enum Teacher {
    Learned(Arc<LearnedValueModel>),
    Spatial(Arc<SpatialValueModel>),
}

impl Teacher {
    #[inline]
    fn eval(&self, b: &FastBoard, rem_g: u16, in_hand: Option<u8>, avail_mask: u8) -> i64 {
        match self {
            Teacher::Learned(m) => m.evaluate_board_with_bag(b, rem_g, in_hand, avail_mask),
            Teacher::Spatial(m) => m.evaluate_board_with_bag(b, rem_g, in_hand, avail_mask),
        }
    }
}

#[derive(Clone)]
struct LeafEval {
    v_cal: Arc<SpatialValueModel>,
    teacher: Teacher,
    v_cal_w: f32,
    t_scale: f32,
    t_shift: f32,
}

impl LeafEval {
    #[inline]
    fn eval_pieces(&self, b: &FastBoard, rem_g: u16, in_hand: Option<u8>, avail_mask: u8) -> f32 {
        let vc = (self.v_cal.evaluate_board_with_bag(b, rem_g, in_hand, avail_mask) as f32) * 0.01;
        let tw = 1.0 - self.v_cal_w;
        if tw <= 1e-4 {
            return vc.max(0.5);
        }
        let ts = (self.teacher.eval(b, rem_g, in_hand, avail_mask) as f32) * 0.01;
        let tc = self.t_scale * ts + self.t_shift;
        (self.v_cal_w * vc + tw * tc).max(0.5)
    }
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

fn extended_queue(stacker: &Stacker, len: usize) -> Vec<blockfish::Color> {
    use std::convert::TryInto;
    let mut sim = stacker.clone();
    let mut out = Vec::with_capacity(len);
    for &c in sim.next() {
        out.push(c.try_into().unwrap());
    }
    let mut ai_cfg = BeamConfig::default();
    ai_cfg.beam_width = 24;
    ai_cfg.max_per_root = 6;
    ai_cfg.rollout_top_k = 0;
    let mut ai = FastAI::new(ai_cfg);
    while out.len() < len {
        let snap = match sim.snapshot() {
            Some(s) => s,
            None => break,
        };
        if let Some(res) = ai.best_move(&snap, Some(0), Some(1)) {
            let mut inputs = res.inputs;
            inputs.pop();
            let had_hold = sim.held().is_some();
            let did_hold = inputs.first() == Some(&blockfish::Input::Hold);
            sim.run(inputs);
            sim.hard_drop();
            let nx = sim.next();
            if did_hold && !had_hold && nx.len() >= 2 {
                out.push(nx[nx.len() - 2].try_into().unwrap());
                out.push(nx[nx.len() - 1].try_into().unwrap());
            } else if let Some(&last) = nx.last() {
                out.push(last.try_into().unwrap());
            } else {
                break;
            }
        } else {
            break;
        }
    }
    out
}

struct ExactSeedData {
    seed: u64,
    holes: Vec<u16>,
    pieces: Vec<u8>,
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
        let did_hold = res.inputs.first() == Some(&blockfish::Input::Hold);
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

#[derive(Clone)]
struct StateFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
    teacher_score: f32,
}

impl StateFeat {
    fn from_board(
        b: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail: u8,
        teacher: &Teacher,
    ) -> Self {
        let mut active_buf = [0u16; SPARSE_DIM];
        let mut dense = [0.0f32; DENSE_DIM];
        let n_act = extract_spatial_features(b, rem_g, in_hand, avail, &mut active_buf, &mut dense);
        let t_sc = (teacher.eval(b, rem_g, in_hand, avail) as f32) * 0.01;
        Self {
            active: active_buf[..n_act].to_vec(),
            dense,
            teacher_score: t_sc,
        }
    }
}

#[inline]
fn consume_bag_bit(mask: &mut u8, p: u8) {
    let bit = 1u8 << p;
    if (*mask & bit) == 0 {
        *mask = 0x7F;
    }
    *mask &= !bit;
    if *mask == 0 {
        *mask = 0x7F;
    }
}

/// Fast board structural cleanliness metrics `(stack_cells, covered_holes, h1_blocks)`
/// used to select diverse frontier beam leaves without full feature extraction.
#[inline]
fn fast_leaf_clean_metrics(b: &FastBoard) -> (i32, i32, i32) {
    let g = b.garbage_rows as usize;
    let n = b.rows as usize;
    let mut stack_cells = 0i32;
    let mut covered_holes = 0i32;
    let mut above_mask: u16 = 0;
    let mut i = n;
    while i > g {
        i -= 1;
        let bits = b.data[i] & 0x3FF;
        stack_cells += bits.count_ones() as i32;
        let rholes = (!bits & 0x3FF) & above_mask;
        covered_holes += rholes.count_ones() as i32;
        above_mask |= bits;
    }
    let mut h1_blocks = 0i32;
    if g >= 1 {
        let c1 = (!b.data[g - 1] & 0x3FF).trailing_zeros() as usize;
        if c1 < 10 {
            for r in g..n {
                if ((b.data[r] >> c1) & 1) != 0 {
                    h1_blocks += 1;
                }
            }
        }
    }
    (stack_cells, covered_holes, h1_blocks)
}

#[derive(Clone)]
struct ValueSample {
    state_idx: usize,
    target: f32,
    weight: f32,
    is_val: bool,
}

#[derive(Clone)]
struct RankPair {
    win_idx: usize,
    lose_idx: usize,
    dq: f32,
    is_trap: bool,
    is_flipped: bool,
    is_val: bool,
}

#[derive(Clone)]
struct CfCand {
    state_idx: usize,
    board: FastBoard,
    held: Option<u8>,
    qc: usize,
    rem_g: u16,
    ref_score: i64,
}

#[derive(Clone)]
struct CfTask {
    base_rng: u64,
    q_start: usize,
    q_win_len: usize,
    legal_q: [u8; 6],
    legal_q_len: usize,
    legal_bag: u8,
    bag_at_q_end: u8,
    spawned_so_far0: usize,
    prev_hole0: u16,
    sc_w_1ply: i64,
    root_cands: Vec<CfCand>,
    leaf_cands: Vec<CfCand>,
    max_spawned_leaf: usize,
    prev_hole_leaf: u16,
}

struct GameGenerated {
    ed: Option<ExactSeedData>,
    pieces: usize,
    ds_cleared: usize,
    topout: bool,
    states: Vec<StateFeat>,
    traj_recs: Vec<(usize, u16)>,
    cf_tasks: Vec<CfTask>,
}

#[inline]
fn avail_mask_from_q(held: Option<u8>, q_cursor: usize, q: &[u8], bag_mask: u8) -> u8 {
    let start = if held.is_some() { q_cursor } else { q_cursor + 1 };
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
fn draw_from_bag(mask: &mut u8, rng: &mut u64) -> u8 {
    if *mask == 0 {
        *mask = 0x7F;
    }
    let cnt = mask.count_ones();
    *rng = rng
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut pick = ((*rng >> 32) as u32) % cnt;
    let mut m = *mask;
    while m != 0 {
        let p = m.trailing_zeros() as u8;
        if pick == 0 {
            *mask &= !(1u8 << p);
            if *mask == 0 {
                *mask = 0x7F;
            }
            return p;
        }
        pick -= 1;
        m &= m - 1;
    }
    0
}

#[inline]
fn spawn_exact_cheese_ed(b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]) -> bool {
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
        let dst_r = num - 1 - k;
        b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
    }
    b.rows = (old_rows + num) as u8;
    b.garbage_rows = (g + num) as u8;
    true
}

/// Finite-horizon (or terminal when `roll_horizon == 0`) CRN rollout from a candidate state.
/// Returns calibrated expected remaining pieces after the candidate placement.
#[allow(clippy::too_many_arguments)]
fn rollout_crn_eval(
    fst: &FastShapeTable,
    pgen: &mut FastPlaceGen,
    tmp_places: &mut Vec<blockfish::ai::fast::FastPlace>,
    b_start: FastBoard,
    held_start: Option<u8>,
    qc_start: usize,
    rem_g_start: u16,
    q_full: &[u8],
    holes: &[u16],
    teacher: &Teacher,
    leaf_eval: &LeafEval,
    roll_horizon: usize,
    roll_bw: usize,
) -> f32 {
    if rem_g_start == 0 {
        let ch = b_start.col_heights();
        let max_h = *ch.iter().max().unwrap_or(&0);
        return if max_h < 20 { 0.0 } else { 25.0 };
    }

    #[derive(Clone, Copy)]
    struct CNode {
        board: FastBoard,
        held: Option<u8>,
        qc: usize,
        rem_g: u16,
        score: i64,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct CKey {
        board: FastBoard,
        held: Option<u8>,
        rem_g: u16,
    }

    let max_bw = roll_bw.max(24);
    let mut beam: Vec<CNode> = Vec::with_capacity(max_bw);
    let mut next_beam: Vec<CNode> = Vec::with_capacity(max_bw * 32);
    let mut seen: HashSet<CKey, FxBuild> =
        HashSet::with_capacity_and_hasher(max_bw * 8, FxBuild::default());

    let in_h0 = held_start.or_else(|| q_full.get(qc_start).copied());
    let av0 = avail_mask_from_q(held_start, qc_start, q_full, 0x7F);
    beam.push(CNode {
        board: b_start,
        held: held_start,
        qc: qc_start,
        rem_g: rem_g_start,
        score: teacher.eval(&b_start, rem_g_start, in_h0, av0),
    });

    let q_len = q_full.len();
    let max_steps = if roll_horizon > 0 {
        roll_horizon.min(q_len.saturating_sub(qc_start))
    } else {
        230usize.min(q_len.saturating_sub(qc_start))
    };
    let mut steps_done = 0usize;

    for step in 1..=max_steps {
        next_beam.clear();
        let mut any_expanded = false;
        let in_endgame = beam
            .first()
            .map_or(false, |n| n.rem_g <= 7 && n.rem_g == n.board.garbage_rows as u16);
        let eff_bw = if in_endgame { roll_bw.max(24) } else { roll_bw };

        for &node in &beam {
            if node.rem_g == 0 {
                continue;
            }
            let qc = node.qc;
            if qc >= q_len {
                continue;
            }
            let (c_nh, c_h) = if node.held.is_some() {
                (q_full.get(qc).copied(), node.held)
            } else {
                (q_full.get(qc).copied(), q_full.get(qc + 1).copied())
            };
            if c_nh.is_none() && c_h.is_none() {
                continue;
            }
            any_expanded = true;
            pgen.generate_for_state(fst, &node.board, c_nh, c_h, tmp_places);
            for &fp in tmp_places.iter() {
                let mut b = node.board;
                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                let rem_g = node.rem_g.saturating_sub(ds as u16);
                if rem_g > 0 && !spawn_exact_cheese_ed(&mut b, rem_g, lines, holes) {
                    continue;
                }
                let (held, nqc) = if node.held.is_some() {
                    if fp.did_hold {
                        (c_nh, (qc + 1).min(q_len))
                    } else {
                        (node.held, (qc + 1).min(q_len))
                    }
                } else if fp.did_hold {
                    (c_nh, (qc + 2).min(q_len))
                } else {
                    (None, (qc + 1).min(q_len))
                };
                if rem_g == 0 && b.rows < 20 {
                    return step as f32;
                }
                let in_h = held.or_else(|| q_full.get(nqc).copied());
                let av = avail_mask_from_q(held, nqc, q_full, 0x7F);
                let sc = teacher.eval(&b, rem_g, in_h, av);
                next_beam.push(CNode {
                    board: b,
                    held,
                    qc: nqc,
                    rem_g,
                    score: sc,
                });
            }
        }
        if !any_expanded || next_beam.is_empty() {
            if let Some(top) = beam.first() {
                let in_h = top.held.or_else(|| q_full.get(top.qc).copied());
                let av = avail_mask_from_q(top.held, top.qc, q_full, 0x7F);
                return (steps_done as f32) + 12.0 + leaf_eval.eval_pieces(&top.board, top.rem_g, in_h, av);
            }
            return (steps_done as f32) + 25.0 + 1.80 * (rem_g_start as f32);
        }
        steps_done = step;
        next_beam.sort_unstable_by_key(|n| n.score);
        beam.clear();
        seen.clear();
        for &ch in &next_beam {
            if beam.len() >= eff_bw {
                break;
            }
            let key = CKey {
                board: ch.board,
                held: ch.held,
                rem_g: ch.rem_g,
            };
            if seen.insert(key) {
                beam.push(ch);
            }
        }
    }

    if beam.is_empty() {
        return (steps_done as f32) + 25.0 + 1.80 * (rem_g_start as f32);
    }
    let mut best_leaf = f32::MAX;
    for node in beam.iter().take(3) {
        let in_h = node.held.or_else(|| q_full.get(node.qc).copied());
        let av = avail_mask_from_q(node.held, node.qc, q_full, 0x7F);
        let lv = leaf_eval.eval_pieces(&node.board, node.rem_g, in_h, av);
        if lv < best_leaf {
            best_leaf = lv;
        }
    }
    (steps_done as f32) + best_leaf
}

fn generate_game_data(
    seed: u64,
    horizon: usize,
    beam_w: usize,
    max_per_root: usize,
    cf_per_game: usize,
    teacher: &Teacher,
) -> GameGenerated {
    let ed = match collect_exact_seed(seed) {
        Some(e) => e,
        None => {
            return GameGenerated {
                ed: None,
                pieces: 400,
                ds_cleared: 0,
                topout: true,
                states: Vec::new(),
                traj_recs: Vec::new(),
                cf_tasks: Vec::new(),
            };
        }
    };
    let fst = FastShapeTable::srs();

    #[derive(Clone, Copy)]
    struct ONode {
        board: FastBoard,
        held: Option<u8>,
        q_cursor: usize,
        rem_g: u16,
        first_idx: u16,
        score: i64,
    }
    #[derive(Clone, Copy, Eq, PartialEq, Hash)]
    struct OKey {
        board: FastBoard,
        held: Option<u8>,
        rem_g: u16,
    }

    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);
    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut bag_tracker = BagTracker::new();
    let mut pgen = FastPlaceGen::new();
    let mut ai_in = FastAI::new(BeamConfig::default());

    let mut root_places = Vec::with_capacity(64);
    let mut tmp_places = Vec::with_capacity(64);
    let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
    let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
    let mut seen: HashSet<OKey, FxBuild> =
        HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

    let mut states: Vec<StateFeat> = Vec::with_capacity(600);
    let mut traj_recs: Vec<(usize, u16)> = Vec::with_capacity(220);
    let mut cf_tasks: Vec<CfTask> = Vec::with_capacity(cf_per_game + 4);

    let mut ds_total = 0usize;
    let mut q_pos = 0usize;
    let mut pcs = 0usize;
    let max_cf_rem_g: u16 = 99;
    let mut next_cf_rem_g: u16 = max_cf_rem_g;
    let cf_step_g: u16 = (92u16 / (cf_per_game.max(1) as u16)).max(2);
    let mut topout = false;

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        let legal_bag = bag_tracker.observe(snap.hold, &snap.queue);
        let legal_q_len = snap.queue.len().min(6);
        let mut legal_q = [0u8; 6];
        for (i, &c) in snap.queue.iter().take(legal_q_len).enumerate() {
            legal_q[i] = color_to_idx(c) as u8;
        }

        let g_rows = garbage_rows(&stacker);
        let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
        let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
        let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
        let q_start = q_pos.min(ed.pieces.len());
        let q_end = (q_start + horizon).min(ed.pieces.len());
        let q = &ed.pieces[q_start..q_end];
        let q_full_len = ed.pieces.len().saturating_sub(q_start);
        let q_len = q.len();
        if q_len == 0 {
            topout = true;
            break;
        }
        let cur0 = Some(q[0]);
        let alt0 = hold0.or_else(|| q.get(1).copied());

        pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
        if root_places.is_empty() {
            topout = true;
            break;
        }

        let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
        let eff_w = if is_end { beam_w.max(320) } else { beam_w };
        let eff_mpr = if is_end { 64 } else { max_per_root };
        let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

        cur_beam.clear();
        seen.clear();
        let mut best_goal: Option<(usize, u16, i64)> = None;
        let mut root_best_goal: Vec<Option<usize>> = vec![None; root_places.len()];
        let mut root_after1: Vec<Option<(FastBoard, Option<u8>, usize, Option<u8>, u16, u8, u8, i64)>> =
            vec![None; root_places.len()];

        for (idx, &fp) in root_places.iter().enumerate() {
            let mut b = root_board;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            let rem_g = rem_g0.saturating_sub(ds as u16);
            if rem_g > 0 && !spawn_exact_cheese_ed(&mut b, rem_g, lines, &ed.holes) {
                continue;
            }
            let (held, qc) = if hold0.is_some() {
                if fp.did_hold {
                    (cur0, 1.min(q_len))
                } else {
                    (hold0, 1.min(q_len))
                }
            } else if fp.did_hold {
                (cur0, 2.min(q_len))
            } else {
                (None, 1.min(q_len))
            };
            if rem_g == 0 && b.rows < 20 {
                let gr = -30_000_000 + 100 + b.rows as i64;
                if best_goal.map_or(true, |x| gr < x.2) {
                    best_goal = Some((1, idx as u16, gr));
                }
                root_best_goal[idx] = Some(1);
            }
            let in_h = held.or_else(|| q.get(qc).copied());
            let sc = teacher.eval(&b, rem_g, in_h, 0x7F);
            let l_in_h = held.or_else(|| legal_q[..legal_q_len].get(qc).copied());
            let l_avail = avail_mask_from_q(held, qc, &legal_q[..legal_q_len], legal_bag);
            let sc_legal_1ply = teacher.eval(&b, rem_g, l_in_h, l_avail);
            root_after1[idx] = Some((b, held, qc, l_in_h, rem_g, l_avail, ds, sc_legal_1ply));

            let key = OKey { board: b, held, rem_g };
            if seen.insert(key) {
                cur_beam.push(ONode {
                    board: b,
                    held,
                    q_cursor: qc,
                    rem_g,
                    first_idx: idx as u16,
                    score: sc,
                });
            }
        }

        for depth in 2..=target_d {
            children.clear();
            for node in &cur_beam {
                if node.rem_g == 0 {
                    continue;
                }
                let qc = node.q_cursor;
                let (c_nh, c_h) = if node.held.is_some() {
                    (q.get(qc).copied(), node.held)
                } else {
                    (q.get(qc).copied(), q.get(qc + 1).copied())
                };
                pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                for &fp in &tmp_places {
                    let mut b = node.board;
                    let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                    let rem_g = node.rem_g.saturating_sub(ds as u16);
                    if rem_g > 0 && !spawn_exact_cheese_ed(&mut b, rem_g, lines, &ed.holes) {
                        continue;
                    }
                    let (held, nqc) = if node.held.is_some() {
                        if fp.did_hold {
                            (c_nh, (qc + 1).min(q_len))
                        } else {
                            (node.held, (qc + 1).min(q_len))
                        }
                    } else if fp.did_hold {
                        (c_nh, (qc + 2).min(q_len))
                    } else {
                        (None, (qc + 1).min(q_len))
                    };
                    if rem_g == 0 && b.rows < 20 {
                        let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                        if best_goal.map_or(true, |x| gr < x.2) {
                            best_goal = Some((depth, node.first_idx, gr));
                        }
                        let ri = node.first_idx as usize;
                        if root_best_goal[ri].map_or(true, |gd| depth < gd) {
                            root_best_goal[ri] = Some(depth);
                        }
                    }
                    let in_h = held.or_else(|| q.get(nqc).copied());
                    let sc = teacher.eval(&b, rem_g, in_h, 0x7F);
                    children.push(ONode {
                        board: b,
                        held,
                        q_cursor: nqc,
                        rem_g,
                        first_idx: node.first_idx,
                        score: sc,
                    });
                }
            }
            if children.is_empty() {
                break;
            }
            children.sort_unstable_by_key(|n| (n.score, n.first_idx));
            cur_beam.clear();
            seen.clear();
            let mut rc = vec![0usize; root_places.len()];
            for &ch in &children {
                if cur_beam.len() >= eff_w {
                    break;
                }
                if rc[ch.first_idx as usize] >= eff_mpr {
                    continue;
                }
                let key = OKey {
                    board: ch.board,
                    held: ch.held,
                    rem_g: ch.rem_g,
                };
                if seen.insert(key) {
                    rc[ch.first_idx as usize] += 1;
                    cur_beam.push(ch);
                }
            }
            if cur_beam.len() < eff_w {
                for &ch in &children {
                    if cur_beam.len() >= eff_w {
                        break;
                    }
                    let key = OKey {
                        board: ch.board,
                        held: ch.held,
                        rem_g: ch.rem_g,
                    };
                    if seen.insert(key) {
                        cur_beam.push(ch);
                    }
                }
            }
        }

        let chosen_idx = if let Some((_, ri, _)) = best_goal {
            ri as usize
        } else if let Some(best) = cur_beam.first() {
            best.first_idx as usize
        } else {
            0
        };

        let mut root_beam_rank: Vec<Option<i64>> = vec![None; root_places.len()];
        for (ri, &gd) in root_best_goal.iter().enumerate() {
            if let Some(d) = gd {
                root_beam_rank[ri] = Some(-30_000_000 + (d as i64) * 100);
            }
        }
        for node in &cur_beam {
            let ri = node.first_idx as usize;
            if root_beam_rank[ri].is_none() {
                root_beam_rank[ri] = Some(node.score);
            }
        }

        if let Some((b_w, held_w, qc_w, l_ih_w, rem_w, l_av_w, ds_w, sc_w_1ply)) = root_after1[chosen_idx] {
            if rem_w > 0 {
                let win_s_idx = states.len();
                states.push(StateFeat::from_board(&b_w, rem_w, l_ih_w, l_av_w, teacher));
                traj_recs.push((win_s_idx, rem_w));

                let do_cf = rem_g0 <= next_cf_rem_g
                    && rem_g0 > 6
                    && cf_tasks.len() < cf_per_game
                    && q_full_len > 24
                    && best_goal.is_none();

                if do_cf {
                    next_cf_rem_g = rem_g0.saturating_sub(cf_step_g);

                    // 1) Root 1-ply candidates (up to 5: chosen vs opp_ds / runner / hole_alt / 1ply_other / clean_alt)
                    let mut best_runner: Option<(usize, i64)> = None;
                    let mut best_opp_ds: Option<(usize, i64)> = None;
                    let mut best_1ply_other: Option<(usize, i64)> = None;
                    let mut best_clean_alt: Option<(usize, i64)> = None;
                    let mut sorted_1ply: Vec<(usize, i64, i32)> = Vec::with_capacity(root_places.len());

                    for ri in 0..root_places.len() {
                        if let Some((b_l, _, _, _, rem_l, _, ds_l, sc_l_1ply)) = root_after1[ri] {
                            if rem_l == 0 {
                                continue;
                            }
                            let (stk_c, cov_h, h1_b) = fast_leaf_clean_metrics(&b_l);
                            if b_l.rows < 20 {
                                sorted_1ply.push((ri, sc_l_1ply, cov_h));
                            }
                            if ri == chosen_idx {
                                continue;
                            }
                            if best_1ply_other.map_or(true, |x| sc_l_1ply < x.1) {
                                best_1ply_other = Some((ri, sc_l_1ply));
                            }
                            if let Some(br) = root_beam_rank[ri] {
                                if best_runner.map_or(true, |x| br < x.1) {
                                    best_runner = Some((ri, br));
                                }
                            }
                            if (ds_l > 0) != (ds_w > 0) {
                                let metric = root_beam_rank[ri].unwrap_or(sc_l_1ply + 500_000);
                                if best_opp_ds.map_or(true, |x| metric < x.1) {
                                    best_opp_ds = Some((ri, metric));
                                }
                            }
                            let clean_metric = (cov_h * 80 + h1_b * 40 + stk_c * 4) as i64;
                            if best_clean_alt.map_or(true, |x| clean_metric < x.1) {
                                best_clean_alt = Some((ri, clean_metric));
                            }
                        }
                    }
                    sorted_1ply.sort_unstable_by_key(|x| x.1);
                    let mut best_hole_alt: Option<usize> = None;
                    for &(ri, _, cov_h) in sorted_1ply.iter().take(15) {
                        if ri != chosen_idx && cov_h > 0 {
                            best_hole_alt = Some(ri);
                            break;
                        }
                    }

                    let mut cand_ris: Vec<usize> = Vec::with_capacity(5);
                    let mut root_cands: Vec<CfCand> = Vec::with_capacity(5);
                    cand_ris.push(chosen_idx);
                    root_cands.push(CfCand {
                        state_idx: win_s_idx,
                        board: b_w,
                        held: held_w,
                        qc: qc_w,
                        rem_g: rem_w,
                        ref_score: sc_w_1ply,
                    });
                    for opt_ri in [
                        best_opp_ds.map(|x| x.0),
                        best_runner.map(|x| x.0),
                        best_hole_alt,
                        best_1ply_other.map(|x| x.0),
                        best_clean_alt.map(|x| x.0),
                    ] {
                        if root_cands.len() >= 5 {
                            break;
                        }
                        if let Some(ri) = opt_ri {
                            if !cand_ris.contains(&ri) {
                                if let Some((b_l, held_l, qc_l, l_ih_l, rem_l, l_av_l, _, sc_l_1ply)) =
                                    root_after1[ri]
                                {
                                    let l_s_idx = states.len();
                                    states.push(StateFeat::from_board(&b_l, rem_l, l_ih_l, l_av_l, teacher));
                                    cand_ris.push(ri);
                                    root_cands.push(CfCand {
                                        state_idx: l_s_idx,
                                        board: b_l,
                                        held: held_l,
                                        qc: qc_l,
                                        rem_g: rem_l,
                                        ref_score: sc_l_1ply,
                                    });
                                }
                            }
                        }
                    }

                    // 2) Frontier diverse beam leaves (up to 4)
                    let mut bag_at_q_end = legal_bag;
                    for &p in q.iter().skip(legal_q_len) {
                        consume_bag_bit(&mut bag_at_q_end, p);
                    }
                    let mut leaf_cands: Vec<CfCand> = Vec::with_capacity(4);
                    let mut max_spawned_leaf = 0usize;
                    let valid_leaves: Vec<usize> = cur_beam
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| n.rem_g > 0 && n.board.rows < 20)
                        .map(|(i, _)| i)
                        .collect();
                    if valid_leaves.len() >= 2 {
                        let best_li = valid_leaves[0];
                        let best_rem_g = cur_beam[best_li].rem_g;
                        let mut opp_rem_li: Option<usize> = None;
                        let mut low_stk_li: Option<(usize, i32)> = None;
                        let mut clean_li: Option<(usize, i32)> = None;

                        for &li in valid_leaves.iter().skip(1) {
                            let nd = &cur_beam[li];
                            if opp_rem_li.is_none() && nd.rem_g != best_rem_g {
                                opp_rem_li = Some(li);
                            }
                            let (stk_c, cov_h, h1_b) = fast_leaf_clean_metrics(&nd.board);
                            let low_m = (nd.rem_g as i32) * 200 + stk_c * 10 + h1_b * 25;
                            if low_stk_li.map_or(true, |x| low_m < x.1) {
                                low_stk_li = Some((li, low_m));
                            }
                            let clean_m = cov_h * 40
                                + h1_b * 25
                                + stk_c * 3
                                + ((nd.rem_g as i32 - best_rem_g as i32).max(0)) * 180;
                            if clean_li.map_or(true, |x| clean_m < x.1) {
                                clean_li = Some((li, clean_m));
                            }
                        }

                        let mut leaf_idxs: Vec<usize> = Vec::with_capacity(4);
                        leaf_idxs.push(best_li);
                        for opt in [opp_rem_li, low_stk_li.map(|x| x.0), clean_li.map(|x| x.0)] {
                            if let Some(li) = opt {
                                if !leaf_idxs.contains(&li) {
                                    leaf_idxs.push(li);
                                }
                            }
                        }
                        if leaf_idxs.len() < 4 {
                            for &li in &valid_leaves {
                                if !leaf_idxs.contains(&li) {
                                    leaf_idxs.push(li);
                                    if leaf_idxs.len() >= 4 {
                                        break;
                                    }
                                }
                            }
                        }
                        if leaf_idxs.len() >= 2 {
                            for &li in &leaf_idxs {
                                let nd = &cur_beam[li];
                                let unsp = (nd.rem_g as usize).saturating_sub(nd.board.garbage_rows as usize);
                                max_spawned_leaf = max_spawned_leaf.max(100usize.saturating_sub(unsp));
                                let l_ih = nd.held.or_else(|| q.get(nd.q_cursor).copied());
                                let l_av = avail_mask_from_q(nd.held, nd.q_cursor, q, bag_at_q_end);
                                let s_idx = states.len();
                                states.push(StateFeat::from_board(&nd.board, nd.rem_g, l_ih, l_av, teacher));
                                leaf_cands.push(CfCand {
                                    state_idx: s_idx,
                                    board: nd.board,
                                    held: nd.held,
                                    qc: nd.q_cursor,
                                    rem_g: nd.rem_g,
                                    ref_score: nd.score,
                                });
                            }
                        }
                    }

                    if root_cands.len() >= 2 || leaf_cands.len() >= 2 {
                        let base_rng = seed.wrapping_mul(0x9E3779B97F4A7C15)
                            ^ ((pcs as u64) + 1).wrapping_mul(0xBF58476D1CE4E5B9);
                        let unspawned0 = (rem_g0 as usize).saturating_sub(g_rows as usize);
                        let spawned_so_far0 = 100usize.saturating_sub(unspawned0);
                        let prev_hole0 = if spawned_so_far0 > 0 && spawned_so_far0 <= ed.holes.len() {
                            ed.holes[spawned_so_far0 - 1]
                        } else if g_rows > 0 {
                            (!root_board.data[0] & 0x3FF).trailing_zeros() as u16
                        } else {
                            4
                        };
                        let prev_hole_leaf = if max_spawned_leaf > 0 && max_spawned_leaf <= ed.holes.len() {
                            ed.holes[max_spawned_leaf - 1]
                        } else {
                            4
                        };
                        cf_tasks.push(CfTask {
                            base_rng,
                            q_start,
                            q_win_len: q_len,
                            legal_q,
                            legal_q_len,
                            legal_bag,
                            bag_at_q_end,
                            spawned_so_far0,
                            prev_hole0,
                            sc_w_1ply,
                            root_cands,
                            leaf_cands,
                            max_spawned_leaf,
                            prev_hole_leaf,
                        });
                    }
                }
            }
        }

        let fp = root_places[chosen_idx];
        let mut inputs = match ai_in.inputs_for_place(&snap.matrix, fp) {
            Some(inp) => inp,
            None => {
                topout = true;
                break;
            }
        };
        inputs.pop();
        if fp.did_hold {
            if hold0.is_none() {
                q_pos += 2;
            } else {
                q_pos += 1;
            }
        } else {
            q_pos += 1;
        }
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        pcs += 1;
        ds_total += ds;
    }

    if ds_total < 100 || topout {
        return GameGenerated {
            ed: None,
            pieces: 400,
            ds_cleared: ds_total,
            topout: true,
            states: Vec::new(),
            traj_recs: Vec::new(),
            cf_tasks: Vec::new(),
        };
    }

    GameGenerated {
        ed: Some(ed),
        pieces: pcs,
        ds_cleared: ds_total,
        topout: false,
        states,
        traj_recs,
        cf_tasks,
    }
}

#[allow(clippy::too_many_arguments)]
fn run_cf_task(
    task: &CfTask,
    ed: &ExactSeedData,
    state_base: usize,
    is_val: bool,
    crn_worlds: usize,
    roll_horizon: usize,
    roll_bw: usize,
    teacher: &Teacher,
    leaf_eval: &LeafEval,
    fst: &FastShapeTable,
    pgen: &mut FastPlaceGen,
    cf_tmp_places: &mut Vec<blockfish::ai::fast::FastPlace>,
) -> (Vec<ValueSample>, Vec<RankPair>) {
    let mut vals = Vec::new();
    let mut pairs = Vec::new();
    let q_full = &ed.pieces[task.q_start..];
    let q_need = if roll_horizon > 0 {
        (task.q_win_len + roll_horizon + 12).min(260)
    } else {
        260
    };

    if task.root_cands.len() >= 2 {
        let mut w_queues: Vec<Vec<u8>> = Vec::with_capacity(crn_worlds);
        let mut w_holes: Vec<Vec<u16>> = Vec::with_capacity(crn_worlds);
        for wi in 0..crn_worlds {
            if wi == 0 {
                w_queues.push(q_full.to_vec());
                w_holes.push(ed.holes.clone());
            } else {
                let mut wr = task.base_rng ^ ((wi as u64) + 1).wrapping_mul(0x94D049BB133111EB);
                let mut qw = Vec::with_capacity(q_need);
                for &p in &task.legal_q[..task.legal_q_len] {
                    qw.push(p);
                }
                let mut bag = task.legal_bag;
                while qw.len() < q_need {
                    qw.push(draw_from_bag(&mut bag, &mut wr));
                }
                let mut hw = ed.holes.clone();
                let mut ph = task.prev_hole0;
                let mut hr = wr ^ 0xD6E8FEB86659FD93;
                for hslot in hw.iter_mut().skip(task.spawned_so_far0) {
                    hr = hr
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let off = (((hr >> 32) as u16) % 9) + 1;
                    ph = (ph + off) % 10;
                    *hslot = ph;
                }
                w_queues.push(qw);
                w_holes.push(hw);
            }
        }

        let mut cand_means: Vec<f32> = Vec::with_capacity(task.root_cands.len());
        for cand in &task.root_cands {
            let mut sum_c = 0.0f32;
            for wi in 0..crn_worlds {
                sum_c += rollout_crn_eval(
                    fst,
                    pgen,
                    cf_tmp_places,
                    cand.board,
                    cand.held,
                    cand.qc,
                    cand.rem_g,
                    &w_queues[wi],
                    &w_holes[wi],
                    teacher,
                    leaf_eval,
                    roll_horizon,
                    roll_bw,
                );
            }
            let mean_c = sum_c / (crn_worlds as f32);
            cand_means.push(mean_c);
            vals.push(ValueSample {
                state_idx: state_base + cand.state_idx,
                target: mean_c.clamp(0.5, 280.0),
                weight: 1.0,
                is_val,
            });
        }

        for a in 0..task.root_cands.len() {
            for b in (a + 1)..task.root_cands.len() {
                let (win_c, lose_c) = if cand_means[a] <= cand_means[b] {
                    (a, b)
                } else {
                    (b, a)
                };
                let delta = (cand_means[lose_c] - cand_means[win_c]).min(10.0);
                if delta >= 0.20 {
                    let is_flipped = win_c != 0 && lose_c == 0;
                    let is_trap = is_flipped
                        || (task.root_cands[win_c].ref_score > task.root_cands[lose_c].ref_score + 10)
                        || (task.sc_w_1ply >= task.root_cands[lose_c].ref_score);
                    pairs.push(RankPair {
                        win_idx: state_base + task.root_cands[win_c].state_idx,
                        lose_idx: state_base + task.root_cands[lose_c].state_idx,
                        dq: delta,
                        is_trap,
                        is_flipped,
                        is_val,
                    });
                }
            }
        }
    }

    if task.leaf_cands.len() >= 2 {
        let q_win_end = (task.q_start + task.q_win_len).min(ed.pieces.len());
        let q_win = &ed.pieces[task.q_start..q_win_end];
        let mut leaf_queues: Vec<Vec<u8>> = Vec::with_capacity(crn_worlds);
        let mut leaf_holes: Vec<Vec<u16>> = Vec::with_capacity(crn_worlds);
        for wi in 0..crn_worlds {
            if wi == 0 {
                leaf_queues.push(q_full.to_vec());
                leaf_holes.push(ed.holes.clone());
            } else {
                let mut wr = (task.base_rng ^ 0xA5A55A5A9E3779B9)
                    ^ ((wi as u64) + 1).wrapping_mul(0x94D049BB133111EB);
                let mut qw = Vec::with_capacity(q_need);
                qw.extend_from_slice(q_win);
                let mut bag = task.bag_at_q_end;
                while qw.len() < q_need {
                    qw.push(draw_from_bag(&mut bag, &mut wr));
                }
                let mut hw = ed.holes.clone();
                let mut ph = task.prev_hole_leaf;
                let mut hr = wr ^ 0xD6E8FEB86659FD93;
                for hslot in hw.iter_mut().skip(task.max_spawned_leaf) {
                    hr = hr
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let off = (((hr >> 32) as u16) % 9) + 1;
                    ph = (ph + off) % 10;
                    *hslot = ph;
                }
                leaf_queues.push(qw);
                leaf_holes.push(hw);
            }
        }

        let mut leaf_means: Vec<f32> = Vec::with_capacity(task.leaf_cands.len());
        for cand in &task.leaf_cands {
            let mut sum_c = 0.0f32;
            for wi in 0..crn_worlds {
                sum_c += rollout_crn_eval(
                    fst,
                    pgen,
                    cf_tmp_places,
                    cand.board,
                    cand.held,
                    cand.qc,
                    cand.rem_g,
                    &leaf_queues[wi],
                    &leaf_holes[wi],
                    teacher,
                    leaf_eval,
                    roll_horizon,
                    roll_bw,
                );
            }
            let mean_c = sum_c / (crn_worlds as f32);
            leaf_means.push(mean_c);
            vals.push(ValueSample {
                state_idx: state_base + cand.state_idx,
                target: mean_c.clamp(0.5, 280.0),
                weight: 1.0,
                is_val,
            });
        }

        for a in 0..task.leaf_cands.len() {
            for b in (a + 1)..task.leaf_cands.len() {
                let (win_c, lose_c) = if leaf_means[a] <= leaf_means[b] {
                    (a, b)
                } else {
                    (b, a)
                };
                let delta = (leaf_means[lose_c] - leaf_means[win_c]).min(10.0);
                if delta >= 0.20 {
                    let is_flipped = win_c != 0 && lose_c == 0;
                    let is_trap = is_flipped
                        || (task.leaf_cands[win_c].ref_score > task.leaf_cands[lose_c].ref_score + 10);
                    pairs.push(RankPair {
                        win_idx: state_base + task.leaf_cands[win_c].state_idx,
                        lose_idx: state_base + task.leaf_cands[lose_c].state_idx,
                        dq: delta,
                        is_trap,
                        is_flipped,
                        is_val,
                    });
                }
            }
        }
    }

    (vals, pairs)
}

struct AdamW {
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    t: i32,
    b1t: f32,
    b2t: f32,
    m_lin_b: f32,
    v_lin_b: f32,
    m_lin_s: [f32; SPARSE_DIM],
    v_lin_s: [f32; SPARSE_DIM],
    m_lin_d: [f32; DENSE_DIM],
    v_lin_d: [f32; DENSE_DIM],
    m_b1: [f32; SPATIAL_H1],
    v_b1: [f32; SPATIAL_H1],
    m_ws: Box<[[f32; SPATIAL_H1]; SPARSE_DIM]>,
    v_ws: Box<[[f32; SPATIAL_H1]; SPARSE_DIM]>,
    m_wd: Box<[[f32; SPATIAL_H1]; DENSE_DIM]>,
    v_wd: Box<[[f32; SPATIAL_H1]; DENSE_DIM]>,
    m_b2: [f32; SPATIAL_H2],
    v_b2: [f32; SPATIAL_H2],
    m_w2: Box<[[f32; SPATIAL_H2]; SPATIAL_H1]>,
    v_w2: Box<[[f32; SPATIAL_H2]; SPATIAL_H1]>,
    m_b3: f32,
    v_b3: f32,
    m_w3: [f32; SPATIAL_H2],
    v_w3: [f32; SPATIAL_H2],
}

impl AdamW {
    fn new(wd: f32) -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            wd,
            t: 0,
            b1t: 1.0,
            b2t: 1.0,
            m_lin_b: 0.0,
            v_lin_b: 0.0,
            m_lin_s: [0.0; SPARSE_DIM],
            v_lin_s: [0.0; SPARSE_DIM],
            m_lin_d: [0.0; DENSE_DIM],
            v_lin_d: [0.0; DENSE_DIM],
            m_b1: [0.0; SPATIAL_H1],
            v_b1: [0.0; SPATIAL_H1],
            m_ws: Box::new([[0.0; SPATIAL_H1]; SPARSE_DIM]),
            v_ws: Box::new([[0.0; SPATIAL_H1]; SPARSE_DIM]),
            m_wd: Box::new([[0.0; SPATIAL_H1]; DENSE_DIM]),
            v_wd: Box::new([[0.0; SPATIAL_H1]; DENSE_DIM]),
            m_b2: [0.0; SPATIAL_H2],
            v_b2: [0.0; SPATIAL_H2],
            m_w2: Box::new([[0.0; SPATIAL_H2]; SPATIAL_H1]),
            v_w2: Box::new([[0.0; SPATIAL_H2]; SPATIAL_H1]),
            m_b3: 0.0,
            v_b3: 0.0,
            m_w3: [0.0; SPATIAL_H2],
            v_w3: [0.0; SPATIAL_H2],
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn step_param(
        w: &mut f32,
        g: f32,
        m: &mut f32,
        v: &mut f32,
        decay: f32,
        lr_cur: f32,
        beta1: f32,
        beta2: f32,
        b1t: f32,
        b2t: f32,
        eps: f32,
    ) {
        *m = beta1 * *m + (1.0 - beta1) * g;
        *v = beta2 * *v + (1.0 - beta2) * g * g;
        let mh = *m / (1.0 - b1t);
        let vh = *v / (1.0 - b2t);
        if decay > 0.0 {
            *w *= 1.0 - lr_cur * decay;
        }
        *w -= lr_cur * mh / (vh.sqrt() + eps);
    }

    fn apply(
        &mut self,
        mdl: &mut SpatialValueModel,
        g: &GradBuf,
        lr_cur: f32,
        lin_lr_mult: f32,
    ) {
        self.t += 1;
        self.b1t *= self.beta1;
        self.b2t *= self.beta2;
        let wd_mlp = self.wd * 4.0;
        let wd_lin = self.wd * 0.5;
        let lr_lin = lr_cur * lin_lr_mult;
        let (b1, b2, b1t, b2t, eps) = (self.beta1, self.beta2, self.b1t, self.b2t, self.eps);

        Self::step_param(&mut mdl.lin_bias, g.lin_bias, &mut self.m_lin_b, &mut self.v_lin_b, 0.0, lr_lin, b1, b2, b1t, b2t, eps);
        for i in 0..SPARSE_DIM {
            Self::step_param(&mut mdl.lin_sparse[i], g.lin_sparse[i], &mut self.m_lin_s[i], &mut self.v_lin_s[i], wd_lin, lr_lin, b1, b2, b1t, b2t, eps);
        }
        for j in 0..DENSE_DIM {
            let d = if j == 0 { 0.0 } else { wd_lin };
            Self::step_param(&mut mdl.lin_dense[j], g.lin_dense[j], &mut self.m_lin_d[j], &mut self.v_lin_d[j], d, lr_lin, b1, b2, b1t, b2t, eps);
        }
        for h in 0..SPATIAL_H1 {
            Self::step_param(&mut mdl.b1[h], g.b1[h], &mut self.m_b1[h], &mut self.v_b1[h], 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        }
        for i in 0..SPARSE_DIM {
            for h in 0..SPATIAL_H1 {
                Self::step_param(&mut mdl.w_sparse[i][h], g.w_sparse[i][h], &mut self.m_ws[i][h], &mut self.v_ws[i][h], wd_mlp, lr_cur, b1, b2, b1t, b2t, eps);
            }
        }
        for j in 1..DENSE_DIM {
            if !dense_mlp_allowed(j) {
                continue;
            }
            for h in 0..SPATIAL_H1 {
                Self::step_param(&mut mdl.w_dense[j][h], g.w_dense[j][h], &mut self.m_wd[j][h], &mut self.v_wd[j][h], wd_mlp, lr_cur, b1, b2, b1t, b2t, eps);
            }
        }
        for k in 0..SPATIAL_H2 {
            Self::step_param(&mut mdl.b2[k], g.b2[k], &mut self.m_b2[k], &mut self.v_b2[k], 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        }
        for h in 0..SPATIAL_H1 {
            for k in 0..SPATIAL_H2 {
                Self::step_param(&mut mdl.w2[h][k], g.w2[h][k], &mut self.m_w2[h][k], &mut self.v_w2[h][k], wd_mlp, lr_cur, b1, b2, b1t, b2t, eps);
            }
        }
        Self::step_param(&mut mdl.b3, g.b3, &mut self.m_b3, &mut self.v_b3, 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        for k in 0..SPATIAL_H2 {
            Self::step_param(&mut mdl.w3[k], g.w3[k], &mut self.m_w3[k], &mut self.v_w3[k], wd_mlp, lr_cur, b1, b2, b1t, b2t, eps);
        }
    }
}

#[inline]
const fn dense_mlp_allowed(j: usize) -> bool {
    !matches!(j, 0 | 18 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 64)
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
        for row in self.w_sparse.iter_mut() {
            row.fill(0.0);
        }
        for row in self.w_dense.iter_mut() {
            row.fill(0.0);
        }
        self.b2.fill(0.0);
        for row in self.w2.iter_mut() {
            row.fill(0.0);
        }
        self.b3 = 0.0;
        self.w3.fill(0.0);
    }
}

#[inline]
fn forward_cache(
    mdl: &SpatialValueModel,
    sf: &StateFeat,
    z_out: &mut [f32; DENSE_DIM],
    a1_out: &mut [f32; SPATIAL_H1],
    a2_out: &mut [f32; SPATIAL_H2],
) -> f32 {
    let mut lin = mdl.lin_bias;
    let mut h1 = mdl.b1;

    for &bit in &sf.active {
        let idx = bit as usize;
        lin += mdl.lin_sparse[idx];
        let row = &mdl.w_sparse[idx];
        for h in 0..SPATIAL_H1 {
            h1[h] += row[h];
        }
    }
    for j in 0..DENSE_DIM {
        let z = (sf.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j];
        z_out[j] = z;
        lin += mdl.lin_dense[j] * z;
        if dense_mlp_allowed(j) {
            let zm = z.clamp(-3.0, 3.0);
            if zm != 0.0 {
                let row = &mdl.w_dense[j];
                for h in 0..SPATIAL_H1 {
                    h1[h] += zm * row[h];
                }
            }
        }
    }
    let mut h2 = mdl.b2;
    for h in 0..SPATIAL_H1 {
        let a1 = h1[h].clamp(0.0, 6.0);
        a1_out[h] = a1;
        if a1 > 0.0 {
            let row = &mdl.w2[h];
            for k in 0..SPATIAL_H2 {
                h2[k] += a1 * row[k];
            }
        }
    }
    let mut mlp = mdl.b3;
    for k in 0..SPATIAL_H2 {
        let a2 = h2[k].clamp(0.0, 6.0);
        a2_out[k] = a2;
        if a2 > 0.0 {
            mlp += a2 * mdl.w3[k];
        }
    }
    lin + (mlp / (1.0 + mlp.abs() / 2.5))
}

#[inline]
fn backward_add(
    mdl: &SpatialValueModel,
    sf: &StateFeat,
    z: &[f32; DENSE_DIM],
    a1: &[f32; SPATIAL_H1],
    a2: &[f32; SPATIAL_H2],
    d_out: f32,
    g: &mut GradBuf,
) {
    if d_out == 0.0 {
        return;
    }
    g.lin_bias += d_out;
    for &bit in &sf.active {
        g.lin_sparse[bit as usize] += d_out;
    }
    for j in 0..DENSE_DIM {
        g.lin_dense[j] += d_out * z[j];
    }
    let mut raw_mlp = mdl.b3;
    for k in 0..SPATIAL_H2 {
        raw_mlp += a2[k] * mdl.w3[k];
    }
    let denom = 1.0 + raw_mlp.abs() / 2.5;
    let d_mlp = d_out / (denom * denom);
    g.b3 += d_mlp;

    let mut d_h2 = [0.0f32; SPATIAL_H2];
    for k in 0..SPATIAL_H2 {
        if a2[k] > 0.0 {
            g.w3[k] += d_mlp * a2[k];
            if a2[k] < 6.0 {
                let dk = d_mlp * mdl.w3[k];
                d_h2[k] = dk;
                g.b2[k] += dk;
            }
        }
    }
    for h in 0..SPATIAL_H1 {
        let act1 = a1[h];
        if act1 > 0.0 {
            let mut dh = 0.0f32;
            let w2_row = &mdl.w2[h];
            let g2_row = &mut g.w2[h];
            for k in 0..SPATIAL_H2 {
                let dk = d_h2[k];
                if dk != 0.0 {
                    g2_row[k] += act1 * dk;
                    if act1 < 6.0 {
                        dh += w2_row[k] * dk;
                    }
                }
            }
            if dh != 0.0 {
                g.b1[h] += dh;
                for &bit in &sf.active {
                    let gs_row = &mut g.w_sparse[bit as usize];
                    gs_row[h] += dh;
                }
                for j in 1..DENSE_DIM {
                    if !dense_mlp_allowed(j) {
                        continue;
                    }
                    let zv = z[j].clamp(-3.0, 3.0);
                    if zv != 0.0 {
                        g.w_dense[j][h] += dh * zv;
                    }
                }
            }
        }
    }
}

struct Xorshift64(u64);
impl Xorshift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / 16777216.0
    }
    fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-6);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
}

fn eval_legal_quick(label: &str, seeds: &[u64], mdl: &Arc<SpatialValueModel>) {
    let t0 = Instant::now();
    let mut cfg = BeamConfig::default();
    cfg.beam_width = 120;
    cfg.max_per_root = 24;
    cfg.rollout_top_k = 12;
    cfg.simulate_cheese_spawn = true;
    cfg.bag_det_extend = true;
    cfg.endgame_wide = true;
    cfg.endgame_solver = true;
    cfg.spatial_model = Some(mdl.clone());

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(seeds.len().max(1));
    let chunk_sz = (seeds.len() + n_threads - 1) / n_threads;
    let pcs: Vec<usize> = std::thread::scope(|sc| {
        let handles: Vec<_> = seeds
            .chunks(chunk_sz)
            .map(|sl| {
                let cfg_ref = &cfg;
                sc.spawn(move || {
                    sl.iter()
                        .map(|&s| {
                            let mut gc = BSConfig::default();
                            gc.garbage.total_lines = Some(100);
                            gc.prng_seed = Some(s);
                            let mut stacker = Stacker::new(Ruleset::guideline().into(), gc);
                            let mut ai = FastAI::new(cfg_ref.clone());
                            let mut ds_total = 0usize;
                            let mut moves = 0usize;
                            while ds_total < 100 && stacker.current_piece_type().is_some() {
                                let snap = stacker.snapshot().unwrap();
                                let g_rows = garbage_rows(&stacker);
                                let rem_g = 100usize.saturating_sub(ds_total) as u16;
                                let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
                                    Some(r) => r,
                                    None => return 400,
                                };
                                let mut inputs = res.inputs;
                                inputs.pop();
                                stacker.run(inputs);
                                let (_, ds) = stacker.hard_drop();
                                moves += 1;
                                ds_total += ds;
                            }
                            if ds_total < 100 { 400 } else { moves }
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut out = Vec::with_capacity(seeds.len());
        for h in handles {
            out.extend(h.join().unwrap());
        }
        out
    });
    let mean = pcs.iter().sum::<usize>() as f64 / pcs.len() as f64;
    println!(
        "{:<26} mean={:6.2} per_seed={:?} dt={:.1}s",
        label, mean, pcs, t0.elapsed().as_secs_f64()
    );
}

fn eval_oracle_quick(
    label: &str,
    seeds: &[u64],
    horizon: usize,
    beam_w: usize,
    mdl: &Teacher,
) -> f64 {
    let t0 = Instant::now();
    let eds: Vec<ExactSeedData> = seeds.iter().filter_map(|&s| collect_exact_seed(s)).collect();
    let run_slice = |sl: &[ExactSeedData]| -> Vec<usize> {
        let fst = FastShapeTable::srs();
        sl.iter()
            .map(|ed| {
                #[derive(Clone, Copy)]
                struct ONode {
                    board: FastBoard,
                    held: Option<u8>,
                    q_cursor: usize,
                    rem_g: u16,
                    first_idx: u16,
                    score: i64,
                }
                #[derive(Clone, Copy, Eq, PartialEq, Hash)]
                struct OKey {
                    board: FastBoard,
                    held: Option<u8>,
                    rem_g: u16,
                }

                let mut gc = BSConfig::default();
                gc.garbage.total_lines = Some(100);
                gc.prng_seed = Some(ed.seed);
                let mut stacker = Stacker::new(Ruleset::guideline().into(), gc);
                let mut pgen = FastPlaceGen::new();
                let mut ai_in = FastAI::new(BeamConfig::default());
                let mut root_places = Vec::with_capacity(64);
                let mut tmp_places = Vec::with_capacity(64);
                let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
                let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
                let mut seen: HashSet<OKey, FxBuild> =
                    HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

                let mut ds_total = 0usize;
                let mut q_pos = 0usize;
                let mut moves = 0usize;
                while ds_total < 100 && stacker.current_piece_type().is_some() {
                    let snap = stacker.snapshot().unwrap();
                    let g_rows = garbage_rows(&stacker);
                    let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
                    let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
                    let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
                    let q_start = q_pos.min(ed.pieces.len());
                    let q_end = (q_start + horizon).min(ed.pieces.len());
                    let q = &ed.pieces[q_start..q_end];
                    let q_len = q.len();
                    if q_len == 0 {
                        return 400;
                    }
                    let cur0 = Some(q[0]);
                    let alt0 = hold0.or_else(|| q.get(1).copied());
                    pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
                    if root_places.is_empty() {
                        return 400;
                    }

                    let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
                    let eff_w = if is_end { beam_w.max(320) } else { beam_w };
                    let eff_mpr = if is_end { 96 } else { (eff_w / 4).max(24) };
                    let target_d = q_len.saturating_sub(if hold0.is_some() { 0 } else { 1 }).max(1);

                    cur_beam.clear();
                    seen.clear();
                    let mut best_goal: Option<(usize, u16, i64)> = None;
                    for (idx, &fp) in root_places.iter().enumerate() {
                        let mut b = root_board;
                        let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                        let rem_g = rem_g0.saturating_sub(ds as u16);
                        if rem_g > 0 && !spawn_exact_cheese_ed(&mut b, rem_g, lines, &ed.holes) {
                            continue;
                        }
                        let (held, qc) = if hold0.is_some() {
                            if fp.did_hold {
                                (cur0, 1.min(q_len))
                            } else {
                                (hold0, 1.min(q_len))
                            }
                        } else if fp.did_hold {
                            (cur0, 2.min(q_len))
                        } else {
                            (None, 1.min(q_len))
                        };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) {
                                best_goal = Some((1, idx as u16, gr));
                            }
                        }
                        let in_h = held.or_else(|| q.get(qc).copied());
                        let sc = mdl.eval(&b, rem_g, in_h, 0x7F);
                        if seen.insert(OKey { board: b, held, rem_g }) {
                            cur_beam.push(ONode {
                                board: b,
                                held,
                                q_cursor: qc,
                                rem_g,
                                first_idx: idx as u16,
                                score: sc,
                            });
                        }
                    }
                    for depth in 2..=target_d {
                        children.clear();
                        for node in &cur_beam {
                            if node.rem_g == 0 {
                                continue;
                            }
                            let qc = node.q_cursor;
                            let (c_nh, c_h) = if node.held.is_some() {
                                (q.get(qc).copied(), node.held)
                            } else {
                                (q.get(qc).copied(), q.get(qc + 1).copied())
                            };
                            pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                            for &fp in &tmp_places {
                                let mut b = node.board;
                                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                                let rem_g = node.rem_g.saturating_sub(ds as u16);
                                if rem_g > 0 && !spawn_exact_cheese_ed(&mut b, rem_g, lines, &ed.holes) {
                                    continue;
                                }
                                let (held, nqc) = if node.held.is_some() {
                                    if fp.did_hold {
                                        (c_nh, (qc + 1).min(q_len))
                                    } else {
                                        (node.held, (qc + 1).min(q_len))
                                    }
                                } else if fp.did_hold {
                                    (c_nh, (qc + 2).min(q_len))
                                } else {
                                    (None, (qc + 1).min(q_len))
                                };
                                if rem_g == 0 && b.rows < 20 {
                                    let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                                    if best_goal.map_or(true, |x| gr < x.2) {
                                        best_goal = Some((depth, node.first_idx, gr));
                                    }
                                }
                                let in_h = held.or_else(|| q.get(nqc).copied());
                                let sc = mdl.eval(&b, rem_g, in_h, 0x7F);
                                children.push(ONode {
                                    board: b,
                                    held,
                                    q_cursor: nqc,
                                    rem_g,
                                    first_idx: node.first_idx,
                                    score: sc,
                                });
                            }
                        }
                        if children.is_empty() {
                            break;
                        }
                        children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                        cur_beam.clear();
                        seen.clear();
                        let mut rc = vec![0usize; root_places.len()];
                        for &ch in &children {
                            if cur_beam.len() >= eff_w {
                                break;
                            }
                            if rc[ch.first_idx as usize] >= eff_mpr {
                                continue;
                            }
                            if seen.insert(OKey {
                                board: ch.board,
                                held: ch.held,
                                rem_g: ch.rem_g,
                            }) {
                                rc[ch.first_idx as usize] += 1;
                                cur_beam.push(ch);
                            }
                        }
                    }
                    let chosen_idx = best_goal
                        .map(|x| x.1 as usize)
                        .unwrap_or_else(|| cur_beam.first().map_or(0, |n| n.first_idx as usize));
                    let fp = root_places[chosen_idx];
                    let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
                    inputs.pop();
                    if fp.did_hold {
                        if hold0.is_none() {
                            q_pos += 2;
                        } else {
                            q_pos += 1;
                        }
                    } else {
                        q_pos += 1;
                    }
                    stacker.run(inputs);
                    let (_, ds) = stacker.hard_drop();
                    moves += 1;
                    ds_total += ds;
                }
                if ds_total < 100 { 400 } else { moves }
            })
            .collect()
    };
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(eds.len().max(1));
    let chunk_sz = (eds.len() + n_threads - 1) / n_threads;
    let pcs: Vec<usize> = std::thread::scope(|sc| {
        let handles: Vec<_> = eds.chunks(chunk_sz).map(|sl| sc.spawn(move || run_slice(sl))).collect();
        let mut out = Vec::with_capacity(eds.len());
        for h in handles {
            out.extend(h.join().unwrap());
        }
        out
    });
    let mean = pcs.iter().sum::<usize>() as f64 / pcs.len().max(1) as f64;
    println!(
        "{:<26} mean={:6.2} per_seed={:?} dt={:.1}s",
        label, mean, pcs, t0.elapsed().as_secs_f64()
    );
    mean
}

/// Gentle sign-consistent bounds for the linear skip connection:
/// prevents collinear sign inversions while leaving magnitudes empirical.
fn compute_gentle_bounds(dense_inv_std: &[f32; DENSE_DIM]) -> (Vec<f64>, Vec<f64>) {
    const P: usize = SPARSE_DIM + DENSE_DIM;
    let mut lb = vec![-6.0f64; P];
    let mut ub = vec![6.0f64; P];

    // Sparse spatial/bag bounds
    for i in 0..80 {
        lb[i] = -0.01;
        ub[i] = 0.15;
    }
    for i in 80..90 {
        lb[i] = 0.0;
        ub[i] = 1.00;
    }
    for i in 90..120 {
        // Prevent overfitting to absolute seed hole column indices
        lb[i] = -0.03;
        ub[i] = 0.03;
    }
    for i in 120..160 {
        lb[i] = -0.15;
        ub[i] = 0.30;
    }
    for i in 160..166 {
        lb[i] = -0.25;
        ub[i] = 0.25;
    }
    for i in 166..180 {
        lb[i] = -0.08;
        ub[i] = 0.08;
    }
    // Constrain hand_O (sparse bit 167) to prevent artificial O-dumping bias
    lb[167] = -0.05;
    ub[167] = 0.08;

    lb[180] = 0.50;
    ub[180] = 5.00; // g == 0 && rem_g > 0
    lb[181] = 0.0;
    ub[181] = 0.60; // g >= 5
    lb[182] = -0.35;
    ub[182] = 0.35;
    for i in 183..186 {
        lb[i] = -0.45;
        ub[i] = 0.0; // h1_open / h2_open are bonuses <= 0
    }

    let set_raw_range = |lb: &mut [f64], ub: &mut [f64], j: usize, min_raw: f64, max_raw: f64| {
        let inv = dense_inv_std[j] as f64;
        if inv > 1e-6 {
            lb[SPARSE_DIM + j] = min_raw / inv;
            ub[SPARSE_DIM + j] = max_raw / inv;
        } else {
            lb[SPARSE_DIM + j] = 0.0;
            ub[SPARSE_DIM + j] = 0.0;
        }
    };

    // Wide empirical bounds for rem_g, g, and stack_cells
    set_raw_range(&mut lb, &mut ub, 0, 1.35, 2.35); // rem_g
    set_raw_range(&mut lb, &mut ub, 1, 0.0, 0.45);  // g

    for j in 2..DENSE_DIM {
        match j {
            // Readiness / open-hole / clean-O / near-full / clearable bonuses <= 0
            33 | 34 | 35 | 41 | 54 | 55 | 56 | 57 | 58 | 59 | 60 | 61 | 67 | 71 => {
                ub[SPARSE_DIM + j] = 0.0;
            }
            // Risk / hole / blocking / clutter / roughness penalties >= 0
            _ => {
                lb[SPARSE_DIM + j] = 0.0;
            }
        }
    }

    set_raw_range(&mut lb, &mut ub, 3, 0.04, 0.25); // stack_cells
    set_raw_range(&mut lb, &mut ub, 19, 0.0, 0.25); // min_stack_h
    set_raw_range(&mut lb, &mut ub, 34, -0.35, 0.0); // hand_fits_h1
    set_raw_range(&mut lb, &mut ub, 35, -0.35, 0.0); // bag_fit_frac
    set_raw_range(&mut lb, &mut ub, 39, 0.0, 0.35); // h2_narrow_1
    set_raw_range(&mut lb, &mut ub, 40, 0.0, 0.35); // h2_narrow_2plus
    set_raw_range(&mut lb, &mut ub, 53, 0.0, 0.35); // two_well_cnt
    set_raw_range(&mut lb, &mut ub, 66, 0.0, 0.60); // o_trapped
    set_raw_range(&mut lb, &mut ub, 68, 0.0, 0.60); // no_i_when_i_only
    set_raw_range(&mut lb, &mut ub, 69, 0.0, 0.60); // i_dep_cols_no_i

    for c in 0..10 {
        set_raw_range(&mut lb, &mut ub, 8 + c, 0.0, 0.08);
    }

    // Minimal safety floors only
    let floors: [(usize, f64); 4] = [
        (22, 0.03),  // ex13_sq
        (23, 0.25),  // covered_holes
        (28, 0.12),  // h1_blocks
        (36, 0.08),  // h2_blocks
    ];
    for &(j, min_raw) in &floors {
        let inv = dense_inv_std[j] as f64;
        if inv > 1e-6 {
            lb[SPARSE_DIM + j] = lb[SPARSE_DIM + j].max(min_raw / inv);
        }
    }

    (lb, ub)
}

/// Fit `SpatialValueModel`'s linear skip `(lin_bias, lin_sparse, lin_dense)` directly to weighted
/// true outcome targets `(state_idx, target, weight)` and optional pairwise CRN differences
/// `(win_idx, lose_idx, dq, weight)` via sign-consistent Ridge regression.
fn ridge_fit_values(
    label: &str,
    mdl: &mut SpatialValueModel,
    all_states: &[StateFeat],
    samples: &[(usize, f32, f32)],
    pair_diffs: &[(usize, usize, f32, f32)],
    min_rem_g: f32,
    max_rem_g: f32,
) {
    const P: usize = SPARSE_DIM + DENSE_DIM;
    const P1: usize = P + 1;
    let mut a = vec![0.0f64; P1 * P1];
    let mut b = vec![0.0f64; P1];
    let mut x = [0.0f64; P1];
    let mut seen_stamp = [0u32; P1];
    let mut stamp: u32 = 0;
    let mut nz: Vec<usize> = Vec::with_capacity(240);
    let mut used = 0usize;
    let mut used_pairs = 0usize;
    let mut sum_w = 0.0f64;

    for &(si, target, weight) in samples {
        if weight <= 0.0 || !target.is_finite() || target.abs() > 1000.0 {
            continue;
        }
        let sf = &all_states[si];
        let rg = sf.dense[0];
        if rg < min_rem_g || rg > max_rem_g {
            continue;
        }
        used += 1;
        let wt = weight as f64;
        sum_w += wt;
        stamp += 1;
        nz.clear();
        for &bit in &sf.active {
            let idx = bit as usize;
            if seen_stamp[idx] != stamp {
                seen_stamp[idx] = stamp;
                nz.push(idx);
            }
            x[idx] += 1.0;
        }
        for j in 0..DENSE_DIM {
            let z = ((sf.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j]) as f64;
            if z.abs() > 1e-9 {
                let idx = SPARSE_DIM + j;
                seen_stamp[idx] = stamp;
                x[idx] = z;
                nz.push(idx);
            }
        }
        seen_stamp[P] = stamp;
        x[P] = 1.0;
        nz.push(P);
        nz.sort_unstable();

        let y = target as f64;
        for (ii, &ia) in nz.iter().enumerate() {
            let wxa = wt * x[ia];
            b[ia] += wxa * y;
            for &ib in &nz[ii..] {
                a[ia * P1 + ib] += wxa * x[ib];
            }
        }
        for &idx in &nz {
            x[idx] = 0.0;
        }
    }

    for &(win_si, lose_si, dq, weight) in pair_diffs {
        if weight <= 0.0 || !dq.is_finite() || dq.abs() > 50.0 {
            continue;
        }
        let sw = &all_states[win_si];
        let sl = &all_states[lose_si];
        if sw.dense[0] < min_rem_g
            || sw.dense[0] > max_rem_g
            || sl.dense[0] < min_rem_g
            || sl.dense[0] > max_rem_g
        {
            continue;
        }
        used_pairs += 1;
        let pw = weight as f64;
        sum_w += pw;
        stamp += 1;
        nz.clear();
        for &bit in &sl.active {
            let idx = bit as usize;
            if seen_stamp[idx] != stamp {
                seen_stamp[idx] = stamp;
                nz.push(idx);
            }
            x[idx] += 1.0;
        }
        for &bit in &sw.active {
            let idx = bit as usize;
            if seen_stamp[idx] != stamp {
                seen_stamp[idx] = stamp;
                nz.push(idx);
            }
            x[idx] -= 1.0;
        }
        for j in 0..DENSE_DIM {
            let dz = ((sl.dense[j] - sw.dense[j]) * mdl.dense_inv_std[j]) as f64;
            if dz.abs() > 1e-9 {
                let idx = SPARSE_DIM + j;
                seen_stamp[idx] = stamp;
                x[idx] = dz;
                nz.push(idx);
            }
        }
        nz.retain(|&idx| x[idx].abs() > 1e-12);
        nz.sort_unstable();

        let y = dq as f64;
        for (ii, &ia) in nz.iter().enumerate() {
            let wxa = pw * x[ia];
            b[ia] += wxa * y;
            for &ib in &nz[ii..] {
                a[ia * P1 + ib] += wxa * x[ib];
            }
        }
        for &idx in &nz {
            x[idx] = 0.0;
        }
    }

    for i in 0..P1 {
        for j in 0..i {
            a[i * P1 + j] = a[j * P1 + i];
        }
    }

    let lam_sparse = (sum_w * 0.015).clamp(30.0, 1200.0);
    let lam_dense = (sum_w * 0.015).clamp(15.0, 900.0);
    for i in 0..SPARSE_DIM {
        let mult = if (90..120).contains(&i) { 4.0 } else { 1.0 };
        a[i * P1 + i] += lam_sparse * mult;
    }
    a[SPARSE_DIM * P1 + SPARSE_DIM] += 0.5;
    for j in 1..DENSE_DIM {
        a[(SPARSE_DIM + j) * P1 + (SPARSE_DIM + j)] += lam_dense;
    }

    let (mut lb, mut ub) = compute_gentle_bounds(&mdl.dense_inv_std);
    lb.push(-1e6);
    ub.push(1e6);
    let mut w = vec![0.0f64; P1];
    for _ in 0..600 {
        for j in 0..P1 {
            let diag = a[j * P1 + j];
            if diag.abs() < 1e-12 {
                w[j] = 0.0;
                continue;
            }
            let mut dot = 0.0f64;
            let row = &a[j * P1..(j + 1) * P1];
            for k in 0..P1 {
                if k != j {
                    dot += row[k] * w[k];
                }
            }
            w[j] = ((b[j] - dot) / diag).clamp(lb[j], ub[j]);
        }
    }

    for i in 0..SPARSE_DIM {
        mdl.lin_sparse[i] = w[i] as f32;
    }
    for j in 0..DENSE_DIM {
        mdl.lin_dense[j] = w[SPARSE_DIM + j] as f32;
    }
    mdl.lin_bias = w[P] as f32;

    let mut abs_err = 0.0f64;
    let mut sq_err = 0.0f64;
    let mut eval_w = 0.0f64;
    for &(si, target, weight) in samples {
        if weight <= 0.0 || !target.is_finite() || target.abs() > 1000.0 {
            continue;
        }
        let sf = &all_states[si];
        let rg = sf.dense[0];
        if rg < min_rem_g || rg > max_rem_g {
            continue;
        }
        let pred = mdl.predict_linear_extracted(&sf.active, &sf.dense) as f64;
        let e = pred - (target as f64);
        let wt = weight as f64;
        abs_err += wt * e.abs();
        sq_err += wt * e * e;
        eval_w += wt;
    }
    let mae = abs_err / eval_w.max(1.0);
    let rmse = (sq_err / eval_w.max(1.0)).sqrt();
    if used_pairs > 0 {
        let mut p_ok = 0usize;
        let mut p_tot = 0usize;
        let mut p_derr = 0.0f64;
        for &(win_si, lose_si, dq, weight) in pair_diffs {
            if weight <= 0.0 || !dq.is_finite() || dq.abs() > 50.0 {
                continue;
            }
            let sw = &all_states[win_si];
            let sl = &all_states[lose_si];
            if sw.dense[0] < min_rem_g
                || sw.dense[0] > max_rem_g
                || sl.dense[0] < min_rem_g
                || sl.dense[0] > max_rem_g
            {
                continue;
            }
            let yw = mdl.predict_linear_extracted(&sw.active, &sw.dense) as f64;
            let yl = mdl.predict_linear_extracted(&sl.active, &sl.dense) as f64;
            p_tot += 1;
            if yw < yl {
                p_ok += 1;
            }
            p_derr += ((yl - yw) - (dq as f64)).abs();
        }
        let p_acc = 100.0 * (p_ok as f64) / (p_tot.max(1) as f64);
        let p_mae = p_derr / (p_tot.max(1) as f64);
        println!(
            "  Ridge [{}] ({} vals, {} pairs, sum_w={:.1}): mae={:.3} rmse={:.3} | train_pair={:.1}% diff_mae={:.3}",
            label, used, used_pairs, sum_w, mae, rmse, p_acc, p_mae
        );
    } else {
        println!(
            "  Ridge [{}] ({} samples, sum_w={:.1}): mae={:.3} rmse={:.3}",
            label, used, sum_w, mae, rmse
        );
    }
}

fn clamp_linear_sanity(mdl: &mut SpatialValueModel) {
    let (lb, ub) = compute_gentle_bounds(&mdl.dense_inv_std);
    for i in 0..SPARSE_DIM {
        mdl.lin_sparse[i] = mdl.lin_sparse[i].clamp(lb[i] as f32, ub[i] as f32);
    }
    for j in 0..DENSE_DIM {
        mdl.lin_dense[j] = mdl.lin_dense[j].clamp(lb[SPARSE_DIM + j] as f32, ub[SPARSE_DIM + j] as f32);
    }
    for k in 0..SPATIAL_H2 {
        mdl.w3[k] = mdl.w3[k].clamp(-0.35, 0.35);
    }
    mdl.b3 = mdl.b3.clamp(-2.0, 2.0);
}

fn print_spatial_diagnostics(label: &str, mdl: &SpatialValueModel) {
    let checks = [
        (0, "rem_g"),
        (1, "g"),
        (2, "pe_net"),
        (3, "stk_cells"),
        (18, "max_sh"),
        (21, "over2_sq"),
        (23, "cov_holes"),
        (25, "blk_over_h"),
        (28, "h1_blk"),
        (36, "h2_blk"),
        (47, "bump"),
        (56, "near9"),
        (61, "clean_sk"),
        (70, "h2_only_rg"),
        (71, "h1_ds_rg_h2"),
    ];
    let mut parts = Vec::new();
    for &(j, name) in &checks {
        parts.push(format!("{}:{:+.3}", name, mdl.lin_dense[j] * mdl.dense_inv_std[j]));
    }
    println!(
        "  [{}] bias={:+.2} g0_sp={:+.2} h_IOT=({:+.3},{:+.3},{:+.3}) | lin_raw: {}",
        label,
        mdl.lin_bias,
        mdl.lin_sparse[180],
        mdl.lin_sparse[166],
        mdl.lin_sparse[167],
        mdl.lin_sparse[168],
        parts.join(" ")
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut teacher_arg = "learned:models/crn_r3_h2stk03.txt".to_string();
    let mut out_path = "models/spatial_v1.txt".to_string();
    let mut init_path: Option<String> = None;
    let mut games: usize = 120;
    let mut horizon: usize = 12;
    let mut beam_w: usize = 140;
    let mut max_per_root: usize = 20;
    let mut roll_horizon: usize = 20;
    let mut roll_bw: usize = 16;
    let mut crn_worlds: usize = 6;
    let mut cf_per_game: usize = 24;
    let mut epochs: usize = 35;
    let mut batch_size: usize = 128;
    let mut lr: f32 = 0.0006;
    let mut wd: f32 = 0.015;
    let mut lin_lr_mult: f32 = 0.45;
    let mut rank_w: f32 = 1.5;
    let mut diff_w: f32 = 0.45;
    let mut val_w: f32 = 0.25;
    let mut cal_v_weight: f32 = 0.55;
    let mut keep_o_bias: bool = false;
    let mut seed_base: u64 = 1000;
    let mut eval_oracle: usize = 5;
    let mut eval_legal: usize = 5;
    let mut eval_teacher: bool = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--teacher" => { teacher_arg = args[i + 1].clone(); i += 2; }
            "--out" => { out_path = args[i + 1].clone(); i += 2; }
            "--init" => { init_path = Some(args[i + 1].clone()); i += 2; }
            "--games" => { games = args[i + 1].parse().unwrap(); i += 2; }
            "--horizon" => { horizon = args[i + 1].parse().unwrap(); i += 2; }
            "--beam" => { beam_w = args[i + 1].parse().unwrap(); i += 2; }
            "--max-per-root" => { max_per_root = args[i + 1].parse().unwrap(); i += 2; }
            "--roll-horizon" => { roll_horizon = args[i + 1].parse().unwrap(); i += 2; }
            "--roll-bw" => { roll_bw = args[i + 1].parse().unwrap(); i += 2; }
            "--crn-worlds" => { crn_worlds = args[i + 1].parse().unwrap(); i += 2; }
            "--cf-per-game" => { cf_per_game = args[i + 1].parse().unwrap(); i += 2; }
            "--epochs" => { epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch_size = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i + 1].parse().unwrap(); i += 2; }
            "--wd" => { wd = args[i + 1].parse().unwrap(); i += 2; }
            "--lin-lr-mult" => { lin_lr_mult = args[i + 1].parse().unwrap(); i += 2; }
            "--rank-w" => { rank_w = args[i + 1].parse().unwrap(); i += 2; }
            "--diff-w" => { diff_w = args[i + 1].parse().unwrap(); i += 2; }
            "--val-w" => { val_w = args[i + 1].parse().unwrap(); i += 2; }
            "--cal-v-weight" => { cal_v_weight = args[i + 1].parse().unwrap(); i += 2; }
            "--keep-o-bias" => { keep_o_bias = true; i += 1; }
            "--seed-base" => { seed_base = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-seeds" => {
                let n: usize = args[i + 1].parse().unwrap();
                eval_oracle = n;
                eval_legal = n;
                i += 2;
            }
            "--eval-oracle" => { eval_oracle = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-legal" => { eval_legal = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-teacher" => { eval_teacher = true; i += 1; }
            other => panic!("Unknown arg: {}", other),
        }
    }

    let teacher = if let Some(p) = teacher_arg.strip_prefix("spatial:") {
        Teacher::Spatial(Arc::new(SpatialValueModel::load_from_file(p).expect("load spatial teacher")))
    } else if let Some(p) = teacher_arg.strip_prefix("learned:") {
        Teacher::Learned(Arc::new(load_learned_model(p, true, !keep_o_bias)))
    } else {
        panic!("--teacher must start with learned: or spatial:");
    };

    let avail_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2);
    let n_threads = avail_threads.min(games.max(1));
    println!(
        "=== Round 3 OOD-Safe Calibrated Finite-Horizon CRN SpatialValueModel RL ===\n  teacher={} no_o_bias={} games={} seeds=[{}..{}] H={} B={} MPR={} roll_h={} roll_bw={} crn_w={} cf/g={} cal_v_w={} lin_lr_mult={} threads={}",
        teacher_arg, !keep_o_bias, games, seed_base, seed_base + games as u64, horizon, beam_w, max_per_root, roll_horizon, roll_bw, crn_worlds, cf_per_game, cal_v_weight, lin_lr_mult, avail_threads
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

    // Phase 1: Play games and collect trajectories + counterfactual tasks
    let t_gen = Instant::now();
    let seeds: Vec<u64> = (seed_base..seed_base + games as u64).collect();
    let game_results: Vec<GameGenerated> = std::thread::scope(|sc| {
        let chunk_sz = (seeds.len() + n_threads - 1) / n_threads;
        let handles: Vec<_> = seeds
            .chunks(chunk_sz)
            .map(|chunk| {
                let t_ref = &teacher;
                sc.spawn(move || {
                    chunk
                        .iter()
                        .map(|&s| {
                            generate_game_data(
                                s,
                                horizon,
                                beam_w,
                                max_per_root,
                                cf_per_game,
                                t_ref,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut all = Vec::with_capacity(games);
        for h in handles {
            all.extend(h.join().unwrap());
        }
        all
    });

    let valid_games: Vec<&GameGenerated> = game_results.iter().filter(|g| !g.topout).collect();
    assert!(!valid_games.is_empty(), "No valid games generated!");
    let topouts = game_results.len() - valid_games.len();
    let sum_pcs: usize = valid_games.iter().map(|g| g.pieces).sum();
    let sum_ds: usize = valid_games.iter().map(|g| g.ds_cleared).sum();
    let sum_cf_tasks: usize = valid_games.iter().map(|g| g.cf_tasks.len()).sum();
    let mean_pcs = sum_pcs as f64 / valid_games.len().max(1) as f64;
    let emp_rate = sum_pcs as f64 / sum_ds.max(1) as f64;
    let min_pcs = valid_games.iter().map(|g| g.pieces).min().unwrap_or(0);
    let max_pcs = valid_games.iter().map(|g| g.pieces).max().unwrap_or(0);
    println!(
        "Phase 1: {} valid games ({} topouts, {} cf_tasks) in {:.1}s | teacher mean={:.2} [{}..{}] emp_rate={:.4}",
        valid_games.len(),
        topouts,
        sum_cf_tasks,
        t_gen.elapsed().as_secs_f64(),
        mean_pcs,
        min_pcs,
        max_pcs,
        emp_rate
    );

    // Phase 2: Train/val split (zero leakage), normalization stats, V_cal MC fit, LeafEval calibration
    let val_start = if valid_games.len() <= 1 {
        valid_games.len()
    } else {
        ((valid_games.len() * 85) / 100).min(valid_games.len() - 1)
    };
    let mut all_states: Vec<StateFeat> = Vec::new();
    let mut game_state_bases: Vec<usize> = Vec::with_capacity(valid_games.len());
    let mut train_state_idxs: Vec<usize> = Vec::new();
    let mut mc_train_samples: Vec<(usize, f32, f32)> = Vec::new();

    for (gi, g) in valid_games.iter().enumerate() {
        let is_val = gi >= val_start;
        let base = all_states.len();
        let end = base + g.states.len();
        game_state_bases.push(base);
        all_states.extend(g.states.iter().cloned());
        if !is_val {
            train_state_idxs.extend(base..end);
            for (step_idx, &(local_si, rem_w)) in g.traj_recs.iter().enumerate() {
                if rem_w == 0 {
                    continue;
                }
                let mc_rem = (g.pieces.saturating_sub(step_idx + 1)) as f32;
                mc_train_samples.push((base + local_si, mc_rem.clamp(0.5, 280.0), 1.0));
            }
        }
    }

    let mut mean = [0.0f64; DENSE_DIM];
    let mut var = [0.0f64; DENSE_DIM];
    let n_tr = train_state_idxs.len().max(1) as f64;
    for &si in &train_state_idxs {
        let s = &all_states[si];
        for j in 0..DENSE_DIM {
            mean[j] += s.dense[j] as f64;
        }
    }
    for j in 0..DENSE_DIM {
        mean[j] /= n_tr;
    }
    for &si in &train_state_idxs {
        let s = &all_states[si];
        for j in 0..DENSE_DIM {
            let d = s.dense[j] as f64 - mean[j];
            var[j] += d * d;
        }
    }
    let mut dense_mean = [0.0f32; DENSE_DIM];
    let mut dense_inv_std = [1.0f32; DENSE_DIM];
    for j in 0..DENSE_DIM {
        dense_mean[j] = mean[j] as f32;
        let std = (var[j] / n_tr).sqrt().max(1e-3);
        dense_inv_std[j] = (1.0 / std) as f32;
    }

    let mut v_cal = SpatialValueModel::default();
    v_cal.dense_mean = dense_mean;
    v_cal.dense_inv_std = dense_inv_std;
    ridge_fit_values("V_cal_MC", &mut v_cal, &all_states, &mc_train_samples, &[], 1.0, 100.0);
    v_cal.prepare_fast();
    print_spatial_diagnostics("V_cal_MC", &v_cal);
    let v_cal_arc = Arc::new(v_cal);

    // Calibrate teacher scaling to empirical pieces-remaining scale via OLS on train MC outcomes
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut n_cal = 0usize;
    for &(si, y_mc, _) in &mc_train_samples {
        let tx = all_states[si].teacher_score as f64;
        if tx.is_finite() && tx.abs() < 1000.0 {
            sum_x += tx;
            sum_y += y_mc as f64;
            n_cal += 1;
        }
    }
    let (t_scale, t_shift) = if n_cal >= 10 {
        let mx = sum_x / (n_cal as f64);
        let my = sum_y / (n_cal as f64);
        let mut cov = 0.0f64;
        let mut vx = 0.0f64;
        for &(si, y_mc, _) in &mc_train_samples {
            let tx = all_states[si].teacher_score as f64;
            if tx.is_finite() && tx.abs() < 1000.0 {
                let dx = tx - mx;
                cov += dx * ((y_mc as f64) - my);
                vx += dx * dx;
            }
        }
        let raw_slope = if vx > 1e-6 { cov / vx } else { emp_rate / 2.1189 };
        let ts = raw_slope.clamp(0.70, 1.0) as f32;
        let sh = (my - (ts as f64) * mx) as f32;
        (ts, sh)
    } else {
        ((emp_rate / 2.1189).clamp(0.70, 1.0) as f32, 0.0f32)
    };
    println!(
        "  LeafEval calibration: emp_rate={:.4} t_scale={:.4} t_shift={:+.2} v_cal_w={:.2}",
        emp_rate, t_scale, t_shift, cal_v_weight
    );
    let leaf_eval = LeafEval {
        v_cal: v_cal_arc.clone(),
        teacher: teacher.clone(),
        v_cal_w: cal_v_weight,
        t_scale,
        t_shift,
    };

    // Build blended 0.5*MC + 0.5*TD(K) trajectory targets (using train-fit V_cal)
    let td_k = if roll_horizon > 0 { roll_horizon } else { 20 };
    let mut val_samples: Vec<ValueSample> = Vec::new();
    for (gi, g) in valid_games.iter().enumerate() {
        let is_val = gi >= val_start;
        let base = game_state_bases[gi];
        for (step_idx, &(local_si, rem_w)) in g.traj_recs.iter().enumerate() {
            if rem_w == 0 {
                continue;
            }
            let mc_rem = (g.pieces.saturating_sub(step_idx + 1)) as f32;
            let targ = if step_idx + td_k < g.traj_recs.len() {
                let (fut_si, _) = g.traj_recs[step_idx + td_k];
                let f_sf = &all_states[base + fut_si];
                let v_fut = v_cal_arc.predict_extracted(&f_sf.active, &f_sf.dense).max(0.5);
                let td_rem = (td_k as f32) + v_fut;
                (0.5 * mc_rem + 0.5 * td_rem).clamp(0.5, 280.0)
            } else {
                mc_rem.clamp(0.5, 280.0)
            };
            val_samples.push(ValueSample {
                state_idx: base + local_si,
                target: targ,
                weight: 0.30,
                is_val,
            });
        }
    }

    // Phase 3: Execute all counterfactual tasks across all threads via work-stealing
    let t_cf = Instant::now();
    let mut jobs: Vec<(usize, usize)> = Vec::with_capacity(sum_cf_tasks);
    for (gi, g) in valid_games.iter().enumerate() {
        for ti in 0..g.cf_tasks.len() {
            jobs.push((gi, ti));
        }
    }
    let next_job = std::sync::atomic::AtomicUsize::new(0);
    let cf_threads = avail_threads.min(jobs.len().max(1));
    let cf_chunks: Vec<(Vec<ValueSample>, Vec<RankPair>)> = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..cf_threads)
            .map(|_| {
                let vg_ref = &valid_games;
                let bases_ref = &game_state_bases;
                let jobs_ref = &jobs;
                let nj_ref = &next_job;
                let t_ref = &teacher;
                let le_ref = &leaf_eval;
                sc.spawn(move || {
                    let fst = FastShapeTable::srs();
                    let mut pgen = FastPlaceGen::new();
                    let mut tmp = Vec::with_capacity(64);
                    let mut out_v = Vec::new();
                    let mut out_p = Vec::new();
                    loop {
                        let ji = nj_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if ji >= jobs_ref.len() {
                            break;
                        }
                        let (gi, ti) = jobs_ref[ji];
                        let g = vg_ref[gi];
                        let ed = g.ed.as_ref().expect("valid game must retain ExactSeedData");
                        let (vs, ps) = run_cf_task(
                            &g.cf_tasks[ti],
                            ed,
                            bases_ref[gi],
                            gi >= val_start,
                            crn_worlds,
                            roll_horizon,
                            roll_bw,
                            t_ref,
                            le_ref,
                            &fst,
                            &mut pgen,
                            &mut tmp,
                        );
                        out_v.extend(vs);
                        out_p.extend(ps);
                    }
                    (out_v, out_p)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut pairs: Vec<RankPair> = Vec::new();
    let mut flipped_cnt = 0usize;
    for (vs, ps) in cf_chunks {
        val_samples.extend(vs);
        for p in ps {
            if p.is_flipped {
                flipped_cnt += 1;
            }
            pairs.push(p);
        }
    }
    let train_vals: Vec<usize> = (0..val_samples.len()).filter(|&i| !val_samples[i].is_val).collect();
    let valid_vals: Vec<usize> = (0..val_samples.len()).filter(|&i| val_samples[i].is_val).collect();
    let train_pairs: Vec<usize> = (0..pairs.len()).filter(|&i| !pairs[i].is_val).collect();
    let valid_pairs: Vec<usize> = (0..pairs.len()).filter(|&i| pairs[i].is_val).collect();
    println!(
        "Phase 3: {} CF jobs finished in {:.1}s | states={} train_vals={} val_vals={} train_pairs={} val_pairs={} crn_flipped={}",
        jobs.len(),
        t_cf.elapsed().as_secs_f64(),
        all_states.len(),
        train_vals.len(),
        valid_vals.len(),
        train_pairs.len(),
        valid_pairs.len(),
        flipped_cnt
    );

    // Phase 4: Unanchored Ridge initialization on true outcomes + joint AdamW RL
    let mut mdl = if let Some(ref ip) = init_path {
        let mut m = SpatialValueModel::load_from_file(ip).expect("load init model");
        m.fast_ready = false;
        for j in 0..DENSE_DIM {
            if !dense_mlp_allowed(j) {
                for h in 0..SPATIAL_H1 {
                    m.w_dense[j][h] = 0.0;
                }
            }
        }
        m
    } else {
        let mut m = SpatialValueModel::default();
        m.dense_mean = dense_mean;
        m.dense_inv_std = dense_inv_std;

        let all_train_val_tuples: Vec<(usize, f32, f32)> = train_vals
            .iter()
            .map(|&vi| {
                let s = &val_samples[vi];
                (s.state_idx, s.target, s.weight)
            })
            .collect();
        let train_pair_diffs: Vec<(usize, usize, f32, f32)> = train_pairs
            .iter()
            .map(|&pi| {
                let pr = &pairs[pi];
                let trap_mult = if pr.is_flipped {
                    2.0
                } else if pr.is_trap {
                    1.5
                } else {
                    1.0
                };
                (pr.win_idx, pr.lose_idx, pr.dq, 1.5 * trap_mult)
            })
            .collect();
        ridge_fit_values(
            "ep0_init",
            &mut m,
            &all_states,
            &all_train_val_tuples,
            &train_pair_diffs,
            1.0,
            100.0,
        );

        // Initialize tiny residual 2-layer MLP on top of unbiased Ridge linear skip
        let mut rng = Xorshift64(0x9E3779B97F4A7C15 ^ seed_base);
        let s1 = (2.0f32 / (DENSE_DIM as f32 + 40.0)).sqrt() * 0.20;
        let s2 = (2.0f32 / SPATIAL_H1 as f32).sqrt() * 0.20;
        let s3 = 0.008f32;
        for i in 0..SPARSE_DIM {
            for h in 0..SPATIAL_H1 {
                m.w_sparse[i][h] = rng.next_normal() * s1 * 0.5;
            }
        }
        for j in 1..DENSE_DIM {
            if !dense_mlp_allowed(j) {
                continue;
            }
            for h in 0..SPATIAL_H1 {
                m.w_dense[j][h] = rng.next_normal() * s1;
            }
        }
        for h in 0..SPATIAL_H1 {
            for k in 0..SPATIAL_H2 {
                m.w2[h][k] = rng.next_normal() * s2;
            }
        }
        for k in 0..SPATIAL_H2 {
            m.w3[k] = rng.next_normal() * s3;
        }
        m.b3 = 0.0;
        m
    };
    mdl.fast_ready = false;
    let ep0_mdl = mdl.clone();
    print_spatial_diagnostics("ep0_init", &mdl);

    let mut opt = AdamW::new(wd);
    let mut grad = GradBuf::new();
    let mut rng = Xorshift64(0xBF58476D1CE4E5B9 ^ seed_base);
    let mut tv_perm = train_vals.clone();
    let mut tp_perm = train_pairs.clone();

    let mut best_score = f32::MAX;
    let mut best_mdl = mdl.clone();

    let mut z_w = [0.0f32; DENSE_DIM];
    let mut a1_w = [0.0f32; SPATIAL_H1];
    let mut a2_w = [0.0f32; SPATIAL_H2];
    let mut z_l = [0.0f32; DENSE_DIM];
    let mut a1_l = [0.0f32; SPATIAL_H1];
    let mut a2_l = [0.0f32; SPATIAL_H2];

    let t_train = Instant::now();
    for ep in 0..=epochs {
        let lr_cur = if ep == 0 {
            0.0
        } else if ep <= 2 {
            lr * (ep as f32 / 2.0)
        } else {
            let progress = (ep as f32) / (epochs as f32);
            lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()))
        };

        if ep > 0 {
            for idx in (1..tv_perm.len()).rev() {
                let j = (rng.next_u64() as usize) % (idx + 1);
                tv_perm.swap(idx, j);
            }
            for idx in (1..tp_perm.len()).rev() {
                let j = (rng.next_u64() as usize) % (idx + 1);
                tp_perm.swap(idx, j);
            }

            let n_batches = (tv_perm.len() + batch_size - 1) / batch_size;
            let pair_batch = if tp_perm.is_empty() {
                0
            } else {
                ((tp_perm.len() + n_batches - 1) / n_batches).max(1)
            };

            let mut p_cursor = 0usize;
            let val_huber = 4.0f32;
            let diff_huber = 2.0f32;
            let tau = 0.30f32;

            for b_idx in 0..n_batches {
                grad.zero();
                let v_start = b_idx * batch_size;
                let v_end = (v_start + batch_size).min(tv_perm.len());
                let v_cnt = (v_end - v_start).max(1) as f32;

                for &vi in &tv_perm[v_start..v_end] {
                    let smp = &val_samples[vi];
                    let sf = &all_states[smp.state_idx];
                    let pred = forward_cache(&mdl, sf, &mut z_w, &mut a1_w, &mut a2_w);
                    let diff = pred - smp.target;
                    let g_raw = if diff.abs() <= val_huber {
                        diff
                    } else {
                        val_huber * diff.signum()
                    };
                    let g_v = (val_w * smp.weight * g_raw) / v_cnt;
                    backward_add(&mdl, sf, &z_w, &a1_w, &a2_w, g_v, &mut grad);
                }

                if pair_batch > 0 && !tp_perm.is_empty() {
                    let p_cnt = pair_batch as f32;
                    for _ in 0..pair_batch {
                        let pi = tp_perm[p_cursor % tp_perm.len()];
                        p_cursor += 1;
                        let pr = &pairs[pi];
                        let sw = &all_states[pr.win_idx];
                        let sl = &all_states[pr.lose_idx];
                        let yw = forward_cache(&mdl, sw, &mut z_w, &mut a1_w, &mut a2_w);
                        let yl = forward_cache(&mdl, sl, &mut z_l, &mut a1_l, &mut a2_l);

                        let pred_d = yl - yw;
                        let err_d = pred_d - pr.dq;
                        let g_cal = if err_d.abs() <= diff_huber {
                            err_d
                        } else {
                            diff_huber * err_d.signum()
                        };

                        let margin = pr.dq.clamp(0.20, 1.50);
                        let u = ((yw - yl + margin) / tau).clamp(-20.0, 20.0);
                        let sig = 1.0 / (1.0 + (-u).exp());
                        let trap_mult = if pr.is_flipped {
                            2.0
                        } else if pr.is_trap {
                            1.5
                        } else {
                            1.0
                        };

                        let g_yw = (-diff_w * g_cal + rank_w * trap_mult * sig) / p_cnt;
                        let g_yl = (diff_w * g_cal - rank_w * trap_mult * sig) / p_cnt;
                        backward_add(&mdl, sw, &z_w, &a1_w, &a2_w, g_yw, &mut grad);
                        backward_add(&mdl, sl, &z_l, &a1_l, &a2_l, g_yl, &mut grad);
                    }
                }

                opt.apply(&mut mdl, &grad, lr_cur, lin_lr_mult);
                clamp_linear_sanity(&mut mdl);
            }
        }

        let tau = 0.30f32;
        let mut cf_val_err = 0.0f32;
        let mut cf_val_wt = 0.0f32;
        for &vi in &valid_vals {
            let smp = &val_samples[vi];
            if smp.weight < 0.5 {
                continue;
            }
            let pred = forward_cache(&mdl, &all_states[smp.state_idx], &mut z_w, &mut a1_w, &mut a2_w);
            cf_val_err += smp.weight * (pred - smp.target).abs();
            cf_val_wt += smp.weight;
        }
        let val_mae = if cf_val_wt > 0.0 {
            cf_val_err / cf_val_wt
        } else {
            0.0
        };

        let mut pair_ok = 0usize;
        let mut trap_ok = 0usize;
        let mut trap_tot = 0usize;
        let mut flip_ok = 0usize;
        let mut flip_tot = 0usize;
        let mut diff_mae = 0.0f32;
        let mut rank_loss_sum = 0.0f32;
        for &pi in &valid_pairs {
            let pr = &pairs[pi];
            let yw = forward_cache(&mdl, &all_states[pr.win_idx], &mut z_w, &mut a1_w, &mut a2_w);
            let yl = forward_cache(&mdl, &all_states[pr.lose_idx], &mut z_l, &mut a1_l, &mut a2_l);
            diff_mae += ((yl - yw) - pr.dq).abs();
            if yw < yl {
                pair_ok += 1;
            }
            if pr.is_trap {
                trap_tot += 1;
                if yw < yl {
                    trap_ok += 1;
                }
            }
            if pr.is_flipped {
                flip_tot += 1;
                if yw < yl {
                    flip_ok += 1;
                }
            }
            let margin = pr.dq.clamp(0.20, 1.50);
            let u = ((yw - yl + margin) / tau).clamp(-20.0, 20.0);
            rank_loss_sum += tau * (1.0 + u.exp()).ln();
        }
        let n_vp = valid_pairs.len().max(1) as f32;
        diff_mae /= n_vp;
        let pair_acc = 100.0 * pair_ok as f32 / n_vp;
        let trap_acc = 100.0 * trap_ok as f32 / trap_tot.max(1) as f32;
        let flip_acc = 100.0 * flip_ok as f32 / flip_tot.max(1) as f32;
        let rank_l = rank_loss_sum / n_vp;
        let comb = 0.15 * val_mae + 0.65 * diff_mae + rank_w * rank_l;
        if comb < best_score {
            best_score = comb;
            best_mdl = mdl.clone();
        }
        if ep <= 1 || ep % 5 == 0 || ep == epochs {
            println!(
                "  ep={:2}/{} lr={:.5} | cf_mae={:.3} diff_mae={:.3} pair={:.1}% trap={:.1}% flip={:.1}% ({}) rank_l={:.3} comb={:.3}",
                ep, epochs, lr_cur, val_mae, diff_mae, pair_acc, trap_acc, flip_acc, flip_tot, rank_l, comb
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    best_mdl.prepare_fast();
    print_spatial_diagnostics("best", &best_mdl);
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    best_mdl.save_to_file(&out_path).expect("save spatial model");
    println!(
        "Saved best SpatialValueModel (best_comb={:.3}) to {} in {:.1}s",
        best_score,
        out_path,
        t_train.elapsed().as_secs_f64()
    );

    if eval_oracle > 0 || eval_legal > 0 {
        let mdl_arc = Arc::new(best_mdl);
        println!("\n=== Post-Training Held-Out Verification ===");
        if eval_oracle > 0 {
            let test_seeds: Vec<u64> = (101..101 + eval_oracle as u64).collect();
            if eval_teacher {
                let _ = eval_oracle_quick("OR_p12_b140_teacher", &test_seeds, 12, 140, &teacher);
            }
            if init_path.is_none() {
                let mut ep0_fast = ep0_mdl;
                ep0_fast.prepare_fast();
                let ep0_t = Teacher::Spatial(Arc::new(ep0_fast));
                let _ = eval_oracle_quick("OR_p12_b140_ep0", &test_seeds, 12, 140, &ep0_t);
            }
            let sp_teacher = Teacher::Spatial(mdl_arc.clone());
            let _ = eval_oracle_quick("OR_p12_b140_best", &test_seeds, 12, 140, &sp_teacher);
        }
        if eval_legal > 0 {
            let test_seeds: Vec<u64> = (101..101 + eval_legal as u64).collect();
            eval_legal_quick("LEGAL_p5_b120_spatial", &test_seeds, &mdl_arc);
        }
    }
}
