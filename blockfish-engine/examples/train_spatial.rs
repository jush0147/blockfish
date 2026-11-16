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

fn load_learned_model(path: &str, nog0: bool) -> LearnedValueModel {
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

    fn is_spatial(&self) -> bool {
        matches!(self, Teacher::Spatial(_))
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
    ai_cfg.beam_width = 16;
    ai_cfg.max_per_root = 4;
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

fn collect_exact_seed(s: u64) -> ExactSeedData {
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
    let mut pieces = Vec::with_capacity(320);
    pieces.push(color_to_idx(stacker.current_piece_type().unwrap().try_into().unwrap()) as u8);
    for &ch in stacker.next() {
        pieces.push(color_to_idx(ch.try_into().unwrap()) as u8);
    }
    let mut c = BeamConfig::default();
    c.beam_width = 16;
    c.max_per_root = 4;
    c.rollout_top_k = 0;
    let mut ai = FastAI::new(c);
    let mut ds_total = 0usize;
    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        let g_before = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = ai.best_move(&snap, Some(g_before), Some(rem_g)).unwrap();
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
    assert_eq!(holes.len(), 100);
    let skip_n = stacker.snapshot().map_or(0, |sn| sn.queue.len());
    if stacker.current_piece_type().is_some() {
        for &col in extended_queue(&stacker, 240).iter().skip(skip_n) {
            pieces.push(color_to_idx(col) as u8);
        }
    }
    ExactSeedData { seed: s, holes, pieces }
}

#[derive(Clone)]
struct StateFeat {
    active: Vec<u16>,
    dense: [f32; DENSE_DIM],
}

impl StateFeat {
    fn from_board(b: &FastBoard, rem_g: u16, in_hand: Option<u8>, avail: u8) -> Self {
        let mut active_buf = [0u16; SPARSE_DIM];
        let mut dense = [0.0f32; DENSE_DIM];
        let n_act = extract_spatial_features(b, rem_g, in_hand, avail, &mut active_buf, &mut dense);
        Self {
            active: active_buf[..n_act].to_vec(),
            dense,
        }
    }
}

#[derive(Clone)]
struct ValueSample {
    state_idx: usize,
    target: f32,
    is_val: bool,
}

#[derive(Clone)]
struct RankPair {
    win_idx: usize,
    lose_idx: usize,
    dq: f32,
    is_trap: bool,
    is_val: bool,
}

struct GameGenerated {
    pieces: usize,
    topout: bool,
    states: Vec<StateFeat>,
    val_targets: Vec<(usize, f32)>,
    pairs: Vec<(usize, usize, f32, bool)>,
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

fn generate_game_data(
    seed: u64,
    horizon: usize,
    beam_w: usize,
    max_per_root: usize,
    teacher: &Teacher,
) -> GameGenerated {
    let ed = collect_exact_seed(seed);
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

    let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
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
    };

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

    struct StepRec {
        win_state_idx: usize,
        q_chosen: f32,
        rem_g_after: u16,
    }

    let mut states: Vec<StateFeat> = Vec::with_capacity(1200);
    let mut step_recs: Vec<StepRec> = Vec::with_capacity(220);
    let mut raw_pairs: Vec<(usize, usize, f32, bool)> = Vec::with_capacity(600);

    let mut ds_total = 0usize;
    let mut q_pos = 0usize;
    let mut pcs = 0usize;
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
        let mut root_after1: Vec<Option<(FastBoard, Option<u8>, u16, u8, i64)>> =
            vec![None; root_places.len()];

        for (idx, &fp) in root_places.iter().enumerate() {
            let mut b = root_board;
            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
            let rem_g = rem_g0.saturating_sub(ds as u16);
            if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
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
            // Compute strict legal 1-ply hand & avail for student features
            let l_in_h = held.or_else(|| legal_q[..legal_q_len].get(qc).copied());
            let l_avail = avail_mask_from_q(held, qc, &legal_q[..legal_q_len], legal_bag);
            let sc_legal_1ply = teacher.eval(&b, rem_g, l_in_h, l_avail);
            root_after1[idx] = Some((b, l_in_h, rem_g, l_avail, sc_legal_1ply));

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
                    if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
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

        // Compute D-ply Q estimate for each root from best leaf / best goal
        let mut root_q: Vec<Option<f32>> = vec![None; root_places.len()];
        for (ri, &gd) in root_best_goal.iter().enumerate() {
            if let Some(d) = gd {
                root_q[ri] = Some(d as f32);
            }
        }
        for node in &cur_beam {
            let ri = node.first_idx as usize;
            if root_q[ri].is_none() && node.score > -9_000_000 && node.score < 4_000_000 {
                root_q[ri] = Some((target_d as f32) + (node.score as f32) * 0.01);
            }
        }

        // Record winner post-move state (with strict legal observable features!)
        if let Some((b_w, ih_w, rem_w, av_w, sc_w_1ply)) = root_after1[chosen_idx] {
            if rem_w > 0 {
                let win_s_idx = states.len();
                states.push(StateFeat::from_board(&b_w, rem_w, ih_w, av_w));
                let q_w = root_q[chosen_idx].unwrap_or(rem_w as f32 * 1.75);
                step_recs.push(StepRec {
                    win_state_idx: win_s_idx,
                    q_chosen: q_w,
                    rem_g_after: rem_w,
                });

                // Collect sibling contrast pairs: surviving suboptimal roots + eliminated 1-ply traps
                if let Some(q_win) = root_q[chosen_idx] {
                    let worst_surv_q = cur_beam
                        .last()
                        .filter(|n| n.score > -9_000_000 && n.score < 4_000_000)
                        .map(|n| (target_d as f32) + (n.score as f32) * 0.01)
                        .unwrap_or(q_win + 1.0);
                    let mut losers: Vec<(usize, f32, i64)> = Vec::new();
                    for ri in 0..root_places.len() {
                        if ri == chosen_idx {
                            continue;
                        }
                        if let Some((_, _, rem_l, _, sc_l_1ply)) = root_after1[ri] {
                            if rem_l == 0 {
                                continue;
                            }
                            if let Some(ql) = root_q[ri] {
                                let dq = ql - q_win;
                                if dq >= 0.20 {
                                    losers.push((ri, dq, sc_l_1ply));
                                }
                            } else if sc_l_1ply <= sc_w_1ply + 25 {
                                // Eliminated before final depth despite looking good at 1-ply
                                let dq = (worst_surv_q - q_win).max(0.80).min(2.0);
                                losers.push((ri, dq, sc_l_1ply));
                            }
                        }
                    }
                    // Sort traps (lowest 1-ply score relative to winner) first
                    losers.sort_by_key(|x| x.2);
                    let mut picked = 0usize;
                    let mut chosen_set = Vec::new();
                    for &(ri, dq, sc_l_1ply) in &losers {
                        if picked >= 2 {
                            break;
                        }
                        if let Some((b_l, ih_l, rem_l, av_l, _)) = root_after1[ri] {
                            let l_idx = states.len();
                            states.push(StateFeat::from_board(&b_l, rem_l, ih_l, av_l));
                            raw_pairs.push((win_s_idx, l_idx, dq, sc_l_1ply <= sc_w_1ply + 20));
                            chosen_set.push(ri);
                            picked += 1;
                        }
                    }
                    // Add 1 clear higher-dq negative for global gradient coverage
                    if let Some(&(ri, dq, sc_l_1ply)) = losers.iter().rfind(|x| !chosen_set.contains(&x.0)) {
                        if let Some((b_l, ih_l, rem_l, av_l, _)) = root_after1[ri] {
                            let l_idx = states.len();
                            states.push(StateFeat::from_board(&b_l, rem_l, ih_l, av_l));
                            raw_pairs.push((win_s_idx, l_idx, dq, sc_l_1ply <= sc_w_1ply + 20));
                        }
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

    if ds_total < 100 {
        return GameGenerated {
            pieces: 400,
            topout: true,
            states: Vec::new(),
            val_targets: Vec::new(),
            pairs: Vec::new(),
        };
    }

    let use_bellman_blend = teacher.is_spatial();
    let mut val_targets = Vec::with_capacity(step_recs.len());
    for (step_idx, sr) in step_recs.iter().enumerate() {
        let mc_rem = (pcs - (step_idx + 1)) as f32;
        if sr.rem_g_after == 0 {
            continue;
        }
        let target = if use_bellman_blend {
            let bell = (sr.q_chosen - 1.0).max(0.5);
            0.65 * mc_rem + 0.35 * bell
        } else {
            mc_rem
        };
        val_targets.push((sr.win_state_idx, target));
    }

    GameGenerated {
        pieces: pcs,
        topout,
        states,
        val_targets,
        pairs: raw_pairs,
    }
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

    fn apply(&mut self, mdl: &mut SpatialValueModel, g: &GradBuf, lr_cur: f32) {
        self.t += 1;
        self.b1t *= self.beta1;
        self.b2t *= self.beta2;
        let wd = self.wd;
        let wd_lin = wd * 0.1;
        let (b1, b2, b1t, b2t, eps) = (self.beta1, self.beta2, self.b1t, self.b2t, self.eps);

        Self::step_param(&mut mdl.lin_bias, g.lin_bias, &mut self.m_lin_b, &mut self.v_lin_b, 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        for i in 0..SPARSE_DIM {
            Self::step_param(&mut mdl.lin_sparse[i], g.lin_sparse[i], &mut self.m_lin_s[i], &mut self.v_lin_s[i], wd_lin, lr_cur, b1, b2, b1t, b2t, eps);
        }
        for j in 0..DENSE_DIM {
            let d = if j == 0 { 0.0 } else { wd_lin };
            Self::step_param(&mut mdl.lin_dense[j], g.lin_dense[j], &mut self.m_lin_d[j], &mut self.v_lin_d[j], d, lr_cur, b1, b2, b1t, b2t, eps);
        }
        for h in 0..SPATIAL_H1 {
            Self::step_param(&mut mdl.b1[h], g.b1[h], &mut self.m_b1[h], &mut self.v_b1[h], 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        }
        for i in 0..SPARSE_DIM {
            for h in 0..SPATIAL_H1 {
                Self::step_param(&mut mdl.w_sparse[i][h], g.w_sparse[i][h], &mut self.m_ws[i][h], &mut self.v_ws[i][h], wd, lr_cur, b1, b2, b1t, b2t, eps);
            }
        }
        for j in 0..DENSE_DIM {
            for h in 0..SPATIAL_H1 {
                Self::step_param(&mut mdl.w_dense[j][h], g.w_dense[j][h], &mut self.m_wd[j][h], &mut self.v_wd[j][h], wd, lr_cur, b1, b2, b1t, b2t, eps);
            }
        }
        for k in 0..SPATIAL_H2 {
            Self::step_param(&mut mdl.b2[k], g.b2[k], &mut self.m_b2[k], &mut self.v_b2[k], 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        }
        for h in 0..SPATIAL_H1 {
            for k in 0..SPATIAL_H2 {
                Self::step_param(&mut mdl.w2[h][k], g.w2[h][k], &mut self.m_w2[h][k], &mut self.v_w2[h][k], wd, lr_cur, b1, b2, b1t, b2t, eps);
            }
        }
        Self::step_param(&mut mdl.b3, g.b3, &mut self.m_b3, &mut self.v_b3, 0.0, lr_cur, b1, b2, b1t, b2t, eps);
        for k in 0..SPATIAL_H2 {
            Self::step_param(&mut mdl.w3[k], g.w3[k], &mut self.m_w3[k], &mut self.v_w3[k], wd, lr_cur, b1, b2, b1t, b2t, eps);
        }
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
        if z != 0.0 {
            let row = &mdl.w_dense[j];
            for h in 0..SPATIAL_H1 {
                h1[h] += z * row[h];
            }
        }
    }
    let mut h2 = mdl.b2;
    for h in 0..SPATIAL_H1 {
        let a1 = h1[h].max(0.0);
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
        let a2 = h2[k].max(0.0);
        a2_out[k] = a2;
        if a2 > 0.0 {
            mlp += a2 * mdl.w3[k];
        }
    }
    lin + mlp
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
    g.b3 += d_out;

    let mut d_h2 = [0.0f32; SPATIAL_H2];
    for k in 0..SPATIAL_H2 {
        if a2[k] > 0.0 {
            g.w3[k] += d_out * a2[k];
            let dk = d_out * mdl.w3[k];
            d_h2[k] = dk;
            g.b2[k] += dk;
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
                    dh += w2_row[k] * dk;
                }
            }
            if dh != 0.0 {
                g.b1[h] += dh;
                for &bit in &sf.active {
                    let gs_row = &mut g.w_sparse[bit as usize];
                    gs_row[h] += dh;
                }
                for j in 0..DENSE_DIM {
                    let zv = z[j];
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
    cfg.endgame_wide = true;
    cfg.spatial_model = Some(mdl.clone());

    let mid = seeds.len() / 2;
    let (s1, s2) = seeds.split_at(mid);
    let run_slice = |sl: &[u64]| -> Vec<usize> {
        sl.iter().map(|&s| {
            let mut gc = BSConfig::default();
            gc.garbage.total_lines = Some(100);
            gc.prng_seed = Some(s);
            let mut stacker = Stacker::new(Ruleset::guideline().into(), gc);
            let mut ai = FastAI::new(cfg.clone());
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
        }).collect()
    };
    let pcs: Vec<usize> = std::thread::scope(|sc| {
        let rs_ref = &run_slice;
        let h = sc.spawn(move || rs_ref(s1));
        let mut r2 = run_slice(s2);
        let mut r1 = h.join().unwrap();
        r1.append(&mut r2);
        r1
    });
    let mean = pcs.iter().sum::<usize>() as f64 / pcs.len() as f64;
    println!(
        "{:<24} mean={:6.2} per_seed={:?} dt={:.1}s",
        label, mean, pcs, t0.elapsed().as_secs_f64()
    );
}

fn eval_oracle_quick(label: &str, seeds: &[u64], horizon: usize, beam_w: usize, mdl: &Arc<SpatialValueModel>) {
    let t0 = Instant::now();
    let eds: Vec<ExactSeedData> = seeds.iter().map(|&s| collect_exact_seed(s)).collect();
    let mid = eds.len() / 2;
    let (e1, e2) = eds.split_at(mid);
    let run_slice = |sl: &[ExactSeedData]| -> Vec<usize> {
        let fst = FastShapeTable::srs();
            sl.iter().map(|ed| {
                #[derive(Clone, Copy)]
                struct ONode { board: FastBoard, held: Option<u8>, q_cursor: usize, rem_g: u16, first_idx: u16, score: i64 }
                #[derive(Clone, Copy, Eq, PartialEq, Hash)]
                struct OKey { board: FastBoard, held: Option<u8>, rem_g: u16 }

                let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
                    let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
                    let g = b.garbage_rows as usize;
                    let unspawned = (rem_g as usize).saturating_sub(g);
                    let num = target_g.saturating_sub(g).min(unspawned);
                    if num == 0 { return true; }
                    let old_rows = b.rows as usize;
                    if old_rows + num >= 32 { return false; }
                    let spawned_so_far = 100usize - unspawned;
                    for r in (0..old_rows).rev() { b.data[r + num] = b.data[r]; }
                    for k in 0..num {
                        b.data[num - 1 - k] = 0xFC00 | (0x3FF & !(1u16 << holes[spawned_so_far + k]));
                    }
                    b.rows = (old_rows + num) as u8;
                    b.garbage_rows = (g + num) as u8;
                    true
                };

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
                let mut seen: HashSet<OKey, FxBuild> = HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

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
                    if q_len == 0 { return 400; }
                    let cur0 = Some(q[0]);
                    let alt0 = hold0.or_else(|| q.get(1).copied());
                    pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
                    if root_places.is_empty() { return 400; }

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
                        if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                        let (held, qc) = if hold0.is_some() {
                            if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                        } else if fp.did_hold { (cur0, 2.min(q_len)) } else { (None, 1.min(q_len)) };
                        if rem_g == 0 && b.rows < 20 {
                            let gr = -30_000_000 + 100 + b.rows as i64;
                            if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                        }
                        let in_h = held.or_else(|| q.get(qc).copied());
                        let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                        if seen.insert(OKey { board: b, held, rem_g }) {
                            cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, first_idx: idx as u16, score: sc });
                        }
                    }
                    for depth in 2..=target_d {
                        children.clear();
                        for node in &cur_beam {
                            if node.rem_g == 0 { continue; }
                            let qc = node.q_cursor;
                            let (c_nh, c_h) = if node.held.is_some() { (q.get(qc).copied(), node.held) } else { (q.get(qc).copied(), q.get(qc + 1).copied()) };
                            pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                            for &fp in &tmp_places {
                                let mut b = node.board;
                                let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                                let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                                let rem_g = node.rem_g.saturating_sub(ds as u16);
                                if rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) { continue; }
                                let (held, nqc) = if node.held.is_some() {
                                    if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                                } else if fp.did_hold { (c_nh, (qc + 2).min(q_len)) } else { (None, (qc + 1).min(q_len)) };
                                if rem_g == 0 && b.rows < 20 {
                                    let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                                    if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                                }
                                let in_h = held.or_else(|| q.get(nqc).copied());
                                let sc = mdl.evaluate_board_with_bag(&b, rem_g, in_h, 0x7F);
                                children.push(ONode { board: b, held, q_cursor: nqc, rem_g, first_idx: node.first_idx, score: sc });
                            }
                        }
                        if children.is_empty() { break; }
                        children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                        cur_beam.clear();
                        seen.clear();
                        let mut rc = vec![0usize; root_places.len()];
                        for &ch in &children {
                            if cur_beam.len() >= eff_w { break; }
                            if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                            if seen.insert(OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g }) {
                                rc[ch.first_idx as usize] += 1;
                                cur_beam.push(ch);
                            }
                        }
                    }
                    let chosen_idx = best_goal.map(|x| x.1 as usize).unwrap_or_else(|| cur_beam.first().map_or(0, |n| n.first_idx as usize));
                    let fp = root_places[chosen_idx];
                    let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
                    inputs.pop();
                    if fp.did_hold { if hold0.is_none() { q_pos += 2; } else { q_pos += 1; } } else { q_pos += 1; }
                    stacker.run(inputs);
                    let (_, ds) = stacker.hard_drop();
                    moves += 1;
                    ds_total += ds;
                }
                if ds_total < 100 { 400 } else { moves }
            }).collect()
    };
    let pcs: Vec<usize> = std::thread::scope(|sc| {
        let h = sc.spawn(|| run_slice(e1));
        let mut r2 = run_slice(e2);
        let mut r1 = h.join().unwrap();
        r1.append(&mut r2);
        r1
    });
    let mean = pcs.iter().sum::<usize>() as f64 / pcs.len() as f64;
    println!(
        "{:<24} mean={:6.2} per_seed={:?} dt={:.1}s",
        label, mean, pcs, t0.elapsed().as_secs_f64()
    );
}

fn ridge_warmstart_linear(
    mdl: &mut SpatialValueModel,
    all_states: &[StateFeat],
    val_samples: &[ValueSample],
    train_vals: &[usize],
) {
    const P: usize = 1 + SPARSE_DIM + DENSE_DIM;
    let mut a = vec![0.0f64; P * P];
    let mut b = vec![0.0f64; P];
    let mut x = [0.0f64; P];
    let mut nz: Vec<usize> = Vec::with_capacity(128);

    for &vi in train_vals {
        let smp = &val_samples[vi];
        let sf = &all_states[smp.state_idx];
        let y = smp.target as f64;
        nz.clear();
        x[0] = 1.0;
        nz.push(0);
        for &bit in &sf.active {
            let idx = 1 + bit as usize;
            x[idx] = 1.0;
            nz.push(idx);
        }
        for j in 0..DENSE_DIM {
            let z = ((sf.dense[j] - mdl.dense_mean[j]) * mdl.dense_inv_std[j]) as f64;
            if z != 0.0 {
                let idx = 1 + SPARSE_DIM + j;
                x[idx] = z;
                nz.push(idx);
            }
        }
        for (ii, &ia) in nz.iter().enumerate() {
            let xa = x[ia];
            b[ia] += xa * y;
            for &ib in &nz[ii..] {
                a[ia * P + ib] += xa * x[ib];
            }
        }
        for &bit in &sf.active {
            x[1 + bit as usize] = 0.0;
        }
        for j in 0..DENSE_DIM {
            x[1 + SPARSE_DIM + j] = 0.0;
        }
    }

    for i in 0..P {
        for j in 0..i {
            a[i * P + j] = a[j * P + i];
        }
    }
    let lambda = (train_vals.len() as f64 * 0.005).max(20.0);
    a[0] += 1e-4;
    for k in 1..P {
        let reg = if k == 1 + SPARSE_DIM { 1.0 } else { lambda };
        a[k * P + k] += reg;
    }

    // Gaussian elimination with partial pivoting
    for col in 0..P {
        let mut piv = col;
        let mut max_v = a[col * P + col].abs();
        for r in (col + 1)..P {
            let v = a[r * P + col].abs();
            if v > max_v {
                max_v = v;
                piv = r;
            }
        }
        if piv != col {
            for c in col..P {
                a.swap(col * P + c, piv * P + c);
            }
            b.swap(col, piv);
        }
        let diag = a[col * P + col];
        if diag.abs() < 1e-12 {
            continue;
        }
        let inv = 1.0 / diag;
        for r in (col + 1)..P {
            let fac = a[r * P + col] * inv;
            if fac == 0.0 {
                continue;
            }
            for c in col..P {
                a[r * P + c] -= fac * a[col * P + c];
            }
            b[r] -= fac * b[col];
        }
    }
    let mut w = vec![0.0f64; P];
    for i in (0..P).rev() {
        let mut s = b[i];
        for c in (i + 1)..P {
            s -= a[i * P + c] * w[c];
        }
        let diag = a[i * P + i];
        w[i] = if diag.abs() >= 1e-12 { s / diag } else { 0.0 };
    }

    mdl.lin_bias = w[0] as f32;
    for i in 0..SPARSE_DIM {
        mdl.lin_sparse[i] = w[1 + i] as f32;
    }
    for j in 0..DENSE_DIM {
        mdl.lin_dense[j] = w[1 + SPARSE_DIM + j] as f32;
    }
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
    let mut epochs: usize = 40;
    let mut batch_size: usize = 128;
    let mut lr: f32 = 0.0025;
    let mut wd: f32 = 1e-4;
    let mut rank_w: f32 = 1.5;
    let mut seed_base: u64 = 1000;
    let mut eval_seeds: usize = 5;

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
            "--epochs" => { epochs = args[i + 1].parse().unwrap(); i += 2; }
            "--batch" => { batch_size = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i + 1].parse().unwrap(); i += 2; }
            "--wd" => { wd = args[i + 1].parse().unwrap(); i += 2; }
            "--rank-w" => { rank_w = args[i + 1].parse().unwrap(); i += 2; }
            "--seed-base" => { seed_base = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-seeds" => { eval_seeds = args[i + 1].parse().unwrap(); i += 2; }
            other => panic!("Unknown arg: {}", other),
        }
    }

    let teacher = if let Some(p) = teacher_arg.strip_prefix("spatial:") {
        Teacher::Spatial(Arc::new(SpatialValueModel::load_from_file(p).expect("load spatial teacher")))
    } else if let Some(p) = teacher_arg.strip_prefix("learned:") {
        Teacher::Learned(Arc::new(load_learned_model(p, true)))
    } else {
        panic!("--teacher must start with learned: or spatial:");
    };

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .min(games.max(1));
    println!(
        "=== SpatialValueModel Expert Iteration ===\n  teacher={} games={} seeds=[{}..{}] H={} B={} MPR={} threads={}",
        teacher_arg, games, seed_base, seed_base + games as u64, horizon, beam_w, max_per_root, n_threads
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

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
                        .map(|&s| generate_game_data(s, horizon, beam_w, max_per_root, t_ref))
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
    let topouts = game_results.len() - valid_games.len();
    let mean_pcs = valid_games.iter().map(|g| g.pieces).sum::<usize>() as f64
        / valid_games.len().max(1) as f64;
    let min_pcs = valid_games.iter().map(|g| g.pieces).min().unwrap_or(0);
    let max_pcs = valid_games.iter().map(|g| g.pieces).max().unwrap_or(0);
    println!(
        "Generated {} valid games ({} topouts) in {:.1}s | teacher mean={:.2} [{}..{}]",
        valid_games.len(),
        topouts,
        t_gen.elapsed().as_secs_f64(),
        mean_pcs,
        min_pcs,
        max_pcs
    );

    let val_start = (valid_games.len() * 9) / 10;
    let mut all_states: Vec<StateFeat> = Vec::new();
    let mut val_samples: Vec<ValueSample> = Vec::new();
    let mut pairs: Vec<RankPair> = Vec::new();
    for (gi, g) in valid_games.iter().enumerate() {
        let is_val = gi >= val_start;
        let base = all_states.len();
        all_states.extend(g.states.iter().cloned());
        for &(si, targ) in &g.val_targets {
            val_samples.push(ValueSample {
                state_idx: base + si,
                target: targ,
                is_val,
            });
        }
        for &(wi, li, dq, is_trap) in &g.pairs {
            pairs.push(RankPair {
                win_idx: base + wi,
                lose_idx: base + li,
                dq,
                is_trap,
                is_val,
            });
        }
    }

    let train_vals: Vec<usize> = (0..val_samples.len()).filter(|&i| !val_samples[i].is_val).collect();
    let valid_vals: Vec<usize> = (0..val_samples.len()).filter(|&i| val_samples[i].is_val).collect();
    let train_pairs: Vec<usize> = (0..pairs.len()).filter(|&i| !pairs[i].is_val).collect();
    let valid_pairs: Vec<usize> = (0..pairs.len()).filter(|&i| pairs[i].is_val).collect();
    println!(
        "Dataset: states={} train_vals={} val_vals={} train_pairs={} val_pairs={}",
        all_states.len(),
        train_vals.len(),
        valid_vals.len(),
        train_pairs.len(),
        valid_pairs.len()
    );

    let mut mdl = if let Some(ref ip) = init_path {
        let mut m = SpatialValueModel::load_from_file(ip).expect("load init model");
        m.fast_ready = false;
        m
    } else {
        let mut m = SpatialValueModel::default();
        // Compute dense mean & inv_std on training trajectory samples
        let mut mean = [0.0f64; DENSE_DIM];
        let mut var = [0.0f64; DENSE_DIM];
        let n_tr = train_vals.len().max(1) as f64;
        for &vi in &train_vals {
            let s = &all_states[val_samples[vi].state_idx];
            for j in 0..DENSE_DIM {
                mean[j] += s.dense[j] as f64;
            }
        }
        for j in 0..DENSE_DIM {
            mean[j] /= n_tr;
        }
        for &vi in &train_vals {
            let s = &all_states[val_samples[vi].state_idx];
            for j in 0..DENSE_DIM {
                let d = s.dense[j] as f64 - mean[j];
                var[j] += d * d;
            }
        }
        for j in 0..DENSE_DIM {
            m.dense_mean[j] = mean[j] as f32;
            let std = (var[j] / n_tr).sqrt().max(1e-3);
            m.dense_inv_std[j] = (1.0 / std) as f32;
        }
        ridge_warmstart_linear(&mut m, &all_states, &val_samples, &train_vals);
        let mut rng = Xorshift64(0x9E3779B97F4A7C15 ^ seed_base);
        let s1 = (2.0f32 / (DENSE_DIM as f32 + 40.0)).sqrt() * 0.5;
        let s2 = (2.0f32 / SPATIAL_H1 as f32).sqrt() * 0.5;
        let s3 = (2.0f32 / SPATIAL_H2 as f32).sqrt() * 0.15;
        for i in 0..SPARSE_DIM {
            for h in 0..SPATIAL_H1 {
                m.w_sparse[i][h] = rng.next_normal() * s1 * 0.5;
            }
        }
        for j in 0..DENSE_DIM {
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
        m
    };
    mdl.fast_ready = false;

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
        // Cosine LR schedule with 2-epoch warmup
        let lr_cur = if ep == 0 {
            0.0
        } else if ep <= 2 {
            lr * (ep as f32 / 2.0)
        } else {
            let progress = (ep as f32) / (epochs as f32);
            lr * (0.05 + 0.95 * 0.5 * (1.0 + (std::f32::consts::PI * progress).cos()))
        };

        if ep > 0 {
        // Shuffle train indices
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
        let huber_delta = 2.0f32;
        let tau = 0.25f32;

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
                let g_v = if diff.abs() <= huber_delta {
                    diff
                } else {
                    huber_delta * diff.signum()
                } / v_cnt;
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
                    let margin = pr.dq.clamp(0.15, 1.50);
                    let u = ((yw - yl + margin) / tau).clamp(-20.0, 20.0);
                    let sig = 1.0 / (1.0 + (-u).exp());
                    let trap_mult = if pr.is_trap { 2.0 } else { 1.0 };
                    let g_diff = (rank_w * trap_mult * sig) / p_cnt;
                    backward_add(&mdl, sw, &z_w, &a1_w, &a2_w, g_diff, &mut grad);
                    backward_add(&mdl, sl, &z_l, &a1_l, &a2_l, -g_diff, &mut grad);
                }
            }

            opt.apply(&mut mdl, &grad, lr_cur);
        }
        }

        // Evaluate validation metrics
        let tau = 0.25f32;
        let mut val_mae = 0.0f32;
        for &vi in &valid_vals {
            let smp = &val_samples[vi];
            let pred = forward_cache(&mdl, &all_states[smp.state_idx], &mut z_w, &mut a1_w, &mut a2_w);
            val_mae += (pred - smp.target).abs();
        }
        val_mae /= valid_vals.len().max(1) as f32;

        let mut pair_ok = 0usize;
        let mut trap_ok = 0usize;
        let mut trap_tot = 0usize;
        let mut rank_loss_sum = 0.0f32;
        for &pi in &valid_pairs {
            let pr = &pairs[pi];
            let yw = forward_cache(&mdl, &all_states[pr.win_idx], &mut z_w, &mut a1_w, &mut a2_w);
            let yl = forward_cache(&mdl, &all_states[pr.lose_idx], &mut z_l, &mut a1_l, &mut a2_l);
            if yw < yl {
                pair_ok += 1;
            }
            if pr.is_trap {
                trap_tot += 1;
                if yw < yl {
                    trap_ok += 1;
                }
            }
            let margin = pr.dq.clamp(0.15, 1.50);
            let u = ((yw - yl + margin) / tau).clamp(-20.0, 20.0);
            rank_loss_sum += tau * (1.0 + u.exp()).ln();
        }
        let pair_acc = 100.0 * pair_ok as f32 / valid_pairs.len().max(1) as f32;
        let trap_acc = 100.0 * trap_ok as f32 / trap_tot.max(1) as f32;
        let rank_l = rank_loss_sum / valid_pairs.len().max(1) as f32;
        let comb = val_mae + rank_w * rank_l;
        if comb < best_score {
            best_score = comb;
            best_mdl = mdl.clone();
        }
        if ep <= 1 || ep % 5 == 0 || ep == epochs {
            println!(
                "  ep={:2}/{} lr={:.5} | val_mae={:.3} pair_acc={:.1}% trap_acc={:.1}% rank_l={:.3} comb={:.3}",
                ep, epochs, lr_cur, val_mae, pair_acc, trap_acc, rank_l, comb
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }
    }

    best_mdl.prepare_fast();
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

    if eval_seeds > 0 {
        let mdl_arc = Arc::new(best_mdl);
        let test_seeds: Vec<u64> = (101..101 + eval_seeds as u64).collect();
        println!("\n=== Post-Training Held-Out Verification (seeds 101..{}) ===", 101 + eval_seeds);
        eval_oracle_quick("OR_p12_b140_spatial", &test_seeds, 12, 140, &mdl_arc);
        eval_legal_quick("LEGAL_p5_b120_spatial", &test_seeds, &mdl_arc);
    }
}
