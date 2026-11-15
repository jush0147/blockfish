use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{
    ai::{
        cheese_val::{
            extract_features_opt, extract_features_with_bag, FEAT_DIM, FEAT_NAMES, MLP_HIDDEN,
        },
        fast::{FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
        fast_eval::{eval_fast_with_bag, EvalWeights},
        BeamConfig, FastAI, LearnedValueModel,
    },
    StackerExt as _,
};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::sync::Arc;
use std::time::Instant;

const CRN_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_models_latest.txt");
const GROUPS_CACHE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r1.bin");
const R2_GROUPS_CACHE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r2.bin");
const R3_GROUPS_CACHE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r3.bin");
const R4_GROUPS_CACHE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r4.bin");

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

fn clean_g0_weights() -> EvalWeights {
    let mut w = EvalWeights::default();
    w.garbage_row = 420;
    w.pe_net_mode = true;
    w.piece_estimate = 75;
    w.stack_cell = -4;
    w.upper_cell = 0;
    w.h1_block = 22;
    w.h1_empty_blocking = 8;
    w.h1_height = 6;
    w.h2_block = 15;
    w.h2_extra_empty = 6;
    w.h3_block = 8;
    w.h3_extra_empty = 0;
    w.h4_block = 10;
    w.covered_hole = 60;
    w.hole_row = 20;
    w.blocks_over_hole = 15;
    w.open_h1_bonus = 45;
    w.combo2_bonus = 60;
    w.h1_h2_adj = 0;
    w.bumpiness = 0;
    w.row_transition = 14;
    w.near_full_9_bonus = 40;
    w.row_g_aligned_bonus = 40;
    w.height_limit = 16;
    w.g0_row0_cell = 16;
    w.hold_bonus = [0, 75, -18, 4, 4, -10, -8];
    w
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

#[derive(Clone, Copy)]
struct CrnStat {
    steps: f64,
    ds_cleared: f64,
    rem_g_end: f64,
    stack_cells_end: f64,
    covered_holes_end: f64,
    h1_blocks_end: f64,
    topout: bool,
}

#[inline]
fn eval_teacher(
    board: &FastBoard,
    rem_g: u16,
    in_hand: Option<u8>,
    avail: u8,
    w: &EvalWeights,
    model: Option<&LearnedValueModel>,
) -> i64 {
    if let Some(m) = model {
        m.evaluate_board_with_bag(board, rem_g, in_hand, avail)
    } else {
        eval_fast_with_bag(board, rem_g, in_hand, avail, w)
    }
}

/// Evaluates one leaf board using 2-ply lookahead over a shared CRN piece sequence & cheese-spawn RNG.
fn rollout_crn_2ply(
    fst: &FastShapeTable,
    pgen: &mut FastPlaceGen,
    places1: &mut Vec<FastPlace>,
    places2: &mut Vec<FastPlace>,
    cand1: &mut Vec<(i64, FastBoard, Option<u8>, u16, u8)>,
    mut board: FastBoard,
    in_hand0: Option<u8>,
    mut rem_g: u16,
    world_pieces: &[u8],
    spawn_seed: u64,
    w: &EvalWeights,
    model: Option<&LearnedValueModel>,
) -> CrnStat {
    let mut held = in_hand0;
    let mut rng = spawn_seed;
    let h = world_pieces.len();
    let mut steps_done = 0usize;
    let start_rem_g = rem_g;

    if rem_g > 0 && (board.garbage_rows as usize) < 4 {
        if !board.spawn_cheese_to(rem_g, 4, &mut rng) {
            return CrnStat {
                steps: 0.0,
                ds_cleared: 0.0,
                rem_g_end: rem_g as f64,
                stack_cells_end: 60.0,
                covered_holes_end: 10.0,
                h1_blocks_end: 5.0,
                topout: true,
            };
        }
    }

    for step in 0..h {
        if rem_g == 0 {
            break;
        }
        let cur_p = world_pieces[step];
        let nxt_p = world_pieces.get(step + 1).copied();
        let mut avail1 = 0u8;
        for &p in &world_pieces[(step + 1)..h.min(step + 4)] {
            avail1 |= 1u8 << p;
        }
        if step + 1 >= h {
            avail1 = 0x7F;
        }

        pgen.generate_for_state(fst, &board, Some(cur_p), held, places1);
        if places1.is_empty() {
            return CrnStat {
                steps: steps_done as f64,
                ds_cleared: (start_rem_g - rem_g) as f64,
                rem_g_end: rem_g as f64,
                stack_cells_end: 60.0,
                covered_holes_end: 10.0,
                h1_blocks_end: 5.0,
                topout: true,
            };
        }

        cand1.clear();
        for &fp in places1.iter() {
            let mut b1 = board;
            let orient = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b1.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem1 = rem_g.saturating_sub(ds as u16);
            let h1 = if fp.color_idx == cur_p { held } else { Some(cur_p) };
            let sc1 = eval_teacher(&b1, rem1, h1, avail1, w, model);
            cand1.push((sc1, b1, h1, rem1, lines));
        }
        cand1.sort_unstable_by_key(|x| x.0);

        let top_k = cand1.len().min(4);
        let mut best_2ply = i64::MAX;
        let mut chosen: Option<(FastBoard, Option<u8>, u16, u8)> = None;

        if let Some(np) = nxt_p {
            let mut avail2 = 0u8;
            for &p in &world_pieces[(step + 2)..h.min(step + 5)] {
                avail2 |= 1u8 << p;
            }
            if step + 2 >= h {
                avail2 = 0x7F;
            }

            for ci in 0..top_k {
                let (sc1, b1, h1, rem1, lines1) = cand1[ci];
                if rem1 == 0 && b1.rows < 20 {
                    chosen = Some((b1, h1, rem1, lines1));
                    break;
                }
                let mut b1_sp = b1;
                let mut r_ply1 = rng;
                if rem1 > 0 && !b1_sp.spawn_cheese_rows(rem1, lines1, &mut r_ply1) {
                    continue;
                }
                pgen.generate_for_state(fst, &b1_sp, Some(np), h1, places2);
                let mut best_child = i64::MAX;
                for &fp2 in places2.iter() {
                    let mut b2 = b1_sp;
                    let o2 = &fst.shapes[fp2.color_idx as usize].orients[fp2.tf.2 as usize];
                    let (_, ds2, _) = b2.place_and_sift(o2, fp2.norm_i, fp2.norm_j);
                    let rem2 = rem1.saturating_sub(ds2 as u16);
                    let h2 = if fp2.color_idx == np { h1 } else { Some(np) };
                    let sc2 = eval_teacher(&b2, rem2, h2, avail2, w, model);
                    if sc2 < best_child {
                        best_child = sc2;
                    }
                }
                // Fall back gently if no 2-ply survived
                let comb = if best_child == i64::MAX { sc1 + 500_000 } else { best_child };
                if comb < best_2ply {
                    best_2ply = comb;
                    chosen = Some((b1, h1, rem1, lines1));
                }
            }
        }

        let (mut next_b, next_h, next_rem, lines) = match chosen {
            Some(c) => c,
            None => {
                let (_, b1, h1, rem1, lines1) = cand1[0];
                (b1, h1, rem1, lines1)
            }
        };

        steps_done += 1;
        if next_rem > 0 {
            if !next_b.spawn_cheese_rows(next_rem, lines, &mut rng) {
                return CrnStat {
                    steps: steps_done as f64,
                    ds_cleared: (start_rem_g - next_rem) as f64,
                    rem_g_end: next_rem as f64,
                    stack_cells_end: 60.0,
                    covered_holes_end: 10.0,
                    h1_blocks_end: 5.0,
                    topout: true,
                };
            }
        }
        board = next_b;
        held = next_h;
        rem_g = next_rem;
    }

    let mut tmp_f = [0.0f32; FEAT_DIM];
    extract_features_with_bag(&board, rem_g, held, 0x7F, &mut tmp_f);
    CrnStat {
        steps: steps_done as f64,
        ds_cleared: (start_rem_g - rem_g) as f64,
        rem_g_end: rem_g as f64,
        stack_cells_end: tmp_f[9] as f64,
        covered_holes_end: tmp_f[20] as f64,
        h1_blocks_end: tmp_f[24] as f64,
        topout: false,
    }
}

#[derive(Clone, Copy)]
struct BeamRollNode {
    board: FastBoard,
    held: Option<u8>,
    rem_g: u16,
    score: i64,
}

struct BeamRollScratch {
    beam: Vec<BeamRollNode>,
    next: Vec<BeamRollNode>,
    places: Vec<FastPlace>,
}

impl BeamRollScratch {
    fn new() -> Self {
        Self {
            beam: Vec::with_capacity(16),
            next: Vec::with_capacity(512),
            places: Vec::with_capacity(64),
        }
    }
}

#[inline]
fn quick_stack_cells(b: &FastBoard) -> usize {
    let mut cnt = 0usize;
    for r in (b.garbage_rows as usize)..(b.rows as usize) {
        cnt += (b.data[r] & 0x3FF).count_ones() as usize;
    }
    cnt
}

fn rollout_crn_beam(
    fst: &FastShapeTable,
    pgen: &mut FastPlaceGen,
    sc: &mut BeamRollScratch,
    mut board: FastBoard,
    in_hand0: Option<u8>,
    rem_g: u16,
    world_pieces: &[u8],
    spawn_seed: u64,
    w: &EvalWeights,
    model: Option<&LearnedValueModel>,
    beam_width: usize,
) -> CrnStat {
    let mut rng = spawn_seed;
    let h = world_pieces.len();
    let start_rem_g = rem_g;

    if rem_g > 0 && (board.garbage_rows as usize) < 4 {
        if !board.spawn_cheese_to(rem_g, 4, &mut rng) {
            return CrnStat {
                steps: 0.0,
                ds_cleared: 0.0,
                rem_g_end: rem_g as f64,
                stack_cells_end: 60.0,
                covered_holes_end: 10.0,
                h1_blocks_end: 5.0,
                topout: true,
            };
        }
    }

    sc.beam.clear();
    sc.beam.push(BeamRollNode {
        board,
        held: in_hand0,
        rem_g,
        score: 0,
    });

    let mut best_done: Option<(f64, CrnStat)> = None;
    let mut last_alive_step = 0usize;

    for step in 0..h {
        if sc.beam.is_empty() {
            break;
        }
        let cur_p = world_pieces[step];
        let mut avail = 0u8;
        for &p in &world_pieces[(step + 1)..h.min(step + 4)] {
            avail |= 1u8 << p;
        }
        if step + 1 >= h {
            avail = 0x7F;
        }
        let step_rng = rng ^ ((step as u64) + 1).wrapping_mul(0x517CC1B727220A95);
        sc.next.clear();

        for bi in 0..sc.beam.len() {
            let node = sc.beam[bi];
            if node.rem_g == 0 {
                continue;
            }
            pgen.generate_for_state(fst, &node.board, Some(cur_p), node.held, &mut sc.places);
            for &fp in &sc.places {
                let mut b1 = node.board;
                let orient = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b1.place_and_sift(orient, fp.norm_i, fp.norm_j);
                let rem1 = node.rem_g.saturating_sub(ds as u16);
                let h1 = if fp.color_idx == cur_p { node.held } else { Some(cur_p) };

                if rem1 == 0 && b1.rows < 20 {
                    let sc_end = quick_stack_cells(&b1) as f64;
                    let cost = (step + 1) as f64 + 0.06 * sc_end;
                    if best_done.as_ref().map_or(true, |(bc, _)| cost < *bc) {
                        best_done = Some((
                            cost,
                            CrnStat {
                                steps: (step + 1) as f64,
                                ds_cleared: start_rem_g as f64,
                                rem_g_end: 0.0,
                                stack_cells_end: sc_end,
                                covered_holes_end: 0.0,
                                h1_blocks_end: 0.0,
                                topout: false,
                            },
                        ));
                    }
                    continue;
                }

                if rem1 > 0 {
                    let mut r_sp = step_rng;
                    if !b1.spawn_cheese_rows(rem1, lines, &mut r_sp) {
                        continue;
                    }
                }
                let val = eval_teacher(&b1, rem1, h1, avail, w, model);
                if val >= 4_000_000 {
                    continue;
                }
                sc.next.push(BeamRollNode {
                    board: b1,
                    held: h1,
                    rem_g: rem1,
                    score: val,
                });
            }
        }

        if sc.next.is_empty() {
            sc.beam.clear();
            break;
        }
        last_alive_step = step + 1;
        sc.next.sort_unstable_by_key(|n| n.score);
        sc.beam.clear();
        for &ch in &sc.next {
            if sc.beam.len() >= beam_width {
                break;
            }
            if !sc
                .beam
                .iter()
                .any(|x| x.board == ch.board && x.held == ch.held && x.rem_g == ch.rem_g)
            {
                sc.beam.push(ch);
            }
        }
    }

    if let Some((_, st)) = best_done {
        return st;
    }

    if let Some(best) = sc.beam.first() {
        let mut tmp_f = [0.0f32; FEAT_DIM];
        extract_features_opt(&best.board, best.rem_g, best.held, 0x7F, false, &mut tmp_f);
        CrnStat {
            steps: h as f64,
            ds_cleared: (start_rem_g - best.rem_g) as f64,
            rem_g_end: best.rem_g as f64,
            stack_cells_end: tmp_f[9] as f64,
            covered_holes_end: tmp_f[20] as f64,
            h1_blocks_end: tmp_f[24] as f64,
            topout: false,
        }
    } else {
        CrnStat {
            steps: last_alive_step as f64,
            ds_cleared: 0.0,
            rem_g_end: start_rem_g as f64,
            stack_cells_end: 60.0,
            covered_holes_end: 10.0,
            h1_blocks_end: 5.0,
            topout: true,
        }
    }
}

#[inline]
fn spawn_cheese_from_stream_local(
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

fn rollout_crn_terminal(
    fst: &FastShapeTable,
    pgen: &mut FastPlaceGen,
    sc: &mut BeamRollScratch,
    mut board: FastBoard,
    in_hand0: Option<u8>,
    rem_g: u16,
    world_pieces: &[u8],
    world_holes: &[u16],
    w: &EvalWeights,
    model: Option<&LearnedValueModel>,
    beam_width: usize,
) -> CrnStat {
    let h = world_pieces.len();
    let start_rem_g = rem_g;
    let unspawned_0 = (rem_g as usize).saturating_sub(board.garbage_rows as usize);

    if rem_g > 0 && (board.garbage_rows as usize) < 4 {
        if !spawn_cheese_from_stream_local(&mut board, rem_g, 1, unspawned_0, world_holes) {
            return CrnStat {
                steps: 0.0,
                ds_cleared: 0.0,
                rem_g_end: rem_g as f64,
                stack_cells_end: 60.0,
                covered_holes_end: 10.0,
                h1_blocks_end: 5.0,
                topout: true,
            };
        }
    }

    sc.beam.clear();
    sc.beam.push(BeamRollNode {
        board,
        held: in_hand0,
        rem_g,
        score: 0,
    });

    let mut best_done: Option<(f64, CrnStat)> = None;
    let mut last_alive_step = 0usize;

    for step in 0..h {
        if sc.beam.is_empty() {
            break;
        }
        let cur_p = world_pieces[step];
        let mut avail = 0u8;
        for &p in &world_pieces[(step + 1)..h.min(step + 4)] {
            avail |= 1u8 << p;
        }
        if step + 1 >= h {
            avail = 0x7F;
        }
        sc.next.clear();

        for bi in 0..sc.beam.len() {
            let node = sc.beam[bi];
            if node.rem_g == 0 {
                continue;
            }
            pgen.generate_for_state(fst, &node.board, Some(cur_p), node.held, &mut sc.places);
            for &fp in &sc.places {
                let mut b1 = node.board;
                let orient = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b1.place_and_sift(orient, fp.norm_i, fp.norm_j);
                let rem1 = node.rem_g.saturating_sub(ds as u16);
                let h1 = if fp.color_idx == cur_p { node.held } else { Some(cur_p) };

                if rem1 == 0 && b1.rows < 20 {
                    let sc_end = quick_stack_cells(&b1) as f64;
                    let cost = (step + 1) as f64 + 0.005 * sc_end;
                    if best_done.as_ref().map_or(true, |(bc, _)| cost < *bc) {
                        best_done = Some((
                            cost,
                            CrnStat {
                                steps: (step + 1) as f64,
                                ds_cleared: start_rem_g as f64,
                                rem_g_end: 0.0,
                                stack_cells_end: sc_end,
                                covered_holes_end: 0.0,
                                h1_blocks_end: 0.0,
                                topout: false,
                            },
                        ));
                    }
                    continue;
                }

                if rem1 > 0
                    && !spawn_cheese_from_stream_local(&mut b1, rem1, lines, unspawned_0, world_holes)
                {
                    continue;
                }
                let val = eval_teacher(&b1, rem1, h1, avail, w, model);
                if val >= 4_000_000 {
                    continue;
                }
                sc.next.push(BeamRollNode {
                    board: b1,
                    held: h1,
                    rem_g: rem1,
                    score: val,
                });
            }
        }

        if let Some((_, st)) = best_done {
            return st;
        }
        if sc.next.is_empty() {
            sc.beam.clear();
            break;
        }
        last_alive_step = step + 1;
        sc.next.sort_unstable_by_key(|n| n.score);
        sc.beam.clear();
        for &ch in &sc.next {
            if sc.beam.len() >= beam_width {
                break;
            }
            if !sc
                .beam
                .iter()
                .any(|x| x.board == ch.board && x.held == ch.held && x.rem_g == ch.rem_g)
            {
                sc.beam.push(ch);
            }
        }
    }

    if let Some(best) = sc.beam.first() {
        let mut tmp_f = [0.0f32; FEAT_DIM];
        extract_features_opt(&best.board, best.rem_g, best.held, 0x7F, false, &mut tmp_f);
        CrnStat {
            steps: h as f64,
            ds_cleared: (start_rem_g - best.rem_g) as f64,
            rem_g_end: best.rem_g as f64,
            stack_cells_end: tmp_f[9] as f64,
            covered_holes_end: tmp_f[20] as f64,
            h1_blocks_end: tmp_f[24] as f64,
            topout: false,
        }
    } else {
        CrnStat {
            steps: last_alive_step as f64,
            ds_cleared: 0.0,
            rem_g_end: start_rem_g as f64,
            stack_cells_end: 60.0,
            covered_holes_end: 10.0,
            h1_blocks_end: 5.0,
            topout: true,
        }
    }
}

struct LeafRec {
    board: FastBoard,
    in_hand: Option<u8>,
    rem_g: u16,
    bag_mask: u8,
    stats: Vec<CrnStat>,
}

struct SiblingGroup {
    seed: u64,
    leaves: Vec<LeafRec>,
}

fn save_groups(groups: &[SiblingGroup], path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut w = BufWriter::new(File::create(path).expect("create groups cache"));
    let ng = groups.len() as u32;
    w.write_all(&ng.to_le_bytes()).unwrap();
    for g in groups {
        w.write_all(&g.seed.to_le_bytes()).unwrap();
        let nl = g.leaves.len() as u8;
        w.write_all(&[nl]).unwrap();
        for l in &g.leaves {
            w.write_all(&[
                l.board.rows,
                l.board.garbage_rows,
                l.in_hand.unwrap_or(255),
                l.bag_mask,
            ])
            .unwrap();
            w.write_all(&l.rem_g.to_le_bytes()).unwrap();
            for r in 0..l.board.rows as usize {
                w.write_all(&l.board.data[r].to_le_bytes()).unwrap();
            }
            let ns = l.stats.len() as u8;
            w.write_all(&[ns]).unwrap();
            for st in &l.stats {
                w.write_all(&st.steps.to_le_bytes()).unwrap();
                w.write_all(&st.ds_cleared.to_le_bytes()).unwrap();
                w.write_all(&st.rem_g_end.to_le_bytes()).unwrap();
                w.write_all(&st.stack_cells_end.to_le_bytes()).unwrap();
                w.write_all(&st.covered_holes_end.to_le_bytes()).unwrap();
                w.write_all(&st.h1_blocks_end.to_le_bytes()).unwrap();
                w.write_all(&[u8::from(st.topout)]).unwrap();
            }
        }
    }
}

fn load_groups(path: &str) -> Vec<SiblingGroup> {
    let mut r = BufReader::new(File::open(path).expect("open groups cache"));
    let mut b4 = [0u8; 4];
    let mut b8 = [0u8; 8];
    let mut b2 = [0u8; 2];
    let mut b1 = [0u8; 1];
    r.read_exact(&mut b4).unwrap();
    let ng = u32::from_le_bytes(b4) as usize;
    let mut groups = Vec::with_capacity(ng);
    for _ in 0..ng {
        r.read_exact(&mut b8).unwrap();
        let seed = u64::from_le_bytes(b8);
        r.read_exact(&mut b1).unwrap();
        let nl = b1[0] as usize;
        let mut leaves = Vec::with_capacity(nl);
        for _ in 0..nl {
            r.read_exact(&mut b4).unwrap();
            let mut board = FastBoard::empty();
            board.rows = b4[0];
            board.garbage_rows = b4[1];
            let in_hand = if b4[2] == 255 { None } else { Some(b4[2]) };
            let bag_mask = b4[3];
            r.read_exact(&mut b2).unwrap();
            let rem_g = u16::from_le_bytes(b2);
            for row in 0..board.rows as usize {
                r.read_exact(&mut b2).unwrap();
                board.data[row] = u16::from_le_bytes(b2);
            }
            r.read_exact(&mut b1).unwrap();
            let ns = b1[0] as usize;
            let mut stats = Vec::with_capacity(ns);
            let mut rd_f64 = |rd: &mut BufReader<File>| -> f64 {
                rd.read_exact(&mut b8).unwrap();
                f64::from_le_bytes(b8)
            };
            for _ in 0..ns {
                let steps = rd_f64(&mut r);
                let ds_cleared = rd_f64(&mut r);
                let rem_g_end = rd_f64(&mut r);
                let stack_cells_end = rd_f64(&mut r);
                let covered_holes_end = rd_f64(&mut r);
                let h1_blocks_end = rd_f64(&mut r);
                r.read_exact(&mut b1).unwrap();
                stats.push(CrnStat {
                    steps,
                    ds_cleared,
                    rem_g_end,
                    stack_cells_end,
                    covered_holes_end,
                    h1_blocks_end,
                    topout: b1[0] != 0,
                });
            }
            leaves.push(LeafRec {
                board,
                in_hand,
                rem_g,
                bag_mask,
                stats,
            });
        }
        groups.push(SiblingGroup { seed, leaves });
    }
    groups
}

fn collect_groups_for_seed(
    seed: u64,
    cfg: &BeamConfig,
    fst: &FastShapeTable,
    w_teacher: &EvalWeights,
    teacher_model: Option<&LearnedValueModel>,
    use_beam_teacher: bool,
) -> Vec<SiblingGroup> {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);

    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());
    let mut pgen = FastPlaceGen::new();
    let mut places1 = Vec::with_capacity(64);
    let mut places2 = Vec::with_capacity(64);
    let mut cand1 = Vec::with_capacity(80);
    let mut beam_sc = BeamRollScratch::new();

    let mut groups = Vec::new();
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    let k_worlds = if use_beam_teacher { 4usize } else { 6usize };
    let horizon = if use_beam_teacher { 18usize } else { 16usize };
    let sample_mod = if use_beam_teacher { 14usize } else { 12usize };

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };

        if pieces % sample_mod == 3 && rem_g >= 8 {
            let sibs = ai.diverse_beam_leaves(4);
            if sibs.len() >= 2 {
                let bag_mask = sibs[0].3;
                let base_rng = seed
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    ^ ((pieces as u64) + 1).wrapping_mul(0xBF58476D1CE4E5B9);

                // Pre-generate shared CRN sequences across all siblings in this group
                let mut world_seqs = vec![[0u8; 20]; k_worlds];
                let mut world_spawns = vec![0u64; k_worlds];

                // Stratify step 0 across bag_mask pieces
                let mut bag_bits = [0u8; 7];
                let mut bag_n = 0usize;
                let mut bm = bag_mask;
                while bm != 0 {
                    bag_bits[bag_n] = bm.trailing_zeros() as u8;
                    bag_n += 1;
                    bm &= bm - 1;
                }
                for w_idx in 0..k_worlds {
                    let mut wr = base_rng ^ ((w_idx as u64) + 1).wrapping_mul(0x94D049BB133111EB);
                    let mut bag = bag_mask;
                    if bag_n > 0 {
                        let p0 = bag_bits[w_idx % bag_n];
                        world_seqs[w_idx][0] = p0;
                        bag &= !(1u8 << p0);
                        if bag == 0 {
                            bag = 0x7F;
                        }
                        for st in 1..horizon {
                            world_seqs[w_idx][st] = draw_from_bag(&mut bag, &mut wr);
                        }
                    } else {
                        for st in 0..horizon {
                            world_seqs[w_idx][st] = draw_from_bag(&mut bag, &mut wr);
                        }
                    }
                    world_spawns[w_idx] = wr ^ 0xD6E8FEB86659FD93;
                }

                let mut grp_leaves = Vec::with_capacity(sibs.len());
                for (b, in_h, rg, bmask) in sibs {
                    let mut stats = Vec::with_capacity(k_worlds);
                    for w_idx in 0..k_worlds {
                        let st = if use_beam_teacher {
                            rollout_crn_beam(
                                fst,
                                &mut pgen,
                                &mut beam_sc,
                                b,
                                in_h,
                                rg,
                                &world_seqs[w_idx][..horizon],
                                world_spawns[w_idx],
                                w_teacher,
                                teacher_model,
                                12,
                            )
                        } else {
                            rollout_crn_2ply(
                                fst,
                                &mut pgen,
                                &mut places1,
                                &mut places2,
                                &mut cand1,
                                b,
                                in_h,
                                rg,
                                &world_seqs[w_idx][..horizon],
                                world_spawns[w_idx],
                                w_teacher,
                                teacher_model,
                            )
                        };
                        stats.push(st);
                    }
                    grp_leaves.push(LeafRec {
                        board: b,
                        in_hand: in_h,
                        rem_g: rg,
                        bag_mask: bmask,
                        stats,
                    });
                }
                groups.push(SiblingGroup {
                    seed,
                    leaves: grp_leaves,
                });
            }
        }

        let mut inputs = res.inputs;
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
    }

    groups
}

fn collect_groups_r3(
    seed: u64,
    cfg: &BeamConfig,
    fst: &FastShapeTable,
    w_teacher: &EvalWeights,
    teacher_model: Option<&LearnedValueModel>,
) -> Vec<SiblingGroup> {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);

    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());
    let mut pgen = FastPlaceGen::new();
    let mut beam_sc = BeamRollScratch::new();

    let mut groups = Vec::new();
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    let k_worlds = 4usize;
    let horizon = 96usize;

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };

        if pieces % 7 == 3 && rem_g >= 6 && rem_g <= 38 && groups.len() < 8 {
            let sibs = ai.diverse_beam_leaves(4);
            if sibs.len() >= 2 {
                let bag_mask = sibs[0].3;
                let base_rng = seed
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    ^ ((pieces as u64) + 1).wrapping_mul(0xBF58476D1CE4E5B9);

                let mut world_seqs = vec![[0u8; 96]; k_worlds];
                let mut world_holes = vec![[0u16; 64]; k_worlds];

                let mut bag_bits = [0u8; 7];
                let mut bag_n = 0usize;
                let mut bm = bag_mask;
                while bm != 0 {
                    bag_bits[bag_n] = bm.trailing_zeros() as u8;
                    bag_n += 1;
                    bm &= bm - 1;
                }
                let root_fb = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
                let init_prev_hole = if root_fb.garbage_rows > 0 {
                    let h0 = (!root_fb.data[0] & 0x3FF).trailing_zeros() as u16;
                    if h0 < 10 { Some(h0) } else { None }
                } else {
                    None
                };

                for w_idx in 0..k_worlds {
                    let mut wr = base_rng ^ ((w_idx as u64) + 1).wrapping_mul(0x94D049BB133111EB);
                    let mut bag = bag_mask;
                    if bag_n > 0 {
                        let p0 = bag_bits[w_idx % bag_n];
                        world_seqs[w_idx][0] = p0;
                        bag &= !(1u8 << p0);
                        if bag == 0 {
                            bag = 0x7F;
                        }
                        for st in 1..horizon {
                            world_seqs[w_idx][st] = draw_from_bag(&mut bag, &mut wr);
                        }
                    } else {
                        for st in 0..horizon {
                            world_seqs[w_idx][st] = draw_from_bag(&mut bag, &mut wr);
                        }
                    }

                    let mut hr = wr ^ 0xD6E8FEB86659FD93;
                    let mut prev_h = match init_prev_hole {
                        Some(ph) => ph,
                        None => {
                            hr = hr.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            ((hr >> 32) as u16) % 10
                        }
                    };
                    for k in 0..64 {
                        hr = hr.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let off = (((hr >> 32) as u16) % 9) + 1;
                        prev_h = (prev_h + off) % 10;
                        world_holes[w_idx][k] = prev_h;
                    }
                }

                let mut grp_leaves = Vec::with_capacity(sibs.len());
                for (b, in_h, rg, bmask) in sibs {
                    let mut stats = Vec::with_capacity(k_worlds);
                    for w_idx in 0..k_worlds {
                        let st = rollout_crn_terminal(
                            fst,
                            &mut pgen,
                            &mut beam_sc,
                            b,
                            in_h,
                            rg,
                            &world_seqs[w_idx][..horizon],
                            &world_holes[w_idx],
                            w_teacher,
                            teacher_model,
                            8,
                        );
                        stats.push(st);
                    }
                    grp_leaves.push(LeafRec {
                        board: b,
                        in_hand: in_h,
                        rem_g: rg,
                        bag_mask: bmask,
                        stats,
                    });
                }
                groups.push(SiblingGroup {
                    seed,
                    leaves: grp_leaves,
                });
            }
        }

        let mut inputs = res.inputs;
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
    }

    groups
}

fn collect_groups_r4(
    seed: u64,
    cfg: &BeamConfig,
    fst: &FastShapeTable,
    w_teacher: &EvalWeights,
    teacher_model: Option<&LearnedValueModel>,
) -> Vec<SiblingGroup> {
    let mut game_cfg = BSConfig::default();
    game_cfg.garbage.total_lines = Some(100);
    game_cfg.prng_seed = Some(seed);

    let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
    let mut ai = FastAI::new(cfg.clone());
    let mut pgen = FastPlaceGen::new();
    let mut beam_sc = BeamRollScratch::new();

    let mut groups = Vec::new();
    let mut ds_total = 0usize;
    let mut pieces = 0usize;

    let k_worlds = 4usize;
    let horizon = 140usize;

    while ds_total < 100 && stacker.current_piece_type().is_some() {
        let snap = stacker.snapshot().unwrap();
        let g_rows = garbage_rows(&stacker);
        let rem_g = 100usize.saturating_sub(ds_total) as u16;
        let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
            Some(r) => r,
            None => break,
        };

        if pieces % 7 == 2 && rem_g >= 8 && rem_g <= 50 && groups.len() < 6 {
            let sibs = ai.diverse_beam_leaves(4);
            if sibs.len() >= 2 {
                let bag_mask = sibs[0].3;
                let base_rng = seed
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    ^ ((pieces as u64) + 1).wrapping_mul(0xBF58476D1CE4E5B9);

                let mut world_seqs = vec![[0u8; 140]; k_worlds];
                let mut world_holes = vec![[0u16; 64]; k_worlds];

                let mut bag_bits = [0u8; 7];
                let mut bag_n = 0usize;
                let mut bm = bag_mask;
                while bm != 0 {
                    bag_bits[bag_n] = bm.trailing_zeros() as u8;
                    bag_n += 1;
                    bm &= bm - 1;
                }
                let root_fb = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
                let init_prev_hole = if root_fb.garbage_rows > 0 {
                    let h0 = (!root_fb.data[0] & 0x3FF).trailing_zeros() as u16;
                    if h0 < 10 { Some(h0) } else { None }
                } else {
                    None
                };

                for w_idx in 0..k_worlds {
                    let mut wr = base_rng ^ ((w_idx as u64) + 1).wrapping_mul(0x94D049BB133111EB);
                    let mut bag = bag_mask;
                    if bag_n > 0 {
                        let p0 = bag_bits[w_idx % bag_n];
                        world_seqs[w_idx][0] = p0;
                        bag &= !(1u8 << p0);
                        if bag == 0 {
                            bag = 0x7F;
                        }
                        for st in 1..horizon {
                            world_seqs[w_idx][st] = draw_from_bag(&mut bag, &mut wr);
                        }
                    } else {
                        for st in 0..horizon {
                            world_seqs[w_idx][st] = draw_from_bag(&mut bag, &mut wr);
                        }
                    }

                    let mut hr = wr ^ 0xD6E8FEB86659FD93;
                    let mut prev_h = match init_prev_hole {
                        Some(ph) => ph,
                        None => {
                            hr = hr.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                            ((hr >> 32) as u16) % 10
                        }
                    };
                    for k in 0..64 {
                        hr = hr.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        let off = (((hr >> 32) as u16) % 9) + 1;
                        prev_h = (prev_h + off) % 10;
                        world_holes[w_idx][k] = prev_h;
                    }
                }

                let mut grp_leaves = Vec::with_capacity(sibs.len());
                for (b, in_h, rg, bmask) in sibs {
                    let mut stats = Vec::with_capacity(k_worlds);
                    for w_idx in 0..k_worlds {
                        let st = rollout_crn_terminal(
                            fst,
                            &mut pgen,
                            &mut beam_sc,
                            b,
                            in_h,
                            rg,
                            &world_seqs[w_idx][..horizon],
                            &world_holes[w_idx],
                            w_teacher,
                            teacher_model,
                            24,
                        );
                        stats.push(st);
                    }
                    grp_leaves.push(LeafRec {
                        board: b,
                        in_hand: in_h,
                        rem_g: rg,
                        bag_mask: bmask,
                        stats,
                    });
                }
                groups.push(SiblingGroup {
                    seed,
                    leaves: grp_leaves,
                });
            }
        }

        let mut inputs = res.inputs;
        inputs.pop();
        stacker.run(inputs);
        let (_, ds) = stacker.hard_drop();
        pieces += 1;
        ds_total += ds;
    }

    groups
}

fn solve_linear_system(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let mut piv = i;
        for r in (i + 1)..n {
            if a[r][i].abs() > a[piv][i].abs() {
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let diag = a[i][i];
        if diag.abs() < 1e-12 {
            continue;
        }
        for c in i..n {
            a[i][c] /= diag;
        }
        b[i] /= diag;
        for r in 0..n {
            if r != i && a[r][i].abs() > 1e-14 {
                let factor = a[r][i];
                for c in i..n {
                    a[r][c] -= factor * a[i][c];
                }
                b[r] -= factor * b[i];
            }
        }
    }
    b
}

fn solve_box_ridge(
    xtx: &[Vec<f64>],
    xty: &[f64],
    lb: &[f64],
    ub: &[f64],
    sweeps: usize,
) -> Vec<f64> {
    let n = xty.len();
    let mut w = vec![0.0f64; n];
    for _ in 0..sweeps {
        for j in 0..n {
            let diag = xtx[j][j];
            if diag.abs() < 1e-12 {
                w[j] = 0.0;
                continue;
            }
            let mut dot = 0.0f64;
            let row = &xtx[j];
            for k in 0..n {
                if k != j {
                    dot += row[k] * w[k];
                }
            }
            let val = (xty[j] - dot) / diag;
            w[j] = val.clamp(lb[j], ub[j]);
        }
    }
    w
}

fn physics_bounds(inv_std: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut lb = vec![-1e9f64; FEAT_DIM];
    let mut ub = vec![1e9f64; FEAT_DIM];
    // Non-negative risk/clutter/hole/height features (>= 0 in cost)
    for &idx in &[
        2, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 28, 29, 35,
        36, 37, 38, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 61, 74,
    ] {
        lb[idx] = 0.0;
    }
    // Non-positive readiness / open-hole bonuses (<= 0 in cost)
    for &idx in &[27, 33, 34, 39, 40, 54, 59, 60] {
        ub[idx] = 0.0;
    }
    // Minimum safety guards on extreme height & covered holes (raw -> std space)
    if inv_std[19] > 1e-6 {
        lb[19] = 0.45 / inv_std[19];
    }
    if inv_std[18] > 1e-6 {
        lb[18] = 0.03 / inv_std[18];
    }
    if inv_std[20] > 1e-6 {
        lb[20] = 0.15 / inv_std[20];
    }
    if inv_std[24] > 1e-6 {
        lb[24] = 0.10 / inv_std[24];
    }
    if inv_std[36] > 1e-6 {
        lb[36] = 0.05 / inv_std[36];
    }
    (lb, ub)
}

fn gentle_physics_bounds(inv_std: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut lb = vec![-1e9f64; FEAT_DIM];
    let mut ub = vec![1e9f64; FEAT_DIM];
    // Zero g0_row0_cells (4) to prevent endgame spurious reward
    lb[4] = 0.0;
    ub[4] = 0.0;
    // Prevent collinear sign inversions on all height/clutter/roughness/hole/blocking risk features;
    // leave stack_cells_row_g (10) and stack_cells_low (93) free!
    for &idx in &[
        2, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 28, 29, 35,
        36, 37, 38, 42, 43, 44,
        45, 46, 47, 48, 49, 50, 51, 52, 53, 55, 56, 61, 73, 74, 75,
        76, 77, 78, 79, 80, 83, 84,
        86, 87, 91, 94, 95, 96, 97,
    ] {
        lb[idx] = 0.0;
    }
    // Readiness / open-hole / clean O / near-full row_g / h2 bonuses <= 0 in cost
    for &idx in &[27, 33, 34, 39, 40, 54, 57, 58, 59, 60, 81, 82, 85, 88, 89, 90, 92] {
        ub[idx] = 0.0;
    }
    if inv_std[18] > 1e-6 {
        lb[18] = 0.02 / inv_std[18];
    }
    if inv_std[19] > 1e-6 {
        lb[19] = 0.45 / inv_std[19];
    }
    if inv_std[20] > 1e-6 {
        lb[20] = 0.12 / inv_std[20];
    }
    if inv_std[22] > 1e-6 {
        lb[22] = 0.03 / inv_std[22];
    }
    if inv_std[24] > 1e-6 {
        lb[24] = 0.02 / inv_std[24];
    }
    (lb, ub)
}

fn save_one_model(m: &LearnedValueModel, path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(path).expect("create model file");
    writeln!(f, "{} {}", m.linear_bias, m.mlp_weight).unwrap();
    for j in 0..FEAT_DIM {
        write!(f, "{} {} {} ", m.mean[j], m.inv_std[j], m.linear_w[j]).unwrap();
    }
    writeln!(f).unwrap();
    writeln!(f, "{}", m.b2).unwrap();
    for h in 0..MLP_HIDDEN {
        write!(f, "{} {} ", m.b1[h], m.w2[h]).unwrap();
        for j in 0..FEAT_DIM {
            write!(f, "{} ", m.w1[h][j]).unwrap();
        }
        writeln!(f).unwrap();
    }
}

fn load_one_model(path: &str) -> LearnedValueModel {
    let f = File::open(path).expect("open model file");
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
    m.prepare_fast();
    m
}

fn save_models(lin: &LearnedValueModel, mlp: &LearnedValueModel) {
    if let Some(parent) = std::path::Path::new(CRN_MODEL_PATH).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = File::create(CRN_MODEL_PATH).expect("create CRN model file");
    for m in [lin, mlp] {
        writeln!(f, "{} {}", m.linear_bias, m.mlp_weight).unwrap();
        for j in 0..FEAT_DIM {
            write!(f, "{} {} {} ", m.mean[j], m.inv_std[j], m.linear_w[j]).unwrap();
        }
        writeln!(f).unwrap();
        writeln!(f, "{}", m.b2).unwrap();
        for h in 0..MLP_HIDDEN {
            write!(f, "{} {} ", m.b1[h], m.w2[h]).unwrap();
            for j in 0..FEAT_DIM {
                write!(f, "{} ", m.w1[h][j]).unwrap();
            }
            writeln!(f).unwrap();
        }
    }
}

fn load_models() -> (LearnedValueModel, LearnedValueModel) {
    let f = File::open(CRN_MODEL_PATH).expect("open CRN model file");
    let mut lines = BufReader::new(f).lines().map(|l| l.unwrap());
    let mut read_one = || -> LearnedValueModel {
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
        for j in 0..FEAT_DIM {
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
            for j in 0..FEAT_DIM {
                m.w1[h][j] = lh[2 + j];
            }
        }
        m.prepare_fast();
        m
    };
    (read_one(), read_one())
}

fn eval_batch(seeds: &[u64], cfg: &BeamConfig) -> (f64, usize, usize, usize, f64, f64) {
    let t0 = Instant::now();
    let mid = seeds.len() / 2;
    let (s1, s2) = seeds.split_at(mid);
    let run_one = |s: u64| -> (usize, usize, bool) {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(s);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai = FastAI::new(cfg.clone());
        let mut ds_total = 0usize;
        let mut pieces = 0usize;
        let mut non_ds = 0usize;
        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
                Some(r) => r,
                None => break,
            };
            let mut inputs = res.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            pieces += 1;
            ds_total += ds;
            non_ds += lines.saturating_sub(ds);
        }
        let topout = ds_total < 100;
        (if topout { 400 } else { pieces }, non_ds, topout)
    };
    let pcs: Vec<(usize, usize, bool)> = std::thread::scope(|sc| {
        let h1 = sc.spawn(|| s1.iter().map(|&s| run_one(s)).collect::<Vec<_>>());
        let mut r2: Vec<(usize, usize, bool)> = s2.iter().map(|&s| run_one(s)).collect();
        let mut r1 = h1.join().unwrap();
        r1.append(&mut r2);
        r1
    });
    let dt = t0.elapsed().as_secs_f64();
    let topouts = pcs.iter().filter(|x| x.2).count();
    let sum: usize = pcs.iter().map(|x| x.0).sum();
    let sum_nds: usize = pcs.iter().map(|x| x.1).sum();
    let min_p = pcs.iter().filter(|x| !x.2).map(|x| x.0).min().unwrap_or(999);
    let max_p = pcs.iter().filter(|x| !x.2).map(|x| x.0).max().unwrap_or(999);
    (
        sum as f64 / pcs.len() as f64,
        min_p,
        max_p,
        topouts,
        sum_nds as f64 / pcs.len() as f64,
        dt,
    )
}

fn h1_info(stacker: &Stacker) -> (Option<u16>, usize, usize, usize) {
    let g = garbage_rows(stacker);
    if g == 0 {
        return (None, 0, 0, 0);
    }
    let mut row_g_minus_1 = [false; 10];
    let mut col_h = [0u16; 10];
    let mut blocks_over = 0usize;
    let mut cells_above = 0usize;
    for ((r, c), _) in stacker.matrix() {
        if r == g - 1 {
            row_g_minus_1[c as usize] = true;
        }
        if r >= g {
            cells_above += 1;
        }
        col_h[c as usize] = col_h[c as usize].max(r + 1);
    }
    let h1_col = (0..10u16).find(|&c| !row_g_minus_1[c as usize]);
    let mut chute_min = 0usize;
    if let Some(c1) = h1_col {
        for ((r, c), _) in stacker.matrix() {
            if r >= g && c == c1 {
                blocks_over += 1;
            }
        }
        let lh = if c1 == 0 { 99 } else { col_h[(c1 - 1) as usize].saturating_sub(g) as usize };
        let rh = if c1 == 9 { 99 } else { col_h[(c1 + 1) as usize].saturating_sub(g) as usize };
        chute_min = lh.min(rh);
    }
    (h1_col, blocks_over, chute_min, cells_above)
}

fn extended_queue(stacker: &Stacker, extra: usize) -> Vec<blockfish::Color> {
    use std::convert::TryInto;
    let mut q: Vec<blockfish::Color> = stacker.snapshot().unwrap().queue;
    if extra == 0 {
        return q;
    }
    let mut clone = stacker.clone();
    for step in 0..extra {
        if clone.current_piece_type().is_none() {
            break;
        }
        let dir = ((step % 5) as i16) - 2;
        for _ in 0..dir.abs() {
            clone.move_horizontal(if dir > 0 { 1 } else { -1 });
        }
        clone.hard_drop();
        if let Some(&last) = clone.next().last() {
            q.push(last.try_into().unwrap());
        }
    }
    q
}

fn run_oracle_probe(seeds: &[u64], extra_q: usize, cfg: &BeamConfig) -> (f64, f64, f64, f64) {
    let t0 = Instant::now();
    let mid = seeds.len() / 2;
    let (s1, s2) = seeds.split_at(mid);
    let run_one = |s: u64| -> (usize, usize, usize) {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(s);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai = FastAI::new(cfg.clone());
        let mut ds_total = 0usize;
        let mut pieces = 0usize;
        let mut non_ds = 0usize;
        let mut zds = 0usize;
        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let mut snap = stacker.snapshot().unwrap();
            if extra_q > 0 {
                snap.queue = extended_queue(&stacker, extra_q);
            }
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
                Some(r) => r,
                None => break,
            };
            let mut inputs = res.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            pieces += 1;
            ds_total += ds;
            non_ds += lines.saturating_sub(ds);
            if ds == 0 {
                zds += 1;
            }
        }
        (if ds_total < 100 { 400 } else { pieces }, non_ds, zds)
    };
    let res: Vec<(usize, usize, usize)> = std::thread::scope(|sc| {
        let h1 = sc.spawn(|| s1.iter().map(|&s| run_one(s)).collect::<Vec<_>>());
        let mut r2: Vec<(usize, usize, usize)> = s2.iter().map(|&s| run_one(s)).collect();
        let mut r1 = h1.join().unwrap();
        r1.append(&mut r2);
        r1
    });
    let n = res.len() as f64;
    (
        res.iter().map(|x| x.0).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.1).sum::<usize>() as f64 / n,
        res.iter().map(|x| x.2).sum::<usize>() as f64 / n,
        t0.elapsed().as_secs_f64(),
    )
}

fn run_0ds_breakdown(seeds: &[u64], cfg: &BeamConfig) {
    let t0 = Instant::now();
    let mut sum_pcs = 0usize;
    let mut sum_nds = 0usize;
    let mut sum_0ds = 0usize;
    let mut zds_h1_cov = 0usize;
    let mut zds_g0 = 0usize;
    let mut zds_endgame = 0usize;
    let mut zds_open_o = 0usize;
    let mut zds_open_narrow_noi = 0usize;
    let mut zds_open_wide = 0usize;
    let mut ds_cov_next = 0usize;
    let mut sum_ds = 0usize;
    let mut ds1_cnt = 0usize;
    let mut ds2_cnt = 0usize;
    let mut ds3_cnt = 0usize;
    let mut sum_cells_above = 0usize;
    let mut printed = 0usize;

    for &s in seeds {
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(s);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai = FastAI::new(cfg.clone());
        let mut ds_total = 0usize;

        while ds_total < 100 && stacker.current_piece_type().is_some() {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
            sum_cells_above += cells_above;
            let cur_c = snap.queue[0].as_char();
            let hold_c = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|c| c.as_char());
            let board_before = if s == seeds[0] && printed < 3 {
                format!("{:?}", stacker)
            } else {
                String::new()
            };

            let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) {
                Some(r) => r,
                None => break,
            };
            let mut inputs = res.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (lines, ds) = stacker.hard_drop();
            sum_pcs += 1;
            ds_total += ds;
            sum_ds += ds;
            sum_nds += lines.saturating_sub(ds);
            if ds == 1 {
                ds1_cnt += 1;
            } else if ds == 2 {
                ds2_cnt += 1;
            } else if ds >= 3 {
                ds3_cnt += 1;
            }

            let (_, new_blocks_over, _, _) = h1_info(&stacker);
            if ds > 0 && new_blocks_over > 0 {
                ds_cov_next += 1;
            }
            if ds == 0 {
                sum_0ds += 1;
                if rem_g <= 7 && rem_g == g_rows {
                    zds_endgame += 1;
                } else if g_rows == 0 {
                    zds_g0 += 1;
                } else if blocks_over > 0 {
                    zds_h1_cov += 1;
                } else {
                    let has_o = cur_c == 'O' || hold_c == Some('O');
                    let has_i = cur_c == 'I' || hold_c == Some('I');
                    if has_o {
                        zds_open_o += 1;
                    } else if chute_min >= 1 && !has_i {
                        zds_open_narrow_noi += 1;
                    } else {
                        zds_open_wide += 1;
                        if s == seeds[0] && printed < 3 {
                            printed += 1;
                            println!(
                                "\n[Seed {} move {} | 0ds_open_wide cur={} hold={:?} chute={} lines={}]\nBEFORE:\n{}TOP ROOTS: {:?}\nAFTER:\n{:?}",
                                s, sum_pcs, cur_c, hold_c, chute_min, lines,
                                board_before,
                                ai.last_top_candidates().iter().take(4).collect::<Vec<_>>(),
                                stacker
                            );
                        }
                    }
                }
            }
        }
    }
    let n = seeds.len() as f64;
    println!("\n=== 0DS Breakdown ({} seeds, {:.1}s) ===", seeds.len(), t0.elapsed().as_secs_f64());
    println!("  pieces/game          : {:.2}", sum_pcs as f64 / n);
    println!("  non_ds/game          : {:.2}", sum_nds as f64 / n);
    println!("  ds moves (1/2/3+)    : {:.2} / {:.2} / {:.2}", ds1_cnt as f64 / n, ds2_cnt as f64 / n, ds3_cnt as f64 / n);
    println!("  total_0ds/game       : {:.2}", sum_0ds as f64 / n);
    println!("    - endgame (rem<=7) : {:.2}", zds_endgame as f64 / n);
    println!("    - g==0 waiting     : {:.2}", zds_g0 as f64 / n);
    println!("    - h1_covered       : {:.2}", zds_h1_cov as f64 / n);
    println!("    - h1_open & had O  : {:.2}", zds_open_o as f64 / n);
    println!("    - h1_open narrow!I : {:.2}", zds_open_narrow_noi as f64 / n);
    println!("    - h1_open wide     : {:.2}", zds_open_wide as f64 / n);
    println!("  ds covered next h1   : {:.1}% ({}/{})", 100.0 * ds_cov_next as f64 / sum_ds.max(1) as f64, ds_cov_next, sum_ds);
    println!("  avg cells above g    : {:.1}", sum_cells_above as f64 / sum_pcs.max(1) as f64);
}

fn make_pure_pimc_cfg(m: LearnedValueModel) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.beam_width = 120;
    c.max_per_root = 16;
    c.rollout_top_k = 16;
    c.endgame_wide = true;
    c.no_g_credit = 0;
    c.value_fast_micro = 0;
    c.value_model = Some(Arc::new(m));
    c.pimc_worlds = 6;
    c.pimc_horizon = 4;
    c.pimc_roots = 6;
    c.pimc_width = 16;
    c.pimc_margin = 450;
    c.pimc_spawn = true;
    c
}

fn make_pure_b120_cfg(m: LearnedValueModel) -> BeamConfig {
    let mut c = BeamConfig::default();
    c.beam_width = 120;
    c.max_per_root = 24;
    c.rollout_top_k = 12;
    c.endgame_wide = true;
    c.no_g_credit = 0;
    c.value_fast_micro = 0;
    c.value_model = Some(Arc::new(m));
    c
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "train".to_string());
    let train20: Vec<u64> = (101..=120).collect();
    let dev20: Vec<u64> = (1..=20).collect();
    let pure_r1_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r1_pure.txt");

    if mode == "dev20_pure_b120" {
        let m = load_one_model(pure_r1_path);
        let (mean, mn, mx, top, nds, dt) = eval_batch(&dev20, &make_pure_b120_cfg(m));
        println!("Dev-20 {:<20} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", "crn_r1_pure_b120", mean, mn, mx, top, nds, dt);
        return;
    }
    if mode == "dev20_pure_pimc" {
        let m = load_one_model(pure_r1_path);
        let (mean, mn, mx, top, nds, dt) = eval_batch(&dev20, &make_pure_pimc_cfg(m));
        println!("Dev-20 {:<20} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", "crn_r1_pure_pimc", mean, mn, mx, top, nds, dt);
        return;
    }
    if mode == "dev20_wclean_pimc" {
        let mut c = BeamConfig::default();
        c.beam_width = 120;
        c.max_per_root = 16;
        c.rollout_top_k = 16;
        c.no_g_credit = 180;
        c.endgame_wide = true;
        c.weights = clean_g0_weights();
        c.pimc_worlds = 6;
        c.pimc_horizon = 4;
        c.pimc_roots = 6;
        c.pimc_width = 16;
        c.pimc_margin = 950;
        c.pimc_spawn = true;
        let (mean, mn, mx, top, nds, dt) = eval_batch(&dev20, &c);
        println!("Dev-20 {:<20} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", "w_clean_pimc", mean, mn, mx, top, nds, dt);
        return;
    }
    if mode == "diag" {
        let m = load_one_model(pure_r1_path);
        let seeds5: Vec<u64> = (101..=105).collect();
        run_0ds_breakdown(&seeds5, &make_pure_pimc_cfg(m.clone()));

        println!("\n=== ORACLE Horizon x Beam-Width Ablation (seeds 101..=105, diagnostic only) ===");
        let mut c_p5_b120 = make_pure_b120_cfg(m.clone());
        c_p5_b120.beam_width = 120;
        c_p5_b120.max_per_root = 24;
        let mut c_p5_b480 = make_pure_b120_cfg(m.clone());
        c_p5_b480.beam_width = 480;
        c_p5_b480.max_per_root = 96;
        c_p5_b480.rollout_top_k = 24;

        let mut c_p10_b160 = make_pure_b120_cfg(m.clone());
        c_p10_b160.beam_width = 160;
        c_p10_b160.max_per_root = 32;
        c_p10_b160.rollout_top_k = 16;

        let mut c_p10_b480 = make_pure_b120_cfg(m);
        c_p10_b480.beam_width = 480;
        c_p10_b480.max_per_root = 96;
        c_p10_b480.rollout_top_k = 24;

        for (label, extra_q, cfg) in [
            ("p5_b120", 0, &c_p5_b120),
            ("p5_b480", 0, &c_p5_b480),
            ("ORACLE_p10_b160", 4, &c_p10_b160),
            ("ORACLE_p10_b480", 4, &c_p10_b480),
        ] {
            let (mp, mnds, mzds, dt) = run_oracle_probe(&seeds5, extra_q, cfg);
            println!("  {:<18} pieces={:6.2}  nds={:5.2}  0ds={:5.2}  dt={:.2}s", label, mp, mnds, mzds, dt);
        }
        return;
    }

    if mode == "diag2" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_pure = load_one_model(pure_r1_path);

        // Fit Box-Constrained Pure-CRN Ridge on cached R1 dataset
        let groups = load_groups(GROUPS_CACHE_PATH);
        let mut sum_steps = 0.0f64;
        let mut sum_ds = 0.0f64;
        for g in &groups {
            for l in &g.leaves {
                for st in &l.stats {
                    if !st.topout {
                        sum_steps += st.steps;
                        sum_ds += st.ds_cleared;
                    }
                }
            }
        }
        let emp_rate = (sum_steps / sum_ds.max(1.0)).clamp(1.65, 2.25);
        let cost_of = |stats: &[CrnStat]| -> f64 {
            let mut sum = 0.0f64;
            for st in stats {
                if st.topout {
                    sum += st.steps + 14.0 + emp_rate * st.rem_g_end;
                } else if st.rem_g_end == 0.0 {
                    sum += st.steps + 0.06 * st.stack_cells_end;
                } else {
                    sum += st.steps
                        + emp_rate * st.rem_g_end
                        + 0.16 * st.stack_cells_end
                        + 0.65 * st.covered_holes_end
                        + 0.35 * st.h1_blocks_end;
                }
            }
            sum / (stats.len() as f64)
        };
        struct PD { a: [f32; FEAT_DIM], b: [f32; FEAT_DIM], dy: f64, d_rem_g: f64 }
        let mut train_feats = Vec::new();
        let mut train_pairs = Vec::new();
        let mut val_pairs = Vec::new();
        for g in &groups {
            let mut fv = Vec::with_capacity(g.leaves.len());
            let mut ys = Vec::with_capacity(g.leaves.len());
            for l in &g.leaves {
                let mut f = [0.0f32; FEAT_DIM];
                extract_features_opt(&l.board, l.rem_g, l.in_hand, l.bag_mask, false, &mut f);
                fv.push(f);
                ys.push(cost_of(&l.stats));
            }
            if g.seed <= 258 {
                train_feats.extend_from_slice(&fv);
            }
            for i in 0..g.leaves.len() {
                for j in (i + 1)..g.leaves.len() {
                    let pd = PD {
                        a: fv[i],
                        b: fv[j],
                        dy: (ys[i] - ys[j]).clamp(-12.0, 12.0),
                        d_rem_g: (g.leaves[i].rem_g as f64) - (g.leaves[j].rem_g as f64),
                    };
                    if g.seed <= 258 { train_pairs.push(pd); } else { val_pairs.push(pd); }
                }
            }
        }
        let n_tr = train_feats.len() as f64;
        let mut mean = [0.0f64; FEAT_DIM];
        let mut var = [0.0f64; FEAT_DIM];
        for f in &train_feats {
            for j in 0..FEAT_DIM { mean[j] += f[j] as f64; }
        }
        for j in 0..FEAT_DIM { mean[j] /= n_tr; }
        for f in &train_feats {
            for j in 0..FEAT_DIM {
                let d = f[j] as f64 - mean[j];
                var[j] += d * d;
            }
        }
        let mut inv_std = [0.0f64; FEAT_DIM];
        for j in 0..FEAT_DIM {
            let sd = (var[j] / n_tr).sqrt();
            inv_std[j] = if sd > 1e-6 { 1.0 / sd } else { 0.0 };
        }
        let mut xtx = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
        let mut xty = vec![0.0f64; FEAT_DIM];
        let mut dz = [0.0f64; FEAT_DIM];
        for p in &train_pairs {
            for j in 0..FEAT_DIM {
                dz[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j];
            }
            for &dc in &[0usize, 1, 8] { dz[dc] = 0.0; }
            let tgt = p.dy - emp_rate * p.d_rem_g;
            for j in 0..FEAT_DIM {
                let v = dz[j];
                if v == 0.0 { continue; }
                xty[j] += v * tgt;
                for k in j..FEAT_DIM { xtx[j][k] += v * dz[k]; }
            }
        }
        for j in 0..FEAT_DIM {
            for k in 0..j { xtx[j][k] = xtx[k][j]; }
        }
        let (lb, ub) = physics_bounds(&inv_std);
        let mut best_box_m = LearnedValueModel::default();
        let mut best_box_rmse = f64::MAX;
        for &lam in &[500.0, 1200.0, 2500.0] {
            let mut a_mat = xtx.clone();
            for j in 0..FEAT_DIM { a_mat[j][j] += lam; }
            let w_box = solve_box_ridge(&a_mat, &xty, &lb, &ub, 350);
            let mut m = LearnedValueModel::default();
            for j in 0..FEAT_DIM {
                m.mean[j] = mean[j] as f32;
                m.inv_std[j] = inv_std[j] as f32;
                m.linear_w[j] = w_box[j] as f32;
            }
            if m.inv_std[0] > 1e-6 { m.linear_w[0] = (emp_rate as f32) / m.inv_std[0]; }
            m.linear_w[1] = 0.0;
            m.linear_w[8] = 0.0;
            m.prepare_fast();
            let mut mse = 0.0f64;
            let mut ok = 0usize;
            let mut tot = 0usize;
            for p in &val_pairs {
                let pred = (m.predict_features(&p.a) - m.predict_features(&p.b)) as f64;
                let err = pred - p.dy;
                mse += err * err;
                if p.dy.abs() >= 0.30 {
                    tot += 1;
                    if (pred > 0.0) == (p.dy > 0.0) { ok += 1; }
                }
            }
            let va_r = (mse / val_pairs.len() as f64).sqrt();
            println!("Box-CRN Ridge (lam={:5.0}) | Val RMSE={:.3} Acc={:.1}%", lam, va_r, 100.0 * ok as f64 / tot as f64);
            if va_r < best_box_rmse {
                best_box_rmse = va_r;
                best_box_m = m;
            }
        }
        save_one_model(&best_box_m, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r1_box.txt"));

        let mut c_wclean = BeamConfig::default();
        c_wclean.beam_width = 120;
        c_wclean.max_per_root = 24;
        c_wclean.rollout_top_k = 12;
        c_wclean.no_g_credit = 180;
        c_wclean.endgame_wide = true;
        c_wclean.weights = clean_g0_weights();

        let mut c_wclean_p10 = c_wclean.clone();
        c_wclean_p10.beam_width = 160;
        c_wclean_p10.max_per_root = 32;
        c_wclean_p10.rollout_top_k = 16;

        let mut c_wclean_p10_sp = c_wclean_p10.clone();
        c_wclean_p10_sp.simulate_cheese_spawn = true;

        let mut c_pure_ng = make_pure_b120_cfg(m_pure.clone());
        c_pure_ng.no_g_credit = 180;

        let mut c_pure_p10_sp = make_pure_b120_cfg(m_pure);
        c_pure_p10_sp.beam_width = 160;
        c_pure_p10_sp.max_per_root = 32;
        c_pure_p10_sp.rollout_top_k = 16;
        c_pure_p10_sp.simulate_cheese_spawn = true;

        let c_box_b120 = make_pure_b120_cfg(best_box_m.clone());
        let mut c_box_b480 = c_box_b120.clone();
        c_box_b480.beam_width = 480;
        c_box_b480.max_per_root = 96;
        c_box_b480.rollout_top_k = 24;
        let mut c_box_p10 = c_box_b120.clone();
        c_box_p10.beam_width = 160;
        c_box_p10.max_per_root = 32;
        c_box_p10.rollout_top_k = 16;
        let mut c_box_p10_sp = c_box_p10.clone();
        c_box_p10_sp.simulate_cheese_spawn = true;
        let c_box_pimc = make_pure_pimc_cfg(best_box_m);

        println!("\n=== Diag2 (seeds 101..=105) ===");
        for (label, extra_q, cfg) in [
            ("wclean_p5_b120", 0, &c_wclean),
            ("wclean_p10_nosp", 4, &c_wclean_p10),
            ("wclean_p10_spawn", 4, &c_wclean_p10_sp),
            ("pure_p5_ng180", 0, &c_pure_ng),
            ("pure_p10_spawn", 4, &c_pure_p10_sp),
            ("box_p5_b120", 0, &c_box_b120),
            ("box_p5_b480", 0, &c_box_b480),
            ("box_p10_nosp", 4, &c_box_p10),
            ("box_p10_spawn", 4, &c_box_p10_sp),
            ("box_pimc_w6h4", 0, &c_box_pimc),
        ] {
            let (mp, mnds, mzds, dt) = run_oracle_probe(&seeds5, extra_q, cfg);
            println!("  {:<18} pieces={:6.2}  nds={:5.2}  0ds={:5.2}  dt={:.2}s", label, mp, mnds, mzds, dt);
        }
        return;
    }

    if mode == "eval" || mode == "eval_pimc" {
        let (lin, mlp) = load_models();
        let mut c_lin = BeamConfig::default();
        c_lin.beam_width = 120;
        c_lin.max_per_root = 24;
        c_lin.rollout_top_k = 12;
        c_lin.endgame_wide = true;
        c_lin.value_model = Some(Arc::new(lin));

        let mut c_mlp = c_lin.clone();
        c_mlp.value_model = Some(Arc::new(mlp));

        if mode == "eval" {
            for (name, cfg) in [("crn_lin_b120", &c_lin), ("crn_mlp_b120", &c_mlp)] {
                let (m, mn, mx, top, nds, dt) = eval_batch(&train20, cfg);
                println!("{:<22} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", name, m, mn, mx, top, nds, dt);
            }
        } else {
            let mut c_pimc = c_mlp.clone();
            c_pimc.max_per_root = 16;
            c_pimc.rollout_top_k = 16;
            c_pimc.pimc_worlds = 6;
            c_pimc.pimc_horizon = 4;
            c_pimc.pimc_roots = 6;
            c_pimc.pimc_width = 16;
            c_pimc.pimc_margin = 950;
            c_pimc.pimc_spawn = true;
            let (m, mn, mx, top, nds, dt) = eval_batch(&train20, &c_pimc);
            println!("{:<22} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", "crn_mlp_pimc_w6h4", m, mn, mx, top, nds, dt);
        }
        return;
    }

    if mode == "diag_ceil_oracle" || mode == "diag_mystery" || mode == "diag_gap" || mode == "diag_178" || mode == "diag_p20" || mode == "diag_regimes" || mode == "diag_hcurve_oracle" || mode == "eval_r3" || mode == "diag_166" || mode == "diag_floor" || mode == "diag_oracle_big" || mode == "diag_joint_r3" || mode == "probe_fast2" || mode == "diag_r4_prep" || mode == "fit_r3_98" || mode == "eval_r4" || mode == "diag_r4_diff" || mode == "eval_h2_stk" {
        use blockfish::ai::fast::{color_to_idx, idx_to_color};
        use std::convert::TryInto;
        let _ = idx_to_color;
        let fst = FastShapeTable::srs();
        let w_clean = clean_g0_weights();
        let m_gbox = if mode == "eval_h2_stk" {
            LearnedValueModel::default()
        } else {
            load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"))
        };
        let mut m_fix = m_gbox.clone();
        m_fix.linear_w[4] = 0.0;
        m_fix.prepare_fast();

        let mut m_175 = m_fix.clone();
        if m_175.inv_std[0] > 1e-6 {
            m_175.linear_w[0] = 1.75 / m_175.inv_std[0];
        }
        m_175.prepare_fast();

        struct ExactSeedData {
            seed: u64,
            holes: Vec<u16>,
            pieces: Vec<u8>,
        }
        let seeds5: Vec<u64> = (101..=105).collect();
        let mut exact_data: Vec<ExactSeedData> = Vec::new();

        let hole_at_row = |st: &Stacker, target_r: u16| -> u16 {
            let mut present = [false; 10];
            for ((r, c), _) in st.matrix() {
                if r == target_r {
                    present[c as usize] = true;
                }
            }
            (0..10u16).find(|&c| !present[c as usize]).unwrap()
        };

        for &s in &seeds5 {
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
            c.beam_width = 64;
            c.max_per_root = 16;
            c.rollout_top_k = 8;
            c.no_g_credit = 180;
            c.endgame_wide = true;
            c.weights = w_clean.clone();
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
                for &col in extended_queue(&stacker, 60).iter().skip(skip_n) {
                    pieces.push(color_to_idx(col) as u8);
                }
            }
            exact_data.push(ExactSeedData { seed: s, holes, pieces });
        }

        #[derive(Clone, Copy)]
        struct ONode {
            board: FastBoard,
            held: Option<u8>,
            q_cursor: usize,
            rem_g: u16,
            no_g: u8,
            first_idx: u16,
            score: i64,
        }
        #[derive(Clone, Copy, Eq, PartialEq, Hash)]
        struct OKey {
            board: FastBoard,
            held: Option<u8>,
            rem_g: u16,
        }
        #[derive(Clone, Copy)]
        struct ExactStat {
            pcs: usize,
            nds: usize,
            zds: usize,
            zds_h1_cov: usize,
            zds_g0: usize,
            zds_end: usize,
            zds_o: usize,
            zds_nar: usize,
            zds_wide: usize,
            ds_cov_next: usize,
            ds_cnt: usize,
            cells_sum: usize,
        }
        type FxBuild = std::hash::BuildHasherDefault<blockfish::ai::beam::FxHasher>;

        let spawn_exact_cheese = |b: &mut FastBoard, rem_g: u16, lines_cleared: u8, holes: &[u16]| -> bool {
            let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
            let g = b.garbage_rows as usize;
            let unspawned = (rem_g as usize).saturating_sub(g);
            let num = target_g.saturating_sub(g).min(unspawned);
            if num == 0 { return true; }
            let old_rows = b.rows as usize;
            if old_rows + num >= 32 { return false; }
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

        let eval_o = |b: &FastBoard, rem_g: u16, in_h: Option<u8>, avail: u8, w: &EvalWeights, mdl: Option<&LearnedValueModel>, ng: u8, ng_cred: i64| -> i64 {
            let sc = if let Some(m) = mdl {
                m.evaluate_board_with_bag(b, rem_g, in_h, avail)
            } else {
                eval_fast_with_bag(b, rem_g, in_h, avail, w)
            };
            sc - ng_cred * (ng as i64)
        };

        let run_one_exact = |ed: &ExactSeedData, horizon: usize, beam_w: usize, exact_cheese: bool, w: &EvalWeights, mdl: Option<&LearnedValueModel>, ng_cred: i64, trace: bool| -> ExactStat {
            let mut game_cfg = BSConfig::default();
            game_cfg.garbage.total_lines = Some(100);
            game_cfg.prng_seed = Some(ed.seed);
            let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
            let mut pgen = FastPlaceGen::new();
            let mut ai_in = FastAI::new(BeamConfig::default());
            let mut root_places = Vec::with_capacity(64);
            let mut tmp_places = Vec::with_capacity(64);
            let mut cur_beam: Vec<ONode> = Vec::with_capacity(beam_w);
            let mut children: Vec<ONode> = Vec::with_capacity(beam_w * 32);
            let mut seen: std::collections::HashSet<OKey, FxBuild> = std::collections::HashSet::with_capacity_and_hasher(beam_w * 4, FxBuild::default());

            let mut ds_total = 0usize;
            let mut pcs = 0usize;
            let mut nds = 0usize;
            let mut zds = 0usize;
            let mut zds_h1_cov = 0usize;
            let mut zds_g0 = 0usize;
            let mut zds_end = 0usize;
            let mut zds_o = 0usize;
            let mut zds_nar = 0usize;
            let mut zds_wide = 0usize;
            let mut ds_cov_next = 0usize;
            let mut ds_cnt = 0usize;
            let mut cells_sum = 0usize;
            let mut q_pos = 0usize;
            let mut pr_wide = 0usize;
            let mut pr_cov = 0usize;

            while ds_total < 100 && stacker.current_piece_type().is_some() {
                let snap = stacker.snapshot().unwrap();
                let g_rows = garbage_rows(&stacker);
                let root_board = FastBoard::from_basic_matrix(&snap.matrix, g_rows);
                let rem_g0 = 100usize.saturating_sub(ds_total) as u16;
                let hold0 = snap.hold.map(|cc| color_to_idx(cc) as u8);
                let q_end = (q_pos + horizon).min(ed.pieces.len());
                let q = &ed.pieces[q_pos..q_end];
                let q_len = q.len();
                let cur0 = if q_len > 0 { Some(q[0]) } else { None };
                let alt0 = if hold0.is_some() { hold0 } else if q_len > 1 { Some(q[1]) } else { None };

                let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
                cells_sum += cells_above;
                let cur_ch = snap.queue[0].as_char();
                let hold_ch = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());
                let board_before = if trace && (pr_wide < 4 || pr_cov < 3) {
                    format!("{:?}", stacker)
                } else {
                    String::new()
                };

                pgen.generate_for_state(&fst, &root_board, cur0, alt0, &mut root_places);
                if root_places.is_empty() { pcs = 400; break; }

                let is_end = rem_g0 <= 7 && rem_g0 == g_rows;
                let eff_w = if is_end { beam_w.max(360) } else { beam_w };
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
                    if exact_cheese && rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
                        continue;
                    }
                    let (held, qc) = if hold0.is_some() {
                        if fp.did_hold { (cur0, 1.min(q_len)) } else { (hold0, 1.min(q_len)) }
                    } else if fp.did_hold {
                        (cur0, 2.min(q_len))
                    } else {
                        (None, 1.min(q_len))
                    };
                    if rem_g == 0 && b.rows < 20 {
                        let gr = -30_000_000 + 100 + b.rows as i64;
                        if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((1, idx as u16, gr)); }
                    }
                    let ng = u8::from(root_board.garbage_rows == 0 && rem_g0 > 0);
                    let in_h = held.or_else(|| q.get(qc).copied());
                    let sc = eval_o(&b, rem_g, in_h, 0x7F, w, mdl, ng, ng_cred);
                    let key = OKey { board: b, held, rem_g };
                    if seen.insert(key) {
                        cur_beam.push(ONode { board: b, held, q_cursor: qc, rem_g, no_g: ng, first_idx: idx as u16, score: sc });
                    }
                }

                for depth in 2..=target_d {
                    children.clear();
                    for node in &cur_beam {
                        if node.rem_g == 0 { continue; }
                        let qc = node.q_cursor;
                        let (c_nh, c_h) = if node.held.is_some() {
                            (q.get(qc).copied(), node.held)
                        } else {
                            (q.get(qc).copied(), q.get(qc + 1).copied())
                        };
                        pgen.generate_for_state(&fst, &node.board, c_nh, c_h, &mut tmp_places);
                        let ng = node.no_g.saturating_add(u8::from(node.board.garbage_rows == 0 && node.rem_g > 0));
                        for &fp in &tmp_places {
                            let mut b = node.board;
                            let o = &fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                            let (lines, ds, _) = b.place_and_sift(o, fp.norm_i, fp.norm_j);
                            let rem_g = node.rem_g.saturating_sub(ds as u16);
                            if exact_cheese && rem_g > 0 && !spawn_exact_cheese(&mut b, rem_g, lines, &ed.holes) {
                                continue;
                            }
                            let (held, nqc) = if node.held.is_some() {
                                if fp.did_hold { (c_nh, (qc + 1).min(q_len)) } else { (node.held, (qc + 1).min(q_len)) }
                            } else if fp.did_hold {
                                (c_nh, (qc + 2).min(q_len))
                            } else {
                                (None, (qc + 1).min(q_len))
                            };
                            if rem_g == 0 && b.rows < 20 {
                                let gr = -30_000_000 + (depth as i64) * 100 + b.rows as i64;
                                if best_goal.map_or(true, |x| gr < x.2) { best_goal = Some((depth, node.first_idx, gr)); }
                            }
                            let in_h = held.or_else(|| q.get(nqc).copied());
                            let sc = eval_o(&b, rem_g, in_h, 0x7F, w, mdl, ng, ng_cred);
                            children.push(ONode { board: b, held, q_cursor: nqc, rem_g, no_g: ng, first_idx: node.first_idx, score: sc });
                        }
                    }
                    if children.is_empty() { break; }
                    children.sort_unstable_by_key(|n| (n.score, n.first_idx));
                    cur_beam.clear();
                    seen.clear();
                    let mut rc = vec![0usize; root_places.len()];
                    if (mode == "diag_166" || mode == "diag_oracle_big") && depth > 1 {
                        let mut g_counts = [0usize; 16];
                        let bucket_cap = (eff_w / 4).max(12);
                        let strat_cap = (eff_w * 3) / 4;
                        let step_mpr = ((eff_mpr * depth) / target_d.max(1)).max(4).min(eff_mpr);
                        for &ch in &children {
                            if cur_beam.len() >= strat_cap { break; }
                            let b_idx = rem_g0.saturating_sub(ch.rem_g).min(15) as usize;
                            if g_counts[b_idx] >= bucket_cap || rc[ch.first_idx as usize] >= step_mpr { continue; }
                            let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                            if seen.insert(key) {
                                g_counts[b_idx] += 1;
                                rc[ch.first_idx as usize] += 1;
                                cur_beam.push(ch);
                            }
                        }
                    }
                    for &ch in &children {
                        if cur_beam.len() >= eff_w { break; }
                        if rc[ch.first_idx as usize] >= eff_mpr { continue; }
                        let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                        if seen.insert(key) {
                            rc[ch.first_idx as usize] += 1;
                            cur_beam.push(ch);
                        }
                    }
                    if cur_beam.len() < eff_w {
                        for &ch in &children {
                            if cur_beam.len() >= eff_w { break; }
                            let key = OKey { board: ch.board, held: ch.held, rem_g: ch.rem_g };
                            if seen.insert(key) { cur_beam.push(ch); }
                        }
                    }
                    if (mode == "diag_166" || mode == "diag_oracle_big") && !cur_beam.is_empty() {
                        cur_beam.sort_unstable_by_key(|n| (n.score, n.first_idx));
                    }
                }

                if trace && mode == "diag_floor" && (pcs == 12 || pcs == 31 || pcs == 32) {
                    println!("\n[diag_floor Seed {} before move {} rem_g={} cur={} hold={:?}]", ed.seed, pcs + 1, rem_g0, cur_ch, hold_ch);
                    println!("{}", format!("{:?}", stacker));
                    let mut per_root: Vec<(u16, i64, u16, usize, usize)> = Vec::new();
                    let mut seen_r = vec![false; root_places.len()];
                    for node in &cur_beam {
                        let ri = node.first_idx as usize;
                        if !seen_r[ri] {
                            seen_r[ri] = true;
                            let mut f = [0.0f32; FEAT_DIM];
                            extract_features_opt(&node.board, node.rem_g, node.held, 0x7F, false, &mut f);
                            per_root.push((node.first_idx, node.score, node.rem_g, f[9] as usize, f[20] as usize));
                        }
                    }
                    for (rank, (ri, sc, rg, sc_cells, cov_h)) in per_root.iter().take(8).enumerate() {
                        let rfp = root_places[*ri as usize];
                        let mut bt = root_board;
                        let o_t = &fst.shapes[rfp.color_idx as usize].orients[rfp.tf.2 as usize];
                        let (l0, ds0, _) = bt.place_and_sift(o_t, rfp.norm_i, rfp.norm_j);
                        println!("  cand#{}: {} (j={}, rot={}) hold={} ds0={} l0={} -> d14 sc={} rem_g={} cells={} cov={}",
                            rank + 1, idx_to_color(rfp.color_idx as usize).as_char(), rfp.norm_j, rfp.tf.2 as u8,
                            rfp.did_hold, ds0, l0, sc, rg, sc_cells, cov_h);
                    }
                }

                let chosen_idx = if let Some((_, ri, _)) = best_goal {
                    ri as usize
                } else if let Some(best) = cur_beam.first() {
                    best.first_idx as usize
                } else {
                    0
                };
                let fp = root_places[chosen_idx];
                let mut inputs = ai_in.inputs_for_place(&snap.matrix, fp).unwrap();
                inputs.pop();
                if fp.did_hold {
                    if hold0.is_none() { q_pos += 2; } else { q_pos += 1; }
                } else {
                    q_pos += 1;
                }
                stacker.run(inputs);
                let (lines, ds) = stacker.hard_drop();
                pcs += 1;
                ds_total += ds;
                nds += lines.saturating_sub(ds);
                if ds > 0 {
                    ds_cnt += 1;
                    let (_, new_bo, _, _) = h1_info(&stacker);
                    if new_bo > 0 {
                        ds_cov_next += 1;
                        if trace && pr_cov < 4 && pcs > 10 {
                            pr_cov += 1;
                            let q_str: String = q.iter().take(8).map(|&x| idx_to_color(x as usize).as_char()).collect();
                            println!("\n[ORACLE Seed {} move {} ds_cov_next rem_g={} cur={} hold={:?} q8={}]\nBEFORE:\n{}CHOSE {:?} (j={}, rot={}) hold={}\nAFTER:\n{:?}",
                                ed.seed, pcs, rem_g0, cur_ch, hold_ch, q_str, board_before, idx_to_color(fp.color_idx as usize).as_char(), fp.norm_j, fp.tf.2 as u8, fp.did_hold, stacker);
                        }
                    }
                } else {
                    zds += 1;
                    if rem_g0 <= 7 && rem_g0 == g_rows { zds_end += 1; }
                    else if g_rows == 0 { zds_g0 += 1; }
                    else if blocks_over > 0 { zds_h1_cov += 1; }
                    else {
                        let has_o = cur_ch == 'O' || hold_ch == Some('O');
                        let has_i = cur_ch == 'I' || hold_ch == Some('I');
                        if has_o { zds_o += 1; }
                        else if chute_min >= 1 && !has_i { zds_nar += 1; }
                        else {
                            zds_wide += 1;
                            if trace && pr_wide < 5 && pcs > 10 {
                                pr_wide += 1;
                                let q_str: String = q.iter().take(8).map(|&x| idx_to_color(x as usize).as_char()).collect();
                                println!("\n[ORACLE Seed {} move {} 0ds_open_wide rem_g={} cur={} hold={:?} lines={} q8={}]\nBEFORE:\n{}CHOSE {:?} (j={}, rot={}) hold={}\nAFTER:\n{:?}",
                                    ed.seed, pcs, rem_g0, cur_ch, hold_ch, lines, q_str, board_before, idx_to_color(fp.color_idx as usize).as_char(), fp.norm_j, fp.tf.2 as u8, fp.did_hold, stacker);
                            }
                        }
                    }
                }
            }
            ExactStat {
                pcs: if ds_total < 100 { 400 } else { pcs },
                nds, zds, zds_h1_cov, zds_g0, zds_end, zds_o, zds_nar, zds_wide,
                ds_cov_next, ds_cnt, cells_sum,
            }
        };

        let run_exact_suite = |label: &str, horizon: usize, beam_w: usize, exact_cheese: bool, w: &EvalWeights, mdl: Option<&LearnedValueModel>, ng_cred: i64| {
            let t0 = Instant::now();
            let (s1, s2) = exact_data.split_at(exact_data.len() / 2);
            let res: Vec<ExactStat> = std::thread::scope(|sc| {
                let h1 = sc.spawn(|| s1.iter().map(|ed| run_one_exact(ed, horizon, beam_w, exact_cheese, w, mdl, ng_cred, false)).collect::<Vec<_>>());
                let mut r2: Vec<ExactStat> = s2.iter().map(|ed| run_one_exact(ed, horizon, beam_w, exact_cheese, w, mdl, ng_cred, false)).collect();
                let mut r1 = h1.join().unwrap();
                r1.append(&mut r2);
                r1
            });
            let n = res.len() as f64;
            let pcs_vec: Vec<usize> = res.iter().map(|x| x.pcs).collect();
            let tot_pcs: usize = pcs_vec.iter().sum();
            let tot_ds: usize = res.iter().map(|x| x.ds_cnt).sum();
            let tot_cov: usize = res.iter().map(|x| x.ds_cov_next).sum();
            let tot_cells: usize = res.iter().map(|x| x.cells_sum).sum();
            println!("  {:<28} pieces={:6.2}  nds={:5.2}  0ds={:5.2} (cov={:.1} O={:.1} nar={:.1} wide={:.1} g0={:.1} end={:.1}) cov_nxt={:.1}% avg_c={:.1} per_seed={:?} dt={:.2}s",
                label,
                tot_pcs as f64 / n,
                res.iter().map(|x| x.nds).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds_h1_cov).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds_o).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds_nar).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds_wide).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds_g0).sum::<usize>() as f64 / n,
                res.iter().map(|x| x.zds_end).sum::<usize>() as f64 / n,
                100.0 * tot_cov as f64 / tot_ds.max(1) as f64,
                tot_cells as f64 / tot_pcs.max(1) as f64,
                pcs_vec,
                t0.elapsed().as_secs_f64());
        };

        if mode == "diag_178" {
            println!("\n=== Seed 101 Trace under ORACLE_p14_EXCH_r2fix_b240 ===");
            let st101 = run_one_exact(&exact_data[0], 14, 240, true, &w_clean, Some(&m_fix), 0, true);
            println!("  Seed 101 summary: pcs={} nds={} 0ds={} cov_nxt={}/{}", st101.pcs, st101.nds, st101.zds, st101.ds_cov_next, st101.ds_cnt);

            let mut m_hyb = m_fix.clone();
            if m_hyb.inv_std[9]  > 1e-6 { m_hyb.linear_w[9]  = 0.04 / m_hyb.inv_std[9]; }
            if m_hyb.inv_std[36] > 1e-6 { m_hyb.linear_w[36] = 0.14 / m_hyb.inv_std[36]; }
            if m_hyb.inv_std[42] > 1e-6 { m_hyb.linear_w[42] = 0.06 / m_hyb.inv_std[42]; }
            m_hyb.prepare_fast();

            println!("\n=== diag_178 Probes (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p6_EXCH_r2fix_b120",    6, 120, true, &w_clean, Some(&m_fix), 0);
            run_exact_suite("ORACLE_p14_EXCH_r2hyb_b240",  14, 240, true, &w_clean, Some(&m_hyb), 0);
            return;
        }

        if mode == "diag_p20" {
            println!("\n=== Horizon-20 Exact-Cheese Oracle Probe (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p20_EXCH_r2fix_b240", 20, 240, true, &w_clean, Some(&m_fix), 0);
            return;
        }

        if mode == "diag_regimes" {
            let mut w_185 = clean_g0_weights();
            w_185.garbage_row = 185;
            w_185.piece_estimate = 45;
            w_185.stack_cell = 3;
            w_185.upper_cell = 3;
            w_185.stack_over2_sq = 12;
            w_185.covered_hole = 75;
            w_185.hole_row = 30;
            w_185.blocks_over_hole = 20;
            w_185.h1_block = 32;
            w_185.h1_empty_blocking = 12;
            w_185.h2_block = 14;
            w_185.h2_extra_empty = 8;
            w_185.h3_block = 6;
            w_185.h4_block = 4;
            w_185.h1_fit_scarcity = 14;
            w_185.h1_unfit_hand = 45;
            w_185.zero_o_spot_pen = 30;
            w_185.open_h1_bonus = 40;
            w_185.combo2_bonus = 50;
            w_185.row_g_aligned_bonus = 35;
            w_185.g0_row0_cell = 0;

            println!("\n=== Eval Regime Probes at p14_EXCH (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p14_EXCH_w185_b240",   14, 240, true, &w_185,   None, 0);
            run_exact_suite("ORACLE_p14_EXCH_wclean_b480", 14, 480, true, &w_clean, None, 180);
            return;
        }

        if mode == "diag_hcurve_oracle" {
            println!("\n=== Horizon Curve under EXCH (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p8_EXCH_r2fix_b120",   8, 120, true, &w_clean, Some(&m_fix), 0);
            run_exact_suite("ORACLE_p10_EXCH_r2fix_b120", 10, 120, true, &w_clean, Some(&m_fix), 0);
            run_exact_suite("ORACLE_p12_EXCH_r2fix_b120", 12, 120, true, &w_clean, Some(&m_fix), 0);
            return;
        }

        if mode == "eval_r3" {
            let m_r3_pure = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_pure.txt"));
            let m_r3_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            let m_r3_mlp = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox_mlp.txt"));
            println!("\n=== R3 Models on ORACLE_p14_EXCH_b240 (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p14_EXCH_r3pure_b240", 14, 240, true, &w_clean, Some(&m_r3_pure), 0);
            run_exact_suite("ORACLE_p14_EXCH_r3gbox_b240", 14, 240, true, &w_clean, Some(&m_r3_gbox), 0);
            run_exact_suite("ORACLE_p14_EXCH_r3mlp_b240",  14, 240, true, &w_clean, Some(&m_r3_mlp),  0);

            println!("\n=== R3 Models on LEGAL_p5_b120 (seeds 101..=105) ===");
            for (lbl, mdl) in [
                ("LEGAL_r3pure_p5_b120", m_r3_pure),
                ("LEGAL_r3gbox_p5_b120", m_r3_gbox),
                ("LEGAL_r3mlp_p5_b120",  m_r3_mlp),
            ] {
                let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &make_pure_b120_cfg(mdl));
                println!("  {:<24} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", lbl, m, mn, mx, top, nds, dt);
            }
            return;
        }

        if mode == "diag_166" {
            let m_r3_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            println!("\n=== Seed 101 Trace under r3gbox (strat p14_EXCH_b120) ===");
            let st101 = run_one_exact(&exact_data[0], 14, 120, true, &w_clean, Some(&m_r3_gbox), 0, true);
            println!("  Seed 101 summary: pcs={} nds={} 0ds={} cov_nxt={}/{}", st101.pcs, st101.nds, st101.zds, st101.ds_cov_next, st101.ds_cnt);

            let mut m_r170 = m_r3_gbox.clone();
            if m_r170.inv_std[0] > 1e-6 { m_r170.linear_w[0] = 1.70 / m_r170.inv_std[0]; }
            m_r170.prepare_fast();

            let mut m_r145 = m_r3_gbox.clone();
            if m_r145.inv_std[0] > 1e-6 { m_r145.linear_w[0] = 1.45 / m_r145.inv_std[0]; }
            m_r145.prepare_fast();

            println!("\n=== Stratified Beam & rem_g-Rate Probes (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p14_strat_r3gb_b120",   14, 120, true, &w_clean, Some(&m_r3_gbox), 0);
            run_exact_suite("ORACLE_p14_strat_r170_b120",   14, 120, true, &w_clean, Some(&m_r170),    0);
            run_exact_suite("ORACLE_p14_strat_r145_b120",   14, 120, true, &w_clean, Some(&m_r145),    0);
            run_exact_suite("ORACLE_p20_strat_r170_b240",   20, 240, true, &w_clean, Some(&m_r170),    0);
            return;
        }

        if mode == "diag_floor" {
            let m_r3_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            println!("\n=== Nonzero Weights Comparison: r3_gbox vs r2_fix ===");
            for j in 0..FEAT_DIM {
                let r3_raw = m_r3_gbox.linear_w[j] * m_r3_gbox.inv_std[j];
                let r2_raw = m_fix.linear_w[j] * m_fix.inv_std[j];
                if r3_raw.abs() > 1e-4 || r2_raw.abs() > 1e-4 {
                    println!("  [{:2}] {:<24} r3_std={:+7.4} r3_raw={:+8.4} | r2_std={:+7.4} r2_raw={:+8.4}",
                        j, FEAT_NAMES[j], m_r3_gbox.linear_w[j], r3_raw, m_fix.linear_w[j], r2_raw);
                }
            }
            println!("\n=== Seed 101 Root Candidates under r3gbox (p14_EXCH_b120) ===");
            let st101 = run_one_exact(&exact_data[0], 14, 120, true, &w_clean, Some(&m_r3_gbox), 0, true);
            println!("  Seed 101 summary: pcs={} nds={} 0ds={} cov_nxt={}/{}", st101.pcs, st101.nds, st101.zds, st101.ds_cov_next, st101.ds_cnt);

            println!("\n=== Horizon-20 Unmodified Value Probes (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p20_EXCH_r3gbox_b240", 20, 240, true, &w_clean, Some(&m_r3_gbox), 0);
            run_exact_suite("ORACLE_p20_EXCH_r2fix_b240",  20, 240, true, &w_clean, Some(&m_fix),     0);
            return;
        }

        if mode == "diag_oracle_big" {
            let m_r3_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            let mut m_r3_nog0 = m_r3_gbox.clone();
            m_r3_nog0.linear_w[2] = 0.0;
            m_r3_nog0.prepare_fast();
            let seeds5: Vec<u64> = (101..=105).collect();
            let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &make_pure_b120_cfg(m_r3_nog0));
            println!("  {:<28} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                "LEGAL_r3nog0_p5_b120", m, mn, mx, top, nds, dt);
            println!("\n=== Oracle Floor Probes (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p20_prog_r3gbox_b240", 20, 240, true, &w_clean, Some(&m_r3_gbox), 0);
            return;
        }

        if mode == "probe_fast2" {
            let m_r3 = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            let m_r3j = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_joint.txt"));
            let seeds2: Vec<u64> = vec![101, 103];
            let mut c_j_g4 = make_pure_b120_cfg(m_r3j);
            c_j_g4.root_pimc_worlds = 8;
            c_j_g4.root_pimc_horizon = 6;
            c_j_g4.root_pimc_roots = 8;
            c_j_g4.root_pimc_width = 10;
            c_j_g4.root_pimc_margin = 1000;
            c_j_g4.root_pimc_weight_pct = 85;
            c_j_g4.root_pimc_steady_g4 = true;
            let (m, mn, mx, top, nds, dt) = eval_batch(&seeds2, &c_j_g4);
            println!("  {:<28} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                "LEGAL_r3j_rpimc_w8h6g4_2s", m, mn, mx, top, nds, dt);

            let t0 = Instant::now();
            let (s101, s103) = std::thread::scope(|sc| {
                let h = sc.spawn(|| run_one_exact(&exact_data[0], 28, 480, true, &w_clean, Some(&m_r3), 0, false));
                let r103 = run_one_exact(&exact_data[2], 28, 480, true, &w_clean, Some(&m_r3), 0, false);
                (h.join().unwrap(), r103)
            });
            println!("  ORACLE_p28_EXCH_r3gb_b480_2s mean={:.2} per_seed=[{}, {}] nds=[{}, {}] 0ds=[{}, {}] cov_nxt=[{}, {}] dt={:.2}s",
                (s101.pcs + s103.pcs) as f64 / 2.0, s101.pcs, s103.pcs, s101.nds, s103.nds, s101.zds, s103.zds,
                s101.ds_cov_next, s103.ds_cov_next, t0.elapsed().as_secs_f64());
            return;
        }

        if mode == "diag_r4_prep" {
            let m_r3 = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            let m_mlp = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox_mlp.txt"));
            let mut m_stk = m_r3.clone();
            if m_stk.inv_std[9] > 1e-6 {
                m_stk.linear_w[9] = 0.05 / m_stk.inv_std[9];
            }
            m_stk.prepare_fast();

            // 1. Measure rollout_crn_terminal raw_rate with r3_gbox + beam=24 on 100 sample states from R3 cache
            let groups = load_groups(R3_GROUPS_CACHE_PATH);
            let sub_groups = &groups[..groups.len().min(60)];
            let t_roll = Instant::now();
            let (s1, s2) = sub_groups.split_at(sub_groups.len() / 2);
            let test_slice = |sl: &[SiblingGroup]| -> (f64, f64, usize) {
                let mut pgen = FastPlaceGen::new();
                let mut sc = BeamRollScratch::new();
                let mut steps = 0.0f64;
                let mut dss = 0.0f64;
                let mut n = 0usize;
                for (gi, g) in sl.iter().enumerate() {
                    let l = &g.leaves[0];
                    let mut wr = g.seed.wrapping_mul(0x9E3779B97F4A7C15) ^ (gi as u64 + 1);
                    let mut bag = l.bag_mask;
                    let mut w_seq = [0u8; 96];
                    for k in 0..96 { w_seq[k] = draw_from_bag(&mut bag, &mut wr); }
                    let mut w_holes = [0u16; 64];
                    let mut prev_h = (!l.board.data[0] & 0x3FF).trailing_zeros() as u16 % 10;
                    for k in 0..64 {
                        wr = wr.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                        prev_h = (prev_h + (((wr >> 32) as u16) % 9) + 1) % 10;
                        w_holes[k] = prev_h;
                    }
                    let st = rollout_crn_terminal(&fst, &mut pgen, &mut sc, l.board, l.in_hand, l.rem_g, &w_seq, &w_holes, &w_clean, Some(&m_r3), 24);
                    if !st.topout {
                        steps += st.steps;
                        dss += st.ds_cleared;
                        n += 1;
                    }
                }
                (steps, dss, n)
            };
            let (a, b) = std::thread::scope(|sc| {
                let h = sc.spawn(|| test_slice(s1));
                let r2 = test_slice(s2);
                (h.join().unwrap(), r2)
            });
            println!("  Teacher r3_gbox beam=24 on 60 R3 leaves (rem_g<=38): rate={:.4} pcs/ds ({}/{} ok) dt={:.2}s",
                (a.0 + b.0) / (a.1 + b.1).max(1.0), a.2 + b.2, sub_groups.len(), t_roll.elapsed().as_secs_f64());

            // 2. Test r3_gbox_mlp at p20_EXCH_b240 on [101, 103]
            let t0 = Instant::now();
            let (s101, s103) = std::thread::scope(|sc| {
                let h = sc.spawn(|| run_one_exact(&exact_data[0], 20, 240, true, &w_clean, Some(&m_mlp), 0, false));
                let r103 = run_one_exact(&exact_data[2], 20, 240, true, &w_clean, Some(&m_mlp), 0, false);
                (h.join().unwrap(), r103)
            });
            println!("  ORACLE_p20_EXCH_r3mlp_b240_2s  mean={:.2} per_seed=[{}, {}] nds=[{}, {}] 0ds=[{}, {}] cov_nxt=[{}, {}] dt={:.2}s",
                (s101.pcs + s103.pcs) as f64 / 2.0, s101.pcs, s103.pcs, s101.nds, s103.nds, s101.zds, s103.zds,
                s101.ds_cov_next, s103.ds_cov_next, t0.elapsed().as_secs_f64());

            // 3. Test r3_gbox + stack_cells=0.05 at p20_EXCH_b240 on [101, 103]
            let t1 = Instant::now();
            let (k101, k103) = std::thread::scope(|sc| {
                let h = sc.spawn(|| run_one_exact(&exact_data[0], 20, 240, true, &w_clean, Some(&m_stk), 0, false));
                let r103 = run_one_exact(&exact_data[2], 20, 240, true, &w_clean, Some(&m_stk), 0, false);
                (h.join().unwrap(), r103)
            });
            println!("  ORACLE_p20_EXCH_r3stk_b240_2s  mean={:.2} per_seed=[{}, {}] nds=[{}, {}] 0ds=[{}, {}] cov_nxt=[{}, {}] dt={:.2}s",
                (k101.pcs + k103.pcs) as f64 / 2.0, k101.pcs, k103.pcs, k101.nds, k103.nds, k101.zds, k103.zds,
                k101.ds_cov_next, k103.ds_cov_next, t1.elapsed().as_secs_f64());
            return;
        }

        if mode == "diag_joint_r3" {
            let groups = load_groups(R3_GROUPS_CACHE_PATH);
            let mut stat_tot = 0usize;
            let mut stat_term = 0usize;
            let mut stat_top = 0usize;
            let mut sum_steps = 0.0f64;
            let mut sum_ds = 0.0f64;
            for g in &groups {
                for l in &g.leaves {
                    for st in &l.stats {
                        stat_tot += 1;
                        if st.topout { stat_top += 1; }
                        else {
                            if st.rem_g_end == 0.0 { stat_term += 1; }
                            sum_steps += st.steps;
                            sum_ds += st.ds_cleared;
                        }
                    }
                }
            }
            let emp_rate = (sum_steps / sum_ds.max(1.0)).clamp(1.60, 2.25);
            println!("Loaded {} R3 groups | term={}/{} top={} raw_rate={:.4}",
                groups.len(), stat_term, stat_tot, stat_top, emp_rate);

            struct JP { a: [f32; FEAT_DIM], b: [f32; FEAT_DIM], dy: f64, both_term: bool }
            let mut tr_feats = Vec::new();
            let mut tr_p = Vec::new();
            let mut va_p = Vec::new();
            for g in &groups {
                let mut fv = Vec::with_capacity(g.leaves.len());
                let mut ys = Vec::with_capacity(g.leaves.len());
                let mut ts = Vec::with_capacity(g.leaves.len());
                for l in &g.leaves {
                    let mut f = [0.0f32; FEAT_DIM];
                    extract_features_with_bag(&l.board, l.rem_g, l.in_hand, l.bag_mask, &mut f);
                    let mut s = 0.0f64;
                    let mut all_t = true;
                    for st in &l.stats {
                        if st.topout {
                            all_t = false;
                            s += st.steps + 14.0 + emp_rate * st.rem_g_end;
                        } else if st.rem_g_end == 0.0 {
                            s += st.steps + 0.005 * st.stack_cells_end;
                        } else {
                            all_t = false;
                            s += st.steps + emp_rate * st.rem_g_end + 0.16 * st.stack_cells_end + 0.65 * st.covered_holes_end + 0.35 * st.h1_blocks_end;
                        }
                    }
                    fv.push(f);
                    ys.push(s / l.stats.len() as f64);
                    ts.push(all_t);
                }
                if g.seed <= 256 { tr_feats.extend_from_slice(&fv); }
                for i in 0..g.leaves.len() {
                    for j in (i + 1)..g.leaves.len() {
                        let pd = JP { a: fv[i], b: fv[j], dy: (ys[i] - ys[j]).clamp(-12.0, 12.0), both_term: ts[i] && ts[j] };
                        if g.seed <= 256 { tr_p.push(pd); } else { va_p.push(pd); }
                    }
                }
            }
            let n_tr = tr_feats.len() as f64;
            let mut mean = [0.0f64; FEAT_DIM];
            let mut var = [0.0f64; FEAT_DIM];
            for f in &tr_feats { for j in 0..FEAT_DIM { mean[j] += f[j] as f64; } }
            for j in 0..FEAT_DIM { mean[j] /= n_tr; }
            for f in &tr_feats { for j in 0..FEAT_DIM { let d = f[j] as f64 - mean[j]; var[j] += d * d; } }
            let mut inv_std = [0.0f64; FEAT_DIM];
            for j in 0..FEAT_DIM { let sd = (var[j] / n_tr).sqrt(); inv_std[j] = if sd > 1e-6 { 1.0 / sd } else { 0.0 }; }

            let (mut lb, mut ub) = gentle_physics_bounds(&inv_std);
            lb[0] = 0.0;
            lb[1] = 0.0; ub[1] = 0.0;
            lb[2] = 0.0; ub[2] = 0.0;
            lb[8] = 0.0; ub[8] = 0.0;

            let mut xtx = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
            let mut xty = vec![0.0f64; FEAT_DIM];
            let mut dz = [0.0f64; FEAT_DIM];
            let mut term_pairs = 0usize;
            for p in &tr_p {
                if !p.both_term { continue; }
                term_pairs += 1;
                for j in 0..FEAT_DIM { dz[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j]; }
                for &dc in &[1usize, 2, 4, 8] { dz[dc] = 0.0; }
                for j in 0..FEAT_DIM {
                    let v = dz[j];
                    if v == 0.0 { continue; }
                    xty[j] += v * p.dy;
                    for k in j..FEAT_DIM { xtx[j][k] += v * dz[k]; }
                }
            }
            for j in 0..FEAT_DIM { for k in 0..j { xtx[j][k] = xtx[k][j]; } }
            println!("Joint Regression on {} both-terminal train pairs (out of {})", term_pairs, tr_p.len());

            let mut best_jm = LearnedValueModel::default();
            let mut best_rmse = f64::MAX;
            for &lam in &[20.0, 150.0, 600.0, 1500.0] {
                let mut a_mat = xtx.clone();
                for j in 0..FEAT_DIM { a_mat[j][j] += if j == 0 { 1.0 } else { lam }; }
                let w_j = solve_box_ridge(&a_mat, &xty, &lb, &ub, 500);
                let mut m = LearnedValueModel::default();
                for j in 0..FEAT_DIM {
                    m.mean[j] = mean[j] as f32;
                    m.inv_std[j] = inv_std[j] as f32;
                    m.linear_w[j] = w_j[j] as f32;
                }
                m.linear_w[1] = 0.0; m.linear_w[2] = 0.0; m.linear_w[4] = 0.0; m.linear_w[8] = 0.0;
                m.prepare_fast();
                let mut mse = 0.0f64; let mut ok = 0usize; let mut tot = 0usize;
                for p in &va_p {
                    let pred = (m.predict_features(&p.a) - m.predict_features(&p.b)) as f64;
                    let e = pred - p.dy;
                    mse += e * e;
                    if p.dy.abs() >= 0.30 { tot += 1; if (pred > 0.0) == (p.dy > 0.0) { ok += 1; } }
                }
                let rmse = (mse / va_p.len() as f64).sqrt();
                let rem_g_raw = m.linear_w[0] * m.inv_std[0];
                println!("  Joint-Box (lam={:5.0}) | rem_g_raw={:.4} | Val RMSE={:.3} Acc={:.1}%",
                    lam, rem_g_raw, rmse, 100.0 * ok as f64 / tot.max(1) as f64);
                if rmse < best_rmse { best_rmse = rmse; best_jm = m; }
            }
            println!("\n=== Joint Model Nonzero Weights ===");
            for j in 0..FEAT_DIM {
                let r = best_jm.linear_w[j] * best_jm.inv_std[j];
                if r.abs() > 1e-4 {
                    println!("  [{:2}] {:<24} std_w={:+7.4} raw={:+8.4}", j, FEAT_NAMES[j], best_jm.linear_w[j], r);
                }
            }
            save_one_model(&best_jm, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_joint.txt"));

            let seeds5: Vec<u64> = (101..=105).collect();
            let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &make_pure_b120_cfg(best_jm.clone()));
            println!("\n  {:<28} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                "LEGAL_r3joint_p5_b120", m, mn, mx, top, nds, dt);
            run_exact_suite("ORACLE_p14_EXCH_r3joint_b240", 14, 240, true, &w_clean, Some(&best_jm), 0);

            let m_r3_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            println!("\n=== Seed 103 (172 pcs) Trace under ORACLE_p20_EXCH_r3gbox_b240 ===");
            let st103 = run_one_exact(&exact_data[2], 20, 240, true, &w_clean, Some(&m_r3_gbox), 0, true);
            println!("  Seed 103 summary: pcs={} nds={} 0ds={} cov_nxt={}/{}", st103.pcs, st103.nds, st103.zds, st103.ds_cov_next, st103.ds_cnt);
            return;
        }

        if mode == "fit_r3_98" {
            let groups = load_groups(R3_GROUPS_CACHE_PATH);
            let mut sum_steps = 0.0f64;
            let mut sum_ds = 0.0f64;
            let mut term_x = 0.0f64;
            let mut term_y = 0.0f64;
            let mut term_xx = 0.0f64;
            let mut term_xy = 0.0f64;
            let mut term_n = 0usize;
            for g in &groups {
                for l in &g.leaves {
                    for st in &l.stats {
                        if !st.topout {
                            sum_steps += st.steps;
                            sum_ds += st.ds_cleared;
                            if st.rem_g_end == 0.0 {
                                let x = l.rem_g as f64;
                                let y = st.steps;
                                term_n += 1;
                                term_x += x;
                                term_y += y;
                                term_xx += x * x;
                                term_xy += x * y;
                            }
                        }
                    }
                }
            }
            let raw_rate = sum_steps / sum_ds.max(1.0);
            let mx_t = term_x / term_n.max(1) as f64;
            let my_t = term_y / term_n.max(1) as f64;
            let cov_xy = term_xy / term_n.max(1) as f64 - mx_t * my_t;
            let var_x = term_xx / term_n.max(1) as f64 - mx_t * mx_t;
            let ols_rate = (cov_xy / var_x.max(1e-6)).clamp(1.55, 2.25);
            let ols_beta = my_t - ols_rate * mx_t;
            println!("R3 cache ({} groups, term={}): raw_rate={:.4}, ols_rate={:.4}, intercept_beta={:.2}",
                groups.len(), term_n, raw_rate, ols_rate, ols_beta);

            struct JP { a: [f32; FEAT_DIM], b: [f32; FEAT_DIM], dy: f64, d_rem_g: f64, both_term: bool }
            let mut tr_feats = Vec::new();
            let mut tr_p = Vec::new();
            let mut va_p = Vec::new();
            for g in &groups {
                let mut fv = Vec::with_capacity(g.leaves.len());
                let mut ys = Vec::with_capacity(g.leaves.len());
                let mut ts = Vec::with_capacity(g.leaves.len());
                for l in &g.leaves {
                    let mut f = [0.0f32; FEAT_DIM];
                    extract_features_with_bag(&l.board, l.rem_g, l.in_hand, l.bag_mask, &mut f);
                    let mut s = 0.0f64;
                    let mut all_t = true;
                    for st in &l.stats {
                        if st.topout {
                            all_t = false;
                            s += st.steps + 14.0 + ols_rate * st.rem_g_end;
                        } else if st.rem_g_end == 0.0 {
                            s += st.steps + 0.005 * st.stack_cells_end;
                        } else {
                            all_t = false;
                            s += st.steps + ols_rate * st.rem_g_end + 0.16 * st.stack_cells_end + 0.65 * st.covered_holes_end + 0.35 * st.h1_blocks_end;
                        }
                    }
                    fv.push(f);
                    ys.push(s / l.stats.len() as f64);
                    ts.push(all_t);
                }
                if g.seed <= 256 { tr_feats.extend_from_slice(&fv); }
                for i in 0..g.leaves.len() {
                    for j in (i + 1)..g.leaves.len() {
                        let pd = JP {
                            a: fv[i], b: fv[j],
                            dy: (ys[i] - ys[j]).clamp(-12.0, 12.0),
                            d_rem_g: (g.leaves[i].rem_g as f64) - (g.leaves[j].rem_g as f64),
                            both_term: ts[i] && ts[j],
                        };
                        if g.seed <= 256 { tr_p.push(pd); } else { va_p.push(pd); }
                    }
                }
            }
            let n_tr = tr_feats.len() as f64;
            let mut mean = [0.0f64; FEAT_DIM];
            let mut var = [0.0f64; FEAT_DIM];
            for f in &tr_feats { for j in 0..FEAT_DIM { mean[j] += f[j] as f64; } }
            for j in 0..FEAT_DIM { mean[j] /= n_tr; }
            for f in &tr_feats { for j in 0..FEAT_DIM { let d = f[j] as f64 - mean[j]; var[j] += d * d; } }
            let mut inv_std = [0.0f64; FEAT_DIM];
            for j in 0..FEAT_DIM { let sd = (var[j] / n_tr).sqrt(); inv_std[j] = if sd > 1e-6 { 1.0 / sd } else { 0.0 }; }

            // 1. Joint-Box (drop [1, 2, 4, 6, 8] so pe_raw cannot steal rem_g slope!)
            let (mut j_lb, mut j_ub) = gentle_physics_bounds(&inv_std);
            j_lb[0] = 0.0;
            for &dc in &[1usize, 2, 4, 6, 8] { j_lb[dc] = 0.0; j_ub[dc] = 0.0; }
            let mut xtx_j = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
            let mut xty_j = vec![0.0f64; FEAT_DIM];
            let mut dz = [0.0f64; FEAT_DIM];
            for p in &tr_p {
                if !p.both_term { continue; }
                for j in 0..FEAT_DIM { dz[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j]; }
                for &dc in &[1usize, 2, 4, 6, 8] { dz[dc] = 0.0; }
                for j in 0..FEAT_DIM {
                    let v = dz[j];
                    if v == 0.0 { continue; }
                    xty_j[j] += v * p.dy;
                    for k in j..FEAT_DIM { xtx_j[j][k] += v * dz[k]; }
                }
            }
            for j in 0..FEAT_DIM { for k in 0..j { xtx_j[j][k] = xtx_j[k][j]; } }

            let mut best_jm = LearnedValueModel::default();
            let mut best_j_rmse = f64::MAX;
            for &lam in &[150.0, 600.0, 1200.0, 2000.0] {
                let mut a_mat = xtx_j.clone();
                for j in 0..FEAT_DIM { a_mat[j][j] += if j == 0 { 1.0 } else { lam }; }
                let w_j = solve_box_ridge(&a_mat, &xty_j, &j_lb, &j_ub, 500);
                let mut m = LearnedValueModel::default();
                for j in 0..FEAT_DIM {
                    m.mean[j] = mean[j] as f32;
                    m.inv_std[j] = inv_std[j] as f32;
                    m.linear_w[j] = w_j[j] as f32;
                }
                for &dc in &[1usize, 2, 4, 6, 8] { m.linear_w[dc] = 0.0; }
                m.prepare_fast();
                let mut mse = 0.0f64; let mut ok = 0usize; let mut tot = 0usize;
                for p in &va_p {
                    let pred = (m.predict_features(&p.a) - m.predict_features(&p.b)) as f64;
                    let e = pred - p.dy;
                    mse += e * e;
                    if p.dy.abs() >= 0.30 { tot += 1; if (pred > 0.0) == (p.dy > 0.0) { ok += 1; } }
                }
                let rmse = (mse / va_p.len() as f64).sqrt();
                println!("  98F-Joint-Box (lam={:5.0}) | rem_g_raw={:.4} | Val RMSE={:.3} Acc={:.1}%",
                    lam, m.linear_w[0] * m.inv_std[0], rmse, 100.0 * ok as f64 / tot.max(1) as f64);
                if rmse < best_j_rmse { best_j_rmse = rmse; best_jm = m; }
            }

            // 2. Anchored GentleBox at ols_rate (drop [0, 1, 2, 4, 6, 8])
            let (mut g_lb, mut g_ub) = gentle_physics_bounds(&inv_std);
            for &dc in &[0usize, 1, 2, 4, 6, 8] { g_lb[dc] = 0.0; g_ub[dc] = 0.0; }
            let mut xtx_g = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
            let mut xty_g = vec![0.0f64; FEAT_DIM];
            for p in &tr_p {
                for j in 0..FEAT_DIM { dz[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j]; }
                for &dc in &[0usize, 1, 2, 4, 6, 8] { dz[dc] = 0.0; }
                let tgt = p.dy - ols_rate * p.d_rem_g;
                for j in 0..FEAT_DIM {
                    let v = dz[j];
                    if v == 0.0 { continue; }
                    xty_g[j] += v * tgt;
                    for k in j..FEAT_DIM { xtx_g[j][k] += v * dz[k]; }
                }
            }
            for j in 0..FEAT_DIM { for k in 0..j { xtx_g[j][k] = xtx_g[k][j]; } }

            let mut best_gm = LearnedValueModel::default();
            let mut best_g_rmse = f64::MAX;
            for &lam in &[500.0, 1200.0, 2500.0] {
                let mut a_mat = xtx_g.clone();
                for j in 0..FEAT_DIM { a_mat[j][j] += lam; }
                let w_g = solve_box_ridge(&a_mat, &xty_g, &g_lb, &g_ub, 400);
                let mut m = LearnedValueModel::default();
                for j in 0..FEAT_DIM {
                    m.mean[j] = mean[j] as f32;
                    m.inv_std[j] = inv_std[j] as f32;
                    m.linear_w[j] = w_g[j] as f32;
                }
                if m.inv_std[0] > 1e-6 { m.linear_w[0] = (ols_rate as f32) / m.inv_std[0]; }
                for &dc in &[1usize, 2, 4, 6, 8] { m.linear_w[dc] = 0.0; }
                m.prepare_fast();
                let mut mse = 0.0f64; let mut ok = 0usize; let mut tot = 0usize;
                for p in &va_p {
                    let pred = (m.predict_features(&p.a) - m.predict_features(&p.b)) as f64;
                    let e = pred - p.dy;
                    mse += e * e;
                    if p.dy.abs() >= 0.30 { tot += 1; if (pred > 0.0) == (p.dy > 0.0) { ok += 1; } }
                }
                let rmse = (mse / va_p.len() as f64).sqrt();
                println!("  98F-OlsGBox   (lam={:5.0}) | rem_g_raw={:.4} | Val RMSE={:.3} Acc={:.1}%",
                    lam, m.linear_w[0] * m.inv_std[0], rmse, 100.0 * ok as f64 / tot.max(1) as f64);
                if rmse < best_g_rmse { best_g_rmse = rmse; best_gm = m; }
            }

            println!("\n=== New Features (86..=97) Weights (98F-Joint vs 98F-OlsGBox) ===");
            for j in 86..FEAT_DIM {
                println!("  [{:2}] {:<24} joint_raw={:+8.4} | olsgb_raw={:+8.4}",
                    j, FEAT_NAMES[j],
                    best_jm.linear_w[j] * best_jm.inv_std[j],
                    best_gm.linear_w[j] * best_gm.inv_std[j]);
            }
            save_one_model(&best_jm, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_98_joint.txt"));
            save_one_model(&best_gm, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_98_olsgb.txt"));

            let seeds5: Vec<u64> = (101..=105).collect();
            for (lbl, mdl) in [("LEGAL_r3_98j_p5_b120", best_jm.clone()), ("LEGAL_r3_98gb_p5_b120", best_gm.clone())] {
                let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &make_pure_b120_cfg(mdl));
                println!("  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", lbl, m, mn, mx, top, nds, dt);
            }
            for (lbl, mdl) in [("ORACLE_p20_98j_b240_2s", &best_jm), ("ORACLE_p20_98gb_b240_2s", &best_gm)] {
                let t0 = Instant::now();
                let (s101, s103) = std::thread::scope(|sc| {
                    let h = sc.spawn(|| run_one_exact(&exact_data[0], 20, 240, true, &w_clean, Some(mdl), 0, false));
                    let r103 = run_one_exact(&exact_data[2], 20, 240, true, &w_clean, Some(mdl), 0, false);
                    (h.join().unwrap(), r103)
                });
                println!("  {:<26} mean={:.2} per_seed=[{}, {}] nds=[{}, {}] 0ds=[{}, {}] cov_nxt=[{}, {}] dt={:.2}s",
                    lbl, (s101.pcs + s103.pcs) as f64 / 2.0, s101.pcs, s103.pcs, s101.nds, s103.nds,
                    s101.zds, s103.zds, s101.ds_cov_next, s103.ds_cov_next, t0.elapsed().as_secs_f64());
            }
            return;
        }

        if mode == "eval_r4" {
            let sub = std::env::args().nth(2).unwrap_or_else(|| "fast".to_string());
            let m_r4_gb = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_gbox.txt"));
            let m_r4_jt = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_joint.txt"));
            let seeds5: Vec<u64> = (101..=105).collect();

            if sub == "fast" || sub == "oracle" {
                println!("\n=== R4 Oracle p20_EXCH_b240 on [101, 103] ===");
                for (lbl, mdl) in [
                    ("ORACLE_p20_r4gb_b240_2s", &m_r4_gb),
                    ("ORACLE_p20_r4jt_b240_2s", &m_r4_jt),
                ] {
                    let t0 = Instant::now();
                    let (s101, s103) = std::thread::scope(|sc| {
                        let h = sc.spawn(|| run_one_exact(&exact_data[0], 20, 240, true, &w_clean, Some(mdl), 0, false));
                        let r103 = run_one_exact(&exact_data[2], 20, 240, true, &w_clean, Some(mdl), 0, false);
                        (h.join().unwrap(), r103)
                    });
                    println!("  {:<26} mean={:.2} per_seed=[{}, {}] nds=[{}, {}] 0ds=[{}, {}] cov_nxt=[{}, {}] dt={:.2}s",
                        lbl, (s101.pcs + s103.pcs) as f64 / 2.0, s101.pcs, s103.pcs, s101.nds, s103.nds,
                        s101.zds, s103.zds, s101.ds_cov_next, s103.ds_cov_next, t0.elapsed().as_secs_f64());
                }
            }

            if sub == "fast" || sub == "legal" {
                println!("\n=== R4 Legal 5-Preview p5_b120 (seeds 101..=105) ===");
                for (lbl, mdl) in [("LEGAL_r4gb_p5_b120", m_r4_gb.clone()), ("LEGAL_r4jt_p5_b120", m_r4_jt.clone())] {
                    let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &make_pure_b120_cfg(mdl));
                    println!("  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", lbl, m, mn, mx, top, nds, dt);
                }
            }

            if sub == "rpimc_gb" || sub == "rpimc_jt" {
                let (lbl, mdl) = if sub == "rpimc_gb" {
                    ("LEGAL_r4gb_rpimc_w8h6g4", m_r4_gb)
                } else {
                    ("LEGAL_r4jt_rpimc_w8h6g4", m_r4_jt)
                };
                let mut c = make_pure_b120_cfg(mdl);
                c.root_pimc_worlds = 8;
                c.root_pimc_horizon = 6;
                c.root_pimc_roots = 8;
                c.root_pimc_width = 10;
                c.root_pimc_margin = 1000;
                c.root_pimc_weight_pct = 85;
                c.root_pimc_steady_g4 = true;
                let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &c);
                println!("  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", lbl, m, mn, mx, top, nds, dt);
            }
            return;
        }

        if mode == "diag_r4_diff" {
            let g3 = load_groups(R3_GROUPS_CACHE_PATH);
            let g4 = load_groups(R4_GROUPS_CACHE_PATH);
            let hist = |gs: &[SiblingGroup], name: &str| {
                let mut b = [0usize; 6]; // <10, 10..19, 20..29, 30..39, 40..49, >=50
                let mut tot = 0usize;
                let mut sum_rg = 0usize;
                for g in gs {
                    for l in &g.leaves {
                        tot += 1;
                        let rg = l.rem_g as usize;
                        sum_rg += rg;
                        let idx = if rg < 10 { 0 } else if rg < 20 { 1 } else if rg < 30 { 2 } else if rg < 40 { 3 } else if rg < 50 { 4 } else { 5 };
                        b[idx] += 1;
                    }
                }
                println!("{} ({} leaves, mean_rem_g={:.1}): <10:{} 10s:{} 20s:{} 30s:{} 40s:{} 50+:{}",
                    name, tot, sum_rg as f64 / tot.max(1) as f64, b[0], b[1], b[2], b[3], b[4], b[5]);
            };
            hist(&g3, "R3");
            hist(&g4, "R4");

            let m_r3 = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            let m_r3_98o = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_98_olsgb.txt"));
            let m_r4_gb = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_gbox.txt"));
            let m_r4_jt = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_joint.txt"));

            println!("\n=== Nonzero Raw Weights: r3_gbox | r3_98_olsgb | r4_gbox | r4_joint ===");
            for j in 0..FEAT_DIM {
                let a = m_r3.linear_w[j] * m_r3.inv_std[j];
                let b = m_r3_98o.linear_w[j] * m_r3_98o.inv_std[j];
                let c = m_r4_gb.linear_w[j] * m_r4_gb.inv_std[j];
                let d = m_r4_jt.linear_w[j] * m_r4_jt.inv_std[j];
                if a.abs().max(b.abs()).max(c.abs()).max(d.abs()) > 0.008 {
                    println!("  [{:2}] {:<24} r3={:+8.4}  r3_98o={:+8.4}  r4_gb={:+8.4}  r4_jt={:+8.4}",
                        j, FEAT_NAMES[j], a, b, c, d);
                }
            }

            let mut m_r3_rg192 = m_r3.clone();
            if m_r3_rg192.inv_std[0] > 1e-6 { m_r3_rg192.linear_w[0] = 1.92 / m_r3_rg192.inv_std[0]; }
            m_r3_rg192.prepare_fast();

            let mut m_r3_h2 = m_r3.clone();
            if m_r4_gb.inv_std[86] > 1e-6 {
                m_r3_h2.mean[86] = m_r4_gb.mean[86]; m_r3_h2.inv_std[86] = m_r4_gb.inv_std[86];
                m_r3_h2.linear_w[86] = m_r4_gb.linear_w[86];
            }
            if m_r4_gb.inv_std[88] > 1e-6 {
                m_r3_h2.mean[88] = m_r4_gb.mean[88]; m_r3_h2.inv_std[88] = m_r4_gb.inv_std[88];
                m_r3_h2.linear_w[88] = m_r4_gb.linear_w[88];
            }
            if m_r3_h2.inv_std[9] > 1e-6 {
                m_r3_h2.linear_w[9] = 0.03 / m_r3_h2.inv_std[9];
            }
            m_r3_h2.prepare_fast();

            let mut m_r4_rg212 = m_r4_gb.clone();
            if m_r4_rg212.inv_std[0] > 1e-6 { m_r4_rg212.linear_w[0] = 2.12 / m_r4_rg212.inv_std[0]; }
            m_r4_rg212.prepare_fast();

            println!("\n=== Diagnostic [101, 103] ORACLE_p20_EXCH_b240 Ablations ===");
            for (lbl, mdl) in [
                ("r3_gbox (rg=1.92)",       &m_r3_rg192),
                ("r3_gbox+h2+stk03",        &m_r3_h2),
                ("r4_gb   (rg=2.12)",       &m_r4_rg212),
            ] {
                let t0 = Instant::now();
                let (s1, s3) = std::thread::scope(|sc| {
                    let h = sc.spawn(|| run_one_exact(&exact_data[0], 20, 240, true, &w_clean, Some(mdl), 0, false));
                    let r = run_one_exact(&exact_data[2], 20, 240, true, &w_clean, Some(mdl), 0, false);
                    (h.join().unwrap(), r)
                });
                let avg_c = (s1.cells_sum + s3.cells_sum) as f64 / (s1.pcs + s3.pcs) as f64;
                println!("  {:<22} pcs=[{}, {}] mean={:.2} nds=[{}, {}] cov=[{}, {}] O=[{}, {}] nar=[{}, {}] wide=[{}, {}] g0=[{}, {}] end=[{}, {}] c={:.1} dt={:.1}s",
                    lbl, s1.pcs, s3.pcs, (s1.pcs + s3.pcs) as f64 / 2.0, s1.nds, s3.nds,
                    s1.ds_cov_next, s3.ds_cov_next, s1.zds_o, s3.zds_o, s1.zds_nar, s3.zds_nar,
                    s1.zds_wide, s3.zds_wide, s1.zds_g0, s3.zds_g0, s1.zds_end, s3.zds_end,
                    avg_c, t0.elapsed().as_secs_f64());
            }
            return;
        }

        if mode == "eval_h2_stk" {
            let sub = std::env::args().nth(2).unwrap_or_else(|| "all5".to_string());
            let m_r3 = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
            let m_r4_gb = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_gbox.txt"));
            let mut m_h2 = m_r3.clone();
            for &j in &[86usize, 88] {
                if m_r4_gb.inv_std[j] > 1e-6 {
                    m_h2.mean[j] = m_r4_gb.mean[j];
                    m_h2.inv_std[j] = m_r4_gb.inv_std[j];
                    m_h2.linear_w[j] = m_r4_gb.linear_w[j];
                }
            }
            if m_h2.inv_std[9] > 1e-6 { m_h2.linear_w[9] = 0.03 / m_h2.inv_std[9]; }
            m_h2.prepare_fast();
            save_one_model(&m_h2, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_h2stk03.txt"));

            let mut m_h2_nog0 = m_h2.clone();
            m_h2_nog0.linear_w[2] = 0.0;
            m_h2_nog0.prepare_fast();

            let seeds5: Vec<u64> = (101..=105).collect();
            if sub == "all5" || sub == "oracle5" {
                println!("\n=== 5-Seed ORACLE_p20_EXCH_b240 (seeds 101..=105) ===");
                run_exact_suite("ORACLE_p20_EXCH_h2stk03", 20, 240, true, &w_clean, Some(&m_h2), 0);
            }
            if sub == "all5" || sub == "legal5" {
                println!("\n=== 5-Seed LEGAL_p5_b120 (seeds 101..=105) ===");
                for (lbl, mdl) in [("LEGAL_h2stk03_p5_b120", m_h2.clone()), ("LEGAL_h2stk03_nog0_p5", m_h2_nog0.clone())] {
                    let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &make_pure_b120_cfg(mdl));
                    println!("  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", lbl, m, mn, mx, top, nds, dt);
                }
            }
            if sub == "rpimc" {
                let mut c = make_pure_b120_cfg(m_h2);
                c.root_pimc_worlds = 8;
                c.root_pimc_horizon = 6;
                c.root_pimc_roots = 8;
                c.root_pimc_width = 10;
                c.root_pimc_margin = 1000;
                c.root_pimc_weight_pct = 85;
                c.root_pimc_steady_g4 = true;
                let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, &c);
                println!("  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                    "LEGAL_h2stk03_rpimc_g4", m, mn, mx, top, nds, dt);
            }
            return;
        }

        if mode == "diag_gap" {
            println!("\n=== Horizon vs Exact-Cheese Gap Decomposition (seeds 101..=105) ===");
            run_exact_suite("ORACLE_p5_EXCH_wclean_b120",   5, 120, true,  &w_clean, None, 180);
            run_exact_suite("ORACLE_p5_EXCH_r2fix_b120",    5, 120, true,  &w_clean, Some(&m_fix), 0);
            run_exact_suite("ORACLE_p14_nosp_r2fix_b240",  14, 240, false, &w_clean, Some(&m_fix), 0);
            run_exact_suite("ORACLE_p14_EXCH_r2fix_b240",  14, 240, true,  &w_clean, Some(&m_fix), 0);
            return;
        }

        println!("\n=== Exact-Cheese Oracle Ceiling Probes (seeds 101..=105) ===");
        run_exact_suite("ORACLE_p14_EXCH_r2fix175_b240", 14, 240, true, &w_clean, Some(&m_175), 0);
        run_exact_suite("ORACLE_p14_EXCH_r2fix_b480",    14, 480, true, &w_clean, Some(&m_fix), 0);
        return;
    }

    if mode == "diag_ceil_legal" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        let mut m_fix = m_gbox;
        m_fix.linear_w[4] = 0.0;
        m_fix.prepare_fast();

        let c_p5_b120 = make_pure_b120_cfg(m_fix.clone());
        let mut c_p5_b480_st = c_p5_b120.clone();
        c_p5_b480_st.beam_width = 480;
        c_p5_b480_st.max_per_root = 96;
        c_p5_b480_st.rollout_top_k = 24;
        c_p5_b480_st.stratify_rem_g = true;

        let mut c_strat_pimc = make_pure_pimc_cfg(m_fix.clone());
        c_strat_pimc.stratify_rem_g = true;
        c_strat_pimc.pimc_seed_leaves = 8;
        c_strat_pimc.pimc_worlds = 10;
        c_strat_pimc.pimc_horizon = 5;
        c_strat_pimc.pimc_roots = 8;
        c_strat_pimc.pimc_width = 20;
        c_strat_pimc.pimc_margin = 950;
        c_strat_pimc.pimc_weight_pct = 90;
        c_strat_pimc.pimc_spawn = true;

        let mut c_sim_b120 = c_p5_b120.clone();
        c_sim_b120.simulate_cheese_spawn = true;

        println!("\n=== Legal p5 r2fix (g0_row0=0) Probes (seeds 101..=105) ===");
        for (label, cfg) in [
            ("LEGAL_r2fix_p5_b120", &c_p5_b120),
            ("LEGAL_r2fix_simsp_b120", &c_sim_b120),
            ("LEGAL_r2fix_p5_b480_st", &c_p5_b480_st),
            ("LEGAL_r2fix_strat_pimc", &c_strat_pimc),
        ] {
            let (mp, mnds, mzds, dt) = run_oracle_probe(&seeds5, 0, cfg);
            println!("  {:<26} pieces={:6.2}  nds={:5.2}  0ds={:5.2}  dt={:.2}s", label, mp, mnds, mzds, dt);
        }
        return;
    }

    if mode == "diag_root_pimc" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        let mut m_fix = m_gbox;
        m_fix.linear_w[4] = 0.0;
        m_fix.prepare_fast();

        let mut c_r2_h99 = make_pure_b120_cfg(m_fix.clone());
        c_r2_h99.beam_width = 80;
        c_r2_h99.max_per_root = 16;
        c_r2_h99.rollout_top_k = 16;
        c_r2_h99.root_pimc_worlds = 8;
        c_r2_h99.root_pimc_horizon = 99;
        c_r2_h99.root_pimc_roots = 6;
        c_r2_h99.root_pimc_width = 12;
        c_r2_h99.root_pimc_margin = 1200;
        c_r2_h99.root_pimc_weight_pct = 100;

        let mut c_r2_h7 = make_pure_b120_cfg(m_fix);
        c_r2_h7.beam_width = 80;
        c_r2_h7.max_per_root = 16;
        c_r2_h7.rollout_top_k = 16;
        c_r2_h7.root_pimc_worlds = 6;
        c_r2_h7.root_pimc_horizon = 7;
        c_r2_h7.root_pimc_roots = 6;
        c_r2_h7.root_pimc_width = 10;
        c_r2_h7.root_pimc_margin = 1200;
        c_r2_h7.root_pimc_weight_pct = 100;

        println!("\n=== Legal Root-CRN-PIMC + 7bag-Rollout Probes (seeds 101..=105) ===");
        for (label, cfg) in [
            ("LEGAL_r2_rpimc_w8_h99_r", &c_r2_h99),
            ("LEGAL_r2_rpimc_w6_h7_r", &c_r2_h7),
        ] {
            let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, cfg);
            println!(
                "  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                label, m, mn, mx, top, nds, dt
            );
        }
        return;
    }

    if mode == "test_steady_g4" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_r3 = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
        let m_r2 = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        let mut m_r2fix = m_r2;
        m_r2fix.linear_w[4] = 0.0;
        m_r2fix.prepare_fast();

        let c_r3_b120 = make_pure_b120_cfg(m_r3.clone());
        let mut c_r3_b240 = c_r3_b120.clone();
        c_r3_b240.beam_width = 240;
        c_r3_b240.max_per_root = 48;
        c_r3_b240.rollout_top_k = 24;

        let mut c_r3_g4_w8 = make_pure_b120_cfg(m_r3.clone());
        c_r3_g4_w8.beam_width = 120;
        c_r3_g4_w8.max_per_root = 24;
        c_r3_g4_w8.rollout_top_k = 12;
        c_r3_g4_w8.root_pimc_worlds = 8;
        c_r3_g4_w8.root_pimc_horizon = 6;
        c_r3_g4_w8.root_pimc_roots = 8;
        c_r3_g4_w8.root_pimc_width = 10;
        c_r3_g4_w8.root_pimc_margin = 1000;
        c_r3_g4_w8.root_pimc_weight_pct = 85;
        c_r3_g4_w8.root_pimc_steady_g4 = true;

        let mut c_r3_g4_w10 = c_r3_g4_w8.clone();
        c_r3_g4_w10.root_pimc_worlds = 10;
        c_r3_g4_w10.root_pimc_horizon = 7;
        c_r3_g4_w10.root_pimc_width = 12;
        c_r3_g4_w10.root_pimc_margin = 1100;

        let mut c_r2_g4_w8 = make_pure_b120_cfg(m_r2fix);
        c_r2_g4_w8.beam_width = 120;
        c_r2_g4_w8.max_per_root = 24;
        c_r2_g4_w8.rollout_top_k = 12;
        c_r2_g4_w8.root_pimc_worlds = 8;
        c_r2_g4_w8.root_pimc_horizon = 6;
        c_r2_g4_w8.root_pimc_roots = 8;
        c_r2_g4_w8.root_pimc_width = 10;
        c_r2_g4_w8.root_pimc_margin = 1000;
        c_r2_g4_w8.root_pimc_weight_pct = 85;
        c_r2_g4_w8.root_pimc_steady_g4 = true;

        println!("\n=== Legal 5-Preview steady_g4 Root-CRN-PIMC Probes (seeds 101..=105) ===");
        for (label, cfg) in [
            ("LEGAL_r3gbox_p5_b120", &c_r3_b120),
            ("LEGAL_r3gbox_p5_b240", &c_r3_b240),
            ("LEGAL_r3_rpimc_w8_h6_g4", &c_r3_g4_w8),
            ("LEGAL_r3_rpimc_w10_h7_g4", &c_r3_g4_w10),
            ("LEGAL_r2_rpimc_w8_h6_g4", &c_r2_g4_w8),
        ] {
            let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, cfg);
            println!(
                "  {:<26} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                label, m, mn, mx, top, nds, dt
            );
        }
        return;
    }

    if mode == "diag_hcurve_legal" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        let mut m_fix = m_gbox;
        m_fix.linear_w[4] = 0.0;
        m_fix.prepare_fast();

        let c_p5_b120 = make_pure_b120_cfg(m_fix.clone());
        let mut c_p5_noroll = c_p5_b120.clone();
        c_p5_noroll.rollout_top_k = 0;

        let mut c_wc_b120 = BeamConfig::default();
        c_wc_b120.beam_width = 120;
        c_wc_b120.max_per_root = 32;
        c_wc_b120.rollout_top_k = 16;
        c_wc_b120.no_g_credit = 180;
        c_wc_b120.endgame_wide = true;
        c_wc_b120.weights = clean_g0_weights();

        let mut c_wc_noroll = c_wc_b120.clone();
        c_wc_noroll.rollout_top_k = 0;

        println!("\n=== Legal Rollout & nosp Ablation (seeds 101..=105) ===");
        for (label, cfg) in [
            ("LEGAL_r2fix_p5_b120", &c_p5_b120),
            ("LEGAL_r2fix_p5_noroll", &c_p5_noroll),
            ("LEGAL_wclean_p5_b120", &c_wc_b120),
            ("LEGAL_wclean_p5_noroll", &c_wc_noroll),
        ] {
            let (m, mn, mx, top, nds, dt) = eval_batch(&seeds5, cfg);
            println!(
                "  {:<24} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
                label, m, mn, mx, top, nds, dt
            );
        }
        return;
    }

    if mode == "diag_r2" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        let c_b120 = make_pure_b120_cfg(m_gbox.clone());
        let mut c_b480 = c_b120.clone();
        c_b480.beam_width = 480;
        c_b480.max_per_root = 96;
        c_b480.rollout_top_k = 24;
        println!(">>> r2_gbox p5_b120 <<<");
        run_0ds_breakdown(&seeds5, &c_b120);
        println!("\n>>> r2_gbox p5_b480 <<<");
        run_0ds_breakdown(&seeds5, &c_b480);

        // Find first 3 moves on Seed 101 where b480 diverges from b120
        let mut game_cfg = BSConfig::default();
        game_cfg.garbage.total_lines = Some(100);
        game_cfg.prng_seed = Some(101);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
        let mut ai120 = FastAI::new(c_b120);
        let mut ai480 = FastAI::new(c_b480);
        let mut ds_total = 0usize;
        let mut mv = 0usize;
        let mut diffs = 0usize;
        while ds_total < 100 && stacker.current_piece_type().is_some() && diffs < 4 {
            let snap = stacker.snapshot().unwrap();
            let g_rows = garbage_rows(&stacker);
            let rem_g = 100usize.saturating_sub(ds_total) as u16;
            let r120 = ai120.best_move(&snap, Some(g_rows), Some(rem_g)).unwrap();
            let r480 = ai480.best_move(&snap, Some(g_rows), Some(rem_g)).unwrap();
            mv += 1;
            if r120.inputs != r480.inputs {
                diffs += 1;
                println!("\n[Seed 101 move {} DIVERGE rem_g={} g={}]\n{:?}", mv, rem_g, g_rows, stacker);
                println!("  b120 top3: {:?}", ai120.last_top_candidates().iter().take(3).collect::<Vec<_>>());
                println!("  b480 top3: {:?}", ai480.last_top_candidates().iter().take(3).collect::<Vec<_>>());
            }
            let mut inputs = r480.inputs;
            inputs.pop();
            stacker.run(inputs);
            let (_, ds) = stacker.hard_drop();
            ds_total += ds;
        }
        return;
    }

    if mode == "diag_w10" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let mut c = BeamConfig::default();
        c.beam_width = 160;
        c.max_per_root = 32;
        c.rollout_top_k = 16;
        c.no_g_credit = 180;
        c.endgame_wide = true;
        c.weights = clean_g0_weights();
        // Use extended_queue(4) in a quick breakdown loop
        let t0 = Instant::now();
        let mut sum_pcs = 0usize;
        let mut sum_nds = 0usize;
        let mut sum_0ds = 0usize;
        let mut zds_h1_cov = 0usize;
        let mut zds_g0 = 0usize;
        let mut zds_endgame = 0usize;
        let mut zds_open_o = 0usize;
        let mut zds_open_narrow_noi = 0usize;
        let mut zds_open_wide = 0usize;
        let mut ds_cov_next = 0usize;
        let mut sum_ds = 0usize;
        let mut sum_cells_above = 0usize;
        for &s in &seeds5 {
            let mut game_cfg = BSConfig::default();
            game_cfg.garbage.total_lines = Some(100);
            game_cfg.prng_seed = Some(s);
            let mut stacker = Stacker::new(Ruleset::guideline().into(), game_cfg);
            let mut ai = FastAI::new(c.clone());
            let mut ds_total = 0usize;
            while ds_total < 100 && stacker.current_piece_type().is_some() {
                let mut snap = stacker.snapshot().unwrap();
                snap.queue = extended_queue(&stacker, 4);
                let g_rows = garbage_rows(&stacker);
                let rem_g = 100usize.saturating_sub(ds_total) as u16;
                let (_, blocks_over, chute_min, cells_above) = h1_info(&stacker);
                sum_cells_above += cells_above;
                let cur_c = snap.queue[0].as_char();
                let hold_c = snap.hold.or_else(|| snap.queue.get(1).copied()).map(|cc| cc.as_char());
                let res = match ai.best_move(&snap, Some(g_rows), Some(rem_g)) { Some(r) => r, None => break };
                let mut inputs = res.inputs;
                inputs.pop();
                stacker.run(inputs);
                let (lines, ds) = stacker.hard_drop();
                sum_pcs += 1;
                ds_total += ds;
                sum_ds += ds;
                sum_nds += lines.saturating_sub(ds);
                let (_, new_blocks_over, _, _) = h1_info(&stacker);
                if ds > 0 && new_blocks_over > 0 { ds_cov_next += 1; }
                if ds == 0 {
                    sum_0ds += 1;
                    if rem_g <= 7 && rem_g == g_rows { zds_endgame += 1; }
                    else if g_rows == 0 { zds_g0 += 1; }
                    else if blocks_over > 0 { zds_h1_cov += 1; }
                    else {
                        let has_o = cur_c == 'O' || hold_c == Some('O');
                        let has_i = cur_c == 'I' || hold_c == Some('I');
                        if has_o { zds_open_o += 1; }
                        else if chute_min >= 1 && !has_i { zds_open_narrow_noi += 1; }
                        else { zds_open_wide += 1; }
                    }
                }
            }
        }
        let n = seeds5.len() as f64;
        println!("=== wclean_p10_nosp Breakdown ({:.1}s) ===", t0.elapsed().as_secs_f64());
        println!("  pieces={:.2} nds={:.2} 0ds={:.2} (end={:.2} g0={:.2} h1_cov={:.2} open_O={:.2} open_nar={:.2} open_wide={:.2}) ds_cov_next={:.1}% avg_cells={:.1}",
            sum_pcs as f64/n, sum_nds as f64/n, sum_0ds as f64/n, zds_endgame as f64/n, zds_g0 as f64/n, zds_h1_cov as f64/n, zds_open_o as f64/n, zds_open_narrow_noi as f64/n, zds_open_wide as f64/n,
            100.0 * ds_cov_next as f64 / sum_ds.max(1) as f64, sum_cells_above as f64 / sum_pcs.max(1) as f64);
        return;
    }

    if mode == "diag_search" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));

        let mut c_wclean_pimc = BeamConfig::default();
        c_wclean_pimc.beam_width = 120;
        c_wclean_pimc.max_per_root = 24;
        c_wclean_pimc.rollout_top_k = 16;
        c_wclean_pimc.no_g_credit = 180;
        c_wclean_pimc.endgame_wide = true;
        c_wclean_pimc.stratify_rem_g = true;
        c_wclean_pimc.pimc_seed_leaves = 8;
        c_wclean_pimc.pimc_worlds = 8;
        c_wclean_pimc.pimc_horizon = 5;
        c_wclean_pimc.pimc_roots = 8;
        c_wclean_pimc.pimc_width = 20;
        c_wclean_pimc.pimc_margin = 950;
        c_wclean_pimc.pimc_weight_pct = 90;
        c_wclean_pimc.pimc_spawn = true;
        c_wclean_pimc.weights = clean_g0_weights();

        let mut c_wclean_p10_480 = BeamConfig::default();
        c_wclean_p10_480.beam_width = 480;
        c_wclean_p10_480.max_per_root = 96;
        c_wclean_p10_480.rollout_top_k = 24;
        c_wclean_p10_480.no_g_credit = 180;
        c_wclean_p10_480.endgame_wide = true;
        c_wclean_p10_480.stratify_rem_g = true;
        c_wclean_p10_480.weights = clean_g0_weights();

        let mut w_low = clean_g0_weights();
        w_low.garbage_row = 240;
        w_low.piece_estimate = 55;
        w_low.stack_cell = 6;
        w_low.upper_cell = 6;
        w_low.covered_hole = 95;
        w_low.h1_block = 38;
        w_low.h2_block = 32;
        w_low.h2_extra_empty = 12;
        w_low.open_h1_bonus = 40;
        w_low.combo2_bonus = 55;
        let mut c_wlow_p10_480 = c_wclean_p10_480.clone();
        c_wlow_p10_480.no_g_credit = 120;
        c_wlow_p10_480.weights = w_low;

        let mut m_damp = m_gbox.clone();
        for &idx in &[32usize, 33, 34, 35, 39, 40, 54, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68] {
            m_damp.linear_w[idx] *= 0.30;
        }
        if m_damp.inv_std[15] > 1e-6 {
            m_damp.linear_w[15] = m_damp.linear_w[15].min(0.15 / m_damp.inv_std[15]);
        }
        m_damp.prepare_fast();
        let mut c_damp_120 = make_pure_b120_cfg(m_damp);
        c_damp_120.stratify_rem_g = true;
        let mut c_damp_480 = c_damp_120.clone();
        c_damp_480.beam_width = 480;
        c_damp_480.max_per_root = 96;
        c_damp_480.rollout_top_k = 24;

        println!("\n=== Diag Search Probes (seeds 101..=105) ===");
        for (label, extra_q, cfg) in [
            ("wclean_strat_pimc", 0, &c_wclean_pimc),
            ("wclean_p10_b480_st", 4, &c_wclean_p10_480),
            ("wlow_p10_b480_st", 4, &c_wlow_p10_480),
            ("r2damped_p5_b120", 0, &c_damp_120),
            ("r2damped_p5_b480", 0, &c_damp_480),
        ] {
            let (mp, mnds, mzds, dt) = run_oracle_probe(&seeds5, extra_q, cfg);
            println!("  {:<20} pieces={:6.2}  nds={:5.2}  0ds={:5.2}  dt={:.2}s", label, mp, mnds, mzds, dt);
        }
        return;
    }

    if mode == "eval_r2" || mode == "eval_r2_pimc" {
        let seeds5: Vec<u64> = (101..=105).collect();
        let m_gbox = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        if mode == "eval_r2" {
            let c_b120 = make_pure_b120_cfg(m_gbox.clone());
            let mut c_b480 = c_b120.clone();
            c_b480.beam_width = 480;
            c_b480.max_per_root = 96;
            c_b480.rollout_top_k = 24;
            let mut c_p10 = c_b120.clone();
            c_p10.beam_width = 160;
            c_p10.max_per_root = 32;
            c_p10.rollout_top_k = 16;
            let c_pimc = make_pure_pimc_cfg(m_gbox);
            for (sub, extra_q, cfg) in [
                ("p5_b120", 0, &c_b120),
                ("p5_b480", 0, &c_b480),
                ("p10_nosp", 4, &c_p10),
                ("pimc_w6h4", 0, &c_pimc),
            ] {
                let (mp, mnds, mzds, dt) = run_oracle_probe(&seeds5, extra_q, cfg);
                println!("  r2_gbox_{:<10} pieces={:6.2}  nds={:5.2}  0ds={:5.2}  dt={:.2}s", sub, mp, mnds, mzds, dt);
            }
        } else {
            let (m, mn, mx, top, nds, dt) = eval_batch(&train20, &make_pure_pimc_cfg(m_gbox));
            println!("Train-20 {:<20} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s", "crn_r2_gbox_pimc", m, mn, mx, top, nds, dt);
        }
        return;
    }

    if mode == "collect_r4" {
        let s0: u64 = std::env::args().nth(2).and_then(|x| x.parse().ok()).unwrap_or(201);
        let s1: u64 = std::env::args().nth(3).and_then(|x| x.parse().ok()).unwrap_or(213);
        let out_path = std::env::args().nth(4).unwrap_or_else(|| format!(concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r4_{}_{}.bin"), s0, s1));
        let t_start = Instant::now();
        let fst = FastShapeTable::srs();
        let w_clean = clean_g0_weights();
        let prev_m = Arc::new(load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_joint.txt")));
        let r4_teacher = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
        let mut collect_cfg = BeamConfig::default();
        collect_cfg.beam_width = 64;
        collect_cfg.max_per_root = 16;
        collect_cfg.rollout_top_k = 8;
        collect_cfg.no_g_credit = 0;
        collect_cfg.value_fast_micro = 0;
        collect_cfg.stratify_rem_g = true;
        collect_cfg.endgame_wide = true;
        collect_cfg.weights = w_clean.clone();
        collect_cfg.value_model = Some(prev_m);

        let all_seeds: Vec<u64> = (s0..=s1).collect();
        let mid = all_seeds.len() / 2;
        let (a1, a2) = all_seeds.split_at(mid);
        let g = std::thread::scope(|sc| {
            let h1 = sc.spawn(|| {
                let mut v = Vec::new();
                for &s in a1 {
                    v.extend(collect_groups_r4(s, &collect_cfg, &fst, &w_clean, Some(&r4_teacher)));
                }
                v
            });
            let mut v2 = Vec::new();
            for &s in a2 {
                v2.extend(collect_groups_r4(s, &collect_cfg, &fst, &w_clean, Some(&r4_teacher)));
            }
            let mut v1 = h1.join().unwrap();
            v1.append(&mut v2);
            v1
        });
        save_groups(&g, &out_path);
        println!("Saved {} R4 shard groups ({}..={}) to {} in {:.1}s", g.len(), s0, s1, out_path, t_start.elapsed().as_secs_f64());
        return;
    }

    let is_r2 = mode == "train_r2" || mode == "train_r2_save" || mode == "fit_r2";
    let is_r3 = mode == "train_r3" || mode == "fit_r3";
    let is_r4 = mode == "train_r4" || mode == "fit_r4";
    let t_start = Instant::now();
    let fst = FastShapeTable::srs();
    let w_clean = clean_g0_weights();

    let prev_model: Option<Arc<LearnedValueModel>> = if is_r4 {
        Some(Arc::new(load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_joint.txt"))))
    } else if is_r3 {
        let mut m = load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt"));
        m.linear_w[4] = 0.0;
        m.prepare_fast();
        Some(Arc::new(m))
    } else if is_r2 {
        Some(Arc::new(load_one_model(pure_r1_path)))
    } else {
        None
    };
    let r4_rollout_teacher: Option<LearnedValueModel> = if is_r4 {
        Some(load_one_model(concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt")))
    } else {
        None
    };

    let cache_path = if is_r4 {
        R4_GROUPS_CACHE_PATH
    } else if is_r3 {
        R3_GROUPS_CACHE_PATH
    } else if is_r2 {
        R2_GROUPS_CACHE_PATH
    } else {
        GROUPS_CACHE_PATH
    };
    let is_cached_fit = mode.starts_with("fit");
    let groups: Vec<SiblingGroup> = if is_r4 && is_cached_fit && !std::path::Path::new(cache_path).exists() {
        let mut merged = Vec::new();
        for p in [
            concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r4_201_213.bin"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r4_214_226.bin"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r4_227_239.bin"),
            concat!(env!("CARGO_MANIFEST_DIR"), "/../data/crn_groups_r4_240_252.bin"),
        ] {
            if std::path::Path::new(p).exists() {
                merged.extend(load_groups(p));
            }
        }
        assert!(!merged.is_empty(), "No R4 groups or shards found");
        save_groups(&merged, cache_path);
        println!("Merged {} R4 sibling groups into {} in {:.2}s", merged.len(), cache_path, t_start.elapsed().as_secs_f64());
        merged
    } else if is_cached_fit && std::path::Path::new(cache_path).exists() {
        let g = load_groups(cache_path);
        println!("Loaded {} cached sibling groups from {} in {:.2}s", g.len(), cache_path, t_start.elapsed().as_secs_f64());
        g
    } else {
        let mut collect_cfg = BeamConfig::default();
        if is_r4 {
            collect_cfg.beam_width = 72;
            collect_cfg.max_per_root = 18;
            collect_cfg.rollout_top_k = 10;
            collect_cfg.no_g_credit = 0;
            collect_cfg.value_fast_micro = 0;
            collect_cfg.stratify_rem_g = true;
        } else if is_r3 || is_r2 {
            collect_cfg.beam_width = 64;
            collect_cfg.max_per_root = 16;
            collect_cfg.rollout_top_k = 8;
            collect_cfg.no_g_credit = 0;
            collect_cfg.value_fast_micro = 0;
            if is_r3 {
                collect_cfg.stratify_rem_g = true;
            }
        } else {
            collect_cfg.beam_width = 80;
            collect_cfg.max_per_root = 20;
            collect_cfg.rollout_top_k = 10;
            collect_cfg.no_g_credit = 180;
        }
        collect_cfg.endgame_wide = true;
        collect_cfg.weights = w_clean.clone();
        if let Some(ref m) = prev_model {
            collect_cfg.value_model = Some(m.clone());
        }

        // Seeds disjoint from Dev-20 (1..=20), Train-20 (101..=120), Val-100 (10001..=10100)
        let all_seeds: Vec<u64> = if is_r4 {
            (201..=250).collect()
        } else if is_r3 {
            (201..=272).collect()
        } else if is_r2 {
            (201..=264).collect()
        } else {
            (201..=275).collect()
        };
        let mid = all_seeds.len() / 2;
        let (a1, a2) = all_seeds.split_at(mid);
        let done_cnt = std::sync::atomic::AtomicUsize::new(0);

        let g = std::thread::scope(|sc| {
            let h1 = sc.spawn(|| {
                let mut v = Vec::new();
                for &s in a1 {
                    if is_r4 {
                        v.extend(collect_groups_r4(
                            s,
                            &collect_cfg,
                            &fst,
                            &w_clean,
                            r4_rollout_teacher.as_ref(),
                        ));
                    } else if is_r3 {
                        v.extend(collect_groups_r3(
                            s,
                            &collect_cfg,
                            &fst,
                            &w_clean,
                            prev_model.as_deref(),
                        ));
                    } else {
                        v.extend(collect_groups_for_seed(
                            s,
                            &collect_cfg,
                            &fst,
                            &w_clean,
                            prev_model.as_deref(),
                            is_r2,
                        ));
                    }
                    let d = done_cnt.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if d % 10 == 0 || d == all_seeds.len() {
                        println!(
                            "CRN collect progress: {}/{} seeds ({:.1}s)",
                            d,
                            all_seeds.len(),
                            t_start.elapsed().as_secs_f64()
                        );
                    }
                }
                v
            });
            let mut v2 = Vec::new();
            for &s in a2 {
                if is_r4 {
                    v2.extend(collect_groups_r4(
                        s,
                        &collect_cfg,
                        &fst,
                        &w_clean,
                        r4_rollout_teacher.as_ref(),
                    ));
                } else if is_r3 {
                    v2.extend(collect_groups_r3(
                        s,
                        &collect_cfg,
                        &fst,
                        &w_clean,
                        prev_model.as_deref(),
                    ));
                } else {
                    v2.extend(collect_groups_for_seed(
                        s,
                        &collect_cfg,
                        &fst,
                        &w_clean,
                        prev_model.as_deref(),
                        is_r2,
                    ));
                }
                let d = done_cnt.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if d % 10 == 0 || d == all_seeds.len() {
                    println!(
                        "CRN collect progress: {}/{} seeds ({:.1}s)",
                        d,
                        all_seeds.len(),
                        t_start.elapsed().as_secs_f64()
                    );
                }
            }
            let mut v1 = h1.join().unwrap();
            v1.append(&mut v2);
            v1
        });
        save_groups(&g, cache_path);
        println!("Saved {} sibling groups to {}", g.len(), cache_path);
        g
    };

    // Compute empirical & OLS piece-per-garbage rate of the CRN rollout teacher
    let mut sum_steps = 0.0f64;
    let mut sum_ds = 0.0f64;
    let mut term_x = 0.0f64;
    let mut term_y = 0.0f64;
    let mut term_xx = 0.0f64;
    let mut term_xy = 0.0f64;
    let mut leaf_cnt = 0usize;
    let mut stat_tot = 0usize;
    let mut stat_term = 0usize;
    let mut stat_top = 0usize;
    for g in &groups {
        for l in &g.leaves {
            leaf_cnt += 1;
            for st in &l.stats {
                stat_tot += 1;
                if st.topout {
                    stat_top += 1;
                } else {
                    if st.rem_g_end == 0.0 {
                        stat_term += 1;
                        let x = l.rem_g as f64;
                        let y = st.steps;
                        term_x += x;
                        term_y += y;
                        term_xx += x * x;
                        term_xy += x * y;
                    }
                    sum_steps += st.steps;
                    sum_ds += st.ds_cleared;
                }
            }
        }
    }
    let raw_rate = if sum_ds > 100.0 { sum_steps / sum_ds } else { 1.90 };
    let ols_rate = if stat_term > 20 {
        let mx = term_x / stat_term as f64;
        let my = term_y / stat_term as f64;
        let cov = term_xy / stat_term as f64 - mx * my;
        let vx = term_xx / stat_term as f64 - mx * mx;
        (cov / vx.max(1e-6)).clamp(1.55, 2.25)
    } else {
        raw_rate.clamp(1.60, 2.25)
    };
    let emp_rate = if is_r4 { ols_rate } else { raw_rate.clamp(1.60, 2.25) };
    println!(
        "Total {} sibling groups ({} leaves, term={}/{}, top={}) | raw_rate={:.4}, ols_rate={:.4}, emp_rate={:.4} pieces/ds",
        groups.len(),
        leaf_cnt,
        stat_term,
        stat_tot,
        stat_top,
        raw_rate,
        ols_rate,
        emp_rate
    );

    let cost_of = |stats: &[CrnStat]| -> f64 {
        let mut sum = 0.0f64;
        for st in stats {
            if st.topout {
                sum += st.steps + 14.0 + emp_rate * st.rem_g_end;
            } else if st.rem_g_end == 0.0 {
                sum += st.steps + (if is_r3 || is_r4 { 0.005 } else { 0.06 }) * st.stack_cells_end;
            } else {
                sum += st.steps
                    + emp_rate * st.rem_g_end
                    + 0.16 * st.stack_cells_end
                    + 0.65 * st.covered_holes_end
                    + 0.35 * st.h1_blocks_end;
            }
        }
        sum / (stats.len() as f64)
    };

    // Precompute features and w_clean fast_eval score for every leaf
    struct PairData {
        a: [f32; FEAT_DIM],
        b: [f32; FEAT_DIM],
        dy: f64,
        d_rem_g: f64,
        d_fast: f64,
        both_term: bool,
    }
    let mut train_feats: Vec<[f32; FEAT_DIM]> = Vec::new();
    let mut train_pairs: Vec<PairData> = Vec::new();
    let mut val_pairs: Vec<PairData> = Vec::new();

    let train_cutoff = if is_r4 { 240u64 } else if is_r3 { 256u64 } else if is_r2 { 250u64 } else { 258u64 };
    for g in &groups {
        let mut feats_vec = Vec::with_capacity(g.leaves.len());
        let mut ys = Vec::with_capacity(g.leaves.len());
        let mut fasts = Vec::with_capacity(g.leaves.len());
        let mut all_ts = Vec::with_capacity(g.leaves.len());
        for l in &g.leaves {
            let mut f = [0.0f32; FEAT_DIM];
            extract_features_with_bag(&l.board, l.rem_g, l.in_hand, l.bag_mask, &mut f);
            let fsc = eval_fast_with_bag(&l.board, l.rem_g, l.in_hand, l.bag_mask, &w_clean) as f64;
            feats_vec.push(f);
            ys.push(cost_of(&l.stats));
            fasts.push(fsc);
            all_ts.push(l.stats.iter().all(|st| !st.topout && st.rem_g_end == 0.0));
        }

        let is_train = g.seed <= train_cutoff;
        if is_train {
            train_feats.extend_from_slice(&feats_vec);
        }
        for i in 0..g.leaves.len() {
            for j in (i + 1)..g.leaves.len() {
                let dy = (ys[i] - ys[j]).clamp(-12.0, 12.0);
                let pd = PairData {
                    a: feats_vec[i],
                    b: feats_vec[j],
                    dy,
                    d_rem_g: (g.leaves[i].rem_g as f64) - (g.leaves[j].rem_g as f64),
                    d_fast: fasts[i] - fasts[j],
                    both_term: all_ts[i] && all_ts[j],
                };
                if is_train {
                    train_pairs.push(pd);
                } else {
                    val_pairs.push(pd);
                }
            }
        }
    }
    println!(
        "Pairs: train={} (seed <= {}), val={} (seed > {})",
        train_pairs.len(),
        train_cutoff,
        val_pairs.len(),
        train_cutoff
    );

    let n_tr = train_feats.len() as f64;
    let mut mean = [0.0f64; FEAT_DIM];
    let mut var = [0.0f64; FEAT_DIM];
    for f in &train_feats {
        for j in 0..FEAT_DIM {
            mean[j] += f[j] as f64;
        }
    }
    for j in 0..FEAT_DIM {
        mean[j] /= n_tr;
    }
    for f in &train_feats {
        for j in 0..FEAT_DIM {
            let d = f[j] as f64 - mean[j];
            var[j] += d * d;
        }
    }
    let mut inv_std = [0.0f64; FEAT_DIM];
    for j in 0..FEAT_DIM {
        let sd = (var[j] / n_tr).sqrt();
        inv_std[j] = if sd > 1e-6 { 1.0 / sd } else { 0.0 };
    }

    // Helper: fit pairwise ridge on target fn(p) -> f64, excluding rem_g (0) & g_rows (1)
    let fit_ridge = |lambda: f64, drop_cols: &[usize], target_of: &dyn Fn(&PairData) -> f64| -> Vec<f64> {
        let mut xtx = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
        let mut xty = vec![0.0f64; FEAT_DIM];
        let mut dz = [0.0f64; FEAT_DIM];
        for p in &train_pairs {
            for j in 0..FEAT_DIM {
                dz[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j];
            }
            for &dc in drop_cols {
                dz[dc] = 0.0;
            }
            let tgt = target_of(p);
            for j in 0..FEAT_DIM {
                let v = dz[j];
                if v == 0.0 {
                    continue;
                }
                xty[j] += v * tgt;
                for k in j..FEAT_DIM {
                    xtx[j][k] += v * dz[k];
                }
            }
        }
        for j in 0..FEAT_DIM {
            for k in 0..j {
                xtx[j][k] = xtx[k][j];
            }
            xtx[j][j] += lambda;
        }
        solve_linear_system(xtx, xty)
    };

    let eval_model_pairs = |pairs: &[PairData], m: &LearnedValueModel, alpha: f64| -> (f64, f64) {
        let mut mse = 0.0f64;
        let mut acc_ok = 0usize;
        let mut acc_tot = 0usize;
        for p in pairs {
            let pa = m.predict_features(&p.a) as f64;
            let pb = m.predict_features(&p.b) as f64;
            let pred = alpha * p.d_fast + (pa - pb);
            let err = pred - p.dy;
            mse += err * err;
            if p.dy.abs() >= 0.30 {
                acc_tot += 1;
                if (pred > 0.0) == (p.dy > 0.0) {
                    acc_ok += 1;
                }
            }
        }
        let rmse = (mse / pairs.len().max(1) as f64).sqrt();
        let acc = if acc_tot > 0 { acc_ok as f64 / acc_tot as f64 } else { 0.0 };
        (rmse, acc)
    };

    // --- Track 1: Pure CRN with exact rem_g decomposition: y_res = dy - emp_rate * d_rem_g ---
    let mut best_pure_lambda = 1500.0;
    let mut best_pure_rmse = f64::MAX;
    let mut best_pure_model = LearnedValueModel::default();
    if !is_r4 {
        for &lam in &[500.0, 1200.0, 2500.0, 5000.0] {
            let w_res = fit_ridge(lam, &[0, 1, 8], &|p| p.dy - emp_rate * p.d_rem_g);
            let mut m = LearnedValueModel::default();
            for j in 0..FEAT_DIM {
                m.mean[j] = mean[j] as f32;
                m.inv_std[j] = inv_std[j] as f32;
                m.linear_w[j] = w_res[j] as f32;
            }
            if m.inv_std[0] > 1e-6 {
                m.linear_w[0] = (emp_rate as f32) / m.inv_std[0];
            }
            m.linear_w[1] = 0.0;
            if is_r3 {
                m.linear_w[4] = 0.0;
            }
            m.linear_w[8] = 0.0;
            if m.inv_std[19] > 1e-6 {
                let min_w19 = 0.45f32 / m.inv_std[19];
                if m.linear_w[19] < min_w19 {
                    m.linear_w[19] = min_w19;
                }
            }
            if m.inv_std[18] > 1e-6 {
                let min_w18 = 0.02f32 / m.inv_std[18];
                if m.linear_w[18] < min_w18 {
                    m.linear_w[18] = min_w18;
                }
            }
            m.prepare_fast();
            let (tr_r, tr_a) = eval_model_pairs(&train_pairs, &m, 0.0);
            let (va_r, va_a) = eval_model_pairs(&val_pairs, &m, 0.0);
            println!(
                "Pure-CRN Ridge (lam={:5.0}) | Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
                lam,
                tr_r,
                tr_a * 100.0,
                va_r,
                va_a * 100.0
            );
            if va_r < best_pure_rmse {
                best_pure_rmse = va_r;
                best_pure_lambda = lam;
                best_pure_model = m;
            }
        }
        println!(
            "\nSelected Pure-CRN Ridge (lam={:.0}) top weights (cp per +1 unit):",
            best_pure_lambda
        );
        let mut ranked_p: Vec<(usize, f32, f32)> = (0..FEAT_DIM)
            .map(|j| {
                let cp = best_pure_model.linear_w[j] * best_pure_model.inv_std[j] * 100.0;
                (j, best_pure_model.linear_w[j], cp)
            })
            .collect();
        ranked_p.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());
        for &(j, sw, cp) in ranked_p.iter().take(20) {
            println!("  {:<22} std_w={:+6.3}  cp={:+8.2}", FEAT_NAMES[j], sw, cp);
        }
    }

    // --- Track 1B: Gentle-Box Pure CRN Ridge ---
    let mut xtx_g = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
    let mut xty_g = vec![0.0f64; FEAT_DIM];
    let mut dz_g = [0.0f64; FEAT_DIM];
    for p in &train_pairs {
        for j in 0..FEAT_DIM {
            dz_g[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j];
        }
        if is_r4 {
            for &dc in &[0usize, 1, 2, 4, 6, 8] {
                dz_g[dc] = 0.0;
            }
        } else {
            for &dc in &[0usize, 1, 8] {
                dz_g[dc] = 0.0;
            }
        }
        let tgt = p.dy - emp_rate * p.d_rem_g;
        for j in 0..FEAT_DIM {
            let v = dz_g[j];
            if v == 0.0 {
                continue;
            }
            xty_g[j] += v * tgt;
            for k in j..FEAT_DIM {
                xtx_g[j][k] += v * dz_g[k];
            }
        }
    }
    for j in 0..FEAT_DIM {
        for k in 0..j {
            xtx_g[j][k] = xtx_g[k][j];
        }
    }
    let (mut g_lb, mut g_ub) = gentle_physics_bounds(&inv_std);
    if is_r4 {
        for &dc in &[1usize, 2, 4, 6, 8] {
            g_lb[dc] = 0.0;
            g_ub[dc] = 0.0;
        }
    }
    let mut best_gbox_rmse = f64::MAX;
    let mut best_gbox_lam = 1200.0;
    let mut best_gbox_model = LearnedValueModel::default();
    for &lam in &[400.0, 1000.0, 2000.0] {
        let mut a_mat = xtx_g.clone();
        for j in 0..FEAT_DIM {
            a_mat[j][j] += lam;
        }
        let w_gbox = solve_box_ridge(&a_mat, &xty_g, &g_lb, &g_ub, 400);
        let mut m = LearnedValueModel::default();
        for j in 0..FEAT_DIM {
            m.mean[j] = mean[j] as f32;
            m.inv_std[j] = inv_std[j] as f32;
            m.linear_w[j] = w_gbox[j] as f32;
        }
        if m.inv_std[0] > 1e-6 {
            m.linear_w[0] = (emp_rate as f32) / m.inv_std[0];
        }
        m.linear_w[1] = 0.0;
        if is_r4 {
            m.linear_w[2] = 0.0;
            m.linear_w[6] = 0.0;
        }
        m.linear_w[8] = 0.0;
        m.prepare_fast();
        let (tr_r, tr_a) = eval_model_pairs(&train_pairs, &m, 0.0);
        let (va_r, va_a) = eval_model_pairs(&val_pairs, &m, 0.0);
        println!(
            "GentleBox-CRN  (lam={:5.0}) | Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
            lam,
            tr_r,
            tr_a * 100.0,
            va_r,
            va_a * 100.0
        );
        if va_r < best_gbox_rmse {
            best_gbox_rmse = va_r;
            best_gbox_lam = lam;
            best_gbox_model = m;
        }
    }
    println!(
        "\nSelected GentleBox-CRN (lam={:.0}) top weights (cp per +1 unit):",
        best_gbox_lam
    );
    let mut ranked_g: Vec<(usize, f32, f32)> = (0..FEAT_DIM)
        .map(|j| {
            let cp = best_gbox_model.linear_w[j] * best_gbox_model.inv_std[j] * 100.0;
            (j, best_gbox_model.linear_w[j], cp)
        })
        .collect();
    ranked_g.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());
    for &(j, sw, cp) in ranked_g.iter().take(20) {
        println!("  {:<22} std_w={:+6.3}  cp={:+8.2}", FEAT_NAMES[j], sw, cp);
    }

    if is_r4 {
        println!("\nR4 GentleBox new features (86..=97) cp per +1 unit:");
        for j in 86..FEAT_DIM {
            let sw = best_gbox_model.linear_w[j];
            let cp = sw * best_gbox_model.inv_std[j] * 100.0;
            println!("  [{:2}] {:<24} std_w={:+6.3}  cp={:+8.2}", j, FEAT_NAMES[j], sw, cp);
        }

        // Track 1C: R4 Joint-Box on both-terminal pairs (dropping [1, 2, 4, 6, 8])
        let (mut j_lb, mut j_ub) = gentle_physics_bounds(&inv_std);
        j_lb[0] = 0.0;
        for &dc in &[1usize, 2, 4, 6, 8] {
            j_lb[dc] = 0.0;
            j_ub[dc] = 0.0;
        }
        let mut xtx_j = vec![vec![0.0f64; FEAT_DIM]; FEAT_DIM];
        let mut xty_j = vec![0.0f64; FEAT_DIM];
        let mut dz_j = [0.0f64; FEAT_DIM];
        let mut term_p = 0usize;
        for p in &train_pairs {
            if !p.both_term {
                continue;
            }
            term_p += 1;
            for j in 0..FEAT_DIM {
                dz_j[j] = (p.a[j] as f64 - p.b[j] as f64) * inv_std[j];
            }
            for &dc in &[1usize, 2, 4, 6, 8] {
                dz_j[dc] = 0.0;
            }
            for j in 0..FEAT_DIM {
                let v = dz_j[j];
                if v == 0.0 {
                    continue;
                }
                xty_j[j] += v * p.dy;
                for k in j..FEAT_DIM {
                    xtx_j[j][k] += v * dz_j[k];
                }
            }
        }
        for j in 0..FEAT_DIM {
            for k in 0..j {
                xtx_j[j][k] = xtx_j[k][j];
            }
        }
        println!("\nR4 Joint-Box on {} both-terminal train pairs:", term_p);
        let mut best_jm = LearnedValueModel::default();
        let mut best_j_rmse = f64::MAX;
        for &lam in &[100.0, 400.0, 1000.0, 2000.0] {
            let mut a_mat = xtx_j.clone();
            for j in 0..FEAT_DIM {
                a_mat[j][j] += if j == 0 { 1.0 } else { lam };
            }
            let w_j = solve_box_ridge(&a_mat, &xty_j, &j_lb, &j_ub, 500);
            let mut m = LearnedValueModel::default();
            for j in 0..FEAT_DIM {
                m.mean[j] = mean[j] as f32;
                m.inv_std[j] = inv_std[j] as f32;
                m.linear_w[j] = w_j[j] as f32;
            }
            for &dc in &[1usize, 2, 4, 6, 8] {
                m.linear_w[dc] = 0.0;
            }
            m.prepare_fast();
            let (tr_r, tr_a) = eval_model_pairs(&train_pairs, &m, 0.0);
            let (va_r, va_a) = eval_model_pairs(&val_pairs, &m, 0.0);
            println!(
                "  R4 Joint-Box (lam={:5.0}) | rem_g_raw={:.4} | Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
                lam,
                m.linear_w[0] * m.inv_std[0],
                tr_r,
                tr_a * 100.0,
                va_r,
                va_a * 100.0
            );
            if va_r < best_j_rmse {
                best_j_rmse = va_r;
                best_jm = m;
            }
        }
        save_one_model(&best_gbox_model, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_gbox.txt"));
        save_one_model(&best_jm,         concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r4_joint.txt"));
        println!(
            "\nSaved R4 models (gbox_val_rmse={:.3}, joint_val_rmse={:.3}, total dt={:.1}s)\n",
            best_gbox_rmse,
            best_j_rmse,
            t_start.elapsed().as_secs_f64()
        );
        return;
    }

    // --- Track 2: Residual over w_clean (`alpha * eval_fast + R_theta`) ---
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for p in &train_pairs {
        num += p.dy * p.d_fast;
        den += p.d_fast * p.d_fast;
    }
    let alpha = if den > 1e-6 { (num / den).max(0.0) } else { 0.0045 };
    let fast_micro = (alpha * 100.0 * 10_000.0).round() as i64;
    let alpha_q = (fast_micro as f64) / 1_000_000.0;
    let zero_m = LearnedValueModel::default();
    let (base_tr_r, base_tr_a) = eval_model_pairs(&train_pairs, &zero_m, alpha_q);
    let (base_va_r, base_va_a) = eval_model_pairs(&val_pairs, &zero_m, alpha_q);
    println!(
        "\nw_clean baseline (alpha={:.5}, fast_micro={}) | Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
        alpha_q,
        fast_micro,
        base_tr_r,
        base_tr_a * 100.0,
        base_va_r,
        base_va_a * 100.0
    );

    let mut best_res_lam = 1500.0;
    let mut best_res_rmse = f64::MAX;
    let mut best_res_lin = LearnedValueModel::default();
    for &lam in &[800.0, 2000.0, 4500.0, 9000.0] {
        let w_res = fit_ridge(lam, &[0, 1, 8], &|p| p.dy - alpha_q * p.d_fast);
        let mut m = LearnedValueModel::default();
        for j in 0..FEAT_DIM {
            m.mean[j] = mean[j] as f32;
            m.inv_std[j] = inv_std[j] as f32;
            m.linear_w[j] = w_res[j] as f32;
        }
        m.linear_w[0] = 0.0;
        m.linear_w[1] = 0.0;
        m.linear_w[8] = 0.0;
        m.prepare_fast();
        let (tr_r, tr_a) = eval_model_pairs(&train_pairs, &m, alpha_q);
        let (va_r, va_a) = eval_model_pairs(&val_pairs, &m, alpha_q);
        println!(
            "Res-CRN Ridge  (lam={:5.0}) | Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
            lam,
            tr_r,
            tr_a * 100.0,
            va_r,
            va_a * 100.0
        );
        if va_r < best_res_rmse {
            best_res_rmse = va_r;
            best_res_lam = lam;
            best_res_lin = m;
        }
    }
    println!(
        "\nSelected Res-CRN Ridge (lam={:.0}) top residual corrections over w_clean (cp):",
        best_res_lam
    );
    let mut ranked_r: Vec<(usize, f32, f32)> = (0..FEAT_DIM)
        .map(|j| {
            let cp = best_res_lin.linear_w[j] * best_res_lin.inv_std[j] * 100.0;
            (j, best_res_lin.linear_w[j], cp)
        })
        .collect();
    ranked_r.sort_by(|x, y| y.1.abs().partial_cmp(&x.1.abs()).unwrap());
    for &(j, sw, cp) in ranked_r.iter().take(20) {
        println!("  {:<22} std_w={:+6.3}  cp={:+8.2}", FEAT_NAMES[j], sw, cp);
    }

    // Train strongly regularized Siamese MLP on top of best_res_lin with Val early-stopping
    let mut mlp_model = best_res_lin.clone();
    mlp_model.mlp_weight = 1.0;
    mlp_model.fast_ready = false;

    let mut rng: u64 = 987654321;
    let mut rand_f32 = || -> f32 {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((rng >> 40) as f32) / 16777216.0;
        u * 2.0 - 1.0
    };
    let scale = (2.0 / FEAT_DIM as f32).sqrt() * 0.25;
    for h in 0..MLP_HIDDEN {
        for j in 0..FEAT_DIM {
            mlp_model.w1[h][j] = rand_f32() * scale;
        }
        mlp_model.w2[h] = rand_f32() * 0.04;
    }

    let z_pair: Vec<([f32; FEAT_DIM], [f32; FEAT_DIM], f32)> = train_pairs
        .iter()
        .map(|p| {
            let mut za = [0.0f32; FEAT_DIM];
            let mut zb = [0.0f32; FEAT_DIM];
            let mut lin_diff = 0.0f32;
            for j in 0..FEAT_DIM {
                za[j] = (p.a[j] - best_res_lin.mean[j]) * best_res_lin.inv_std[j];
                zb[j] = (p.b[j] - best_res_lin.mean[j]) * best_res_lin.inv_std[j];
                lin_diff += best_res_lin.linear_w[j] * (za[j] - zb[j]);
            }
            za[0] = 0.0;
            za[1] = 0.0;
            zb[0] = 0.0;
            zb[1] = 0.0;
            (za, zb, ((p.dy - alpha_q * p.d_fast) as f32) - lin_diff)
        })
        .collect();

    let mut best_mlp = mlp_model.clone();
    let mut best_mlp_val_rmse = best_res_rmse;
    let mut order: Vec<usize> = (0..z_pair.len()).collect();
    let epochs = 18;
    let l2 = 3.5e-2f32;
    for epoch in 0..epochs {
        for i in (1..order.len()).rev() {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng as usize) % (i + 1);
            order.swap(i, j);
        }
        let lr = 3.5e-4f32 * (1.0 - 0.6 * (epoch as f32 / epochs as f32));
        for &idx in &order {
            let (za, zb, target) = &z_pair[idx];
            let mut ha = [0.0f32; MLP_HIDDEN];
            let mut hb = [0.0f32; MLP_HIDDEN];
            let mut oa = mlp_model.b2;
            let mut ob = mlp_model.b2;
            for h in 0..MLP_HIDDEN {
                let mut sa = mlp_model.b1[h];
                let mut sb = mlp_model.b1[h];
                let row = &mlp_model.w1[h];
                for j in 0..FEAT_DIM {
                    sa += row[j] * za[j];
                    sb += row[j] * zb[j];
                }
                if sa > 0.0 {
                    ha[h] = sa;
                    oa += mlp_model.w2[h] * sa;
                }
                if sb > 0.0 {
                    hb[h] = sb;
                    ob += mlp_model.w2[h] * sb;
                }
            }
            let err = ((oa - ob) - *target).clamp(-5.0, 5.0);
            for h in 0..MLP_HIDDEN {
                let act_diff = ha[h] - hb[h];
                let w2_h = mlp_model.w2[h];
                mlp_model.w2[h] -= lr * (err * act_diff + l2 * w2_h);
                let ga = if ha[h] > 0.0 { err * w2_h } else { 0.0 };
                let gb = if hb[h] > 0.0 { -err * w2_h } else { 0.0 };
                if ga != 0.0 || gb != 0.0 {
                    mlp_model.b1[h] -= lr * (ga + gb);
                    let row = &mut mlp_model.w1[h];
                    for j in 0..FEAT_DIM {
                        row[j] -= lr * (ga * za[j] + gb * zb[j] + l2 * row[j]);
                    }
                }
            }
        }
        let (mr_tr, ma_tr) = eval_model_pairs(&train_pairs, &mlp_model, alpha_q);
        let (mr_va, ma_va) = eval_model_pairs(&val_pairs, &mlp_model, alpha_q);
        if mr_va < best_mlp_val_rmse {
            best_mlp_val_rmse = mr_va;
            best_mlp = mlp_model.clone();
        }
        if epoch == 0 || epoch == 5 || epoch == 11 || epoch == epochs - 1 {
            println!(
                "  Res-MLP epoch {:2}: Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
                epoch + 1,
                mr_tr,
                ma_tr * 100.0,
                mr_va,
                ma_va * 100.0
            );
        }
    }

    best_mlp.prepare_fast();
    if is_r3 {
        println!("\nR3 GentleBox new features (76..=85) cp per +1 unit:");
        for j in 76..FEAT_DIM {
            let sw = best_gbox_model.linear_w[j];
            let cp = sw * best_gbox_model.inv_std[j] * 100.0;
            println!("  [{:2}] {:<24} std_w={:+6.3}  cp={:+8.2}", j, FEAT_NAMES[j], sw, cp);
        }

        // Train Siamese ReLU MLP residual on top of best_gbox_model
        let mut gmlp = best_gbox_model.clone();
        gmlp.mlp_weight = 1.0;
        gmlp.fast_ready = false;
        let mut mrng: u64 = 1357924680;
        let mut rf32 = || -> f32 {
            mrng = mrng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((mrng >> 40) as f32) / 16777216.0) * 2.0 - 1.0
        };
        let sc_init = (2.0 / FEAT_DIM as f32).sqrt() * 0.20;
        for h in 0..MLP_HIDDEN {
            for j in 0..FEAT_DIM {
                gmlp.w1[h][j] = rf32() * sc_init;
            }
            gmlp.w2[h] = rf32() * 0.03;
        }
        let g_zpairs: Vec<([f32; FEAT_DIM], [f32; FEAT_DIM], f32)> = train_pairs
            .iter()
            .map(|p| {
                let mut za = [0.0f32; FEAT_DIM];
                let mut zb = [0.0f32; FEAT_DIM];
                for j in 0..FEAT_DIM {
                    za[j] = (p.a[j] - best_gbox_model.mean[j]) * best_gbox_model.inv_std[j];
                    zb[j] = (p.b[j] - best_gbox_model.mean[j]) * best_gbox_model.inv_std[j];
                }
                let lin_diff = best_gbox_model.predict_features(&p.a) - best_gbox_model.predict_features(&p.b);
                za[0] = 0.0; za[1] = 0.0; za[4] = 0.0; za[8] = 0.0;
                zb[0] = 0.0; zb[1] = 0.0; zb[4] = 0.0; zb[8] = 0.0;
                (za, zb, (p.dy as f32) - lin_diff)
            })
            .collect();
        let mut best_gmlp = gmlp.clone();
        let mut best_gmlp_va = best_gbox_rmse;
        let mut ord: Vec<usize> = (0..g_zpairs.len()).collect();
        let g_epochs = 20;
        let g_l2 = 4.0e-2f32;
        for ep in 0..g_epochs {
            for i in (1..ord.len()).rev() {
                mrng = mrng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (mrng as usize) % (i + 1);
                ord.swap(i, j);
            }
            let lr = 3.5e-4f32 * (1.0 - 0.65 * (ep as f32 / g_epochs as f32));
            for &idx in &ord {
                let (za, zb, tgt) = &g_zpairs[idx];
                let mut ha = [0.0f32; MLP_HIDDEN];
                let mut hb = [0.0f32; MLP_HIDDEN];
                let mut oa = gmlp.b2;
                let mut ob = gmlp.b2;
                for h in 0..MLP_HIDDEN {
                    let mut sa = gmlp.b1[h];
                    let mut sb = gmlp.b1[h];
                    let row = &gmlp.w1[h];
                    for j in 0..FEAT_DIM {
                        sa += row[j] * za[j];
                        sb += row[j] * zb[j];
                    }
                    if sa > 0.0 { ha[h] = sa; oa += gmlp.w2[h] * sa; }
                    if sb > 0.0 { hb[h] = sb; ob += gmlp.w2[h] * sb; }
                }
                let err = ((oa - ob) - *tgt).clamp(-4.0, 4.0);
                for h in 0..MLP_HIDDEN {
                    let w2_h = gmlp.w2[h];
                    gmlp.w2[h] -= lr * (err * (ha[h] - hb[h]) + g_l2 * w2_h);
                    let ga = if ha[h] > 0.0 { err * w2_h } else { 0.0 };
                    let gb = if hb[h] > 0.0 { -err * w2_h } else { 0.0 };
                    if ga != 0.0 || gb != 0.0 {
                        gmlp.b1[h] -= lr * (ga + gb);
                        let row = &mut gmlp.w1[h];
                        for j in 0..FEAT_DIM {
                            row[j] -= lr * (ga * za[j] + gb * zb[j] + g_l2 * row[j]);
                        }
                    }
                }
            }
            let (tr_r, tr_a) = eval_model_pairs(&train_pairs, &gmlp, 0.0);
            let (va_r, va_a) = eval_model_pairs(&val_pairs, &gmlp, 0.0);
            if va_r < best_gmlp_va {
                best_gmlp_va = va_r;
                best_gmlp = gmlp.clone();
            }
            if ep == 0 || ep == 5 || ep == 11 || ep == g_epochs - 1 {
                println!(
                    "  GBox-MLP epoch {:2}: Train RMSE={:.3} Acc={:.1}% | Val RMSE={:.3} Acc={:.1}%",
                    ep + 1, tr_r, tr_a * 100.0, va_r, va_a * 100.0
                );
            }
        }
        best_gmlp.prepare_fast();
        save_one_model(&best_pure_model, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_pure.txt"));
        save_one_model(&best_gbox_model, concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox.txt"));
        save_one_model(&best_gmlp,       concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r3_gbox_mlp.txt"));
        println!(
            "\nSaved R3 models (best_gbox_val_rmse={:.3}, best_gmlp_val_rmse={:.3}, total dt={:.1}s)\n",
            best_gbox_rmse,
            best_gmlp_va,
            t_start.elapsed().as_secs_f64()
        );
        return;
    }

    save_models(&best_res_lin, &best_mlp);
    let pure_save = if is_r2 {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_pure.txt")
    } else {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r1_pure.txt")
    };
    let gbox_save = if is_r2 {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r2_gbox.txt")
    } else {
        concat!(env!("CARGO_MANIFEST_DIR"), "/../models/crn_r1_gbox.txt")
    };
    save_one_model(&best_pure_model, pure_save);
    save_one_model(&best_gbox_model, gbox_save);
    println!(
        "\nSaved models to {}, {}, {} (best_mlp_val_rmse={:.3}, total dt={:.1}s)\n",
        CRN_MODEL_PATH,
        pure_save,
        gbox_save,
        best_mlp_val_rmse,
        t_start.elapsed().as_secs_f64()
    );
    if mode == "fit_save" || mode == "train_r2_save" || mode == "fit_r2" {
        return;
    }

    // Train-20 Evaluation Suite
    let mut c_base = BeamConfig::default();
    c_base.beam_width = 120;
    c_base.max_per_root = 24;
    c_base.rollout_top_k = 12;
    c_base.no_g_credit = 180;
    c_base.endgame_wide = true;
    c_base.weights = w_clean.clone();

    let mut c_pure = c_base.clone();
    c_pure.no_g_credit = 0;
    c_pure.value_fast_micro = 0;
    c_pure.value_model = Some(Arc::new(best_pure_model));

    let res_ng_credit = (180 * fast_micro) / 10_000;

    let mut c_res_lin = c_base.clone();
    c_res_lin.no_g_credit = res_ng_credit;
    c_res_lin.value_fast_micro = fast_micro;
    c_res_lin.value_model = Some(Arc::new(best_res_lin));

    let mut c_res_mlp = c_base.clone();
    c_res_mlp.no_g_credit = res_ng_credit;
    c_res_mlp.value_fast_micro = fast_micro;
    c_res_mlp.value_model = Some(Arc::new(best_mlp));

    let mut c_pure_pimc = c_pure.clone();
    c_pure_pimc.max_per_root = 16;
    c_pure_pimc.rollout_top_k = 16;
    c_pure_pimc.pimc_worlds = 6;
    c_pure_pimc.pimc_horizon = 4;
    c_pure_pimc.pimc_roots = 6;
    c_pure_pimc.pimc_width = 16;
    c_pure_pimc.pimc_margin = 450;
    c_pure_pimc.pimc_spawn = true;

    let mut c_res_pimc = c_res_lin.clone();
    c_res_pimc.max_per_root = 16;
    c_res_pimc.rollout_top_k = 16;
    c_res_pimc.pimc_worlds = 6;
    c_res_pimc.pimc_horizon = 4;
    c_res_pimc.pimc_roots = 6;
    c_res_pimc.pimc_width = 16;
    c_res_pimc.pimc_margin = 450;
    c_res_pimc.pimc_spawn = true;

    if mode == "fit_pimc_pure" {
        let (m, mn, mx, top, nds, dt) = eval_batch(&train20, &c_pure_pimc);
        println!(
            "{:<22} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
            "crn_pure_lin_pimc", m, mn, mx, top, nds, dt
        );
        return;
    }
    if mode == "fit_pimc_res" || mode == "fit_pimc" {
        let (m, mn, mx, top, nds, dt) = eval_batch(&train20, &c_res_pimc);
        println!(
            "{:<22} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
            "crn_res_lin_pimc", m, mn, mx, top, nds, dt
        );
        return;
    }

    for (name, cfg) in [
        ("crn_pure_lin_b120", &c_pure),
        ("crn_res_lin_b120", &c_res_lin),
        ("crn_res_lin_pimc", &c_res_pimc),
    ] {
        let (m, mn, mx, top, nds, dt) = eval_batch(&train20, cfg);
        println!(
            "{:<22} mean={:6.2} min={:3} max={:3} top={:2} nds={:5.2} dt={:.2}s",
            name, m, mn, mx, top, nds, dt
        );
    }
}
