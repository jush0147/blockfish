use super::{
    cheese_val::LearnedValueModel,
    fast::{
        color_to_idx, idx_to_color, FastBoard, FastPlace, FastPlaceGen, FastShapeTable,
        EMPTY_ROW,
    },
    fast_eval::{eval_fast_with_bag, EvalWeights},
    spatial_val::SpatialValueModel,
    Snapshot,
};
use crate::{finesse::FinesseFinder, shape::ShapeTable, BasicMatrix, Color, Input};
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::Arc;

#[derive(Default)]
pub struct FxHasher {
    hash: u64,
}

impl Hasher for FxHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut h = self.hash;
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let k = u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]);
            h = (h.rotate_left(5) ^ k).wrapping_mul(0x517cc1b727220a95);
        }
        for &b in chunks.remainder() {
            h = (h.rotate_left(5) ^ (b as u64)).wrapping_mul(0x517cc1b727220a95);
        }
        self.hash = h;
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.hash = (self.hash.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.hash = (self.hash.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517cc1b727220a95);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(0x517cc1b727220a95);
    }
}

type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// Legal game state context observable on screen / HUD.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GameContext {
    /// Number of bottom rows that are garbage lines (gray 'G' rows).
    /// If `None`, inferred as contiguous bottom rows with 9 cells.
    pub garbage_rows: Option<u16>,
    /// Remaining garbage lines in the Cheese Race (HUD counter, e.g. `100 - ds`).
    /// If `None`, defaults to `garbage_rows`.
    pub remaining_garbage: Option<u16>,
    /// Bitmask (bits 0..7) of pieces still remaining in the current 7-bag after visible queue.
    /// If `None`, defaults to all 7 pieces (`0x7F`).
    pub next_bag_mask: Option<u8>,
}

/// Legally tracks the 7-bag state purely from visible `(hold, queue)` observations.
#[derive(Clone, Debug)]
pub struct BagTracker {
    bag_mask: u8,
    initialized: bool,
    prev_had_hold: bool,
    prev_queue_tail: Option<Vec<Color>>,
}

impl Default for BagTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BagTracker {
    pub fn new() -> Self {
        Self {
            bag_mask: 0x7F,
            initialized: false,
            prev_had_hold: false,
            prev_queue_tail: None,
        }
    }

    fn consume_piece(&mut self, c: Color) {
        let bit = 1u8 << color_to_idx(c);
        if (self.bag_mask & bit) == 0 {
            // Bag rolled over (or desynced); refill to full bag before consuming
            self.bag_mask = 0x7F;
        }
        self.bag_mask &= !bit;
        if self.bag_mask == 0 {
            self.bag_mask = 0x7F;
        }
    }

    /// Observe a legal `Snapshot` and return the bitmask of pieces remaining in the current bag
    /// after the visible queue.
    pub fn observe(&mut self, hold: Option<Color>, queue: &[Color]) -> u8 {
        if !self.initialized || queue.is_empty() {
            self.bag_mask = 0x7F;
            for &c in queue {
                self.consume_piece(c);
            }
            self.initialized = true;
            self.prev_had_hold = hold.is_some();
            self.prev_queue_tail = Some(queue.to_vec());
            return self.bag_mask;
        }

        // Normally 1 new piece appears at the end of `queue` after each hard drop.
        // If hold transitioned from None -> Some, an extra piece was drawn from `queue` during hold.
        let newly_drawn = if !self.prev_had_hold && hold.is_some() && queue.len() >= 2 {
            2
        } else {
            1
        };

        let start = queue.len().saturating_sub(newly_drawn);
        for &c in &queue[start..] {
            self.consume_piece(c);
        }
        self.prev_had_hold = hold.is_some();
        self.prev_queue_tail = Some(queue.to_vec());
        self.bag_mask
    }
}

#[derive(Clone, Debug)]
pub struct BeamConfig {
    pub beam_width: usize,
    pub max_per_root: usize,
    pub rollout_top_k: usize,
    pub rollout_horizon: usize,
    pub rollout_samples: usize,
    pub rerank_roots: usize,
    pub rerank_margin: i64,
    pub rollout_spawn_g0: bool,
    pub endgame_wide: bool,
    pub pimc_worlds: usize,
    pub pimc_horizon: usize,
    pub pimc_roots: usize,
    pub pimc_width: usize,
    pub pimc_margin: i64,
    pub pimc_spawn: bool,
    pub pimc_weight_pct: i64,
    pub progressive_mpr: bool,
    pub stratify_rem_g: bool,
    pub pimc_seed_leaves: usize,
    pub simulate_cheese_spawn: bool,
    pub no_g_credit: i64,
    pub root_pimc_worlds: usize,
    pub root_pimc_horizon: usize,
    pub root_pimc_roots: usize,
    pub root_pimc_width: usize,
    pub root_pimc_margin: i64,
    pub root_pimc_weight_pct: i64,
    pub root_pimc_steady_g4: bool,
    pub bag_det_extend: bool,
    pub fog_spawn: bool,
    pub endgame_solver: bool,
    pub rollout_per_root: usize,
    pub weights: EvalWeights,
    pub value_model: Option<Arc<LearnedValueModel>>,
    pub spatial_model: Option<Arc<SpatialValueModel>>,
    pub value_fast_micro: i64,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 240,
            max_per_root: 48,
            rollout_top_k: 24,
            rollout_horizon: 0,
            rollout_samples: 1,
            rerank_roots: 0,
            rerank_margin: 900,
            rollout_spawn_g0: false,
            endgame_wide: false,
            pimc_worlds: 0,
            pimc_horizon: 0,
            pimc_roots: 4,
            pimc_width: 12,
            pimc_margin: 700,
            pimc_spawn: false,
            pimc_weight_pct: 67,
            progressive_mpr: false,
            stratify_rem_g: false,
            pimc_seed_leaves: 3,
            simulate_cheese_spawn: false,
            no_g_credit: 0,
            root_pimc_worlds: 0,
            root_pimc_horizon: 0,
            root_pimc_roots: 6,
            root_pimc_width: 8,
            root_pimc_margin: 1200,
            root_pimc_weight_pct: 100,
            root_pimc_steady_g4: false,
            bag_det_extend: false,
            fog_spawn: false,
            endgame_solver: false,
            rollout_per_root: 1,
            weights: EvalWeights::default(),
            value_model: None,
            spatial_model: None,
            value_fast_micro: 0,
        }
    }
}

#[derive(Clone, Copy)]
struct BeamNode {
    board: FastBoard,
    held: Option<u8>,
    q_cursor: u8,
    rem_garbage: u16,
    no_g_moves: u8,
    first_idx: u16,
    score: i64,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct DedupKey {
    board: FastBoard,
    held: Option<u8>,
    rem_garbage: u16,
}

pub struct BeamResult {
    pub best_place: FastPlace,
    pub inputs: Vec<Input>,
    pub rating: i64,
    pub nodes_evaluated: usize,
}

pub struct BeamSearcher {
    fst: FastShapeTable,
    pgen: FastPlaceGen,
    ffind: FinesseFinder,
    cur_beam: Vec<BeamNode>,
    children: Vec<BeamNode>,
    root_places: Vec<FastPlace>,
    tmp_places: Vec<FastPlace>,
    roll_places: Vec<FastPlace>,
    seen: HashSet<DedupKey, FxBuildHasher>,
    root_counts: Vec<usize>,
    per_root_rolled: HashMap<u16, usize, FxBuildHasher>,
    best_per_root: HashMap<u16, i64, FxBuildHasher>,
    best_leaf_idx: HashMap<u16, usize, FxBuildHasher>,
    pimc_beam: Vec<BeamNode>,
    pimc_next: Vec<BeamNode>,
    pimc_seen: HashSet<DedupKey, FxBuildHasher>,
    last_top: Vec<(char, i16, i16, u8, bool, i64)>,
    last_top_leaves: Vec<(u16, u16, i64, u8, u8)>,
    last_q: [u8; 16],
    last_q_len: usize,
    last_bag_mask: u8,
}

impl BeamSearcher {
    pub fn new(shtb: &ShapeTable) -> Self {
        Self {
            fst: FastShapeTable::from_shape_table(shtb),
            pgen: FastPlaceGen::new(),
            ffind: FinesseFinder::new(),
            cur_beam: Vec::with_capacity(512),
            children: Vec::with_capacity(16384),
            root_places: Vec::with_capacity(64),
            tmp_places: Vec::with_capacity(64),
            roll_places: Vec::with_capacity(64),
            seen: HashSet::with_capacity_and_hasher(16384, FxBuildHasher::default()),
            root_counts: Vec::with_capacity(64),
            per_root_rolled: HashMap::with_capacity_and_hasher(64, FxBuildHasher::default()),
            best_per_root: HashMap::with_capacity_and_hasher(64, FxBuildHasher::default()),
            best_leaf_idx: HashMap::with_capacity_and_hasher(64, FxBuildHasher::default()),
            pimc_beam: Vec::with_capacity(64),
            pimc_next: Vec::with_capacity(512),
            pimc_seen: HashSet::with_capacity_and_hasher(512, FxBuildHasher::default()),
            last_top: Vec::with_capacity(16),
            last_top_leaves: Vec::with_capacity(16),
            last_q: [0u8; 16],
            last_q_len: 0,
            last_bag_mask: 0x7F,
        }
    }

    pub fn last_top_candidates(&self) -> &[(char, i16, i16, u8, bool, i64)] {
        &self.last_top
    }

    pub fn last_top_leaf_info(&self) -> &[(u16, u16, i64, u8, u8)] {
        &self.last_top_leaves
    }

    pub fn diverse_beam_leaves(&self, max_siblings: usize) -> Vec<(FastBoard, Option<u8>, u16, u8)> {
        let mut out = Vec::with_capacity(max_siblings);
        let mut used_roots: Vec<u16> = Vec::with_capacity(max_siblings);
        for node in &self.cur_beam {
            if out.len() >= max_siblings {
                break;
            }
            if node.rem_garbage == 0 || used_roots.contains(&node.first_idx) {
                continue;
            }
            used_roots.push(node.first_idx);
            let hc = Self::hand_piece(node.held, node.q_cursor as usize, &self.last_q[..self.last_q_len]);
            out.push((node.board, hc, node.rem_garbage, self.last_bag_mask));
        }
        if out.len() < max_siblings {
            for node in self.cur_beam.iter().step_by(5) {
                if out.len() >= max_siblings {
                    break;
                }
                if node.rem_garbage == 0 {
                    continue;
                }
                let hc = Self::hand_piece(node.held, node.q_cursor as usize, &self.last_q[..self.last_q_len]);
                if !out.iter().any(|x| x.0 == node.board && x.1 == hc && x.2 == node.rem_garbage) {
                    out.push((node.board, hc, node.rem_garbage, self.last_bag_mask));
                }
            }
        }
        out
    }

    pub fn search(
        &mut self,
        shtb: &ShapeTable,
        snapshot: &Snapshot,
        ctx: GameContext,
        cfg: &BeamConfig,
    ) -> Option<BeamResult> {
        let garbage_rows = ctx
            .garbage_rows
            .unwrap_or_else(|| FastBoard::infer_garbage_rows(&snapshot.matrix));
        let root_board = FastBoard::from_basic_matrix(&snapshot.matrix, garbage_rows);
        let rem_g0 = ctx
            .remaining_garbage
            .unwrap_or(garbage_rows)
            .max(garbage_rows);
        let is_endgame_exact = cfg.endgame_solver && rem_g0 > 0 && rem_g0 == garbage_rows && rem_g0 <= 9;
        let is_pure_endgame =
            is_endgame_exact || (cfg.endgame_wide && rem_g0 <= 7 && rem_g0 == garbage_rows);
        let eff_beam_width = if is_endgame_exact {
            cfg.beam_width.max(480)
        } else if is_pure_endgame {
            cfg.beam_width.max(360)
        } else {
            cfg.beam_width
        };
        let eff_max_per_root = if is_endgame_exact {
            cfg.max_per_root.max(96)
        } else if is_pure_endgame {
            cfg.max_per_root.max(80)
        } else {
            cfg.max_per_root
        };
        let mut bag_mask = match ctx.next_bag_mask {
            Some(m) if m != 0 => m,
            _ => 0x7F,
        };

        let hold0 = snapshot.hold.map(|c| color_to_idx(c) as u8);
        let mut q = [0u8; 16];
        let mut q_len = snapshot.queue.len().min(16);
        for (i, &c) in snapshot.queue.iter().take(q_len).enumerate() {
            q[i] = color_to_idx(c) as u8;
        }
        // Legal 7-bag deterministic extension: when only 1 piece remains in the bag after the visible queue,
        // the next piece is 100% uniquely determined!
        if cfg.bag_det_extend && q_len < 16 && bag_mask.count_ones() == 1 {
            q[q_len] = bag_mask.trailing_zeros() as u8;
            q_len += 1;
            bag_mask = 0x7F;
        }
        self.last_q = q;
        self.last_q_len = q_len;
        self.last_bag_mask = bag_mask;
        let total_known = q_len + usize::from(hold0.is_some());
        if total_known == 0 {
            return None;
        }
        // Stop main beam with 1 known piece left in hand so 7-bag rollout can test it with piece #7,
        // unless there is only 1 piece total or we are in exact endgame solver mode.
        let target_depth = if is_endgame_exact {
            total_known.max(1)
        } else if total_known > 1 {
            total_known - 1
        } else {
            1
        };

        // Generate root moves
        let cur0 = if q_len > 0 { Some(q[0]) } else { None };
        let alt0 = if hold0.is_some() {
            hold0
        } else if q_len > 1 {
            Some(q[1])
        } else {
            None
        };
        self.pgen
            .generate_for_state(&self.fst, &root_board, cur0, alt0, &mut self.root_places);
        if self.root_places.is_empty() {
            return None;
        }

        let mut nodes_evaluated = 0usize;
        let mut best_goal: Option<(usize, u16, i64)> = None; // (depth, root_idx, rating)

        self.cur_beam.clear();
        self.seen.clear();

        let mut crn_seed: u64 = 0x9E3779B97F4A7C15u64
            ^ (rem_g0 as u64).wrapping_mul(0xBF58476D1CE4E5B9)
            ^ ((bag_mask as u64) << 16);
        for r in 0..(root_board.rows as usize) {
            crn_seed = crn_seed.rotate_left(7) ^ (root_board.data[r] as u64);
        }
        let beam_spawn_rng = crn_seed ^ 0x94D049BB133111EB;

        for (idx, &fp) in self.root_places.iter().enumerate() {
            let mut b = root_board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem_g = rem_g0.saturating_sub(ds as u16);
            if !is_endgame_exact && rem_g > 0 {
                if cfg.fog_spawn {
                    if !b.spawn_fog_rows(rem_g, lines) {
                        continue;
                    }
                } else if cfg.simulate_cheese_spawn {
                    let mut r_local = beam_spawn_rng;
                    if !b.spawn_cheese_rows(rem_g, lines, &mut r_local) {
                        continue;
                    }
                }
            }

            let (held, q_cursor) = if hold0.is_some() {
                if fp.did_hold {
                    (cur0, if cur0.is_some() { 1 } else { 0 })
                } else {
                    (hold0, 1)
                }
            } else if fp.did_hold {
                (cur0, 2)
            } else {
                (None, 1)
            };

            if rem_g == 0 && b.rows < 20 {
                let g_rating = -30_000_000 + 100 + (b.rows as i64);
                if best_goal.map_or(true, |(_, _, r)| g_rating < r) {
                    best_goal = Some((1, idx as u16, g_rating));
                }
            }

            let no_g_inc = u8::from(root_board.garbage_rows == 0 && rem_g0 > 0);
            let in_hand = Self::hand_piece(held, q_cursor, &q[..q_len]);
            let avail = Self::avail_mask_from_q(held, q_cursor, &q[..q_len], bag_mask);
            let sc = if is_endgame_exact {
                Self::eval_state(&b, rem_g, in_hand, avail, cfg)
            } else {
                Self::eval_state(&b, rem_g, in_hand, avail, cfg) - cfg.no_g_credit * (no_g_inc as i64)
            };
            nodes_evaluated += 1;

            let key = DedupKey {
                board: b,
                held,
                rem_garbage: rem_g,
            };
            if self.seen.insert(key) {
                self.cur_beam.push(BeamNode {
                    board: b,
                    held,
                    q_cursor: q_cursor as u8,
                    rem_garbage: rem_g,
                    no_g_moves: no_g_inc,
                    first_idx: idx as u16,
                    score: sc,
                });
            }
        }

        // Expand depths 2 ..= target_depth
        if !(is_endgame_exact && best_goal.is_some()) {
        for depth in 2..=target_depth {
            self.children.clear();
            let depth_rng_base =
                beam_spawn_rng.wrapping_add((depth as u64).wrapping_mul(0x517CC1B727220A95));
            for node_idx in 0..self.cur_beam.len() {
                let node = self.cur_beam[node_idx];
                if node.rem_garbage == 0 {
                    continue;
                }
                let qc = node.q_cursor as usize;
                let (c_nh, c_h) = if node.held.is_some() {
                    let c1 = if qc < q_len { Some(q[qc]) } else { None };
                    (c1, node.held)
                } else {
                    let c1 = if qc < q_len { Some(q[qc]) } else { None };
                    let c2 = if qc + 1 < q_len { Some(q[qc + 1]) } else { None };
                    (c1, c2)
                };

                self.pgen
                    .generate_for_state(&self.fst, &node.board, c_nh, c_h, &mut self.tmp_places);

                let no_g_inc = u8::from(node.board.garbage_rows == 0 && node.rem_garbage > 0);
                let ng = node.no_g_moves.saturating_add(no_g_inc);

                for &fp in &self.tmp_places {
                    let mut b = node.board;
                    let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b.place_and_sift(orient, fp.norm_i, fp.norm_j);
                    let rem_g = node.rem_garbage.saturating_sub(ds as u16);
                    if !is_endgame_exact && rem_g > 0 {
                        if cfg.fog_spawn {
                            if !b.spawn_fog_rows(rem_g, lines) {
                                continue;
                            }
                        } else if cfg.simulate_cheese_spawn {
                            let mut r_local = depth_rng_base;
                            if !b.spawn_cheese_rows(rem_g, lines, &mut r_local) {
                                continue;
                            }
                        }
                    }

                    let (held, next_qc) = if node.held.is_some() {
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

                    if rem_g == 0 && b.rows < 20 {
                        let g_rating = -30_000_000 + (depth as i64) * 100 + (b.rows as i64);
                        if best_goal.map_or(true, |(_, _, r)| g_rating < r) {
                            best_goal = Some((depth, node.first_idx, g_rating));
                        }
                    }

                    let in_hand = Self::hand_piece(held, next_qc, &q[..q_len]);
                    let avail = Self::avail_mask_from_q(held, next_qc, &q[..q_len], bag_mask);
                    let sc = if is_endgame_exact {
                        Self::eval_state(&b, rem_g, in_hand, avail, cfg)
                    } else {
                        Self::eval_state(&b, rem_g, in_hand, avail, cfg)
                            - cfg.no_g_credit * (ng as i64)
                    };
                    nodes_evaluated += 1;

                    self.children.push(BeamNode {
                        board: b,
                        held,
                        q_cursor: next_qc as u8,
                        rem_garbage: rem_g,
                        no_g_moves: ng,
                        first_idx: node.first_idx,
                        score: sc,
                    });
                }
            }

            if self.children.is_empty() {
                break;
            }

            // Sort children by ascending score (best first)
            self.children.sort_unstable_by_key(|n| (n.score, n.first_idx));

            // Select next beam with deduplication and per-root diversity cap
            let step_mpr = if cfg.progressive_mpr && !is_pure_endgame {
                ((eff_max_per_root * depth) / target_depth.max(1)).max(4).min(eff_max_per_root)
            } else {
                eff_max_per_root
            };
            self.cur_beam.clear();
            self.seen.clear();
            self.root_counts.clear();
            self.root_counts.resize(self.root_places.len(), 0);

            if cfg.stratify_rem_g && !is_pure_endgame {
                let mut g_counts = [0usize; 16];
                let bucket_cap = (eff_beam_width / 4).max(12);
                let strat_total_cap = (eff_beam_width * 3) / 4;
                for i in 0..self.children.len() {
                    if self.cur_beam.len() >= strat_total_cap {
                        break;
                    }
                    let ch = self.children[i];
                    let b_idx = rem_g0.saturating_sub(ch.rem_garbage).min(15) as usize;
                    if g_counts[b_idx] >= bucket_cap
                        || self.root_counts[ch.first_idx as usize] >= step_mpr
                    {
                        continue;
                    }
                    let key = DedupKey {
                        board: ch.board,
                        held: ch.held,
                        rem_garbage: ch.rem_garbage,
                    };
                    if self.seen.insert(key) {
                        g_counts[b_idx] += 1;
                        self.root_counts[ch.first_idx as usize] += 1;
                        self.cur_beam.push(ch);
                    }
                }
            }

            for i in 0..self.children.len() {
                if self.cur_beam.len() >= eff_beam_width {
                    break;
                }
                let ch = self.children[i];
                if self.root_counts[ch.first_idx as usize] >= step_mpr {
                    continue;
                }
                let key = DedupKey {
                    board: ch.board,
                    held: ch.held,
                    rem_garbage: ch.rem_garbage,
                };
                if self.seen.insert(key) {
                    self.root_counts[ch.first_idx as usize] += 1;
                    self.cur_beam.push(ch);
                }
            }

            // Fill remaining beam slots if diversity cap left space
            if self.cur_beam.len() < eff_beam_width {
                for i in 0..self.children.len() {
                    if self.cur_beam.len() >= eff_beam_width {
                        break;
                    }
                    let ch = self.children[i];
                    let key = DedupKey {
                        board: ch.board,
                        held: ch.held,
                        rem_garbage: ch.rem_garbage,
                    };
                    if self.seen.insert(key) {
                        self.cur_beam.push(ch);
                    }
                }
            }

            if cfg.stratify_rem_g && !is_pure_endgame {
                self.cur_beam.sort_unstable_by_key(|n| (n.score, n.first_idx));
            }

            if is_endgame_exact && best_goal.is_some() {
                break;
            }
        }
        }

        // Also check deterministic final-piece goal if near endgame (e.g. <= 4 garbage lines left)
        if rem_g0 <= (total_known as u16) + 1 && best_goal.is_none() {
            for i in 0..self.cur_beam.len() {
                let node = self.cur_beam[i];
                if node.rem_garbage == 0 || node.rem_garbage > 1 {
                    continue;
                }
                if let Some(hand_c) = Self::hand_piece(node.held, node.q_cursor as usize, &q[..q_len])
                {
                    self.pgen.generate_for_state(
                        &self.fst,
                        &node.board,
                        Some(hand_c),
                        None,
                        &mut self.tmp_places,
                    );
                    for &fp in &self.tmp_places {
                        let mut b = node.board;
                        let orient =
                            &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (_, ds, _) = b.place_and_sift(orient, fp.norm_i, fp.norm_j);
                        if node.rem_garbage.saturating_sub(ds as u16) == 0 && b.rows < 20 {
                            let g_rating = -30_000_000
                                + ((target_depth + 1) as i64) * 100
                                + (b.rows as i64);
                            if best_goal.map_or(true, |(_, _, r)| g_rating < r) {
                                best_goal = Some((target_depth + 1, node.first_idx, g_rating));
                            }
                        }
                    }
                }
            }
        }

        self.best_per_root.clear();
        self.best_leaf_idx.clear();

        if !self.cur_beam.is_empty() && total_known > 1 && best_goal.is_none() {
            // Stage 1: 1-ply 7-bag rollout on top K leaves AND best `rollout_per_root` leaves per root
            // (If rerank_roots > 0, defer deep rollout_horizon to Stage 2 top roots)
            let s1_horizon = if cfg.rerank_roots > 0 { 0 } else { cfg.rollout_horizon };
            let max_per_root = if is_endgame_exact {
                cfg.rollout_per_root.max(2)
            } else {
                cfg.rollout_per_root.max(1)
            };
            self.per_root_rolled.clear();

            for i in 0..self.cur_beam.len() {
                let node = self.cur_beam[i];
                let rolled_cnt = self.per_root_rolled.get(&node.first_idx).copied().unwrap_or(0);
                let need_for_root = rolled_cnt < max_per_root;
                if i >= cfg.rollout_top_k && !need_for_root {
                    continue;
                }

                let hand_c = Self::hand_piece(node.held, node.q_cursor as usize, &q[..q_len]);
                let sc = if is_endgame_exact {
                    let (r_sc, evals) = self.endgame_rollout_crn(
                        node.board,
                        hand_c,
                        node.rem_garbage,
                        bag_mask,
                        crn_seed,
                        8,
                        cfg,
                    );
                    nodes_evaluated += evals;
                    r_sc
                } else if let Some(hc) = hand_c {
                    let (r_sc, evals) = self.rollout_7bag(
                        node.board,
                        hc,
                        node.rem_garbage,
                        node.no_g_moves,
                        bag_mask,
                        crn_seed,
                        s1_horizon,
                        cfg,
                    );
                    nodes_evaluated += evals;
                    r_sc
                } else {
                    node.score
                };

                *self.per_root_rolled.entry(node.first_idx).or_insert(0) += 1;

                let entry = self.best_per_root.entry(node.first_idx).or_insert(i64::MAX);
                if sc < *entry {
                    *entry = sc;
                    self.best_leaf_idx.insert(node.first_idx, i);
                }
            }
        } else {
            for (i, &node) in self.cur_beam.iter().enumerate() {
                let entry = self.best_per_root.entry(node.first_idx).or_insert(i64::MAX);
                if node.score < *entry {
                    *entry = node.score;
                    self.best_leaf_idx.insert(node.first_idx, i);
                }
            }
        }
        if let Some((_, root_idx, g_rating)) = best_goal {
            let entry = self.best_per_root.entry(root_idx).or_insert(i64::MAX);
            if g_rating < *entry {
                *entry = g_rating;
            }
        }

        // Sort candidate root moves by best rating and verify finesse reachability
        let mut candidates: Vec<(u16, i64)> = self.best_per_root.iter().map(|(&k, &v)| (k, v)).collect();
        candidates.sort_unstable_by_key(|&(idx, r)| (r, idx));

        // Stage 2: Deep CRN multi-ply rollout reranking on close top roots
        if cfg.rerank_roots > 0
            && cfg.rollout_horizon > 0
            && total_known > 1
            && best_goal.is_none()
            && !is_endgame_exact
            && candidates.len() > 1
        {
            let best_s1 = candidates[0].1;
            let k_limit = cfg.rerank_roots.min(candidates.len());
            let mut reranked: Vec<(usize, i64)> = Vec::with_capacity(k_limit);
            for (ci, &(root_idx, s1_sc)) in candidates.iter().take(k_limit).enumerate() {
                if ci > 0 && s1_sc - best_s1 > cfg.rerank_margin {
                    break;
                }
                if let Some(&leaf_i) = self.best_leaf_idx.get(&root_idx) {
                    let node = self.cur_beam[leaf_i];
                    if let Some(hc) = Self::hand_piece(node.held, node.q_cursor as usize, &q[..q_len]) {
                        let (deep_sc, evals) = self.rollout_7bag(
                            node.board,
                            hc,
                            node.rem_garbage,
                            node.no_g_moves,
                            bag_mask,
                            crn_seed,
                            cfg.rollout_horizon,
                            cfg,
                        );
                        nodes_evaluated += evals;
                        reranked.push((ci, deep_sc));
                    }
                }
            }
            if reranked.len() > 1 {
                let top_deep_min = reranked.iter().map(|x| x.1).min().unwrap_or(0);
                // Re-order only the reranked prefix while keeping scores comparable to Stage 1
                let shift = best_s1 - top_deep_min;
                for (ci, deep_sc) in reranked {
                    candidates[ci].1 = (candidates[ci].1 + 2 * (deep_sc + shift)) / 3;
                }
                candidates.sort_unstable_by_key(|&(idx, r)| (r, idx));
            }
        }

        // Stage 3: PIMC Multi-World Beam Continuation on close top roots
        if cfg.pimc_worlds > 0
            && cfg.pimc_horizon > 0
            && total_known > 1
            && best_goal.is_none()
            && !is_endgame_exact
            && candidates.len() > 1
        {
            let best_s1 = candidates[0].1;
            let k_limit = cfg.pimc_roots.max(1).min(candidates.len());
            let mut active_k = 0usize;
            for (ci, &(_, s1_sc)) in candidates.iter().take(k_limit).enumerate() {
                if ci > 0 && s1_sc - best_s1 > cfg.pimc_margin {
                    break;
                }
                active_k += 1;
            }
            if active_k > 1 {
                let h = cfg.pimc_horizon.min(8);
                let w_cnt = cfg.pimc_worlds.min(16);
                let mut world_seqs = [[0u8; 8]; 16];
                let mut world_spawns = [0u64; 16];

                // Stratify first unknown piece across active bits of bag_mask to cut MC variance
                let mut bag_bits = [0u8; 7];
                let mut bag_n = 0usize;
                let mut bm = bag_mask;
                while bm != 0 {
                    bag_bits[bag_n] = bm.trailing_zeros() as u8;
                    bag_n += 1;
                    bm &= bm - 1;
                }
                let mut shuffle_rng = crn_seed ^ 0xA0761D6478BD642F;
                for i in (1..bag_n).rev() {
                    shuffle_rng = shuffle_rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = ((shuffle_rng >> 32) as usize) % (i + 1);
                    bag_bits.swap(i, j);
                }

                for w in 0..w_cnt {
                    let mut bag = bag_mask;
                    let mut w_rng = crn_seed
                        ^ ((w as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15)
                        ^ 0xBF58476D1CE4E5B9;
                    if bag_n > 0 {
                        let p0 = bag_bits[w % bag_n];
                        world_seqs[w][0] = p0;
                        bag &= !(1u8 << p0);
                        if bag == 0 {
                            bag = 0x7F;
                        }
                        for step in 1..h {
                            world_seqs[w][step] = Self::draw_from_bag(&mut bag, &mut w_rng);
                        }
                    } else {
                        for step in 0..h {
                            world_seqs[w][step] = Self::draw_from_bag(&mut bag, &mut w_rng);
                        }
                    }
                    world_spawns[w] = w_rng ^ 0xD6E8FEB86659FD93;
                }

                let mut pimc_scores: [i64; 16] = [0; 16];
                let eval_k = active_k.min(16);
                for ci in 0..eval_k {
                    let root_idx = candidates[ci].0;
                    let target_seeds = cfg.pimc_seed_leaves.clamp(1, 12);
                    let mut seeds = [BeamNode {
                        board: root_board,
                        held: None,
                        q_cursor: 0,
                        rem_garbage: 0,
                        no_g_moves: 0,
                        first_idx: root_idx,
                        score: 0,
                    }; 12];
                    let mut seed_cnt = 0usize;
                    if let Some(&leaf_i) = self.best_leaf_idx.get(&root_idx) {
                        seeds[0] = self.cur_beam[leaf_i];
                        seed_cnt = 1;
                    }
                    // First pass: ensure diverse rem_garbage leaves enter PIMC for this root
                    for &node in &self.cur_beam {
                        if seed_cnt >= target_seeds {
                            break;
                        }
                        if node.first_idx == root_idx
                            && !seeds[..seed_cnt].iter().any(|s| s.rem_garbage == node.rem_garbage)
                        {
                            seeds[seed_cnt] = node;
                            seed_cnt += 1;
                        }
                    }
                    // Second pass: fill remaining seed slots by best score
                    for &node in &self.cur_beam {
                        if seed_cnt >= target_seeds {
                            break;
                        }
                        if node.first_idx == root_idx
                            && !seeds[..seed_cnt].iter().any(|s| {
                                s.board == node.board && s.held == node.held && s.rem_garbage == node.rem_garbage
                            })
                        {
                            seeds[seed_cnt] = node;
                            seed_cnt += 1;
                        }
                    }

                    let mut sum_w: i64 = 0;
                    for w in 0..w_cnt {
                        let (w_sc, evals) = self.pimc_eval_root(
                            &seeds[..seed_cnt],
                            &q[..q_len],
                            &world_seqs[w][..h],
                            world_spawns[w],
                            cfg,
                        );
                        nodes_evaluated += evals;
                        sum_w += w_sc;
                    }
                    pimc_scores[ci] = sum_w / (w_cnt as i64);
                }

                let min_pimc = pimc_scores[..eval_k].iter().copied().min().unwrap_or(0);
                let shift = best_s1 - min_pimc;
                let pw = cfg.pimc_weight_pct.clamp(0, 100);
                for ci in 0..eval_k {
                    candidates[ci].1 = ((100 - pw) * candidates[ci].1 + pw * (pimc_scores[ci] + shift)) / 100;
                }
                candidates[..eval_k].sort_unstable_by_key(|&(idx, r)| (r, idx));
            }
        }

        // Stage 4: Root-Level CRN-PIMC (evaluate top candidate roots directly from depth 1 across
        // W deterministic (hole-stream, bag-tail) worlds with true Cheese spawn at every step).
        let skip_root_pimc = is_endgame_exact || (rem_g0 <= (total_known as u16) && rem_g0 == garbage_rows);
        if cfg.root_pimc_worlds > 0
            && cfg.root_pimc_horizon > 0
            && total_known > 1
            && best_goal.is_none()
            && !skip_root_pimc
            && candidates.len() > 1
        {
            let best_s1 = candidates[0].1;
            let k_limit = cfg.root_pimc_roots.clamp(1, 14).min(candidates.len());
            let mut active_roots: Vec<(usize, i64)> = Vec::with_capacity(k_limit + 2);
            let mut has_0ds = false;
            let mut has_ds1 = false;

            for (ci, &(root_idx, s1_sc)) in candidates.iter().enumerate() {
                let fp = self.root_places[root_idx as usize];
                let mut b_tmp = root_board;
                let o_tmp = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (_, ds_tmp, _) = b_tmp.place_and_sift(o_tmp, fp.norm_i, fp.norm_j);
                if active_roots.len() < k_limit && (ci == 0 || s1_sc - best_s1 <= cfg.root_pimc_margin)
                {
                    active_roots.push((ci, s1_sc));
                    if ds_tmp == 0 {
                        has_0ds = true;
                    } else {
                        has_ds1 = true;
                    }
                } else {
                    // Ensure at least one competitive 0ds and ds>0 alternative enters CRN-PIMC
                    if !has_0ds && ds_tmp == 0 && s1_sc - best_s1 <= cfg.root_pimc_margin * 2 {
                        active_roots.push((ci, s1_sc));
                        has_0ds = true;
                    } else if !has_ds1 && ds_tmp > 0 && s1_sc - best_s1 <= cfg.root_pimc_margin * 2 {
                        active_roots.push((ci, s1_sc));
                        has_ds1 = true;
                    }
                    if active_roots.len() >= k_limit + 2 || (has_0ds && has_ds1 && active_roots.len() >= k_limit) {
                        break;
                    }
                }
            }

            if active_roots.len() > 1 {
                let w_cnt = cfg.root_pimc_worlds.clamp(1, 24);
                let h = if cfg.root_pimc_horizon == 99 {
                    target_depth.saturating_sub(1).max(1)
                } else {
                    cfg.root_pimc_horizon.clamp(1, 16)
                };
                let need_q_len = (q_len + h + 3).min(24);
                let mut world_qs = [[0u8; 24]; 24];
                let mut world_bags = [[0x7Fu8; 24]; 24];
                let mut world_holes = [[0u16; 32]; 24];
                let mut world_end_bags = [0x7Fu8; 24];

                // Stratify first unknown piece across active bits of bag_mask
                let mut bag_bits = [0u8; 7];
                let mut bag_n = 0usize;
                let mut bm = bag_mask;
                while bm != 0 {
                    bag_bits[bag_n] = bm.trailing_zeros() as u8;
                    bag_n += 1;
                    bm &= bm - 1;
                }
                let mut shuffle_rng = crn_seed ^ 0xA0761D6478BD642F;
                for i in (1..bag_n).rev() {
                    shuffle_rng = shuffle_rng
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = ((shuffle_rng >> 32) as usize) % (i + 1);
                    bag_bits.swap(i, j);
                }

                let init_prev_hole = if root_board.garbage_rows > 0 {
                    let h0 = (!root_board.data[0] & 0x3FF).trailing_zeros() as u16;
                    if h0 < 10 { Some(h0) } else { None }
                } else {
                    None
                };

                // Stratify 1st-hole offsets without replacement over shuffled 1..=9 blocks
                let mut first_offs = [1u16; 24];
                let mut hole_shuf_rng = crn_seed ^ 0x94D049BB133111EB;
                for blk in 0..3usize {
                    let mut perm = [1u16, 2, 3, 4, 5, 6, 7, 8, 9];
                    for i in (1..9usize).rev() {
                        hole_shuf_rng = hole_shuf_rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let j = ((hole_shuf_rng >> 32) as usize) % (i + 1);
                        perm.swap(i, j);
                    }
                    for r in 0..9usize {
                        let idx = blk * 9 + r;
                        if idx < 24 {
                            first_offs[idx] = perm[r];
                        }
                    }
                }

                for w in 0..w_cnt {
                    for i in 0..q_len {
                        world_qs[w][i] = q[i];
                        world_bags[w][i] = bag_mask;
                    }
                    let mut bag = bag_mask;
                    let mut w_rng = crn_seed
                        ^ ((w as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15)
                        ^ 0xBF58476D1CE4E5B9;
                    if q_len < need_q_len {
                        if bag_n > 0 {
                            let p0 = bag_bits[w % bag_n];
                            world_qs[w][q_len] = p0;
                            bag &= !(1u8 << p0);
                            if bag == 0 {
                                bag = 0x7F;
                            }
                            world_bags[w][q_len] = bag;
                            for step in (q_len + 1)..need_q_len {
                                world_qs[w][step] = Self::draw_from_bag(&mut bag, &mut w_rng);
                                world_bags[w][step] = bag;
                            }
                        } else {
                            for step in q_len..need_q_len {
                                world_qs[w][step] = Self::draw_from_bag(&mut bag, &mut w_rng);
                                world_bags[w][step] = bag;
                            }
                        }
                    }
                    world_end_bags[w] = bag;

                    let mut h_rng = w_rng ^ 0xD6E8FEB86659FD93;
                    let mut prev_h = match init_prev_hole {
                        Some(ph) => ph,
                        None => {
                            h_rng = h_rng
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            ((h_rng >> 32) as u16) % 10
                        }
                    };
                    for k in 0..32 {
                        let off = if k == 0 && init_prev_hole.is_some() {
                            first_offs[w]
                        } else {
                            h_rng = h_rng
                                .wrapping_mul(6364136223846793005)
                                .wrapping_add(1442695040888963407);
                            (((h_rng >> 32) as u16) % 9) + 1
                        };
                        prev_h = (prev_h + off) % 10;
                        world_holes[w][k] = prev_h;
                    }
                }

                let eval_k = active_roots.len();
                let mut pimc_scores = vec![0i64; eval_k];
                let unspawned_0 = (rem_g0 as usize).saturating_sub(root_board.garbage_rows as usize);
                for (ai, &(ci, _)) in active_roots.iter().enumerate() {
                    let root_idx = candidates[ci].0;
                    let fp = self.root_places[root_idx as usize];
                    let mut sum_w: i64 = 0;
                    for w in 0..w_cnt {
                        let w_seed = crn_seed ^ ((w as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15);
                        let (w_sc, evals) = self.root_pimc_eval_world(
                            &root_board,
                            hold0,
                            cur0,
                            q_len,
                            rem_g0,
                            unspawned_0,
                            fp,
                            &world_qs[w][..need_q_len],
                            &world_bags[w][..need_q_len],
                            world_end_bags[w],
                            &world_holes[w],
                            h,
                            w_seed,
                            cfg,
                        );
                        nodes_evaluated += evals;
                        sum_w += w_sc;
                    }
                    pimc_scores[ai] = sum_w / (w_cnt as i64);
                }

                let min_pimc = pimc_scores.iter().copied().min().unwrap_or(0);
                let shift = best_s1 - min_pimc;
                let pw = cfg.root_pimc_weight_pct.clamp(0, 100);
                // Put reranked roots into a fresh sorted order and place unselected roots after them
                let mut selected_mask = vec![false; candidates.len()];
                let mut reranked_list: Vec<(u16, i64)> = Vec::with_capacity(eval_k);
                for (ai, &(ci, s1_sc)) in active_roots.iter().enumerate() {
                    selected_mask[ci] = true;
                    let comb = ((100 - pw) * s1_sc + pw * (pimc_scores[ai] + shift)) / 100;
                    reranked_list.push((candidates[ci].0, comb));
                }
                reranked_list.sort_unstable_by_key(|&(idx, r)| (r, idx));
                let mut rest_list: Vec<(u16, i64)> = Vec::with_capacity(candidates.len() - eval_k);
                for (ci, &pair) in candidates.iter().enumerate() {
                    if !selected_mask[ci] {
                        rest_list.push(pair);
                    }
                }
                candidates.clear();
                candidates.extend_from_slice(&reranked_list);
                candidates.extend_from_slice(&rest_list);
            }
        }

        self.last_top.clear();
        self.last_top_leaves.clear();
        for &(root_idx, rating) in candidates.iter().take(10) {
            let fp = self.root_places[root_idx as usize];
            let c = idx_to_color(fp.color_idx as usize).as_char();
            self.last_top
                .push((c, fp.norm_i, fp.norm_j, fp.tf.2 as u8, fp.did_hold, rating));
            if let Some(leaf) = self.cur_beam.iter().find(|n| n.first_idx == root_idx) {
                self.last_top_leaves.push((
                    root_idx,
                    leaf.rem_garbage,
                    leaf.score,
                    leaf.board.rows,
                    leaf.board.garbage_rows,
                ));
            }
        }

        for (root_idx, rating) in candidates {
            let fp = self.root_places[root_idx as usize];
            if let Some(inputs) = self.build_inputs(shtb, &snapshot.matrix, fp) {
                return Some(BeamResult {
                    best_place: fp,
                    inputs,
                    rating,
                    nodes_evaluated,
                });
            }
        }

        None
    }

    #[inline]
    fn eval_core(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
    ) -> i64 {
        if let Some(ref sm) = cfg.spatial_model {
            return sm.evaluate_board_with_bag(board, rem_g, in_hand, avail_mask);
        }
        if let Some(ref model) = cfg.value_model {
            let m_sc = model.evaluate_board_with_bag(board, rem_g, in_hand, avail_mask);
            if cfg.value_fast_micro != 0 && m_sc > -9_000_000 && m_sc < 4_000_000 {
                let f_sc = eval_fast_with_bag(board, rem_g, in_hand, avail_mask, &cfg.weights);
                (f_sc * cfg.value_fast_micro) / 10_000 + m_sc
            } else {
                m_sc
            }
        } else {
            eval_fast_with_bag(board, rem_g, in_hand, avail_mask, &cfg.weights)
        }
    }

    #[inline]
    fn eval_state(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
    ) -> i64 {
        if cfg.fog_spawn && board.has_opaque_top_garbage() {
            let g = board.garbage_rows as usize;
            let mut sum: i64 = 0;
            let mut b_tmp = *board;
            for c in 0..10u16 {
                b_tmp.data[g - 1] = EMPTY_ROW | (0x3FF & !(1u16 << c));
                sum += Self::eval_core(&b_tmp, rem_g, in_hand, avail_mask, cfg);
            }
            return sum / 10;
        }
        Self::eval_core(board, rem_g, in_hand, avail_mask, cfg)
    }

    #[allow(dead_code)]
    #[inline]
    fn eval_endgame(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        _avail_mask: u8,
        _cfg: &BeamConfig,
    ) -> i64 {
        if rem_g == 0 {
            return if board.rows < 20 {
                -20_000_000 + (board.rows as i64) * 5
            } else {
                5_000_000 + (board.rows as i64) * 10_000
            };
        }
        let col_h = board.col_heights();
        let max_h = *col_h.iter().max().unwrap() as i64;
        let center_max_h = col_h[3..=6].iter().copied().max().unwrap() as i64;
        if max_h >= 20 || center_max_h >= 19 {
            return 5_000_000 + max_h * 10_000 + center_max_h * 10_000;
        }

        let g = board.garbage_rows as usize;
        let n = board.rows as usize;
        let mut stack_cells: i64 = 0;
        let mut covered_holes: i64 = 0;
        let mut above_mask: u16 = 0;
        let mut row_empty_cnt = [0i64; 32];

        let mut i = n;
        while i > g {
            i -= 1;
            let bits = board.data[i] & 0x3FF;
            let cnt = bits.count_ones() as i64;
            stack_cells += cnt;
            row_empty_cnt[i - g] = 10 - cnt;
            let rholes = (!bits & 0x3FF) & above_mask;
            covered_holes += rholes.count_ones() as i64;
            above_mask |= bits;
        }

        let mut h1_blocks: i64 = 0;
        let mut h1_empty: i64 = 0;
        let mut h1_open = false;
        let mut h1_hand_fits = false;
        let mut h2_blocks: i64 = 0;
        let mut h2_empty: i64 = 0;
        let mut h2_open = false;
        let mut h3_blocks: i64 = 0;

        if g >= 1 {
            let c1 = (!board.data[g - 1] & 0x3FF).trailing_zeros() as usize;
            if c1 < 10 {
                let ch = col_h[c1] as usize;
                if ch <= g {
                    h1_open = true;
                    if let Some(p) = in_hand {
                        let l1 = if c1 == 0 { 99 } else { (col_h[c1 - 1] as usize).saturating_sub(g) };
                        let r1 = if c1 == 9 { 99 } else { (col_h[c1 + 1] as usize).saturating_sub(g) };
                        let fits = match p {
                            0 => true,                              // I
                            2 => l1 == 0 || r1 == 0,                // T
                            3 => l1 == 0,                           // S
                            4 => r1 == 0,                           // Z
                            5 => r1 <= 1 || l1 == 0,                // J
                            6 => l1 <= 1 || r1 == 0,                // L
                            _ => false,                             // O cannot clear 1-wide
                        };
                        h1_hand_fits = fits;
                    }
                } else {
                    for r in g..ch {
                        if ((board.data[r] >> c1) & 1) != 0 {
                            h1_blocks += 1;
                        }
                        h1_empty += row_empty_cnt[r - g];
                    }
                }
            }
        }
        if g >= 2 {
            let c2 = (!board.data[g - 2] & 0x3FF).trailing_zeros() as usize;
            if c2 < 10 {
                let ch = col_h[c2] as usize;
                if ch <= g {
                    h2_open = true;
                } else {
                    for r in g..ch {
                        if ((board.data[r] >> c2) & 1) != 0 {
                            h2_blocks += 1;
                        }
                        h2_empty += row_empty_cnt[r - g];
                    }
                }
            }
        }
        if g >= 3 {
            let c3 = (!board.data[g - 3] & 0x3FF).trailing_zeros() as usize;
            if c3 < 10 {
                let ch = col_h[c3] as usize;
                for r in g..ch {
                    if ((board.data[r] >> c3) & 1) != 0 {
                        h3_blocks += 1;
                    }
                }
            }
        }

        let mut bumpiness: i64 = 0;
        for c in 0..9 {
            let h_a = (col_h[c] as i64 - g as i64).max(0);
            let h_b = (col_h[c + 1] as i64 - g as i64).max(0);
            bumpiness += (h_a - h_b).abs();
        }

        (rem_g as i64) * 260
            + covered_holes * 95
            + h1_blocks * 85
            + h1_empty * 28
            + h2_blocks * 55
            + h2_empty * 16
            + h3_blocks * 25
            + stack_cells * 6
            + bumpiness * 4
            - if h1_open { 50 } else { 0 }
            - if h2_open { 25 } else { 0 }
            - if h1_hand_fits { 45 } else { 0 }
    }

    fn endgame_rollout_crn(
        &mut self,
        board: FastBoard,
        in_hand_opt: Option<u8>,
        rem_g: u16,
        start_bag_mask: u8,
        crn_rng_seed: u64,
        horizon: usize,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        if rem_g == 0 {
            return (
                if board.rows < 20 {
                    -20_000_000 + (board.rows as i64) * 5
                } else {
                    5_000_000
                },
                0,
            );
        }
        let eff_mask = if start_bag_mask == 0 { 0x7F } else { start_bag_mask };
        let mut bag_bits = [0u8; 7];
        let mut bag_n = 0usize;
        let mut bm = eff_mask;
        while bm != 0 {
            bag_bits[bag_n] = bm.trailing_zeros() as u8;
            bag_n += 1;
            bm &= bm - 1;
        }
        let h = horizon.clamp(1, 12);
        let w_cnt = bag_n.max(1);
        let end_beam_w = 8usize;
        let mut evals = 0usize;
        let mut sum_w: i64 = 0;

        for w in 0..w_cnt {
            let mut bag = eff_mask;
            let mut rng = crn_rng_seed
                ^ ((w as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0xA24BAED4963EE407;
            let p0 = bag_bits[w % bag_n];
            bag &= !(1u8 << p0);
            if bag == 0 {
                bag = 0x7F;
            }
            let init_hand = match in_hand_opt {
                Some(ih) => ih,
                None => p0,
            };
            let mut wq = [0u8; 12];
            let mut w_bags = [0x7Fu8; 12];
            if in_hand_opt.is_some() {
                wq[0] = p0;
                w_bags[0] = bag;
                for s in 1..h {
                    wq[s] = Self::draw_from_bag(&mut bag, &mut rng);
                    w_bags[s] = bag;
                }
            } else {
                for s in 0..h {
                    wq[s] = Self::draw_from_bag(&mut bag, &mut rng);
                    w_bags[s] = bag;
                }
            }

            self.pimc_beam.clear();
            self.pimc_beam.push(BeamNode {
                board,
                held: Some(init_hand),
                q_cursor: 0,
                rem_garbage: rem_g,
                no_g_moves: 0,
                first_idx: 0,
                score: 0,
            });
            let mut best_done = i64::MAX;

            for step in 0..h {
                if self.pimc_beam.is_empty() {
                    break;
                }
                let next_p = wq[step];
                let step_bag = w_bags[step];
                self.pimc_next.clear();

                for ni in 0..self.pimc_beam.len() {
                    let node = self.pimc_beam[ni];
                    if node.rem_garbage == 0 {
                        continue;
                    }
                    let cur_h = node.held.unwrap_or(init_hand);
                    self.pgen.generate_for_state(
                        &self.fst,
                        &node.board,
                        Some(cur_h),
                        Some(next_p),
                        &mut self.roll_places,
                    );
                    for &fp in &self.roll_places {
                        let mut b2 = node.board;
                        let orient =
                            &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (_, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                        let rem2 = node.rem_garbage.saturating_sub(ds as u16);
                        if rem2 == 0 && b2.rows < 20 {
                            let done_sc = ((step as i64) + 1) * 1000 + (b2.rows as i64) * 5;
                            if done_sc < best_done {
                                best_done = done_sc;
                            }
                            continue;
                        }
                        let new_hand = if fp.color_idx == next_p { cur_h } else { next_p };
                        let sc = Self::eval_endgame(&b2, rem2, Some(new_hand), step_bag, cfg);
                        evals += 1;
                        self.pimc_next.push(BeamNode {
                            board: b2,
                            held: Some(new_hand),
                            q_cursor: 0,
                            rem_garbage: rem2,
                            no_g_moves: 0,
                            first_idx: 0,
                            score: sc,
                        });
                    }
                }

                if best_done != i64::MAX {
                    break;
                }
                if self.pimc_next.is_empty() {
                    self.pimc_beam.clear();
                    break;
                }
                self.pimc_next.sort_unstable_by_key(|n| n.score);
                self.pimc_beam.clear();
                self.pimc_seen.clear();
                for ni in 0..self.pimc_next.len() {
                    if self.pimc_beam.len() >= end_beam_w {
                        break;
                    }
                    let ch = self.pimc_next[ni];
                    let key = DedupKey {
                        board: ch.board,
                        held: ch.held,
                        rem_garbage: ch.rem_garbage,
                    };
                    if self.pimc_seen.insert(key) {
                        self.pimc_beam.push(ch);
                    }
                }
            }

            let w_cost = if best_done != i64::MAX {
                best_done
            } else if let Some(top) = self.pimc_beam.first() {
                ((h as i64) + 1) * 1000 + top.score.max(0).min(200_000)
            } else {
                1_000_000
            };
            sum_w += w_cost;
        }

        (-20_000_000 + (sum_w / (w_cnt as i64)), evals)
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
    fn hand_piece(held: Option<u8>, q_cursor: usize, q: &[u8]) -> Option<u8> {
        if held.is_some() {
            held
        } else {
            q.get(q_cursor).copied()
        }
    }

    #[inline]
    fn draw_from_bag(mask: &mut u8, rng: &mut u64) -> u8 {
        if *mask == 0 {
            *mask = 0x7F;
        }
        let cnt = mask.count_ones();
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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

    fn rollout_greedy_crn(
        &mut self,
        mut board: FastBoard,
        mut in_hand: u8,
        mut rem_g: u16,
        mut ng: u8,
        start_bag_mask: u8,
        crn_rng_seed: u64,
        horizon: usize,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        let mut bag = if start_bag_mask == 0 { 0x7F } else { start_bag_mask };
        let mut rng = crn_rng_seed;
        if cfg.fog_spawn {
            board.materialize_fog_holes(&mut rng);
        }
        let mut evals = 0usize;

        for step in 0..horizon {
            if rem_g == 0 {
                return (-15_000_000 + (step as i64) * 100 + (board.rows as i64) * 5, evals);
            }
            if cfg.simulate_cheese_spawn || cfg.fog_spawn {
                // handled after step
            } else if cfg.rollout_spawn_g0 && board.garbage_rows == 0 {
                if !board.spawn_cheese_to(rem_g, 4, &mut rng) {
                    return (5_000_000, evals);
                }
            }
            ng = ng.saturating_add(u8::from(board.garbage_rows == 0 && rem_g > 0));
            let ng_bonus = cfg.no_g_credit * (ng as i64);

            let next_p = Self::draw_from_bag(&mut bag, &mut rng);
            self.pgen.generate_for_state(
                &self.fst,
                &board,
                Some(in_hand),
                Some(next_p),
                &mut self.roll_places,
            );
            let mut best_sc = i64::MAX;
            let mut best_state: Option<(FastBoard, u8, u16, u8)> = None;
            for &fp in &self.roll_places {
                let mut b2 = board;
                let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                let rem2 = rem_g.saturating_sub(ds as u16);
                let new_hand = if fp.color_idx == next_p { in_hand } else { next_p };
                let sc = Self::eval_state(&b2, rem2, Some(new_hand), bag, cfg) - ng_bonus;
                evals += 1;
                if sc < best_sc {
                    best_sc = sc;
                    best_state = Some((b2, new_hand, rem2, lines));
                }
            }
            match best_state {
                Some((mut b2, nh, r2, lines)) => {
                    if r2 > 0 {
                        if cfg.simulate_cheese_spawn || cfg.fog_spawn {
                            if !b2.spawn_cheese_rows(r2, lines, &mut rng) {
                                return (5_000_000, evals);
                            }
                        }
                    } else if b2.rows < 20 {
                        return (
                            -15_000_000 + ((step as i64) + 1) * 100 + (b2.rows as i64) * 5,
                            evals,
                        );
                    }
                    board = b2;
                    in_hand = nh;
                    rem_g = r2;
                }
                None => return (5_000_000, evals),
            }
        }
        (
            Self::eval_state(&board, rem_g, Some(in_hand), bag, cfg)
                - cfg.no_g_credit * (ng as i64),
            evals,
        )
    }

    fn rollout_7bag(
        &mut self,
        board: FastBoard,
        in_hand: u8,
        rem_g: u16,
        no_g_moves: u8,
        bag_mask: u8,
        crn_seed: u64,
        horizon: usize,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        if rem_g == 0 {
            return (-15_000_000 + (board.rows as i64) * 5, 0);
        }
        let mut total_best: i64 = 0;
        let mut bag_count: i64 = 0;
        let mut evals: usize = 0;
        let mut mask = bag_mask;
        let samples = cfg.rollout_samples.max(1);
        let ng_next = no_g_moves.saturating_add(u8::from(board.garbage_rows == 0 && rem_g > 0));
        let ng_bonus = cfg.no_g_credit * (ng_next as i64);

        while mask != 0 {
            let next_p = mask.trailing_zeros() as u8;
            mask &= mask - 1;
            let rem_after_p = {
                let r = bag_mask & !(1u8 << next_p);
                if r == 0 { 0x7F } else { r }
            };

            self.pgen.generate_for_state(
                &self.fst,
                &board,
                Some(in_hand),
                Some(next_p),
                &mut self.tmp_places,
            );

            let mut best_for_p = i64::MAX;
            let mut second_for_p = i64::MAX;
            let mut best1: Option<(FastBoard, u8, u16, u8)> = None;
            let mut best2: Option<(FastBoard, u8, u16, u8)> = None;
            for &fp in &self.tmp_places {
                let mut b2 = board;
                let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                let rem2 = rem_g.saturating_sub(ds as u16);
                let new_hand = if fp.color_idx == next_p {
                    in_hand
                } else {
                    next_p
                };
                let sc = Self::eval_state(&b2, rem2, Some(new_hand), rem_after_p, cfg) - ng_bonus;
                evals += 1;
                if sc < best_for_p {
                    second_for_p = best_for_p;
                    best2 = best1;
                    best_for_p = sc;
                    if horizon > 0 {
                        best1 = Some((b2, new_hand, rem2, lines));
                    }
                } else if horizon > 0 && sc < second_for_p {
                    second_for_p = sc;
                    best2 = Some((b2, new_hand, rem2, lines));
                }
            }
            if best_for_p == i64::MAX {
                best_for_p = 500_000;
            } else if horizon > 0 {
                let rem_mask = bag_mask & !(1u8 << next_p);
                let mut best_deep = i64::MAX;
                for (b7, h7, r7, lines7) in best1.into_iter().chain(best2) {
                    let deep_val = if r7 == 0 && b7.rows < 20 {
                        -15_000_000 + (b7.rows as i64) * 5
                    } else {
                        let mut sum_s: i64 = 0;
                        let mut valid_s: i64 = 0;
                        for s in 0..samples {
                            let s_seed = crn_seed
                                ^ (next_p as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)
                                ^ (s as u64 + 1).wrapping_mul(0xBF58476D1CE4E5B9);
                            let mut b_start = b7;
                            let mut spawn_rng = s_seed ^ 0x123456789ABCDEF;
                            if cfg.fog_spawn {
                                b_start.materialize_fog_holes(&mut spawn_rng);
                            }
                            if cfg.simulate_cheese_spawn || cfg.fog_spawn {
                                if !b_start.spawn_cheese_rows(r7, lines7, &mut spawn_rng) {
                                    sum_s += 5_000_000;
                                    valid_s += 1;
                                    continue;
                                }
                            } else if cfg.rollout_spawn_g0 && b_start.garbage_rows == 0 && r7 > 0 {
                                if !b_start.spawn_cheese_to(r7, 4, &mut spawn_rng) {
                                    sum_s += 5_000_000;
                                    valid_s += 1;
                                    continue;
                                }
                            }
                            let (r_sc, r_ev) = self.rollout_greedy_crn(
                                b_start,
                                h7,
                                r7,
                                ng_next,
                                rem_mask,
                                s_seed,
                                horizon,
                                cfg,
                            );
                            evals += r_ev;
                            sum_s += r_sc;
                            valid_s += 1;
                        }
                        if valid_s > 0 { sum_s / valid_s } else { best_for_p }
                    };
                    if deep_val < best_deep {
                        best_deep = deep_val;
                    }
                }
                if best_deep != i64::MAX {
                    best_for_p = best_deep;
                }
            }
            total_best += best_for_p;
            bag_count += 1;
        }

        if bag_count == 0 {
            (
                Self::eval_state(&board, rem_g, Some(in_hand), bag_mask, cfg)
                    - cfg.no_g_credit * (no_g_moves as i64),
                evals,
            )
        } else {
            (total_best / bag_count, evals)
        }
    }

    fn pimc_eval_root(
        &mut self,
        seeds: &[BeamNode],
        q: &[u8],
        world_seq: &[u8],
        spawn_seed: u64,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        self.pimc_beam.clear();
        self.pimc_seen.clear();
        let mut best_done = i64::MAX;
        let mut evals = 0usize;

        for &s in seeds {
            if s.rem_garbage == 0 && s.board.rows < 20 {
                best_done = best_done.min(-15_000_000 + (s.board.rows as i64) * 5);
                continue;
            }
            let hc = Self::hand_piece(s.held, s.q_cursor as usize, q);
            let mut n = s;
            n.held = hc;
            let key = DedupKey {
                board: n.board,
                held: n.held,
                rem_garbage: n.rem_garbage,
            };
            if self.pimc_seen.insert(key) {
                self.pimc_beam.push(n);
            }
        }
        if best_done != i64::MAX && self.pimc_beam.is_empty() {
            return (best_done, evals);
        }

        let width = cfg.pimc_width.max(4);
        for (step, &cur_p) in world_seq.iter().enumerate() {
            if self.pimc_beam.is_empty() {
                break;
            }
            let mut step_avail = 0u8;
            for &np in &world_seq[(step + 1)..] {
                step_avail |= 1u8 << np;
            }
            if step + 1 >= world_seq.len() {
                step_avail = 0x7F;
            }
            self.pimc_next.clear();
            let step_rng_base =
                spawn_seed ^ ((step as u64) + 1).wrapping_mul(0x517CC1B727220A95);

            for i in 0..self.pimc_beam.len() {
                let mut node = self.pimc_beam[i];
                if node.rem_garbage == 0 {
                    continue;
                }
                if cfg.pimc_spawn
                    && (node.board.garbage_rows == 0
                        || (step == 0 && (node.board.garbage_rows as usize) < 4))
                {
                    let mut r_pre = step_rng_base ^ 0xA5A5A5A5A5A5A5A5;
                    if !node.board.spawn_cheese_to(node.rem_garbage, 4, &mut r_pre) {
                        continue;
                    }
                }
                let ng = node
                    .no_g_moves
                    .saturating_add(u8::from(node.board.garbage_rows == 0 && node.rem_garbage > 0));
                let ng_bonus = cfg.no_g_credit * (ng as i64);

                self.pgen.generate_for_state(
                    &self.fst,
                    &node.board,
                    Some(cur_p),
                    node.held,
                    &mut self.tmp_places,
                );
                for &fp in &self.tmp_places {
                    let mut b2 = node.board;
                    let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                    let rem2 = node.rem_garbage.saturating_sub(ds as u16);
                    let new_held = if fp.color_idx == cur_p {
                        node.held
                    } else {
                        Some(cur_p)
                    };
                    if cfg.pimc_spawn && rem2 > 0 {
                        let mut r_post = step_rng_base;
                        if !b2.spawn_cheese_rows(rem2, lines, &mut r_post) {
                            continue;
                        }
                    }
                    if rem2 == 0 && b2.rows < 20 {
                        let g_sc = -15_000_000 + ((step as i64) + 1) * 100 + (b2.rows as i64) * 5;
                        if g_sc < best_done {
                            best_done = g_sc;
                        }
                    }
                    let sc = Self::eval_state(&b2, rem2, new_held, step_avail, cfg) - ng_bonus;
                    evals += 1;
                    self.pimc_next.push(BeamNode {
                        board: b2,
                        held: new_held,
                        q_cursor: node.q_cursor,
                        rem_garbage: rem2,
                        no_g_moves: ng,
                        first_idx: node.first_idx,
                        score: sc,
                    });
                }
            }

            if self.pimc_next.is_empty() {
                self.pimc_beam.clear();
                break;
            }
            self.pimc_next.sort_unstable_by_key(|n| n.score);
            self.pimc_beam.clear();
            self.pimc_seen.clear();
            for i in 0..self.pimc_next.len() {
                if self.pimc_beam.len() >= width {
                    break;
                }
                let ch = self.pimc_next[i];
                let key = DedupKey {
                    board: ch.board,
                    held: ch.held,
                    rem_garbage: ch.rem_garbage,
                };
                if self.pimc_seen.insert(key) {
                    self.pimc_beam.push(ch);
                }
            }
        }

        let beam_best = self.pimc_beam.first().map_or(5_000_000, |n| n.score);
        (best_done.min(beam_best), evals)
    }

    #[inline]
    fn spawn_cheese_from_stream(
        b: &mut FastBoard,
        rem_g: u16,
        lines_cleared: u8,
        steady_g4: bool,
        unspawned_0: usize,
        holes: &[u16],
    ) -> bool {
        let target_g: usize = if steady_g4 || lines_cleared > 0 { 4 } else { 9 };
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

    fn root_pimc_eval_world(
        &mut self,
        root_board: &FastBoard,
        hold0: Option<u8>,
        cur0: Option<u8>,
        q_len: usize,
        rem_g0: u16,
        unspawned_0: usize,
        root_fp: FastPlace,
        world_q: &[u8],
        world_bags: &[u8],
        end_bag: u8,
        world_holes: &[u16],
        horizon: usize,
        w_seed: u64,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        let mut b0 = *root_board;
        let orient0 = &self.fst.shapes[root_fp.color_idx as usize].orients[root_fp.tf.2 as usize];
        let (lines0, ds0, _) = b0.place_and_sift(orient0, root_fp.norm_i, root_fp.norm_j);
        let rem0 = rem_g0.saturating_sub(ds0 as u16);
        if rem0 == 0 && b0.rows < 20 {
            return (-15_000_000 + 100 + (b0.rows as i64) * 5, 1);
        }
        if rem0 > 0
            && !Self::spawn_cheese_from_stream(
                &mut b0,
                rem0,
                lines0,
                cfg.root_pimc_steady_g4,
                unspawned_0,
                world_holes,
            )
        {
            return (5_000_000, 1);
        }

        let (held0, qc0) = if hold0.is_some() {
            if root_fp.did_hold {
                (cur0, if cur0.is_some() { 1 } else { 0 })
            } else {
                (hold0, 1.min(q_len))
            }
        } else if root_fp.did_hold {
            (cur0, 2.min(q_len))
        } else {
            (None, 1.min(q_len))
        };

        let ng0 = u8::from(root_board.garbage_rows == 0 && rem_g0 > 0);
        let in_hand0 = Self::hand_piece(held0, qc0, world_q);
        let avail0 = Self::avail_mask_from_q(held0, qc0, world_q, end_bag);
        let sc0 = Self::eval_state(&b0, rem0, in_hand0, avail0, cfg) - cfg.no_g_credit * (ng0 as i64);

        self.pimc_beam.clear();
        self.pimc_beam.push(BeamNode {
            board: b0,
            held: held0,
            q_cursor: qc0 as u8,
            rem_garbage: rem0,
            no_g_moves: ng0,
            first_idx: 0,
            score: sc0,
        });

        let mut best_done = i64::MAX;
        let mut evals = 1usize;
        let width = cfg.root_pimc_width.max(4);
        let wq_len = world_q.len();

        for step in 1..=horizon {
            if self.pimc_beam.is_empty() {
                break;
            }
            self.pimc_next.clear();

            for i in 0..self.pimc_beam.len() {
                let node = self.pimc_beam[i];
                if node.rem_garbage == 0 {
                    continue;
                }
                let qc = node.q_cursor as usize;
                let (c_nh, c_h) = if node.held.is_some() {
                    let c1 = if qc < wq_len { Some(world_q[qc]) } else { None };
                    (c1, node.held)
                } else {
                    let c1 = if qc < wq_len { Some(world_q[qc]) } else { None };
                    let c2 = if qc + 1 < wq_len {
                        Some(world_q[qc + 1])
                    } else {
                        None
                    };
                    (c1, c2)
                };

                self.pgen
                    .generate_for_state(&self.fst, &node.board, c_nh, c_h, &mut self.tmp_places);

                let ng = node
                    .no_g_moves
                    .saturating_add(u8::from(node.board.garbage_rows == 0 && node.rem_garbage > 0));
                let ng_bonus = cfg.no_g_credit * (ng as i64);

                for &fp in &self.tmp_places {
                    let mut b2 = node.board;
                    let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                    let rem2 = node.rem_garbage.saturating_sub(ds as u16);

                    if rem2 == 0 && b2.rows < 20 {
                        let g_sc = -15_000_000 + ((step as i64) + 1) * 100 + (b2.rows as i64) * 5;
                        if g_sc < best_done {
                            best_done = g_sc;
                        }
                    }

                    if rem2 > 0
                        && !Self::spawn_cheese_from_stream(
                            &mut b2,
                            rem2,
                            lines,
                            cfg.root_pimc_steady_g4,
                            unspawned_0,
                            world_holes,
                        )
                    {
                        continue;
                    }

                    let (new_held, next_qc) = if node.held.is_some() {
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

                    let in_hand2 = Self::hand_piece(new_held, next_qc, world_q);
                    let avail2 = Self::avail_mask_from_q(new_held, next_qc, world_q, end_bag);
                    let sc = Self::eval_state(&b2, rem2, in_hand2, avail2, cfg) - ng_bonus;
                    evals += 1;

                    self.pimc_next.push(BeamNode {
                        board: b2,
                        held: new_held,
                        q_cursor: next_qc as u8,
                        rem_garbage: rem2,
                        no_g_moves: ng,
                        first_idx: 0,
                        score: sc,
                    });
                }
            }

            if self.pimc_next.is_empty() {
                self.pimc_beam.clear();
                break;
            }
            self.pimc_next.sort_unstable_by_key(|n| n.score);
            self.pimc_beam.clear();
            self.pimc_seen.clear();
            for i in 0..self.pimc_next.len() {
                if self.pimc_beam.len() >= width {
                    break;
                }
                let ch = self.pimc_next[i];
                let key = DedupKey {
                    board: ch.board,
                    held: ch.held,
                    rem_garbage: ch.rem_garbage,
                };
                if self.pimc_seen.insert(key) {
                    self.pimc_beam.push(ch);
                }
            }
        }

        if best_done != i64::MAX {
            return (best_done, evals);
        }
        if self.pimc_beam.is_empty() {
            return (5_000_000, evals);
        }

        if cfg.rollout_top_k > 0 {
            let check_k = self.pimc_beam.len().min(2);
            let mut best_roll = i64::MAX;
            for li in 0..check_k {
                let leaf = self.pimc_beam[li];
                let hc = Self::hand_piece(leaf.held, leaf.q_cursor as usize, world_q);
                let l_sc = if let Some(in_h) = hc {
                    let bag_idx = if leaf.held.is_some() {
                        (leaf.q_cursor as usize).saturating_sub(1)
                    } else {
                        leaf.q_cursor as usize
                    };
                    let l_bag = if bag_idx < world_bags.len() {
                        world_bags[bag_idx]
                    } else {
                        end_bag
                    };
                    let (r_sc, r_ev) = self.rollout_7bag(
                        leaf.board,
                        in_h,
                        leaf.rem_garbage,
                        leaf.no_g_moves,
                        l_bag,
                        w_seed,
                        0,
                        cfg,
                    );
                    evals += r_ev;
                    r_sc
                } else {
                    leaf.score
                };
                if l_sc < best_roll {
                    best_roll = l_sc;
                }
            }
            (best_roll, evals)
        } else {
            (self.pimc_beam[0].score, evals)
        }
    }

    pub fn inputs_for_place(
        &mut self,
        shtb: &ShapeTable,
        matrix: &BasicMatrix,
        fp: FastPlace,
    ) -> Option<Vec<Input>> {
        self.build_inputs(shtb, matrix, fp)
    }

    fn build_inputs(
        &mut self,
        shtb: &ShapeTable,
        matrix: &BasicMatrix,
        fp: FastPlace,
    ) -> Option<Vec<Input>> {
        let color = idx_to_color(fp.color_idx as usize);
        let sr = shtb.shape(color)?;
        let norm = sr.normalize(fp.tf);
        let path = self.ffind.find(matrix, sr, norm).ok()?;
        let mut inputs = Vec::with_capacity(path.len() + 2);
        if fp.did_hold {
            inputs.push(Input::Hold);
        }
        inputs.extend(path);
        inputs.push(Input::HD);
        Some(inputs)
    }
}

#[cfg(all(test, feature = "block-stacker"))]
mod test {
    use super::*;
    use crate::{shape::srs, StackerExt};
    use block_stacker::{Config as BSConfig, Ruleset, Stacker};

    #[test]
    fn test_bag_tracker_matches_7bag_invariants() {
        let mut cfg = BSConfig::default();
        cfg.prng_seed = Some(42);
        cfg.garbage.total_lines = Some(100);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), cfg);
        let mut tracker = BagTracker::new();

        for step in 0..70 {
            let snap = match stacker.snapshot() {
                Some(s) => s,
                None => break,
            };
            let mask = tracker.observe(snap.hold, &snap.queue);
            if step == 0 {
                assert_eq!(mask.count_ones(), 1);
            }
            assert!(mask != 0 && mask <= 0x7F);
            if step % 3 == 0 && stacker.held().is_none() {
                stacker.hold();
            }
            let dir = (step % 7) as i16 - 3;
            for _ in 0..dir.abs() {
                stacker.move_horizontal(if dir > 0 { 1 } else { -1 });
            }
            stacker.hard_drop();
        }
    }

    #[test]
    fn test_beam_search_outputs_valid_inputs() {
        let shtb = srs();
        let mut searcher = BeamSearcher::new(&shtb);
        let mut cfg = BSConfig::default();
        cfg.prng_seed = Some(1);
        cfg.garbage.total_lines = Some(100);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), cfg);
        let mut tracker = BagTracker::new();
        let bcfg = BeamConfig::default();

        for _ in 0..15 {
            let snap = stacker.snapshot().unwrap();
            let garbage_rows = {
                let mut max_g = 0u16;
                let mut any_g = false;
                for ((r, _), cc) in stacker.matrix() {
                    if cc == 'G' {
                        any_g = true;
                        max_g = max_g.max(r + 1);
                    }
                }
                if any_g { max_g } else { 0 }
            };
            let mask = tracker.observe(snap.hold, &snap.queue);
            let ctx = GameContext {
                garbage_rows: Some(garbage_rows),
                remaining_garbage: Some(100),
                next_bag_mask: Some(mask),
            };
            let res = searcher.search(&shtb, &snap, ctx, &bcfg).expect("beam move");
            let mut inputs = res.inputs;
            assert_eq!(inputs.pop(), Some(Input::HD));
            stacker.run(inputs);
            stacker.hard_drop();
        }
    }

    #[test]
    fn test_fog_spawn_bag_det_and_endgame_solver() {
        use crate::ai::fast::OPAQUE_GARBAGE_ROW;
        // Verify FastBoard fog spawn & materialization invariants
        let mut fb = FastBoard::empty();
        assert!(!fb.has_opaque_top_garbage());
        assert!(fb.spawn_fog_rows(20, 0));
        assert_eq!(fb.garbage_rows, 9);
        assert_eq!(fb.rows, 9);
        assert!(fb.has_opaque_top_garbage());
        for r in 0..9 {
            assert_eq!(fb.data[r], OPAQUE_GARBAGE_ROW);
        }
        let mut rng = 12345u64;
        fb.materialize_fog_holes(&mut rng);
        assert!(!fb.has_opaque_top_garbage());
        for r in 0..9 {
            let h = (!fb.data[r] & 0x3FF).trailing_zeros();
            assert!(h < 10);
            assert_eq!((fb.data[r] & 0x3FF).count_ones(), 9);
            if r > 0 {
                let h_below = (!fb.data[r - 1] & 0x3FF).trailing_zeros();
                assert_ne!(h, h_below);
            }
        }

        // Verify BeamSearcher with bag_det_extend + fog_spawn + endgame_solver
        let shtb = srs();
        let mut searcher = BeamSearcher::new(&shtb);
        let mut cfg = BSConfig::default();
        cfg.prng_seed = Some(7);
        cfg.garbage.total_lines = Some(10);
        let mut stacker = Stacker::new(Ruleset::guideline().into(), cfg);
        let mut tracker = BagTracker::new();
        let mut bcfg = BeamConfig::default();
        bcfg.beam_width = 120;
        bcfg.max_per_root = 24;
        bcfg.bag_det_extend = true;
        bcfg.fog_spawn = true;
        bcfg.endgame_solver = true;
        bcfg.rollout_per_root = 2;

        let mut ds_total = 0usize;
        for _ in 0..40 {
            if ds_total >= 10 {
                break;
            }
            let snap = stacker.snapshot().unwrap();
            let g_rows = {
                let mut max_g = 0u16;
                let mut any_g = false;
                for ((r, _), cc) in stacker.matrix() {
                    if cc == 'G' {
                        any_g = true;
                        max_g = max_g.max(r + 1);
                    }
                }
                if any_g { max_g } else { 0 }
            };
            let rem_g = 10usize.saturating_sub(ds_total) as u16;
            let mask = tracker.observe(snap.hold, &snap.queue);
            let ctx = GameContext {
                garbage_rows: Some(g_rows),
                remaining_garbage: Some(rem_g),
                next_bag_mask: Some(mask),
            };
            let res = searcher.search(&shtb, &snap, ctx, &bcfg).expect("beam move");
            let mut inputs = res.inputs;
            assert_eq!(inputs.pop(), Some(Input::HD));
            stacker.run(inputs);
            let (_, ds) = stacker.hard_drop();
            ds_total += ds;
        }
        assert_eq!(ds_total, 10);
    }
}
