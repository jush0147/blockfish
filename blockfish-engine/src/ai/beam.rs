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
    pub spatial_rerank_only: bool,
    pub anneal_mid_pen: i64,
    pub root_top_k_mean: usize,
    pub root_pimc_prescreen_worlds: usize,
    pub root_pimc_fine_roots: usize,
    pub root_pimc_cvar_pct: i64,
    pub spatial_fast_ew_internal: bool,
    pub spatial_no_pe_internal: bool,
    pub root_pimc_ds_target: usize,
    pub nds_pen: i64,
    pub stage1_weights: Option<EvalWeights>,
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
            pimc_weight_pct: 67,
            pimc_spawn: false,
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
            spatial_rerank_only: false,
            anneal_mid_pen: 0,
            root_top_k_mean: 1,
            root_pimc_prescreen_worlds: 0,
            root_pimc_fine_roots: 8,
            root_pimc_cvar_pct: 0,
            spatial_fast_ew_internal: false,
            spatial_no_pe_internal: false,
            root_pimc_ds_target: 0,
            nds_pen: 0,
            stage1_weights: None,
        }
    }
}

#[derive(Clone, Copy)]
struct BeamNode {
    board: FastBoard,
    held: Option<u8>,
    q_cursor: u8,
    rem_garbage: u16,
    nds_used: u16,
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
    unique_root_idxs: Vec<(u16, i64)>,
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
            unique_root_idxs: Vec::with_capacity(64),
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
            is_endgame_exact || (cfg.endgame_wide && rem_g0 <= 9 && rem_g0 == garbage_rows);
        let eff_beam_width = if is_endgame_exact {
            cfg.beam_width.max(360)
        } else if is_pure_endgame {
            cfg.beam_width.max(300)
        } else {
            cfg.beam_width
        };
        let eff_max_per_root = if is_endgame_exact {
            cfg.max_per_root.max(80)
        } else if is_pure_endgame {
            cfg.max_per_root.max(60)
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

        let mut crn_seed: u64 = 0x9E3779B97F4A7C15u64
            ^ (rem_g0 as u64).wrapping_mul(0xBF58476D1CE4E5B9)
            ^ ((bag_mask as u64) << 16);
        for r in 0..(root_board.rows as usize) {
            crn_seed = crn_seed.rotate_left(7) ^ (root_board.data[r] as u64);
        }

        if is_endgame_exact {
            if let Some(res) = self.solve_endgame_ds(
                shtb,
                snapshot,
                &root_board,
                rem_g0,
                hold0,
                &q[..q_len],
                bag_mask,
                crn_seed,
            ) {
                return Some(res);
            }
        }

        let mut nodes_evaluated = 0usize;
        let mut best_goal: Option<(usize, u16, i64)> = None; // (depth, root_idx, rating)

        self.cur_beam.clear();
        self.unique_root_idxs.clear();
        self.seen.clear();

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
                let res_cells: i64 = (0..b.rows as usize)
                    .map(|r| (b.data[r] & 0x3FF).count_ones() as i64)
                    .sum();
                let g_rating = -30_000_000 + 200 + res_cells;
                if best_goal.map_or(true, |(_, _, r)| g_rating < r) {
                    best_goal = Some((1, idx as u16, g_rating));
                }
            }

            let nds0 = lines.saturating_sub(ds) as u16;
            let no_g_inc = u8::from(root_board.garbage_rows == 0 && rem_g0 > 0);
            let in_hand = Self::hand_piece(held, q_cursor, &q[..q_len]);
            let avail = Self::avail_mask_from_q(held, q_cursor, &q[..q_len], bag_mask);
            let mut sc = Self::eval_state_s1(&b, rem_g, in_hand, avail, cfg)
                + (nds0 as i64) * cfg.nds_pen
                - cfg.no_g_credit * (no_g_inc as i64);
            if cfg.anneal_mid_pen > 0 && !is_endgame_exact && target_depth > 1 {
                sc += (rem_g0.saturating_sub(rem_g) as i64) * cfg.anneal_mid_pen;
            }
            nodes_evaluated += 1;

            let key = DedupKey {
                board: b,
                held,
                rem_garbage: rem_g,
            };
            if self.seen.insert(key) {
                self.unique_root_idxs.push((idx as u16, sc));
                self.cur_beam.push(BeamNode {
                    board: b,
                    held,
                    q_cursor: q_cursor as u8,
                    rem_garbage: rem_g,
                    nds_used: nds0,
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
            let mid_pen = if cfg.anneal_mid_pen > 0 && !is_endgame_exact && depth < target_depth {
                (cfg.anneal_mid_pen * ((target_depth - depth) as i64)) / (target_depth as i64)
            } else {
                0
            };
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
                    let nds_new = node.nds_used.saturating_add(lines.saturating_sub(ds) as u16);
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
                        let res_cells: i64 = (0..b.rows as usize)
                            .map(|r| (b.data[r] & 0x3FF).count_ones() as i64)
                            .sum();
                        let g_rating = -30_000_000 + (depth as i64) * 200 + res_cells;
                        if best_goal.map_or(true, |(_, _, r)| g_rating < r) {
                            best_goal = Some((depth, node.first_idx, g_rating));
                        }
                    }

                    let in_hand = Self::hand_piece(held, next_qc, &q[..q_len]);
                    let avail = Self::avail_mask_from_q(held, next_qc, &q[..q_len], bag_mask);
                    let mut sc = Self::eval_state_s1(&b, rem_g, in_hand, avail, cfg)
                        + (nds_new as i64) * cfg.nds_pen
                        - cfg.no_g_credit * (ng as i64);
                    if mid_pen > 0 {
                        sc += (rem_g0.saturating_sub(rem_g) as i64) * mid_pen;
                    }
                    nodes_evaluated += 1;

                    self.children.push(BeamNode {
                        board: b,
                        held,
                        q_cursor: next_qc as u8,
                        rem_garbage: rem_g,
                        nds_used: nds_new,
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
                            let res_cells: i64 = (0..b.rows as usize)
                                .map(|r| (b.data[r] & 0x3FF).count_ones() as i64)
                                .sum();
                            let g_rating = -30_000_000
                                + ((target_depth + 1) as i64) * 200
                                + res_cells;
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

        if !self.cur_beam.is_empty() && total_known > 1 && best_goal.is_none() && (is_endgame_exact || cfg.rollout_top_k > 0) {
            // Stage 1: 1-ply 7-bag rollout on top K leaves AND best `rollout_per_root` leaves per root
            // (If rerank_roots > 0, defer deep rollout_horizon to Stage 2 top roots)
            let s1_horizon = if cfg.rerank_roots > 0 { 0 } else { cfg.rollout_horizon };
            let max_per_root = if is_endgame_exact {
                1
            } else {
                cfg.rollout_per_root.max(1)
            };
            let eff_roll_top_k = if is_endgame_exact {
                cfg.rollout_top_k.max(12)
            } else {
                cfg.rollout_top_k
            };
            self.per_root_rolled.clear();

            for i in 0..self.cur_beam.len() {
                let node = self.cur_beam[i];
                let rolled_cnt = self.per_root_rolled.get(&node.first_idx).copied().unwrap_or(0);
                if is_endgame_exact && rolled_cnt == 0 && self.per_root_rolled.len() >= 12 {
                    self.best_per_root.entry(node.first_idx).or_insert(-20_000_000 + 100_000 + node.score.clamp(-20_000, 20_000));
                    self.best_leaf_idx.entry(node.first_idx).or_insert(i);
                    continue;
                }
                let need_for_root = rolled_cnt < max_per_root;
                if i >= eff_roll_top_k && !need_for_root {
                    continue;
                }

                let hand_c = Self::hand_piece(node.held, node.q_cursor as usize, &q[..q_len]);
                let sc = if is_endgame_exact {
                    let end_h = (node.rem_garbage as usize + 5).clamp(10, 14);
                    let (r_sc, evals) = self.endgame_rollout_crn(
                        node.board,
                        hand_c,
                        node.rem_garbage,
                        bag_mask,
                        crn_seed,
                        end_h,
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
        } else if cfg.root_top_k_mean > 1 && !self.cur_beam.is_empty() && best_goal.is_none() {
            let k_mean = cfg.root_top_k_mean;
            let mut sums: HashMap<u16, i64, FxBuildHasher> =
                HashMap::with_capacity_and_hasher(self.root_places.len(), FxBuildHasher::default());
            let mut cnts: HashMap<u16, usize, FxBuildHasher> =
                HashMap::with_capacity_and_hasher(self.root_places.len(), FxBuildHasher::default());
            let mut min_sc: HashMap<u16, i64, FxBuildHasher> =
                HashMap::with_capacity_and_hasher(self.root_places.len(), FxBuildHasher::default());
            for (i, &node) in self.cur_beam.iter().enumerate() {
                let ri = node.first_idx;
                let c = cnts.entry(ri).or_insert(0);
                if *c < k_mean {
                    let leaf_sc = if cfg.spatial_rerank_only && cfg.spatial_model.is_some() {
                        let hc = Self::hand_piece(node.held, node.q_cursor as usize, &q[..q_len]);
                        let av = Self::avail_mask_from_q(node.held, node.q_cursor as usize, &q[..q_len], bag_mask);
                        nodes_evaluated += 1;
                        (node.nds_used as i64) * cfg.nds_pen
                            + Self::eval_leaf_state(&node.board, node.rem_garbage, hc, av, cfg)
                            - cfg.no_g_credit * (node.no_g_moves as i64)
                    } else {
                        node.score
                    };
                    let mn = min_sc.entry(ri).or_insert(i64::MAX);
                    if leaf_sc < *mn {
                        *mn = leaf_sc;
                        self.best_leaf_idx.insert(ri, i);
                    }
                    *sums.entry(ri).or_insert(0) += leaf_sc;
                    *c += 1;
                }
            }
            for (ri, sum) in sums {
                let cnt = cnts[&ri];
                let mn = min_sc[&ri];
                let padded = sum + ((k_mean - cnt) as i64) * (mn + 35);
                self.best_per_root.insert(ri, padded / (k_mean as i64));
            }
        } else {
            for (i, &node) in self.cur_beam.iter().enumerate() {
                let sc = if cfg.spatial_rerank_only && cfg.spatial_model.is_some() {
                    let hc = Self::hand_piece(node.held, node.q_cursor as usize, &q[..q_len]);
                    let av = Self::avail_mask_from_q(node.held, node.q_cursor as usize, &q[..q_len], bag_mask);
                    nodes_evaluated += 1;
                    (node.nds_used as i64) * cfg.nds_pen
                        + Self::eval_leaf_state(&node.board, node.rem_garbage, hc, av, cfg)
                        - cfg.no_g_credit * (node.no_g_moves as i64)
                } else {
                    node.score
                };
                let entry = self.best_per_root.entry(node.first_idx).or_insert(i64::MAX);
                if sc < *entry {
                    *entry = sc;
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
                        nds_used: 0,
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
            && (candidates.len() > 1 || self.unique_root_idxs.len() > 1)
        {
            if cfg.root_pimc_prescreen_worlds > 0 && self.unique_root_idxs.len() > candidates.len() {
                let tail_base = candidates.last().map_or(0, |x| x.1) + 500;
                let min_d1 = self.unique_root_idxs.iter().map(|x| x.1).min().unwrap_or(0);
                let mut extra: Vec<(u16, i64)> = self
                    .unique_root_idxs
                    .iter()
                    .copied()
                    .filter(|&(ur_idx, _)| !candidates.iter().any(|&(idx, _)| idx == ur_idx))
                    .collect();
                extra.sort_unstable_by_key(|x| (x.1, x.0));
                for (ur_idx, d1_sc) in extra {
                    candidates.push((ur_idx, tail_base + (d1_sc - min_d1)));
                }
            }

            let best_s1 = candidates[0].1;
            let max_k = if cfg.root_pimc_prescreen_worlds > 0 { 48 } else { 14 };
            let k_limit = cfg.root_pimc_roots.clamp(1, max_k).min(candidates.len());
            let mut active_roots: Vec<(usize, i64)> = Vec::with_capacity(k_limit + 2);
            let mut has_0ds = false;
            let mut has_ds1 = false;

            for (ci, &(root_idx, s1_sc)) in candidates.iter().enumerate() {
                let fp = self.root_places[root_idx as usize];
                let mut b_tmp = root_board;
                let o_tmp = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (_, ds_tmp, _) = b_tmp.place_and_sift(o_tmp, fp.norm_i, fp.norm_j);
                let within_margin = cfg.root_pimc_prescreen_worlds > 0
                    || ci == 0
                    || s1_sc - best_s1 <= cfg.root_pimc_margin;
                if active_roots.len() < k_limit && within_margin {
                    active_roots.push((ci, s1_sc));
                    if ds_tmp == 0 {
                        has_0ds = true;
                    } else {
                        has_ds1 = true;
                    }
                } else if cfg.root_pimc_prescreen_worlds == 0 {
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
                } else {
                    break;
                }
            }

            if active_roots.len() > 1 {
                let w_cnt = cfg.root_pimc_worlds.clamp(1, 24);
                let h = if cfg.root_pimc_horizon == 99 {
                    target_depth.saturating_sub(1).max(1)
                } else {
                    cfg.root_pimc_horizon.clamp(1, 16)
                };
                let ds_q_need = if cfg.root_pimc_ds_target > 0 {
                    cfg.root_pimc_ds_target * 2 + 6
                } else {
                    0
                };
                let need_q_len = (q_len + h.max(ds_q_need) + 3).min(24);
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

                let unspawned_0 = (rem_g0 as usize).saturating_sub(root_board.garbage_rows as usize);

                // Phase 4A: Coarse prescreen over all active_roots -> prune to fine_k before deep eval
                let fine_k = cfg.root_pimc_fine_roots.clamp(2, 16);
                let mut fine_roots = active_roots;
                if cfg.root_pimc_prescreen_worlds > 0 && fine_roots.len() > fine_k {
                    let w_pre = cfg.root_pimc_prescreen_worlds.min(w_cnt);
                    let h_pre = h.min(6);
                    let width_pre = (cfg.root_pimc_width / 2).clamp(6, 12);
                    let mut pre_ranked: Vec<(usize, i64, i64)> = Vec::with_capacity(fine_roots.len());
                    for &(ci, s1_sc) in &fine_roots {
                        let root_idx = candidates[ci].0;
                        let fp = self.root_places[root_idx as usize];
                        let mut sum_w: i64 = 0;
                        for w in 0..w_pre {
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
                                h_pre,
                                width_pre,
                                false,
                                w_seed,
                                cfg,
                            );
                            nodes_evaluated += evals;
                            sum_w += w_sc;
                        }
                        pre_ranked.push((ci, s1_sc, sum_w / (w_pre as i64)));
                    }
                    pre_ranked.sort_unstable_by_key(|x| (x.2, x.1, x.0));
                    let mut next_fine: Vec<(usize, i64)> = Vec::with_capacity(fine_k + 2);
                    let mut f_has_0ds = false;
                    let mut f_has_ds1 = false;
                    for &(ci, s1_sc, _) in pre_ranked.iter().take(fine_k) {
                        let fp = self.root_places[candidates[ci].0 as usize];
                        let mut b_tmp = root_board;
                        let o_tmp = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (_, ds_tmp, _) = b_tmp.place_and_sift(o_tmp, fp.norm_i, fp.norm_j);
                        if ds_tmp == 0 { f_has_0ds = true; } else { f_has_ds1 = true; }
                        next_fine.push((ci, s1_sc));
                    }
                    if !f_has_0ds || !f_has_ds1 {
                        for &(ci, s1_sc, _) in pre_ranked.iter().skip(fine_k) {
                            let fp = self.root_places[candidates[ci].0 as usize];
                            let mut b_tmp = root_board;
                            let o_tmp = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                            let (_, ds_tmp, _) = b_tmp.place_and_sift(o_tmp, fp.norm_i, fp.norm_j);
                            if !f_has_0ds && ds_tmp == 0 {
                                next_fine.push((ci, s1_sc));
                                f_has_0ds = true;
                            } else if !f_has_ds1 && ds_tmp > 0 {
                                next_fine.push((ci, s1_sc));
                                f_has_ds1 = true;
                            }
                            if f_has_0ds && f_has_ds1 { break; }
                        }
                    }
                    fine_roots = next_fine;
                }

                let eval_k = fine_roots.len();
                let mut pimc_scores = vec![0i64; eval_k];
                let width_fine = cfg.root_pimc_width.max(4);
                let cvar_pct = cfg.root_pimc_cvar_pct.clamp(0, 100);
                for (ai, &(ci, _)) in fine_roots.iter().enumerate() {
                    let root_idx = candidates[ci].0;
                    let fp = self.root_places[root_idx as usize];
                    let mut sum_w: i64 = 0;
                    let mut w_scores = [0i64; 24];
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
                            width_fine,
                            true,
                            w_seed,
                            cfg,
                        );
                        nodes_evaluated += evals;
                        sum_w += w_sc;
                        w_scores[w] = w_sc;
                    }
                    let mean_sc = sum_w / (w_cnt as i64);
                    pimc_scores[ai] = if cvar_pct > 0 && w_cnt >= 4 {
                        let tail_k = (w_cnt + 3) / 4;
                        w_scores[..w_cnt].sort_unstable_by(|a, b| b.cmp(a));
                        let tail_sum: i64 = w_scores[..tail_k].iter().sum();
                        let tail_sc = tail_sum / (tail_k as i64);
                        ((100 - cvar_pct) * mean_sc + cvar_pct * tail_sc) / 100
                    } else {
                        mean_sc
                    };
                }

                let min_pimc = pimc_scores.iter().copied().min().unwrap_or(0);
                let shift = best_s1 - min_pimc;
                let pw = cfg.root_pimc_weight_pct.clamp(0, 100);
                // Put reranked roots into a fresh sorted order and place unselected roots after them
                let mut selected_mask = vec![false; candidates.len()];
                let mut reranked_list: Vec<(u16, i64)> = Vec::with_capacity(eval_k);
                for (ai, &(ci, s1_sc)) in fine_roots.iter().enumerate() {
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
    fn eval_core_ex(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
        allow_spatial: bool,
    ) -> i64 {
        if allow_spatial {
            if let Some(ref sm) = cfg.spatial_model {
                return sm.evaluate_board_with_bag(board, rem_g, in_hand, avail_mask);
            }
        }
        if let Some(ref model) = cfg.value_model {
            let m_sc = model.evaluate_board_with_bag(board, rem_g, in_hand, avail_mask);
            if cfg.value_fast_micro != 0 && m_sc > -9_000_000 && m_sc < 4_000_000 {
                let f_sc = eval_fast_with_bag(board, rem_g, in_hand, avail_mask, &cfg.weights);
                (f_sc * cfg.value_fast_micro) / 10_000 + m_sc
            } else {
                m_sc
            }
        } else if let Some(ref sm) = cfg.spatial_model {
            if allow_spatial {
                sm.evaluate_board_with_bag(board, rem_g, in_hand, avail_mask)
            } else if cfg.spatial_fast_ew_internal {
                eval_fast_with_bag(board, rem_g, in_hand, avail_mask, &cfg.weights)
            } else if cfg.spatial_no_pe_internal {
                sm.evaluate_linear_board_with_bag_no_pe(board, rem_g, in_hand, avail_mask)
            } else {
                sm.evaluate_linear_board_with_bag(board, rem_g, in_hand, avail_mask)
            }
        } else {
            eval_fast_with_bag(board, rem_g, in_hand, avail_mask, &cfg.weights)
        }
    }

    #[inline]
    fn eval_state_ex(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
        allow_spatial: bool,
    ) -> i64 {
        if cfg.fog_spawn && board.has_opaque_top_garbage() {
            let g = board.garbage_rows as usize;
            let mut sum: i64 = 0;
            let mut b_tmp = *board;
            for c in 0..10u16 {
                b_tmp.data[g - 1] = EMPTY_ROW | (0x3FF & !(1u16 << c));
                sum += Self::eval_core_ex(&b_tmp, rem_g, in_hand, avail_mask, cfg, allow_spatial);
            }
            return sum / 10;
        }
        Self::eval_core_ex(board, rem_g, in_hand, avail_mask, cfg, allow_spatial)
    }

    #[inline]
    fn eval_state(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
    ) -> i64 {
        Self::eval_state_ex(board, rem_g, in_hand, avail_mask, cfg, !cfg.spatial_rerank_only)
    }

    #[inline]
    fn eval_leaf_state(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
    ) -> i64 {
        Self::eval_state_ex(board, rem_g, in_hand, avail_mask, cfg, true)
    }

    #[inline]
    fn eval_state_s1(
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        cfg: &BeamConfig,
    ) -> i64 {
        if let Some(ref s1w) = cfg.stage1_weights {
            if cfg.spatial_rerank_only || cfg.spatial_model.is_none() {
                if cfg.fog_spawn && board.has_opaque_top_garbage() {
                    let g = board.garbage_rows as usize;
                    let mut sum: i64 = 0;
                    let mut b_tmp = *board;
                    for c in 0..10u16 {
                        b_tmp.data[g - 1] = EMPTY_ROW | (0x3FF & !(1u16 << c));
                        sum += eval_fast_with_bag(&b_tmp, rem_g, in_hand, avail_mask, s1w);
                    }
                    return sum / 10;
                }
                return eval_fast_with_bag(board, rem_g, in_hand, avail_mask, s1w);
            }
        }
        Self::eval_state(board, rem_g, in_hand, avail_mask, cfg)
    }


    fn endgame_eval_weights() -> EvalWeights {
        let mut w = EvalWeights::default();
        w.garbage_row = 240;
        w.piece_estimate = 20;
        w.pe_net_mode = true;
        w.stack_cell = 10;
        w.upper_cell = 6;
        w.stack_h_sq = 0;
        w.stack_over2_sq = 4;
        w.max_stack_h_excess = 14;
        w.covered_hole = 90;
        w.hole_row = 24;
        w.blocks_over_hole = 24;
        w.h1_block = 42;
        w.h1_contig_cover = 120;
        w.h1_empty_blocking = 10;
        w.h1_height = 12;
        w.h2_block = 26;
        w.h2_contig_cover = 70;
        w.h2_extra_empty = 6;
        w.h3_block = 14;
        w.h3_extra_empty = 3;
        w.h4_block = 6;
        w.h1_narrow_1 = 16;
        w.h1_narrow_2plus = 24;
        w.h1_fit_scarcity = 6;
        w.h1_unfit_hand = 20;
        w.zero_o_spot_pen = 6;
        w.bag_no_i_chute_pen = 14;
        w.open_h1_bonus = 24;
        w.combo2_bonus = 22;
        w.bumpiness = 4;
        w.sharp_diff = 5;
        w.i_dep = 35;
        w.two_well = 10;
        w.i_dep_multi = 50;
        w.i_dep_chute = 12;
        w.near_full_9_bonus = 18;
        w.near_full_8_bonus = 5;
        w.endgame_cell_taper = 18;
        w.endgame_cell_taper_rem = 5;
        w
    }

    fn endgame_world_ds(
        &mut self,
        root_board: &FastBoard,
        hold0: Option<u8>,
        cur0: Option<u8>,
        q_len: usize,
        rem_g0: u16,
        root_fp: FastPlace,
        world_q: &[u8],
        end_bag: u8,
        end_w: &EvalWeights,
    ) -> (i64, usize) {
        let target_ds = (rem_g0 as usize).min(9);
        let mut b0 = *root_board;
        let orient0 = &self.fst.shapes[root_fp.color_idx as usize].orients[root_fp.tf.2 as usize];
        let (_, ds0, _) = b0.place_and_sift(orient0, root_fp.norm_i, root_fp.norm_j);
        let rem0 = rem_g0.saturating_sub(ds0 as u16);
        if rem0 == 0 && b0.rows < 20 {
            let res_cells: i64 = (0..b0.rows as usize)
                .map(|r| (b0.data[r] & 0x3FF).count_ones() as i64)
                .sum();
            return (1000 + res_cells * 2, 1);
        }
        if b0.rows >= 20 {
            return (1_000_000, 1);
        }

        let (held0, qc0) = if hold0.is_some() {
            if root_fp.did_hold {
                (cur0, if cur0.is_some() { 1.min(q_len) } else { 0 })
            } else {
                (hold0, 1.min(q_len))
            }
        } else if root_fp.did_hold {
            (cur0, 2.min(q_len))
        } else {
            (None, 1.min(q_len))
        };

        let in_hand0 = Self::hand_piece(held0, qc0, world_q);
        let avail0 = Self::avail_mask_from_q(held0, qc0, world_q, end_bag);
        let sc0 = 100 + eval_fast_with_bag(&b0, rem0, in_hand0, avail0, end_w);

        let ds0_u = (ds0 as usize).min(target_ds);
        let mut buckets: [Vec<BeamNode>; 10] = Default::default();
        buckets[ds0_u].push(BeamNode {
            board: b0,
            held: held0,
            q_cursor: qc0 as u8,
            rem_garbage: rem0,
            nds_used: 0,
            no_g_moves: 1, // stores pieces_used
            first_idx: 0,
            score: sc0,
        });

        let mut evals = 1usize;
        let mut best_win: Option<(u16, i64)> = None; // (pieces_used, res_cells)
        let mut best_rem_g = rem0;
        let mut best_fb_score = sc0;
        let wq_len = world_q.len();
        let beam_w = 18usize;
        let sub_w = 10usize;
        let z_limit = 3usize;

        for ds_cur in ds0_u..target_ds {
            let mut bucket = std::mem::take(&mut buckets[ds_cur]);
            if bucket.is_empty() {
                continue;
            }
            bucket.sort_unstable_by_key(|n| (n.score, n.no_g_moves));
            if bucket[0].rem_garbage < best_rem_g
                || (bucket[0].rem_garbage == best_rem_g && bucket[0].score < best_fb_score)
            {
                best_rem_g = bucket[0].rem_garbage;
                best_fb_score = bucket[0].score;
            }

            self.pimc_beam.clear();
            self.pimc_seen.clear();
            for ch in bucket {
                if self.pimc_beam.len() >= beam_w {
                    break;
                }
                if let Some((wp, _)) = best_win {
                    if ch.no_g_moves as u16 >= wp {
                        continue;
                    }
                }
                let key = DedupKey {
                    board: ch.board,
                    held: ch.held,
                    rem_garbage: ch.q_cursor as u16,
                };
                if self.pimc_seen.insert(key) {
                    self.pimc_beam.push(ch);
                }
            }

            let mut best_next_ds_sc = i64::MAX;
            for z_sub in 0..=z_limit {
                if self.pimc_beam.is_empty() {
                    break;
                }
                if z_sub > 0
                    && best_next_ds_sc != i64::MAX
                    && self.pimc_beam[0].score > best_next_ds_sc + 600
                {
                    break;
                }
                self.pimc_next.clear();

                for i in 0..self.pimc_beam.len() {
                    let node = self.pimc_beam[i];
                    if node.rem_garbage == 0 {
                        continue;
                    }
                    let np = (node.no_g_moves as u16) + 1;
                    if let Some((wp, _)) = best_win {
                        if np > wp {
                            continue;
                        }
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
                    if c_nh.is_none() && c_h.is_none() {
                        continue;
                    }

                    self.pgen
                        .generate_for_state(&self.fst, &node.board, c_nh, c_h, &mut self.roll_places);

                    for &fp in &self.roll_places {
                        let mut b2 = node.board;
                        let orient =
                            &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (_, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                        let rem2 = node.rem_garbage.saturating_sub(ds as u16);

                        if rem2 == 0 && b2.rows < 20 {
                            let res_cells: i64 = (0..b2.rows as usize)
                                .map(|r| (b2.data[r] & 0x3FF).count_ones() as i64)
                                .sum();
                            let better = match best_win {
                                None => true,
                                Some((wp, wc)) => (np, res_cells) < (wp, wc),
                            };
                            if better {
                                best_win = Some((np, res_cells));
                            }
                            continue;
                        }
                        if b2.rows >= 20 {
                            continue;
                        }
                        if let Some((wp, _)) = best_win {
                            if np >= wp {
                                continue;
                            }
                        }

                        let (new_held, next_qc) = if node.held.is_some() {
                            if fp.did_hold {
                                (c_nh, if c_nh.is_some() { (qc + 1).min(wq_len) } else { qc })
                            } else {
                                (node.held, (qc + 1).min(wq_len))
                            }
                        } else if fp.did_hold {
                            (c_nh, (qc + 2).min(wq_len))
                        } else {
                            (None, (qc + 1).min(wq_len))
                        };

                        let in_hand2 = Self::hand_piece(new_held, next_qc, world_q);
                        let avail2 = Self::avail_mask_from_q(new_held, next_qc, world_q, end_bag);
                        let sc = (np as i64) * 100
                            + eval_fast_with_bag(&b2, rem2, in_hand2, avail2, end_w);
                        evals += 1;

                        let out_node = BeamNode {
                            board: b2,
                            held: new_held,
                            q_cursor: next_qc as u8,
                            rem_garbage: rem2,
                            nds_used: 0,
                            no_g_moves: np as u8,
                            first_idx: 0,
                            score: sc,
                        };
                        if ds > 0 {
                            let nds = (ds_cur + ds as usize).min(target_ds);
                            if sc < best_next_ds_sc {
                                best_next_ds_sc = sc;
                            }
                            buckets[nds].push(out_node);
                        } else if z_sub < z_limit && (np as usize) < wq_len + 1 {
                            self.pimc_next.push(out_node);
                        }
                    }
                }

                if self.pimc_next.is_empty() {
                    self.pimc_beam.clear();
                    break;
                }
                self.pimc_next
                    .sort_unstable_by_key(|n| (n.score, n.no_g_moves));
                self.pimc_beam.clear();
                self.pimc_seen.clear();
                for i in 0..self.pimc_next.len() {
                    if self.pimc_beam.len() >= sub_w {
                        break;
                    }
                    let ch = self.pimc_next[i];
                    let key = DedupKey {
                        board: ch.board,
                        held: ch.held,
                        rem_garbage: ch.q_cursor as u16,
                    };
                    if self.pimc_seen.insert(key) {
                        self.pimc_beam.push(ch);
                    }
                }
            }
        }

        if let Some((wp, res_cells)) = best_win {
            ((wp as i64) * 1000 + res_cells * 2, evals)
        } else {
            (
                (wq_len as i64 + 2) * 1000
                    + (best_rem_g as i64) * 250
                    + best_fb_score.clamp(-5000, 100_000),
                evals,
            )
        }
    }

    fn solve_endgame_ds(
        &mut self,
        shtb: &ShapeTable,
        snapshot: &Snapshot,
        root_board: &FastBoard,
        rem_g0: u16,
        hold0: Option<u8>,
        q: &[u8],
        bag_mask: u8,
        crn_seed: u64,
    ) -> Option<BeamResult> {
        let q_len = q.len();
        let target_ds = (rem_g0 as usize).min(9);
        if target_ds == 0 || self.root_places.is_empty() {
            return None;
        }
        let end_w = Self::endgame_eval_weights();
        let n_roots = self.root_places.len();
        let cur0 = if q_len > 0 { Some(q[0]) } else { None };
        let total_known = q_len + usize::from(hold0.is_some());

        let mut buckets: [Vec<BeamNode>; 10] = Default::default();
        let mut root_best_prog = vec![i64::MAX; n_roots];
        let mut root_ds0 = vec![0u8; n_roots];
        let mut best_win: Option<(u16, i64, u16, u16)> = None; // (pieces_used, res_cells, nds, root_idx)
        let mut nodes_evaluated = 0usize;

        for (idx, &fp) in self.root_places.iter().enumerate() {
            let mut b = *root_board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (lines, ds, _) = b.place_and_sift(orient, fp.norm_i, fp.norm_j);
            root_ds0[idx] = ds;
            let rem_g = rem_g0.saturating_sub(ds as u16);
            if rem_g == 0 && b.rows < 20 {
                let res_cells: i64 = (0..b.rows as usize)
                    .map(|r| (b.data[r] & 0x3FF).count_ones() as i64)
                    .sum();
                let nds0 = lines.saturating_sub(ds) as u16;
                let better = match best_win {
                    None => true,
                    Some((wp, wc, wn, _)) => (1u16, res_cells, nds0) < (wp, wc, wn),
                };
                if better {
                    best_win = Some((1, res_cells, nds0, idx as u16));
                }
                continue;
            }
            if b.rows >= 20 {
                continue;
            }
            let (held, qc) = if hold0.is_some() {
                if fp.did_hold {
                    (cur0, if cur0.is_some() { 1 } else { 0 })
                } else {
                    (hold0, 1.min(q_len))
                }
            } else if fp.did_hold {
                (cur0, 2.min(q_len))
            } else {
                (None, 1.min(q_len))
            };
            let in_hand = Self::hand_piece(held, qc, q);
            let avail = Self::avail_mask_from_q(held, qc, q, bag_mask);
            let sc = 100 + eval_fast_with_bag(&b, rem_g, in_hand, avail, &end_w);
            nodes_evaluated += 1;
            if sc < root_best_prog[idx] {
                root_best_prog[idx] = sc;
            }
            let ds_u = (ds as usize).min(target_ds);
            buckets[ds_u].push(BeamNode {
                board: b,
                held,
                q_cursor: qc as u8,
                rem_garbage: rem_g,
                nds_used: lines.saturating_sub(ds) as u16,
                no_g_moves: 1,
                first_idx: idx as u16,
                score: sc,
            });
        }

        // Step 1: Exact DS-bucketed BFS over known pieces (hold + previews)
        if best_win.map_or(true, |(wp, _, _, _)| wp > 1) {
            let end_bw = 180usize;
            let end_sub_w = 140usize;
            let per_root_cap = 16usize;
            let z_limit = 4usize;

            for ds_cur in 0..target_ds {
                let mut bucket = std::mem::take(&mut buckets[ds_cur]);
                if bucket.is_empty() {
                    continue;
                }
                bucket.sort_unstable_by_key(|n| (n.score, n.no_g_moves, n.first_idx));

                self.cur_beam.clear();
                self.seen.clear();
                self.root_counts.clear();
                self.root_counts.resize(n_roots, 0);

                for &ch in &bucket {
                    if self.cur_beam.len() >= end_bw {
                        break;
                    }
                    if let Some((wp, _, _, _)) = best_win {
                        if ch.no_g_moves as u16 >= wp {
                            continue;
                        }
                    }
                    if self.root_counts[ch.first_idx as usize] >= per_root_cap {
                        continue;
                    }
                    let key = DedupKey {
                        board: ch.board,
                        held: ch.held,
                        rem_garbage: ch.q_cursor as u16,
                    };
                    if self.seen.insert(key) {
                        self.root_counts[ch.first_idx as usize] += 1;
                        self.cur_beam.push(ch);
                    }
                }
                if self.cur_beam.len() < end_bw {
                    for &ch in &bucket {
                        if self.cur_beam.len() >= end_bw {
                            break;
                        }
                        if let Some((wp, _, _, _)) = best_win {
                            if ch.no_g_moves as u16 >= wp {
                                continue;
                            }
                        }
                        let key = DedupKey {
                            board: ch.board,
                            held: ch.held,
                            rem_garbage: ch.q_cursor as u16,
                        };
                        if self.seen.insert(key) {
                            self.cur_beam.push(ch);
                        }
                    }
                }

                let mut best_next_ds_sc = i64::MAX;
                for z_sub in 0..=z_limit {
                    if self.cur_beam.is_empty() {
                        break;
                    }
                    if z_sub > 0
                        && best_next_ds_sc != i64::MAX
                        && self.cur_beam[0].score > best_next_ds_sc + 700
                    {
                        break;
                    }
                    self.children.clear();

                    for ni in 0..self.cur_beam.len() {
                        let node = self.cur_beam[ni];
                        if node.rem_garbage == 0 {
                            continue;
                        }
                        let np = (node.no_g_moves as u16) + 1;
                        if let Some((wp, _, _, _)) = best_win {
                            if np > wp {
                                continue;
                            }
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
                        if c_nh.is_none() && c_h.is_none() {
                            continue;
                        }

                        self.pgen.generate_for_state(
                            &self.fst,
                            &node.board,
                            c_nh,
                            c_h,
                            &mut self.tmp_places,
                        );

                        for &fp in &self.tmp_places {
                            let mut b = node.board;
                            let orient =
                                &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                            let (lines, ds, _) = b.place_and_sift(orient, fp.norm_i, fp.norm_j);
                            let rem_g = node.rem_garbage.saturating_sub(ds as u16);
                            let nds_new = node.nds_used.saturating_add(lines.saturating_sub(ds) as u16);

                            if rem_g == 0 && b.rows < 20 {
                                let res_cells: i64 = (0..b.rows as usize)
                                    .map(|r| (b.data[r] & 0x3FF).count_ones() as i64)
                                    .sum();
                                let better = match best_win {
                                    None => true,
                                    Some((wp, wc, wn, _)) => {
                                        (np, res_cells, nds_new) < (wp, wc, wn)
                                    }
                                };
                                if better {
                                    best_win = Some((np, res_cells, nds_new, node.first_idx));
                                }
                                continue;
                            }
                            if b.rows >= 20 {
                                continue;
                            }
                            if let Some((wp, _, _, _)) = best_win {
                                if np >= wp {
                                    continue;
                                }
                            }

                            let (held, next_qc) = if node.held.is_some() {
                                if fp.did_hold {
                                    (c_nh, if c_nh.is_some() { (qc + 1).min(q_len) } else { qc })
                                } else {
                                    (node.held, (qc + 1).min(q_len))
                                }
                            } else if fp.did_hold {
                                (c_nh, (qc + 2).min(q_len))
                            } else {
                                (None, (qc + 1).min(q_len))
                            };

                            let in_hand = Self::hand_piece(held, next_qc, q);
                            let avail = Self::avail_mask_from_q(held, next_qc, q, bag_mask);
                            let sc = (np as i64) * 100
                                + eval_fast_with_bag(&b, rem_g, in_hand, avail, &end_w);
                            nodes_evaluated += 1;

                            let ri = node.first_idx as usize;
                            if sc < root_best_prog[ri] {
                                root_best_prog[ri] = sc;
                            }

                            let out_node = BeamNode {
                                board: b,
                                held,
                                q_cursor: next_qc as u8,
                                rem_garbage: rem_g,
                                nds_used: nds_new,
                                no_g_moves: np as u8,
                                first_idx: node.first_idx,
                                score: sc,
                            };
                            if ds > 0 {
                                let nds = (ds_cur + ds as usize).min(target_ds);
                                if sc < best_next_ds_sc {
                                    best_next_ds_sc = sc;
                                }
                                buckets[nds].push(out_node);
                            } else if z_sub < z_limit {
                                self.children.push(out_node);
                            }
                        }
                    }

                    if self.children.is_empty() {
                        self.cur_beam.clear();
                        break;
                    }
                    self.children
                        .sort_unstable_by_key(|n| (n.score, n.no_g_moves, n.first_idx));
                    self.cur_beam.clear();
                    self.seen.clear();
                    for i in 0..self.children.len() {
                        if self.cur_beam.len() >= end_sub_w {
                            break;
                        }
                        let ch = self.children[i];
                        let key = DedupKey {
                            board: ch.board,
                            held: ch.held,
                            rem_garbage: ch.q_cursor as u16,
                        };
                        if self.seen.insert(key) {
                            self.cur_beam.push(ch);
                        }
                    }
                }
            }
        }

        if let Some((wp, res_cells, _, win_ri)) = best_win {
            let fp = self.root_places[win_ri as usize];
            if let Some(inputs) = self.build_inputs(shtb, &snapshot.matrix, fp) {
                let rating = -30_000_000 + (wp as i64) * 200 + res_cells;
                self.last_top.clear();
                self.last_top_leaves.clear();
                let c = idx_to_color(fp.color_idx as usize).as_char();
                self.last_top
                    .push((c, fp.norm_i, fp.norm_j, fp.tf.2 as u8, fp.did_hold, rating));
                return Some(BeamResult {
                    best_place: fp,
                    inputs,
                    rating,
                    nodes_evaluated,
                });
            }
        }

        // Step 2: Not finishable within known queue -> CRN bag-stratified DS endgame solver
        let mut root_order: Vec<(u16, i64)> = (0..n_roots)
            .filter(|&ri| root_best_prog[ri] != i64::MAX)
            .map(|ri| (ri as u16, root_best_prog[ri]))
            .collect();
        if root_order.is_empty() {
            return None;
        }
        root_order.sort_unstable_by_key(|&(ri, sc)| (sc, ri));

        let mut cand_roots: Vec<u16> = Vec::with_capacity(12);
        let mut has_0ds = false;
        let mut has_ds1 = false;
        for &(ri, _) in root_order.iter().take(10) {
            cand_roots.push(ri);
            if root_ds0[ri as usize] == 0 {
                has_0ds = true;
            } else {
                has_ds1 = true;
            }
        }
        if !has_0ds || !has_ds1 {
            for &(ri, _) in root_order.iter().skip(10) {
                let ds0 = root_ds0[ri as usize];
                if !has_0ds && ds0 == 0 {
                    cand_roots.push(ri);
                    has_0ds = true;
                } else if !has_ds1 && ds0 > 0 {
                    cand_roots.push(ri);
                    has_ds1 = true;
                }
                if has_0ds && has_ds1 {
                    break;
                }
            }
        }

        let eff_mask = if bag_mask == 0 { 0x7F } else { bag_mask };
        let mut bag_bits = [0u8; 7];
        let mut bag_n = 0usize;
        let mut bm = eff_mask;
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

        let w_cnt = if bag_n <= 2 { 4 } else { 5 };
        let need_q_len = (rem_g0 as usize + 8).clamp(total_known + 3, 15).min(16);
        let mut world_qs = [[0u8; 16]; 8];
        let mut world_end_bags = [0x7Fu8; 8];
        for w in 0..w_cnt {
            for i in 0..q_len {
                world_qs[w][i] = q[i];
            }
            let mut bag = eff_mask;
            let mut w_rng = crn_seed
                ^ ((w as u64) + 1).wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0xBF58476D1CE4E5B9;
            if q_len < need_q_len {
                if bag_n == 2 {
                    let first_i = w % 2;
                    world_qs[w][q_len] = bag_bits[first_i];
                    if q_len + 1 < need_q_len {
                        world_qs[w][q_len + 1] = bag_bits[1 - first_i];
                    }
                    bag = 0x7F;
                    for step in (q_len + 2)..need_q_len {
                        world_qs[w][step] = Self::draw_from_bag(&mut bag, &mut w_rng);
                    }
                } else {
                    let p0 = bag_bits[w % bag_n.max(1)];
                    world_qs[w][q_len] = p0;
                    bag &= !(1u8 << p0);
                    if bag == 0 {
                        bag = 0x7F;
                    }
                    for step in (q_len + 1)..need_q_len {
                        world_qs[w][step] = Self::draw_from_bag(&mut bag, &mut w_rng);
                    }
                }
            }
            world_end_bags[w] = bag;
        }

        let mut scored_roots: Vec<(u16, i64)> = Vec::with_capacity(cand_roots.len());
        for &ri in &cand_roots {
            let fp = self.root_places[ri as usize];
            let mut sum_w: i64 = 0;
            let mut worst_w: i64 = i64::MIN;
            for w in 0..w_cnt {
                let (w_sc, evals) = self.endgame_world_ds(
                    root_board,
                    hold0,
                    cur0,
                    q_len,
                    rem_g0,
                    fp,
                    &world_qs[w][..need_q_len],
                    world_end_bags[w],
                    &end_w,
                );
                nodes_evaluated += evals;
                sum_w += w_sc;
                if w_sc > worst_w {
                    worst_w = w_sc;
                }
            }
            let mean_sc = sum_w / (w_cnt as i64);
            let comb_sc = (mean_sc * 85 + worst_w * 15) / 100;
            scored_roots.push((ri, comb_sc));
        }
        scored_roots.sort_unstable_by_key(|&(ri, sc)| (sc, ri));

        self.last_top.clear();
        self.last_top_leaves.clear();
        for &(ri, sc) in scored_roots.iter().take(10) {
            let fp = self.root_places[ri as usize];
            let c = idx_to_color(fp.color_idx as usize).as_char();
            self.last_top
                .push((c, fp.norm_i, fp.norm_j, fp.tf.2 as u8, fp.did_hold, -20_000_000 + sc));
        }

        for &(ri, sc) in &scored_roots {
            let fp = self.root_places[ri as usize];
            if let Some(inputs) = self.build_inputs(shtb, &snapshot.matrix, fp) {
                return Some(BeamResult {
                    best_place: fp,
                    inputs,
                    rating: -20_000_000 + sc,
                    nodes_evaluated,
                });
            }
        }

        None
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
            let res_cells: i64 = (0..board.rows as usize)
                .map(|r| (board.data[r] & 0x3FF).count_ones() as i64)
                .sum();
            return (
                if board.rows < 20 {
                    -20_000_000 + res_cells
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
        let h = horizon.clamp(1, 16);
        let mult = if bag_n <= 2 { 2 } else { 1 };
        let w_cnt = (bag_n.max(1) * mult).min(8);
        let end_beam_w = if rem_g <= 5 { 56usize } else { 40usize };
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
            let mut wq = [0u8; 16];
            let mut w_bags = [0x7Fu8; 16];
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
                nds_used: 0,
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
                        let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                        let rem2 = node.rem_garbage.saturating_sub(ds as u16);
                        let nds_new = node.nds_used.saturating_add(lines.saturating_sub(ds) as u16);
                        if rem2 == 0 && b2.rows < 20 {
                            let res_cells: i64 = (0..b2.rows as usize)
                                .map(|r| (b2.data[r] & 0x3FF).count_ones() as i64)
                                .sum();
                            let done_sc = ((step as i64) + 1) * 200 + res_cells;
                            if done_sc < best_done {
                                best_done = done_sc;
                            }
                            continue;
                        }
                        let new_hand = if fp.color_idx == next_p { cur_h } else { next_p };
                        let sc = Self::eval_state_ex(&b2, rem2, Some(new_hand), step_bag, cfg, false)
                            + (nds_new as i64) * cfg.nds_pen;
                        evals += 1;
                        self.pimc_next.push(BeamNode {
                            board: b2,
                            held: Some(new_hand),
                            q_cursor: 0,
                            rem_garbage: rem2,
                            nds_used: nds_new,
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
                50_000 + (top.rem_garbage as i64) * 300 + top.score.clamp(-30_000, 30_000)
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
            Self::eval_leaf_state(&board, rem_g, Some(in_hand), bag, cfg)
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
                let sc = if horizon == 0 {
                    if rem2 == 0 && b2.rows < 20 {
                        -15_000_000 + (b2.rows as i64) * 5
                    } else {
                        let ok_spawn = if rem2 > 0 && (cfg.simulate_cheese_spawn || cfg.fog_spawn) {
                            let mut spawn_rng = crn_seed
                                ^ (next_p as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15)
                                ^ 0x123456789ABCDEF;
                            if cfg.fog_spawn {
                                b2.materialize_fog_holes(&mut spawn_rng);
                            }
                            b2.spawn_cheese_rows(rem2, lines, &mut spawn_rng)
                        } else {
                            true
                        };
                        if !ok_spawn {
                            5_000_000
                        } else {
                            Self::eval_leaf_state(&b2, rem2, Some(new_hand), rem_after_p, cfg)
                                - ng_bonus
                        }
                    }
                } else {
                    Self::eval_state(&b2, rem2, Some(new_hand), rem_after_p, cfg) - ng_bonus
                };
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
                Self::eval_leaf_state(&board, rem_g, Some(in_hand), bag_mask, cfg)
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
                    let nds_new = node.nds_used.saturating_add(lines.saturating_sub(ds) as u16);
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
                    let sc = Self::eval_state(&b2, rem2, new_held, step_avail, cfg)
                        + (nds_new as i64) * cfg.nds_pen
                        - ng_bonus;
                    evals += 1;
                    self.pimc_next.push(BeamNode {
                        board: b2,
                        held: new_held,
                        q_cursor: node.q_cursor,
                        rem_garbage: rem2,
                        nds_used: nds_new,
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

    fn root_pimc_eval_world_ds(
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
        width: usize,
        allow_rollout: bool,
        w_seed: u64,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        let target_ds = (if allow_rollout {
            cfg.root_pimc_ds_target
        } else {
            cfg.root_pimc_ds_target.min(2)
        })
        .clamp(1, 8)
        .min(rem_g0 as usize);

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

        let nds0 = lines0.saturating_sub(ds0) as u16;
        let ng0 = u8::from(root_board.garbage_rows == 0 && rem_g0 > 0);
        let in_hand0 = Self::hand_piece(held0, qc0, world_q);
        let avail0 = Self::avail_mask_from_q(held0, qc0, world_q, end_bag);
        let sc0 = 100
            + (nds0 as i64) * cfg.nds_pen
            + Self::eval_state(&b0, rem0, in_hand0, avail0, cfg)
            - cfg.no_g_credit * (ng0 as i64);

        let ds0_u = (ds0 as usize).min(target_ds);
        let mut buckets: [Vec<BeamNode>; 9] = Default::default();
        buckets[ds0_u].push(BeamNode {
            board: b0,
            held: held0,
            q_cursor: qc0 as u8,
            rem_garbage: rem0,
            nds_used: nds0,
            no_g_moves: ng0,
            first_idx: 1, // stores pieces_used
            score: sc0,
        });

        let mut evals = 1usize;
        let mut best_done = i64::MAX;
        let mut best_fallback = i64::MAX;
        let mut best_fallback_node: Option<(BeamNode, usize)> = None;
        let width = width.max(4);
        let sub_width = (width * 2 / 3).max(5);
        let wq_len = world_q.len();
        let z_limit = if allow_rollout {
            (horizon / 3).clamp(3, 4)
        } else {
            2
        };
        let max_pieces = wq_len.saturating_sub(1).min(target_ds * 2 + 6).max(3);

        for ds_cur in ds0_u..target_ds {
            let mut bucket = std::mem::take(&mut buckets[ds_cur]);
            if bucket.is_empty() {
                continue;
            }
            bucket.sort_unstable_by_key(|n| (n.score, n.first_idx));
            let fb = bucket[0].score + ((target_ds - ds_cur) as i64) * 350;
            if fb < best_fallback {
                best_fallback = fb;
                best_fallback_node = Some((bucket[0], ds_cur));
            }
            self.pimc_beam.clear();
            self.pimc_seen.clear();
            for ch in bucket {
                if self.pimc_beam.len() >= width {
                    break;
                }
                let key = DedupKey {
                    board: ch.board,
                    held: ch.held,
                    rem_garbage: ch.q_cursor as u16,
                };
                if self.pimc_seen.insert(key) {
                    self.pimc_beam.push(ch);
                }
            }

            let mut best_next_ds_sc = i64::MAX;
            let eff_z_limit = z_limit;
            for z_sub in 0..=eff_z_limit {
                if self.pimc_beam.is_empty() {
                    break;
                }
                if z_sub > 0
                    && best_next_ds_sc != i64::MAX
                    && self.pimc_beam[0].score > best_next_ds_sc + 550
                {
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
                    if c_nh.is_none() && c_h.is_none() {
                        continue;
                    }

                    self.pgen
                        .generate_for_state(&self.fst, &node.board, c_nh, c_h, &mut self.tmp_places);

                    let ng = node
                        .no_g_moves
                        .saturating_add(u8::from(node.board.garbage_rows == 0 && node.rem_garbage > 0));
                    let ng_bonus = cfg.no_g_credit * (ng as i64);
                    let np = node.first_idx + 1;

                    for &fp in &self.tmp_places {
                        let mut b2 = node.board;
                        let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                        let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                        let rem2 = node.rem_garbage.saturating_sub(ds as u16);
                        let nds_new = node.nds_used.saturating_add(lines.saturating_sub(ds) as u16);

                        if rem2 == 0 && b2.rows < 20 {
                            let g_sc = -15_000_000 + (np as i64) * 100 + (b2.rows as i64) * 5;
                            if g_sc < best_done {
                                best_done = g_sc;
                            }
                            continue;
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
                                (c_nh, if c_nh.is_some() { (qc + 1).min(wq_len) } else { qc })
                            } else {
                                (node.held, (qc + 1).min(wq_len))
                            }
                        } else if fp.did_hold {
                            (c_nh, (qc + 2).min(wq_len))
                        } else {
                            (None, (qc + 1).min(wq_len))
                        };

                        let in_hand2 = Self::hand_piece(new_held, next_qc, world_q);
                        let avail2 = Self::avail_mask_from_q(new_held, next_qc, world_q, end_bag);
                        let sc = (np as i64) * 100
                            + (nds_new as i64) * cfg.nds_pen
                            + Self::eval_state(&b2, rem2, in_hand2, avail2, cfg)
                            - ng_bonus;
                        evals += 1;

                        let out_node = BeamNode {
                            board: b2,
                            held: new_held,
                            q_cursor: next_qc as u8,
                            rem_garbage: rem2,
                            nds_used: nds_new,
                            no_g_moves: ng,
                            first_idx: np,
                            score: sc,
                        };
                        if ds > 0 {
                            let nds = (ds_cur + ds as usize).min(target_ds);
                            if sc < best_next_ds_sc {
                                best_next_ds_sc = sc;
                            }
                            buckets[nds].push(out_node);
                        } else if z_sub < eff_z_limit && (np as usize) < max_pieces {
                            self.pimc_next.push(out_node);
                        }
                    }
                }

                if self.pimc_next.is_empty() {
                    self.pimc_beam.clear();
                    break;
                }
                self.pimc_next.sort_unstable_by_key(|n| (n.score, n.first_idx));
                self.pimc_beam.clear();
                self.pimc_seen.clear();
                for i in 0..self.pimc_next.len() {
                    if self.pimc_beam.len() >= sub_width {
                        break;
                    }
                    let ch = self.pimc_next[i];
                    let key = DedupKey {
                        board: ch.board,
                        held: ch.held,
                        rem_garbage: ch.q_cursor as u16,
                    };
                    if self.pimc_seen.insert(key) {
                        self.pimc_beam.push(ch);
                    }
                }
            }
        }

        if best_done != i64::MAX {
            return (best_done, evals);
        }

        if !buckets[target_ds].is_empty() {
            buckets[target_ds].sort_unstable_by_key(|n| (n.score, n.first_idx));
            if allow_rollout && cfg.rollout_top_k > 0 {
                let check_k = buckets[target_ds].len().min(2);
                let mut best_roll = i64::MAX;
                for li in 0..check_k {
                    let leaf = buckets[target_ds][li];
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
                        (leaf.first_idx as i64) * 100 + (leaf.nds_used as i64) * cfg.nds_pen + r_sc
                    } else {
                        leaf.score
                    };
                    if l_sc < best_roll {
                        best_roll = l_sc;
                    }
                }
                return (best_roll, evals);
            } else if allow_rollout && cfg.spatial_rerank_only && cfg.spatial_model.is_some() {
                self.pimc_seen.clear();
                let mut best_leaf = i64::MAX;
                let mut checked = 0usize;
                let sm = cfg.spatial_model.as_ref().unwrap();
                for &leaf in buckets[target_ds].iter().take(24) {
                    let key = DedupKey {
                        board: leaf.board,
                        held: leaf.held,
                        rem_garbage: leaf.q_cursor as u16,
                    };
                    if !self.pimc_seen.insert(key) {
                        continue;
                    }
                    let hc = Self::hand_piece(leaf.held, leaf.q_cursor as usize, world_q);
                    let av = Self::avail_mask_from_q(leaf.held, leaf.q_cursor as usize, world_q, end_bag);
                    evals += 1;
                    let leaf_v = if allow_rollout {
                        Self::eval_leaf_state(&leaf.board, leaf.rem_garbage, hc, av, cfg)
                    } else {
                        sm.evaluate_linear_board_with_bag(&leaf.board, leaf.rem_garbage, hc, av)
                    };
                    let sc = (leaf.first_idx as i64) * 100
                        + (leaf.nds_used as i64) * cfg.nds_pen
                        + leaf_v
                        - cfg.no_g_credit * (leaf.no_g_moves as i64);
                    if sc < best_leaf {
                        best_leaf = sc;
                    }
                    checked += 1;
                    if checked >= 6 {
                        break;
                    }
                }
                return (best_leaf, evals);
            }
            return (buckets[target_ds][0].score, evals);
        }

        if let Some((fb_node, fb_ds)) = best_fallback_node {
            if allow_rollout && cfg.spatial_rerank_only && cfg.spatial_model.is_some() {
                let hc = Self::hand_piece(fb_node.held, fb_node.q_cursor as usize, world_q);
                let av = Self::avail_mask_from_q(
                    fb_node.held,
                    fb_node.q_cursor as usize,
                    world_q,
                    end_bag,
                );
                evals += 1;
                let leaf_v =
                    Self::eval_leaf_state(&fb_node.board, fb_node.rem_garbage, hc, av, cfg);
                let sc = (fb_node.first_idx as i64) * 100
                    + (fb_node.nds_used as i64) * cfg.nds_pen
                    + leaf_v
                    - cfg.no_g_credit * (fb_node.no_g_moves as i64)
                    + ((target_ds - fb_ds) as i64) * 450;
                return (sc, evals);
            }
            (best_fallback, evals)
        } else {
            (5_000_000, evals)
        }
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
        width: usize,
        allow_rollout: bool,
        w_seed: u64,
        cfg: &BeamConfig,
    ) -> (i64, usize) {
        if cfg.root_pimc_ds_target > 0 {
            return self.root_pimc_eval_world_ds(
                root_board,
                hold0,
                cur0,
                q_len,
                rem_g0,
                unspawned_0,
                root_fp,
                world_q,
                world_bags,
                end_bag,
                world_holes,
                horizon,
                width,
                allow_rollout,
                w_seed,
                cfg,
            );
        }
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

        let nds0 = lines0.saturating_sub(ds0) as u16;
        let ng0 = u8::from(root_board.garbage_rows == 0 && rem_g0 > 0);
        let in_hand0 = Self::hand_piece(held0, qc0, world_q);
        let avail0 = Self::avail_mask_from_q(held0, qc0, world_q, end_bag);
        let sc0 = (nds0 as i64) * cfg.nds_pen
            + Self::eval_state(&b0, rem0, in_hand0, avail0, cfg)
            - cfg.no_g_credit * (ng0 as i64);

        self.pimc_beam.clear();
        self.pimc_beam.push(BeamNode {
            board: b0,
            held: held0,
            q_cursor: qc0 as u8,
            rem_garbage: rem0,
            nds_used: nds0,
            no_g_moves: ng0,
            first_idx: 0,
            score: sc0,
        });

        let mut best_done = i64::MAX;
        let mut evals = 1usize;
        let width = width.max(4);
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
                    let nds_new = node.nds_used.saturating_add(lines.saturating_sub(ds) as u16);

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
                    let sc = (nds_new as i64) * cfg.nds_pen
                        + Self::eval_state(&b2, rem2, in_hand2, avail2, cfg)
                        - ng_bonus;
                    evals += 1;

                    self.pimc_next.push(BeamNode {
                        board: b2,
                        held: new_held,
                        q_cursor: next_qc as u8,
                        rem_garbage: rem2,
                        nds_used: nds_new,
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

        if allow_rollout && cfg.rollout_top_k > 0 {
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
                    (leaf.nds_used as i64) * cfg.nds_pen + r_sc
                } else {
                    leaf.score
                };
                if l_sc < best_roll {
                    best_roll = l_sc;
                }
            }
            (best_roll, evals)
        } else if allow_rollout && cfg.spatial_rerank_only && cfg.spatial_model.is_some() {
            let check_k = self.pimc_beam.len().min(10);
            let mut best_leaf = i64::MAX;
            for li in 0..check_k {
                let leaf = self.pimc_beam[li];
                let hc = Self::hand_piece(leaf.held, leaf.q_cursor as usize, world_q);
                let av = Self::avail_mask_from_q(leaf.held, leaf.q_cursor as usize, world_q, end_bag);
                evals += 1;
                let sc = (leaf.nds_used as i64) * cfg.nds_pen
                    + Self::eval_leaf_state(&leaf.board, leaf.rem_garbage, hc, av, cfg)
                    - cfg.no_g_credit * (leaf.no_g_moves as i64);
                if sc < best_leaf {
                    best_leaf = sc;
                }
            }
            (best_leaf, evals)
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
