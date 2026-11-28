//! Zero-Strategy-Fusion Single-Observer Information-Set MCTS (SO-ISMCTS / Expecti-MCTS)
//! for Legal `p=5` Jstris 100L Cheese Race.
//!
//! Design guarantees:
//! 1. **Strict Legal `p=5` Observability**: Never reads beyond `snapshot.queue[..6]` (cur + 5 previews)
//!    or `ctx.next_bag_mask`. Every `DecisionNode` stores its own exact observable preview window
//!    `vis_q[..vis_len]` and `bag_mask`, updated strictly when chance events reveal new pieces.
//! 2. **Zero Strategy Fusion**: Every decision node represents an observable information set
//!    reached *before* unspawned garbage holes or unrevealed future pieces occur. All simulations
//!    reaching the same decision node share the exact same PUCT action statistics `(N, Q, P)` and
//!    cannot condition their move on unrevealed future outcomes.
//! 3. **Depth-Consistent Cost-to-Go**: Uses Frozen BC-4 FULL-MLP (`SpatialValueModel::evaluate_board_with_bag`)
//!    for leaf values (`J(v) = depth * step_cost_cp + V_bc4(v)`, centipieces, lower is better) and
//!    Frozen BC-4 fast linear evaluation for 1-ply action priors `P(u, a)`.

use std::collections::HashMap;
use std::hash::BuildHasherDefault;
use std::sync::Arc;

use super::{
    beam::{BagTracker, FxHasher, GameContext},
    fast::{color_to_idx, idx_to_color, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
    spatial_val::SpatialValueModel,
    Snapshot,
};
use crate::{finesse::FinesseFinder, shape::ShapeTable, Color, Input};

type FxHashMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// Configuration for `IsMctsSearcher`.
#[derive(Clone, Debug)]
pub struct IsMctsConfig {
    /// Number of MCTS simulations per root move.
    pub simulations: usize,
    /// Maximum search depth (in plies from root).
    pub max_depth: usize,
    /// Maximum candidate actions retained per expanded decision node.
    pub max_actions: usize,
    /// Maximum candidate actions retained at the root decision node.
    pub root_max_actions: usize,
    /// PUCT exploration constant `c_puct`.
    pub c_puct: f64,
    /// Temperature (in centipieces) for 1-ply softmax prior `P(u, a)`.
    pub prior_temp: f64,
    /// First-Play Urgency penalty (in centipieces added to parent value for unvisited actions).
    pub fpu_penalty: f64,
    /// Value normalization scale (centipieces) for PUCT Q-term.
    pub q_scale: f64,
    /// Topout penalty (centipieces above root reference value).
    pub topout_penalty: f64,
    /// Whether to use steady-state `target_g = 4` cheese spawning (Jstris 100L Cheese Race).
    pub steady_g4: bool,
    /// If true, select root move by highest visit count (tie-break by lowest Q).
    /// If false, select root move by lowest Q among actions with at least `min_select_visits`.
    pub select_by_visits: bool,
    /// Backup mode at decision nodes: `0` = standard MCTS mean backup, `1` = Bellman Expecti-Min,
    /// `2` = 50/50 blend, `3` = conservative Bellman.
    pub backup_mode: u8,
    /// Per-ply step cost in centipieces added to leaf `V_bc4` (`depth * step_cost_cp + V_bc4`).
    pub step_cost_cp: f64,
    /// If true, chance events branch on every newly revealed tail preview piece so every child
    /// `DecisionNode` maintains the exact full 6-piece Legal `p=5` window (`cur + next5`).
    /// If false, child nodes retain all already-revealed pieces from the parent's `vis_q` and
    /// only reveal new pieces from `bag_mask` once needed to keep `(cur, next1)` known.
    pub split_tail_preview: bool,
    /// If true, run a fast deterministic Legal `p=5` exact clear solver whenever all remaining
    /// garbage is already visible on the board (`rem_g <= 9 && rem_g == garbage_rows`).
    pub use_endgame_solver: bool,
    /// If true, share identical `(board, hold, rem_g, vis_q, vis_len, bag_mask, depth)` nodes
    /// across paths via an acyclic Transposition DAG.
    pub use_transposition: bool,
    /// If true (default), propagate shared-child `v_bellman` updates to ALL parents in exact
    /// reverse-depth topological order so no parent caches stale child values.
    pub dag_propagate_parents: bool,
    /// Chance-Node Progressive Widening coefficient `c_pw` (active when `cpw_alpha > 0.0`).
    /// Caps distinct expanded chance outcomes on an edge at `ceil(cpw_c * (N + 1)^cpw_alpha)`.
    pub cpw_c: f64,
    /// Chance-Node Progressive Widening exponent `alpha_pw` (`0.0` = disabled; e.g. `0.5` = sqrt).
    pub cpw_alpha: f64,
    /// Optional separate CPW coefficient for pure tail-preview chance events (`n_spawn == 0`).
    /// Active when `cpw_tail_alpha > 0.0`; otherwise falls back to `(cpw_c, cpw_alpha)`.
    pub cpw_tail_c: f64,
    /// Optional separate CPW exponent for pure tail-preview chance events (`n_spawn == 0`).
    pub cpw_tail_alpha: f64,
    /// Frozen or iterated spatial value model (used for full-MLP leaf evaluation).
    pub spatial_model: Option<Arc<SpatialValueModel>>,
    /// Optional separate policy prior model (used ONLY for 1-ply linear prior `P(u, a)` in `expand_node`;
    /// falls back to `spatial_model` when `None`).
    pub policy_prior: Option<Arc<SpatialValueModel>>,
}

impl Default for IsMctsConfig {
    fn default() -> Self {
        Self {
            simulations: 400,
            max_depth: 10,
            max_actions: 10,
            root_max_actions: 16,
            c_puct: 1.5,
            prior_temp: 120.0,
            fpu_penalty: 80.0,
            q_scale: 250.0,
            topout_penalty: 4000.0,
            steady_g4: true,
            select_by_visits: false,
            backup_mode: 1,
            step_cost_cp: 100.0,
            split_tail_preview: true,
            use_endgame_solver: false,
            use_transposition: false,
            dag_propagate_parents: true,
            cpw_c: 1.0,
            cpw_alpha: 0.0,
            cpw_tail_c: 0.5,
            cpw_tail_alpha: 0.0,
            spatial_model: None,
            policy_prior: None,
        }
    }
}

/// Output of an `IsMctsSearcher::search_with_ctx` call.
#[derive(Clone, Debug)]
pub struct IsMctsResult {
    pub inputs: Vec<Input>,
    pub best_q: i64,
    pub best_visits: u32,
    pub total_simulations: usize,
    pub nodes_expanded: usize,
    pub mlp_evals: usize,
    /// Average depth (in plies from root) reached per simulation.
    pub avg_sim_depth: f64,
    /// Maximum depth reached across all simulations.
    pub max_sim_depth: usize,
    /// Average additional depth reached *after* the first garbage spawn (among simulations
    /// that triggered at least one garbage spawn).
    pub post_spawn_avg_depth: f64,
    /// Number of simulations that triggered at least one garbage spawn.
    pub spawn_sims: usize,
    /// Number of transposition DAG hits (when `use_transposition == true`).
    pub trans_hits: usize,
    /// Whether the move was chosen by the deterministic endgame clear solver.
    pub used_endgame_solver: bool,
    pub root_candidates: Vec<(FastPlace, u32, f64, f64)>, // (place, visits, q, prior)
}

#[derive(Clone, Debug)]
struct ActionEdge {
    fp: FastPlace,
    prior: f64,
    visits: u32,
    q_sum: f64,
    q_bellman: f64,
    term_visits: u32,
    term_q_sum: f64,
    /// Chance outcome `(chance_key, child_node_id, edge_local_visits)`.
    /// Storing `edge_local_visits` ensures Transposition DAG Bellman backups weight each chance
    /// outcome strictly by this parent edge's own transition count, never by `child.visits`.
    children: Vec<(u64, u32, u32)>,
}

impl ActionEdge {
    #[inline]
    fn q_mean(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.q_sum / (self.visits as f64)
        }
    }

    #[inline]
    fn q_val(&self, backup_mode: u8) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            match backup_mode {
                1 | 4 => self.q_bellman,
                2 => 0.5 * self.q_bellman + 0.5 * self.q_mean(),
                3 => self.q_bellman + 100.0 / (self.visits as f64).sqrt(),
                _ => self.q_mean(),
            }
        }
    }
}

#[derive(Clone)]
struct DecisionNode {
    board: FastBoard,
    hold: Option<u8>,
    rem_g: u16,
    depth: u8,
    /// Queue cursor in the determinized world `world.q` corresponding to `vis_q[0]`.
    qc: u8,
    /// Observable piece window at this decision node (`vis_q[0] = cur`, `vis_q[1..vis_len] = next previews`).
    vis_q: [u8; 6],
    vis_len: u8,
    /// Legal 7-bag remaining piece mask *after* `vis_q[..vis_len]`.
    bag_mask: u8,
    /// Cached `depth * step_cost_cp + V_bc4(node)` in centipieces.
    leaf_val: f64,
    v_bellman: f64,
    visits: u32,
    v_sum: f64,
    actions: Vec<ActionEdge>,
    /// Parent edges `(parent_node_id, parent_action_idx)` pointing to this node in the acyclic DAG.
    parents: Vec<(u32, u16)>,
    expanded: bool,
}

#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct TransKey {
    board: FastBoard,
    hold: Option<u8>,
    rem_g: u16,
    vis_q: [u8; 6],
    vis_len: u8,
    bag_mask: u8,
    depth: u8,
}

#[derive(Clone, Copy)]
struct SampledWorld {
    q: [u8; 40],
    bags: [u8; 40],
    holes: [u16; 48],
}

#[derive(Clone, Copy)]
struct EndgameNode {
    board: FastBoard,
    hold: Option<u8>,
    rem_g: u16,
    q_off: u8,
    first_idx: u16,
    score: f64,
}

/// Reusable Single-Observer IS-MCTS searcher.
pub struct IsMctsSearcher {
    shtb: ShapeTable,
    fst: FastShapeTable,
    pgen: FastPlaceGen,
    ff: FinesseFinder,
    bag_tracker: BagTracker,
    nodes: Vec<DecisionNode>,
    trans_map: FxHashMap<TransKey, u32>,
    tmp_places: Vec<FastPlace>,
}

impl IsMctsSearcher {
    pub fn new(shapes: ShapeTable) -> Self {
        let fst = FastShapeTable::from_shape_table(&shapes);
        Self {
            shtb: shapes,
            fst,
            pgen: FastPlaceGen::new(),
            ff: FinesseFinder::new(),
            bag_tracker: BagTracker::new(),
            nodes: Vec::with_capacity(4096),
            trans_map: FxHashMap::default(),
            tmp_places: Vec::with_capacity(128),
        }
    }

    pub fn new_srs() -> Self {
        Self::new(crate::shape::srs())
    }

    pub fn reset_bag(&mut self) {
        self.bag_tracker = BagTracker::new();
    }

    #[inline]
    fn next_rng(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s
    }

    #[inline]
    fn draw_from_bag(bag_mask: &mut u8, rng: &mut u64) -> u8 {
        if *bag_mask == 0 {
            *bag_mask = 0x7F;
        }
        let mut bits = [0u8; 7];
        let mut n = 0usize;
        for p in 0..7u8 {
            if (*bag_mask & (1u8 << p)) != 0 {
                bits[n] = p;
                n += 1;
            }
        }
        let r = ((Self::next_rng(rng) >> 32) as usize) % n;
        let pick = bits[r];
        *bag_mask &= !(1u8 << pick);
        if *bag_mask == 0 {
            *bag_mask = 0x7F;
        }
        pick
    }

    #[inline]
    fn bottom_hole_col(b: &FastBoard) -> Option<u16> {
        if b.garbage_rows == 0 {
            return None;
        }
        let inv = (!b.data[0]) & 0x3FF;
        if inv.count_ones() == 1 {
            Some(inv.trailing_zeros() as u16)
        } else {
            None
        }
    }

    /// Exact Jstris Cheese Race garbage spawn from a sampled hole stream.
    /// Returns `Some((num_spawned, packed_holes_key))`, or `None` on topout.
    #[inline]
    fn spawn_cheese_from_stream(
        b: &mut FastBoard,
        rem_g: u16,
        lines_cleared: u8,
        steady_g4: bool,
        holes: &[u16],
        hole_cursor: &mut usize,
    ) -> Option<(usize, u64)> {
        let target_g: usize = if steady_g4 || lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 {
            return Some((0, 0));
        }
        let old_rows = b.rows as usize;
        if old_rows + num >= 20 || *hole_cursor + num > holes.len() {
            return None;
        }
        for r in (0..old_rows).rev() {
            b.data[r + num] = b.data[r];
        }
        let mut key: u64 = 0;
        for k in 0..num {
            let hole = holes[*hole_cursor + k];
            let dst_r = num - 1 - k;
            b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
            key = (key << 4) | ((hole as u64) + 1);
        }
        *hole_cursor += num;
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        Some((num, key))
    }

    /// Legal observable `avail_mask` derived strictly from this node's own `vis_q[..vis_len]` + `bag_mask`.
    #[inline]
    fn avail_mask_from_vis(hold: Option<u8>, vis_q: &[u8], bag_mask: u8) -> u8 {
        let mut mask = 0u8;
        if let Some(h) = hold {
            mask |= 1u8 << h;
        }
        for &p in vis_q {
            mask |= 1u8 << p;
        }
        if vis_q.len() < 3 {
            mask |= bag_mask;
        }
        if mask == 0 {
            0x7F
        } else {
            mask
        }
    }

    #[inline]
    fn eval_leaf_full(
        sm: Option<&SpatialValueModel>,
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        depth: usize,
        step_cost_cp: f64,
        root_ref: f64,
        topout_penalty: f64,
    ) -> f64 {
        if rem_g == 0 && board.rows < 20 {
            return (depth as f64) * step_cost_cp + (board.rows as f64) * 5.0;
        }
        if board.rows >= 20 {
            return root_ref + topout_penalty;
        }
        if let Some(m) = sm {
            let v = m.evaluate_board_with_bag(board, rem_g, in_hand, avail_mask);
            if v >= 4_000_000 {
                return root_ref + topout_penalty;
            }
            if v <= -9_000_000 {
                return (depth as f64) * step_cost_cp + (board.rows as f64) * 5.0;
            }
            (depth as f64) * step_cost_cp + (v as f64)
        } else {
            (depth as f64) * step_cost_cp + (rem_g as f64) * 185.0 + (board.rows as f64) * 20.0
        }
    }

    #[inline]
    fn eval_prior_linear(
        sm: Option<&SpatialValueModel>,
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
    ) -> f64 {
        if rem_g == 0 && board.rows < 20 {
            return -10_000.0 + (board.rows as f64) * 5.0;
        }
        if board.rows >= 20 {
            return 50_000.0;
        }
        if let Some(m) = sm {
            let v = m.evaluate_linear_board_with_bag_no_pe(board, rem_g, in_hand, avail_mask);
            if v >= 4_000_000 {
                return 50_000.0;
            }
            if v <= -9_000_000 {
                return -10_000.0 + (board.rows as f64) * 5.0;
            }
            v as f64
        } else {
            (rem_g as f64) * 185.0 + (board.rows as f64) * 20.0
        }
    }

    /// Sample one legal determinization consistent with `(root_q[..q_len], bag_mask, init_prev_hole)`.
    fn sample_world(
        root_q: &[u8],
        bag_mask: u8,
        init_prev_hole: Option<u16>,
        sim_idx: usize,
        base_seed: u64,
    ) -> SampledWorld {
        let mut w = SampledWorld {
            q: [0u8; 40],
            bags: [0x7Fu8; 40],
            holes: [0u16; 48],
        };
        let q_len = root_q.len().min(6);
        for i in 0..q_len {
            w.q[i] = root_q[i];
            w.bags[i] = bag_mask;
        }

        let mut rng = base_seed
            ^ ((sim_idx as u64).wrapping_add(1)).wrapping_mul(0x9E3779B97F4A7C15)
            ^ 0xBF58476D1CE4E5B9;

        let mut bag_bits = [0u8; 7];
        let mut bag_n = 0usize;
        for p in 0..7u8 {
            if (bag_mask & (1u8 << p)) != 0 {
                bag_bits[bag_n] = p;
                bag_n += 1;
            }
        }
        let mut bag = bag_mask;
        if q_len < 40 {
            if bag_n > 0 {
                let p0 = bag_bits[sim_idx % bag_n];
                w.q[q_len] = p0;
                bag &= !(1u8 << p0);
                if bag == 0 {
                    bag = 0x7F;
                }
                w.bags[q_len] = bag;
                for step in (q_len + 1)..40 {
                    w.q[step] = Self::draw_from_bag(&mut bag, &mut rng);
                    w.bags[step] = bag;
                }
            } else {
                for step in q_len..40 {
                    w.q[step] = Self::draw_from_bag(&mut bag, &mut rng);
                    w.bags[step] = bag;
                }
            }
        }

        let mut h_rng = rng ^ 0xD6E8FEB86659FD93;
        let mut prev_h = match init_prev_hole {
            Some(ph) => ph,
            None => ((Self::next_rng(&mut h_rng) >> 32) as u16) % 10,
        };
        for k in 0..48 {
            let off = if k == 0 && init_prev_hole.is_some() {
                ((sim_idx % 9) as u16) + 1
            } else {
                (((Self::next_rng(&mut h_rng) >> 32) as u16) % 9) + 1
            };
            prev_h = (prev_h + off) % 10;
            w.holes[k] = prev_h;
        }
        w
    }

    /// Deterministic Legal `p=5` endgame exact-clear solver when all remaining garbage is
    /// already on the board (`rem_g <= 9 && rem_g == root_board.garbage_rows`).
    /// Uses ONLY the currently observable `root_q[..q_len]` + `hold0` (zero future peeking).
    /// Returns `Some((best_fp, exact_score))` if a guaranteed exact clear (`rem_g == 0`) is found.
    fn solve_deterministic_endgame(
        &mut self,
        root_board: &FastBoard,
        hold0: Option<u8>,
        root_q: &[u8],
        bag_mask: u8,
        rem_g0: u16,
        cfg: &IsMctsConfig,
    ) -> Option<(FastPlace, i64)> {
        let q_len = root_q.len();
        if q_len == 0 || rem_g0 == 0 || rem_g0 > 9 || rem_g0 != root_board.garbage_rows as u16 {
            return None;
        }

        let sm = cfg.spatial_model.as_deref();
        let cur0 = Some(root_q[0]);
        let (c_nh0, c_h0) = if hold0.is_some() {
            (cur0, hold0)
        } else {
            (cur0, root_q.get(1).copied())
        };

        let mut root_places = Vec::with_capacity(128);
        self.pgen
            .generate_for_state(&self.fst, root_board, c_nh0, c_h0, &mut root_places);
        if root_places.is_empty() {
            return None;
        }

        let mut cur_beam: Vec<EndgameNode> = Vec::with_capacity(128);
        let mut best_clear: Option<(usize, usize, u8)> = None; // (plies, first_idx, rows)

        for (idx, &fp) in root_places.iter().enumerate() {
            let mut b2 = *root_board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem2 = rem_g0.saturating_sub(ds as u16);
            if rem2 == 0 && b2.rows < 20 {
                let cand = (1usize, idx, b2.rows);
                if best_clear.map_or(true, |bc| (cand.0, cand.2) < (bc.0, bc.2)) {
                    best_clear = Some(cand);
                }
                continue;
            }
            if b2.rows >= 20 {
                continue;
            }
            let (nh, nq) = if hold0.is_some() {
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
            let vis_slice = &root_q[nq.min(q_len)..];
            let ih = nh.or_else(|| vis_slice.first().copied());
            let av = Self::avail_mask_from_vis(nh, vis_slice, bag_mask);
            let sc = Self::eval_prior_linear(sm, &b2, rem2, ih, av);
            cur_beam.push(EndgameNode {
                board: b2,
                hold: nh,
                rem_g: rem2,
                q_off: nq as u8,
                first_idx: idx as u16,
                score: sc,
            });
        }

        if let Some((plies, fidx, rows)) = best_clear {
            return Some((root_places[fidx], (plies as i64) * 100 + (rows as i64) * 5));
        }

        let max_plies = q_len + usize::from(hold0.is_some());
        let width = 64usize;
        cur_beam.sort_by(|a, b| a.score.total_cmp(&b.score));
        if cur_beam.len() > width {
            cur_beam.truncate(width);
        }

        let mut nxt_beam: Vec<EndgameNode> = Vec::with_capacity(512);
        let mut seen: FxHashMap<(FastBoard, Option<u8>, u8), ()> = FxHashMap::default();

        for ply in 2..=max_plies {
            nxt_beam.clear();
            seen.clear();
            for &node in &cur_beam {
                let qo = node.q_off as usize;
                let cur = root_q.get(qo).copied();
                let (c_nh, c_h) = if node.hold.is_some() {
                    (cur, node.hold)
                } else {
                    (cur, root_q.get(qo + 1).copied())
                };
                if c_nh.is_none() && c_h.is_none() {
                    continue;
                }
                self.pgen
                    .generate_for_state(&self.fst, &node.board, c_nh, c_h, &mut self.tmp_places);
                for &fp in &self.tmp_places {
                    let mut b2 = node.board;
                    let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                    let rem2 = node.rem_g.saturating_sub(ds as u16);
                    if rem2 == 0 && b2.rows < 20 {
                        let cand = (ply, node.first_idx as usize, b2.rows);
                        if best_clear.map_or(true, |bc| (cand.0, cand.2) < (bc.0, bc.2)) {
                            best_clear = Some(cand);
                        }
                        continue;
                    }
                    if b2.rows >= 20 {
                        continue;
                    }
                    // Optimistic pruning: ds <= 1 per piece, so if remaining plies < rem2, cannot clear.
                    if max_plies - ply < rem2 as usize {
                        continue;
                    }
                    let (nh, nq) = if node.hold.is_some() {
                        if fp.did_hold {
                            (cur, if cur.is_some() { (qo + 1).min(q_len) } else { qo })
                        } else {
                            (node.hold, (qo + 1).min(q_len))
                        }
                    } else if fp.did_hold {
                        (cur, (qo + 2).min(q_len))
                    } else {
                        (None, (qo + 1).min(q_len))
                    };
                    if seen.insert((b2, nh, nq as u8), ()).is_some() {
                        continue;
                    }
                    let vis_slice = &root_q[nq.min(q_len)..];
                    let ih = nh.or_else(|| vis_slice.first().copied());
                    let av = Self::avail_mask_from_vis(nh, vis_slice, bag_mask);
                    let sc = Self::eval_prior_linear(sm, &b2, rem2, ih, av);
                    nxt_beam.push(EndgameNode {
                        board: b2,
                        hold: nh,
                        rem_g: rem2,
                        q_off: nq as u8,
                        first_idx: node.first_idx,
                        score: sc,
                    });
                }
            }
            if let Some((plies, fidx, rows)) = best_clear {
                return Some((root_places[fidx], (plies as i64) * 100 + (rows as i64) * 5));
            }
            if nxt_beam.is_empty() {
                break;
            }
            nxt_beam.sort_by(|a, b| a.score.total_cmp(&b.score));
            if nxt_beam.len() > width {
                nxt_beam.truncate(width);
            }
            std::mem::swap(&mut cur_beam, &mut nxt_beam);
        }
        None
    }

    /// Expand a decision node using ONLY this node's own observable `vis_q[..vis_len]` and `bag_mask`.
    fn expand_node(&mut self, node_idx: usize, max_actions: usize, cfg: &IsMctsConfig) {
        let (board, hold, rem_g, vis_q, vis_len, bag_mask) = {
            let n = &self.nodes[node_idx];
            if n.expanded {
                return;
            }
            (
                n.board,
                n.hold,
                n.rem_g,
                n.vis_q,
                n.vis_len as usize,
                n.bag_mask,
            )
        };

        let cur_piece = if vis_len > 0 { Some(vis_q[0]) } else { None };
        let next1_piece = if vis_len > 1 { Some(vis_q[1]) } else { None };
        let (c_nh, c_h) = if hold.is_some() {
            (cur_piece, hold)
        } else {
            (cur_piece, next1_piece)
        };

        self.pgen
            .generate_for_state(&self.fst, &board, c_nh, c_h, &mut self.tmp_places);

        if self.tmp_places.is_empty() {
            self.nodes[node_idx].expanded = true;
            return;
        }

        let sm = cfg
            .policy_prior
            .as_deref()
            .or(cfg.spatial_model.as_deref());
        let mut scored: Vec<(FastPlace, f64, bool)> = Vec::with_capacity(self.tmp_places.len());
        for &fp in &self.tmp_places {
            let mut b2 = board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem2 = rem_g.saturating_sub(ds as u16);

            let (nh, consumed) = if hold.is_some() {
                if fp.did_hold {
                    (cur_piece, usize::from(cur_piece.is_some()))
                } else {
                    (hold, 1.min(vis_len))
                }
            } else if fp.did_hold {
                (cur_piece, 2.min(vis_len))
            } else {
                (None, 1.min(vis_len))
            };

            // Strictly use the remaining already-observable slice of `node.vis_q` (zero future peeking!)
            let rem_vis = &vis_q[consumed.min(vis_len)..vis_len];
            let in_hand = nh.or_else(|| rem_vis.first().copied());
            let avail = Self::avail_mask_from_vis(nh, rem_vis, bag_mask);
            let s1 = cfg.step_cost_cp + Self::eval_prior_linear(sm, &b2, rem2, in_hand, avail);
            scored.push((fp, s1, ds > 0));
        }

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Keep top `max_actions`, preserving at least one `ds > 0` and one `ds == 0` if present.
        let keep = max_actions.max(2).min(scored.len());
        let mut selected: Vec<(FastPlace, f64)> = Vec::with_capacity(keep + 1);
        let mut has_ds = false;
        let mut has_0ds = false;
        for &(fp, sc, is_ds) in scored.iter().take(keep) {
            if is_ds {
                has_ds = true;
            } else {
                has_0ds = true;
            }
            selected.push((fp, sc));
        }
        if !has_ds {
            if let Some(&(fp, sc, _)) = scored.iter().skip(keep).find(|x| x.2) {
                selected.push((fp, sc));
            }
        }
        if !has_0ds {
            if let Some(&(fp, sc, _)) = scored.iter().skip(keep).find(|x| !x.2) {
                selected.push((fp, sc));
            }
        }

        let min_s1 = selected
            .iter()
            .map(|x| x.1)
            .fold(f64::INFINITY, f64::min);
        let temp = cfg.prior_temp.max(10.0);
        let mut weights = Vec::with_capacity(selected.len());
        let mut sum_w = 0.0f64;
        for &(_, sc) in &selected {
            let z = (-(sc - min_s1) / temp).clamp(-30.0, 0.0);
            let w = z.exp();
            weights.push(w);
            sum_w += w;
        }
        let inv_sum = if sum_w > 0.0 {
            1.0 / sum_w
        } else {
            1.0 / (selected.len() as f64)
        };

        let mut edges = Vec::with_capacity(selected.len());
        for (i, &(fp, _)) in selected.iter().enumerate() {
            edges.push(ActionEdge {
                fp,
                prior: weights[i] * inv_sum,
                visits: 0,
                q_sum: 0.0,
                q_bellman: 0.0,
                term_visits: 0,
                term_q_sum: 0.0,
                children: Vec::new(),
            });
        }

        let node = &mut self.nodes[node_idx];
        node.actions = edges;
        node.expanded = true;
    }

    /// Select action index at decision node `node_idx` via PUCT (lower Q is better).
    /// Strictly uses ONLY `self.nodes[node_idx]` statistics — zero Strategy Fusion!
    #[inline]
    fn select_puct(&self, node_idx: usize, cfg: &IsMctsConfig) -> usize {
        let node = &self.nodes[node_idx];
        let parent_n = (node.visits.max(1)) as f64;
        let sqrt_n = parent_n.sqrt();

        let mut min_visited_q = f64::INFINITY;
        let mut max_visited_q = f64::NEG_INFINITY;
        for e in &node.actions {
            if e.visits > 0 {
                let q = e.q_val(cfg.backup_mode);
                if q < min_visited_q {
                    min_visited_q = q;
                }
                if q > max_visited_q {
                    max_visited_q = q;
                }
            }
        }
        let q_ref = if min_visited_q.is_finite() {
            min_visited_q
        } else {
            node.leaf_val
        };
        let fpu_q = q_ref + cfg.fpu_penalty;
        let scale = cfg
            .q_scale
            .max(if max_visited_q.is_finite() && min_visited_q.is_finite() {
                (max_visited_q - min_visited_q).max(50.0)
            } else {
                cfg.q_scale
            });

        let mut best_idx = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for (i, e) in node.actions.iter().enumerate() {
            let q = if e.visits > 0 {
                e.q_val(cfg.backup_mode)
            } else {
                fpu_q
            };
            let exploit = -(q - q_ref) / scale;
            let explore = cfg.c_puct * e.prior * sqrt_n / (1.0 + (e.visits as f64));
            let score = exploit + explore;
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Run Legal `p=5` SO-ISMCTS from `snapshot` with `ctx`.
    pub fn search_with_ctx(
        &mut self,
        snapshot: &Snapshot,
        ctx: GameContext,
        cfg: &IsMctsConfig,
    ) -> IsMctsResult {
        // Strictly truncate to Legal p=5 (`cur` + 5 previews = 6 pieces max).
        let q_len = snapshot.queue.len().min(6);
        let mut root_vis_q = [0u8; 6];
        for i in 0..q_len {
            root_vis_q[i] = color_to_idx(snapshot.queue[i]) as u8;
        }
        let root_q = &root_vis_q[..q_len];

        let g_rows = ctx
            .garbage_rows
            .unwrap_or_else(|| FastBoard::infer_garbage_rows(&snapshot.matrix));
        let rem_g0 = ctx.remaining_garbage.unwrap_or(g_rows);
        let root_board = FastBoard::from_basic_matrix(&snapshot.matrix, g_rows);
        let hold0 = snapshot.hold.map(|c| color_to_idx(c) as u8);
        let bag_mask = ctx
            .next_bag_mask
            .unwrap_or_else(|| self.bag_tracker.observe(snapshot.hold, &snapshot.queue[..q_len]));

        // Optional deterministic exact-clear endgame solver when all remaining garbage is visible.
        if cfg.use_endgame_solver {
            if let Some((eg_fp, eg_sc)) =
                self.solve_deterministic_endgame(&root_board, hold0, root_q, bag_mask, rem_g0, cfg)
            {
                let inputs = self.fp_to_inputs(&snapshot.matrix, eg_fp);
                return IsMctsResult {
                    inputs,
                    best_q: eg_sc,
                    best_visits: 1,
                    total_simulations: 0,
                    nodes_expanded: 1,
                    mlp_evals: 0,
                    avg_sim_depth: 0.0,
                    max_sim_depth: 0,
                    post_spawn_avg_depth: 0.0,
                    spawn_sims: 0,
                    trans_hits: 0,
                    used_endgame_solver: true,
                    root_candidates: vec![(eg_fp, 1, eg_sc as f64, 1.0)],
                };
            }
        }

        // Deterministic CRN seed derived purely from observable root state.
        let mut base_seed: u64 = 0x517cc1b727220a95
            ^ ((rem_g0 as u64) << 16)
            ^ ((root_board.rows as u64) << 8)
            ^ (bag_mask as u64);
        for r in 0..(root_board.rows as usize) {
            base_seed = base_seed
                .rotate_left(7)
                .wrapping_mul(0x9E3779B97F4A7C15)
                ^ (root_board.data[r] as u64);
        }
        for &p in root_q {
            base_seed = base_seed.rotate_left(5) ^ (p as u64);
        }

        let sm = cfg.spatial_model.as_deref();
        let avail0 = Self::avail_mask_from_vis(hold0, root_q, bag_mask);
        let in_hand0 = hold0.or_else(|| root_q.first().copied());
        let root_ref = Self::eval_leaf_full(
            sm,
            &root_board,
            rem_g0,
            in_hand0,
            avail0,
            0,
            cfg.step_cost_cp,
            20_000.0,
            cfg.topout_penalty,
        );

        self.nodes.clear();
        if cfg.use_transposition {
            self.trans_map.clear();
        }
        self.nodes.push(DecisionNode {
            board: root_board,
            hold: hold0,
            rem_g: rem_g0,
            depth: 0,
            qc: 0,
            vis_q: root_vis_q,
            vis_len: q_len as u8,
            bag_mask,
            leaf_val: root_ref,
            v_bellman: root_ref,
            visits: 0,
            v_sum: 0.0,
            actions: Vec::new(),
            parents: Vec::new(),
            expanded: false,
        });
        self.expand_node(0, cfg.root_max_actions, cfg);

        let mut mlp_evals = 1usize;
        if self.nodes[0].actions.is_empty() {
            return IsMctsResult {
                inputs: Vec::new(),
                best_q: (root_ref + cfg.topout_penalty) as i64,
                best_visits: 0,
                total_simulations: 0,
                nodes_expanded: 1,
                mlp_evals,
                avg_sim_depth: 0.0,
                max_sim_depth: 0,
                post_spawn_avg_depth: 0.0,
                spawn_sims: 0,
                trans_hits: 0,
                used_endgame_solver: false,
                root_candidates: Vec::new(),
            };
        }

        let init_prev_hole = Self::bottom_hole_col(&root_board);
        let sims = cfg.simulations.max(1);
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(cfg.max_depth + 2);
        let mut sim_depth_sum = 0usize;
        let mut max_sim_depth = 0usize;
        let mut post_spawn_depth_sum = 0usize;
        let mut spawn_sims = 0usize;
        let mut trans_hits = 0usize;

        for sim in 0..sims {
            let world = Self::sample_world(root_q, bag_mask, init_prev_hole, sim, base_seed);
            let mut sim_p_rng = base_seed
                ^ ((sim as u64).wrapping_add(1)).wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0xA24BAED4963EE407;
            let mut sim_h_rng = sim_p_rng ^ 0xD6E8FEB86659FD93;
            let mut node_idx = 0usize;
            let mut hole_cursor = 0usize;
            let mut reanchored = false;
            let mut terminal_on_last_edge = false;
            let mut first_spawn_depth: Option<usize> = None;
            let mut reached_depth = 0usize;
            path.clear();

            let backed_val: f64 = loop {
                if !self.nodes[node_idx].expanded {
                    self.expand_node(node_idx, cfg.max_actions, cfg);
                }
                if self.nodes[node_idx].actions.is_empty() {
                    let v = root_ref + cfg.topout_penalty;
                    let n = &mut self.nodes[node_idx];
                    n.visits += 1;
                    n.v_sum += v;
                    n.v_bellman = v;
                    break v;
                }
                let depth = self.nodes[node_idx].depth as usize;
                if depth >= cfg.max_depth {
                    let v = self.nodes[node_idx].leaf_val;
                    let n = &mut self.nodes[node_idx];
                    n.visits += 1;
                    n.v_sum += v;
                    break v;
                }

                // 1. PUCT action selection (strictly observable node stats only — ZERO Strategy Fusion!)
                let act_idx = self.select_puct(node_idx, cfg);
                path.push((node_idx, act_idx));

                let (board, hold, rem_g, qc, vis_q, vis_len, bag_cur, fp) = {
                    let n = &self.nodes[node_idx];
                    (
                        n.board,
                        n.hold,
                        n.rem_g,
                        n.qc as usize,
                        n.vis_q,
                        n.vis_len as usize,
                        n.bag_mask,
                        n.actions[act_idx].fp,
                    )
                };
                let cur_piece = if vis_len > 0 { Some(vis_q[0]) } else { None };

                // 2. Deterministic placement
                let mut b2 = board;
                let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                let rem2 = rem_g.saturating_sub(ds as u16);
                let next_depth = depth + 1;
                reached_depth = next_depth;

                if rem2 == 0 && b2.rows < 20 {
                    terminal_on_last_edge = true;
                    break (next_depth as f64) * cfg.step_cost_cp + (b2.rows as f64) * 5.0;
                }
                if b2.rows >= 20 {
                    terminal_on_last_edge = true;
                    break root_ref + cfg.topout_penalty;
                }

                // Determine deterministically whether this transition triggers chance events
                let target_g: usize = if cfg.steady_g4 || lines > 0 { 4 } else { 9 };
                let g_after = b2.garbage_rows as usize;
                let unspawned = (rem2 as usize).saturating_sub(g_after);
                let n_spawn = if rem2 > 0 {
                    target_g.saturating_sub(g_after).min(unspawned)
                } else {
                    0
                };
                if n_spawn > 0 && (b2.rows as usize) + n_spawn >= 20 {
                    terminal_on_last_edge = true;
                    break root_ref + cfg.topout_penalty;
                }

                let (nh, consumed) = if hold.is_some() {
                    if fp.did_hold {
                        (cur_piece, usize::from(cur_piece.is_some()))
                    } else {
                        (hold, 1)
                    }
                } else if fp.did_hold {
                    (cur_piece, 2)
                } else {
                    (None, 1)
                };
                let nqc = qc + consumed;
                let kept = vis_len.saturating_sub(consumed);
                let target_vis = if cfg.split_tail_preview {
                    6usize.min(world.q.len().saturating_sub(nqc))
                } else {
                    let min_need = if nh.is_some() { 1 } else { 2 };
                    kept.max(min_need).min(world.q.len().saturating_sub(nqc))
                };
                let is_chance = n_spawn > 0 || target_vis > kept;

                // 3. Chance-Node Progressive Widening (CPW):
                //    Never merges distinct observable states. When `k >= ceil(c_pw * (N+1)^alpha)`,
                //    re-selects the least-visited already-expanded child (strictly unbiased equal
                //    weighting among i.i.d. expanded outcomes) and re-anchors future sampling legally.
                if is_chance && (cfg.cpw_alpha > 0.0 || (n_spawn == 0 && cfg.cpw_tail_alpha > 0.0)) {
                    let (pw_c, pw_a) = if n_spawn == 0 && cfg.cpw_tail_alpha > 0.0 {
                        (cfg.cpw_tail_c, cfg.cpw_tail_alpha)
                    } else {
                        (cfg.cpw_c, cfg.cpw_alpha)
                    };
                    if pw_a > 0.0 {
                        let k = self.nodes[node_idx].actions[act_idx].children.len();
                        let e_vis = self.nodes[node_idx].actions[act_idx].visits;
                        let k_max = (pw_c * ((e_vis + 1) as f64).powf(pw_a))
                            .ceil()
                            .max(1.0) as usize;
                        if k >= k_max && k > 0 {
                        if n_spawn > 0 && first_spawn_depth.is_none() {
                            first_spawn_depth = Some(next_depth);
                        }
                        // Pick least-visited existing child to maintain equal-weight unbiasedness
                        let start_off = (sim + node_idx + act_idx) % k;
                        let mut best_ci = start_off;
                        let mut best_en = u32::MAX;
                        for off in 0..k {
                            let ci = (start_off + off) % k;
                            let en = self.nodes[node_idx].actions[act_idx].children[ci].2;
                            if en < best_en {
                                best_en = en;
                                best_ci = ci;
                            }
                        }
                        let cid = {
                            let ch = &mut self.nodes[node_idx].actions[act_idx].children[best_ci];
                            ch.2 += 1;
                            ch.1 as usize
                        };
                            node_idx = cid;
                            reanchored = true;
                            continue;
                        }
                    }
                }

                // 4. Chance Event 1: Garbage spawn (stratified `world.holes` or legal on-the-fly)
                let spawn_key = if n_spawn > 0 {
                    if first_spawn_depth.is_none() {
                        first_spawn_depth = Some(next_depth);
                    }
                    if !reanchored {
                        match Self::spawn_cheese_from_stream(
                            &mut b2,
                            rem2,
                            lines,
                            cfg.steady_g4,
                            &world.holes,
                            &mut hole_cursor,
                        ) {
                            Some((_ns, sk)) => sk,
                            None => {
                                terminal_on_last_edge = true;
                                break root_ref + cfg.topout_penalty;
                            }
                        }
                    } else {
                        // Legal on-the-fly sampling strictly respecting `bottom_hole_col(&b2)`
                        let old_rows = b2.rows as usize;
                        let mut prev_h = Self::bottom_hole_col(&b2);
                        for r in (0..old_rows).rev() {
                            b2.data[r + n_spawn] = b2.data[r];
                        }
                        let mut sk: u64 = 0;
                        for kk in 0..n_spawn {
                            let hole = match prev_h {
                                Some(ph) => {
                                    let off = (((Self::next_rng(&mut sim_h_rng) >> 32) as u16) % 9) + 1;
                                    (ph + off) % 10
                                }
                                None => ((Self::next_rng(&mut sim_h_rng) >> 32) as u16) % 10,
                            };
                            prev_h = Some(hole);
                            let dst_r = n_spawn - 1 - kk;
                            b2.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
                            sk = (sk << 4) | ((hole as u64) + 1);
                        }
                        b2.rows = (old_rows + n_spawn) as u8;
                        b2.garbage_rows = (g_after + n_spawn) as u8;
                        sk
                    }
                } else {
                    0u64
                };

                // 5. Chance Event 2: Update observable preview window `vis_q_next` & `bag_next`
                let mut vis_q_next = [0u8; 6];
                for i in 0..kept {
                    vis_q_next[i] = vis_q[consumed + i];
                }
                let mut piece_key: u64 = 0;
                let mut bag_next = bag_cur;
                if target_vis > kept {
                    if !reanchored {
                        for i in kept..target_vis {
                            let p_new = world.q[nqc + i];
                            vis_q_next[i] = p_new;
                            piece_key = (piece_key << 4) | ((p_new as u64) + 1);
                        }
                        bag_next = world.bags[(nqc + target_vis - 1).min(world.bags.len() - 1)];
                    } else {
                        for i in kept..target_vis {
                            let p_new = Self::draw_from_bag(&mut bag_next, &mut sim_p_rng);
                            vis_q_next[i] = p_new;
                            piece_key = (piece_key << 4) | ((p_new as u64) + 1);
                        }
                    }
                }
                let vis_len_next = target_vis;

                let chance_key = (spawn_key << 12) | piece_key;

                // 6. Look up existing child edge `(node_idx, act_idx, chance_key)`
                let existing_slot = self.nodes[node_idx].actions[act_idx]
                    .children
                    .iter()
                    .position(|&(k, _, _)| k == chance_key);

                if let Some(ci) = existing_slot {
                    let cid = {
                        let ch = &mut self.nodes[node_idx].actions[act_idx].children[ci];
                        ch.2 += 1;
                        ch.1 as usize
                    };
                    node_idx = cid;
                } else {
                    // Optional Transposition DAG lookup (acyclic because `depth` is in `TransKey`)
                    let trans_hit = if cfg.use_transposition {
                        let tkey = TransKey {
                            board: b2,
                            hold: nh,
                            rem_g: rem2,
                            vis_q: vis_q_next,
                            vis_len: vis_len_next as u8,
                            bag_mask: bag_next,
                            depth: next_depth as u8,
                        };
                        self.trans_map.get(&tkey).copied().map(|id| id as usize)
                    } else {
                        None
                    };

                    if let Some(cid) = trans_hit {
                        trans_hits += 1;
                        self.nodes[node_idx].actions[act_idx]
                            .children
                            .push((chance_key, cid as u32, 1));
                        let parent_link = (node_idx as u32, act_idx as u16);
                        if !self.nodes[cid].parents.contains(&parent_link) {
                            self.nodes[cid].parents.push(parent_link);
                        }
                        let v_reuse = if cfg.backup_mode > 0 {
                            self.nodes[cid].v_bellman
                        } else if self.nodes[cid].visits > 0 {
                            self.nodes[cid].v_sum / (self.nodes[cid].visits as f64)
                        } else {
                            self.nodes[cid].leaf_val
                        };
                        break v_reuse;
                    }

                    let vis_slice = &vis_q_next[..vis_len_next];
                    let in_hand_next = nh.or_else(|| vis_slice.first().copied());
                    let avail_next = Self::avail_mask_from_vis(nh, vis_slice, bag_next);
                    let leaf_v = Self::eval_leaf_full(
                        sm,
                        &b2,
                        rem2,
                        in_hand_next,
                        avail_next,
                        next_depth,
                        cfg.step_cost_cp,
                        root_ref,
                        cfg.topout_penalty,
                    );
                    mlp_evals += 1;

                    let new_id = self.nodes.len();
                    self.nodes.push(DecisionNode {
                        board: b2,
                        hold: nh,
                        rem_g: rem2,
                        depth: next_depth as u8,
                        qc: nqc as u8,
                        vis_q: vis_q_next,
                        vis_len: vis_len_next as u8,
                        bag_mask: bag_next,
                        leaf_val: leaf_v,
                        v_bellman: leaf_v,
                        visits: 1,
                        v_sum: leaf_v,
                        actions: Vec::new(),
                        parents: vec![(node_idx as u32, act_idx as u16)],
                        expanded: false,
                    });
                    if cfg.use_transposition {
                        let tkey = TransKey {
                            board: b2,
                            hold: nh,
                            rem_g: rem2,
                            vis_q: vis_q_next,
                            vis_len: vis_len_next as u8,
                            bag_mask: bag_next,
                            depth: next_depth as u8,
                        };
                        self.trans_map.insert(tkey, new_id as u32);
                    }
                    self.nodes[node_idx].actions[act_idx]
                        .children
                        .push((chance_key, new_id as u32, 1));
                    break leaf_v;
                }
            };

            sim_depth_sum += reached_depth;
            if reached_depth > max_sim_depth {
                max_sim_depth = reached_depth;
            }
            if let Some(sd) = first_spawn_depth {
                spawn_sims += 1;
                post_spawn_depth_sum += reached_depth.saturating_sub(sd);
            }

            // 7. Backup along visited `(node_idx, act_idx)` path + acyclic DAG parent propagation
            if terminal_on_last_edge {
                if let Some(&(last_n, last_a)) = path.last() {
                    let e = &mut self.nodes[last_n].actions[last_a];
                    e.term_visits += 1;
                    e.term_q_sum += backed_val;
                }
            }
            let mut changed_nodes: Vec<usize> = Vec::new();
            for &(n_id, a_id) in path.iter().rev() {
                let old_v = self.nodes[n_id].v_bellman;
                {
                    let n = &mut self.nodes[n_id];
                    n.visits += 1;
                    n.v_sum += backed_val;
                    let e = &mut n.actions[a_id];
                    e.visits += 1;
                    e.q_sum += backed_val;
                }
                if cfg.backup_mode > 0 {
                    self.recompute_edge_and_node_bellman(n_id, a_id, cfg.backup_mode, backed_val);
                    if cfg.dag_propagate_parents
                        && cfg.use_transposition
                        && self.nodes[n_id].parents.len() > 1
                        && (self.nodes[n_id].v_bellman - old_v).abs() > 1e-9
                    {
                        changed_nodes.push(n_id);
                    }
                }
            }
            if !changed_nodes.is_empty() {
                self.propagate_bellman_to_all_parents(&changed_nodes, cfg.backup_mode);
            }
        }

        // Select best root action
        let root = &self.nodes[0];
        let bm = cfg.backup_mode;
        let mut root_candidates: Vec<(FastPlace, u32, f64, f64)> = root
            .actions
            .iter()
            .map(|e| (e.fp, e.visits, e.q_val(bm), e.prior))
            .collect();

        let best_edge = if cfg.select_by_visits {
            root.actions
                .iter()
                .max_by(|a, b| {
                    a.visits
                        .cmp(&b.visits)
                        .then_with(|| b.q_val(bm).total_cmp(&a.q_val(bm)))
                })
                .unwrap()
        } else {
            let max_v = root.actions.iter().map(|e| e.visits).max().unwrap_or(0);
            let min_v_req = (max_v / 4).max(1);
            root.actions
                .iter()
                .filter(|e| e.visits >= min_v_req)
                .min_by(|a, b| a.q_val(bm).total_cmp(&b.q_val(bm)))
                .unwrap_or_else(|| &root.actions[0])
        };

        let best_fp = best_edge.fp;
        let best_q = best_edge.q_val(bm).round() as i64;
        let best_visits = best_edge.visits;

        root_candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.2.total_cmp(&b.2)));

        let inputs = self.fp_to_inputs(&snapshot.matrix, best_fp);

        IsMctsResult {
            inputs,
            best_q,
            best_visits,
            total_simulations: sims,
            nodes_expanded: self.nodes.len(),
            mlp_evals,
            avg_sim_depth: sim_depth_sum as f64 / sims as f64,
            max_sim_depth,
            post_spawn_avg_depth: if spawn_sims > 0 {
                post_spawn_depth_sum as f64 / spawn_sims as f64
            } else {
                0.0
            },
            spawn_sims,
            trans_hits,
            used_endgame_solver: false,
            root_candidates,
        }
    }

    #[inline]
    fn recompute_edge_and_node_bellman(
        &mut self,
        n_id: usize,
        a_id: usize,
        backup_mode: u8,
        fallback_val: f64,
    ) {
        let mut sum_w = self.nodes[n_id].actions[a_id].term_q_sum;
        let mut cnt_w = self.nodes[n_id].actions[a_id].term_visits as f64;
        let n_ch = self.nodes[n_id].actions[a_id].children.len();
        for ci in 0..n_ch {
            let (_, cid, edge_n) = self.nodes[n_id].actions[a_id].children[ci];
            let cn = &self.nodes[cid as usize];
            let w = edge_n as f64;
            sum_w += w * cn.v_bellman;
            cnt_w += w;
        }
        let q_b = if cnt_w > 0.0 {
            sum_w / cnt_w
        } else {
            fallback_val
        };
        let n = &mut self.nodes[n_id];
        n.actions[a_id].q_bellman = q_b;
        if backup_mode == 4 {
            // Visit-Weighted DAG Value (exact DAG generalization of MCTS mean backup):
            // eliminates Bellman min winner's curse while preserving full DAG parent propagation.
            let mut v_wsum = n.leaf_val;
            let mut v_wcnt = 1.0f64;
            for e in &n.actions {
                if e.visits > 0 {
                    let w = e.visits as f64;
                    v_wsum += w * e.q_bellman;
                    v_wcnt += w;
                }
            }
            n.v_bellman = v_wsum / v_wcnt;
        } else {
            let mut best_v = f64::INFINITY;
            for e in &n.actions {
                if e.visits > 0 {
                    let cand = if backup_mode == 3 {
                        e.q_bellman + 100.0 / (e.visits as f64).sqrt()
                    } else {
                        e.q_bellman
                    };
                    if cand < best_v {
                        best_v = cand;
                    }
                }
            }
            if best_v.is_finite() {
                n.v_bellman = best_v;
            }
        }
    }

    /// Propagate shared-child `v_bellman` updates to ALL ancestors in exact reverse-depth
    /// topological order (acyclic because `TransKey` includes `depth`). Prevents stale parent caches.
    fn propagate_bellman_to_all_parents(&mut self, seeds: &[usize], backup_mode: u8) {
        let mut buckets: [Vec<usize>; 32] = Default::default();
        let mut max_d = 0usize;
        for &cid in seeds {
            let d = (self.nodes[cid].depth as usize).min(31);
            if !buckets[d].contains(&cid) {
                buckets[d].push(cid);
                if d > max_d {
                    max_d = d;
                }
            }
        }
        for d in (1..=max_d).rev() {
            let layer = std::mem::take(&mut buckets[d]);
            for cid in layer {
                let n_par = self.nodes[cid].parents.len();
                for pi in 0..n_par {
                    let (p_id_u32, p_a_u16) = self.nodes[cid].parents[pi];
                    let p_id = p_id_u32 as usize;
                    let p_a = p_a_u16 as usize;
                    let old_v = self.nodes[p_id].v_bellman;
                    let fallback = self.nodes[p_id].leaf_val;
                    self.recompute_edge_and_node_bellman(p_id, p_a, backup_mode, fallback);
                    if (self.nodes[p_id].v_bellman - old_v).abs() > 1e-9 && d > 1 {
                        let pd = (self.nodes[p_id].depth as usize).min(31);
                        if !buckets[pd].contains(&p_id) {
                            buckets[pd].push(p_id);
                        }
                    }
                }
            }
        }
    }

    fn fp_to_inputs(&mut self, matrix: &crate::BasicMatrix, fp: FastPlace) -> Vec<Input> {
        let color: Color = idx_to_color(fp.color_idx as usize);
        let mut inputs = Vec::new();
        if fp.did_hold {
            inputs.push(Input::Hold);
        }
        if let Some(sr) = self.shtb.shape(color) {
            let norm = sr.normalize(fp.tf);
            if let Ok(path) = self.ff.find(matrix, sr, norm) {
                inputs.extend(path);
            }
        }
        inputs.push(Input::HD);
        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BasicMatrix, Color};

    fn make_test_board(hole_cols: &[u16]) -> BasicMatrix {
        let mut m = BasicMatrix::with_cols(10);
        for &h in hole_cols {
            let mut row = [true; 10];
            row[h as usize] = false;
            m.push_row(row.iter().copied());
        }
        m
    }

    #[test]
    fn test_ismcts_zero_future_peeking_beyond_p5() {
        for split_tail in [true, false] {
            let mut s1 = IsMctsSearcher::new_srs();
            let mut s2 = IsMctsSearcher::new_srs();

            let mat = make_test_board(&[2, 7, 4, 1]);
            let snap1 = Snapshot {
                hold: Some(Color::n('I')),
                queue: vec![
                    Color::n('T'),
                    Color::n('L'),
                    Color::n('J'),
                    Color::n('S'),
                    Color::n('Z'),
                    Color::n('O'),
                    // Illegal future beyond p=5:
                    Color::n('I'),
                    Color::n('I'),
                    Color::n('I'),
                ],
                matrix: mat.clone(),
            };
            let snap2 = Snapshot {
                hold: Some(Color::n('I')),
                queue: vec![
                    Color::n('T'),
                    Color::n('L'),
                    Color::n('J'),
                    Color::n('S'),
                    Color::n('Z'),
                    Color::n('O'),
                    // Completely different illegal future beyond p=5:
                    Color::n('O'),
                    Color::n('Z'),
                    Color::n('S'),
                ],
                matrix: mat,
            };
            let ctx = GameContext {
                garbage_rows: Some(4),
                remaining_garbage: Some(50),
                next_bag_mask: Some(0x7F),
            };
            let cfg = IsMctsConfig {
                simulations: 64,
                max_depth: 8,
                split_tail_preview: split_tail,
                ..Default::default()
            };

            let r1 = s1.search_with_ctx(&snap1, ctx, &cfg);
            let r2 = s2.search_with_ctx(&snap2, ctx, &cfg);

            assert_eq!(r1.inputs, r2.inputs, "must not depend on queue[6..]");
            assert_eq!(r1.best_q, r2.best_q, "Q must not depend on queue[6..]");
            assert_eq!(r1.best_visits, r2.best_visits);
            assert_eq!(r1.nodes_expanded, r2.nodes_expanded);
        }
    }

    #[test]
    fn test_ismcts_zero_strategy_fusion_branches_only_after_spawn() {
        let mut searcher = IsMctsSearcher::new_srs();
        let mat = make_test_board(&[5, 2, 8, 0]);
        let snap = Snapshot {
            hold: Some(Color::n('I')),
            queue: vec![
                Color::n('I'),
                Color::n('T'),
                Color::n('L'),
                Color::n('J'),
                Color::n('S'),
                Color::n('Z'),
            ],
            matrix: mat,
        };
        let ctx = GameContext {
            garbage_rows: Some(4),
            remaining_garbage: Some(40),
            next_bag_mask: Some(0x7F),
        };
        let cfg = IsMctsConfig {
            simulations: 120,
            max_depth: 6,
            split_tail_preview: false,
            ..Default::default()
        };

        let res = searcher.search_with_ctx(&snap, ctx, &cfg);
        assert!(!res.inputs.is_empty());

        let root = &searcher.nodes[0];
        let mut saw_deterministic = false;
        let mut saw_spawn_branching = false;
        for e in &root.actions {
            if e.children.is_empty() {
                continue;
            }
            let mut b2 = root.board;
            let orient = &searcher.fst.shapes[e.fp.color_idx as usize].orients[e.fp.tf.2 as usize];
            let (_lines, ds, _) = b2.place_and_sift(orient, e.fp.norm_i, e.fp.norm_j);
            if ds == 0 {
                assert_eq!(e.children.len(), 1);
                assert_eq!(e.children[0].0, 0);
                saw_deterministic = true;
            } else if e.visits >= 2 {
                for &(k, _, _) in &e.children {
                    assert_ne!(k, 0, "spawn-triggering move must key children by spawned hole");
                }
                if e.children.len() >= 2 {
                    saw_spawn_branching = true;
                }
            }
        }
        assert!(saw_deterministic || saw_spawn_branching);
    }

    #[test]
    fn test_ismcts_uses_full_next5_preview_information() {
        let mut s_a = IsMctsSearcher::new_srs();
        let mut s_b = IsMctsSearcher::new_srs();

        let mat = make_test_board(&[0, 1, 2, 3, 4]);
        let snap_a = Snapshot {
            hold: Some(Color::n('O')),
            queue: vec![
                Color::n('O'),
                Color::n('O'),
                Color::n('I'),
                Color::n('I'),
                Color::n('I'),
                Color::n('I'),
            ],
            matrix: mat.clone(),
        };
        let snap_b = Snapshot {
            hold: Some(Color::n('O')),
            queue: vec![
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
            ],
            matrix: mat,
        };
        let ctx = GameContext {
            garbage_rows: Some(5),
            remaining_garbage: Some(5),
            next_bag_mask: Some(0x7F),
        };
        let cfg = IsMctsConfig {
            simulations: 240,
            max_depth: 8,
            split_tail_preview: true,
            ..Default::default()
        };

        let ra = s_a.search_with_ctx(&snap_a, ctx, &cfg);
        let rb = s_b.search_with_ctx(&snap_b, ctx, &cfg);

        assert!(
            ra.best_q < rb.best_q,
            "expected legal next2..next5 I-pieces to yield strictly better Q than O-pieces: {} vs {}",
            ra.best_q,
            rb.best_q
        );
        for n in &s_a.nodes {
            assert_eq!(n.vis_len, 6, "every DecisionNode must maintain full 6-piece Legal p=5 window");
        }
    }

    #[test]
    fn test_ismcts_endgame_solver_and_transposition_dag() {
        let mut s = IsMctsSearcher::new_srs();
        let mat = make_test_board(&[0]);
        let snap = Snapshot {
            hold: Some(Color::n('T')),
            queue: vec![
                Color::n('I'),
                Color::n('L'),
                Color::n('J'),
                Color::n('S'),
                Color::n('Z'),
                Color::n('O'),
            ],
            matrix: mat,
        };
        let ctx = GameContext {
            garbage_rows: Some(1),
            remaining_garbage: Some(1),
            next_bag_mask: Some(0x7F),
        };
        let cfg = IsMctsConfig {
            simulations: 64,
            max_depth: 6,
            use_endgame_solver: true,
            ..Default::default()
        };
        let res = s.search_with_ctx(&snap, ctx, &cfg);
        assert!(res.used_endgame_solver);
        assert!(!res.inputs.is_empty());
    }

    #[test]
    fn test_ismcts_dag_edge_local_chance_weights_and_backup() {
        // Part A: Minimal reproducible synthetic 2-parent DAG with shared child_A.
        // Parent u1 has action 0 with 2 chance outcomes:
        //   - outcome 1 -> child_A (v_bellman = 1000.0), visited 1 time via u1
        //   - outcome 2 -> child_B (v_bellman = 2000.0), visited 1 time via u1
        // Another parent u2 visited child_A 50 times, so child_A.visits == 51.
        // True edge-local Bellman expectation for u1 must be 0.5*1000 + 0.5*2000 = 1500.0
        // (NOT contaminated by child_A.visits == 51, which would give ~1019.2).
        let mut s = IsMctsSearcher::new_srs();
        let dummy_fp = FastPlace {
            color_idx: 0,
            did_hold: false,
            tf: (0, 0, crate::Orientation::R0),
            norm_i: 0,
            norm_j: 0,
        };
        let make_node =
            |v: f64, visits: u32, depth: u8, actions: Vec<ActionEdge>, parents: Vec<(u32, u16)>| {
                DecisionNode {
                    board: FastBoard::empty(),
                    hold: None,
                    rem_g: 10,
                    depth,
                    qc: depth,
                    vis_q: [0; 6],
                    vis_len: 6,
                    bag_mask: 0x7F,
                    leaf_val: v,
                    v_bellman: v,
                    visits,
                    v_sum: v * (visits as f64),
                    actions,
                    parents,
                    expanded: true,
                }
            };

        s.nodes.clear();
        // node 0: u1
        s.nodes.push(make_node(
            1500.0,
            2,
            0,
            vec![ActionEdge {
                fp: dummy_fp,
                prior: 1.0,
                visits: 2,
                q_sum: 3000.0,
                q_bellman: 0.0,
                term_visits: 0,
                term_q_sum: 0.0,
                children: vec![(1, 2, 1), (2, 3, 1)], // edge-local counts: 1 and 1
            }],
            Vec::new(),
        ));
        // node 1: u2 (visits child_A 50 times)
        s.nodes.push(make_node(
            1000.0,
            50,
            0,
            vec![ActionEdge {
                fp: dummy_fp,
                prior: 1.0,
                visits: 50,
                q_sum: 50_000.0,
                q_bellman: 1000.0,
                term_visits: 0,
                term_q_sum: 0.0,
                children: vec![(1, 2, 50)],
            }],
            Vec::new(),
        ));
        // node 2: shared child_A (parents: u1 and u2; total visits = 51!)
        s.nodes.push(make_node(1000.0, 51, 1, Vec::new(), vec![(0, 0), (1, 0)]));
        // node 3: child_B (parent: u1; total visits = 1)
        s.nodes.push(make_node(2000.0, 1, 1, Vec::new(), vec![(0, 0)]));

        s.recompute_edge_and_node_bellman(0, 0, 1, 0.0);
        let q_u1 = s.nodes[0].actions[0].q_bellman;
        assert!(
            (q_u1 - 1500.0).abs() < 1e-9,
            "expected exact edge-local expectation 1500.0, got {} (contaminated by child.visits!)",
            q_u1
        );
        assert!((s.nodes[0].v_bellman - 1500.0).abs() < 1e-9);

        // Counterexample test for stale parent cache:
        // Update shared child_A via u2's search ONLY (never calling backup directly on u1),
        // then call propagate_bellman_to_all_parents(&[2], 1).
        // u1's cached q_bellman and v_bellman MUST immediately update to 0.5*1200 + 0.5*2000 = 1600.0!
        s.nodes[2].v_bellman = 1200.0;
        s.propagate_bellman_to_all_parents(&[2], 1);
        assert!(
            (s.nodes[0].actions[0].q_bellman - 1600.0).abs() < 1e-9,
            "stale parent u1 cache not refreshed: got {}",
            s.nodes[0].actions[0].q_bellman
        );
        assert!((s.nodes[0].v_bellman - 1600.0).abs() < 1e-9);
        assert!((s.nodes[1].actions[0].q_bellman - 1200.0).abs() < 1e-9);
        assert!((s.nodes[1].v_bellman - 1200.0).abs() < 1e-9);

        // Part B: Full live Transposition DAG search conservation invariant check.
        let mat4 = make_test_board(&[3, 7, 1, 5]);
        let snap4 = Snapshot {
            hold: Some(Color::n('T')),
            queue: vec![
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
                Color::n('O'),
            ],
            matrix: mat4,
        };
        let ctx4 = GameContext {
            garbage_rows: Some(4),
            remaining_garbage: Some(40),
            next_bag_mask: Some(0x7F),
        };
        let cfg_t = IsMctsConfig {
            simulations: 400,
            max_depth: 10,
            split_tail_preview: false,
            use_transposition: true,
            dag_propagate_parents: true,
            backup_mode: 1,
            ..Default::default()
        };
        let r_t = s.search_with_ctx(&snap4, ctx4, &cfg_t);
        assert!(r_t.trans_hits > 0, "expected DAG transposition hits");

        for (nid, node) in s.nodes.iter().enumerate() {
            for (aid, e) in node.actions.iter().enumerate() {
                let sum_edge_n: u32 = e.children.iter().map(|&(_, _, en)| en).sum();
                assert_eq!(
                    e.term_visits + sum_edge_n,
                    e.visits,
                    "edge-local conservation violated at node {} action {}",
                    nid,
                    aid
                );
            }
        }
    }

    #[test]
    fn test_cpw_synthetic_chance_tree_unbiased_and_convergent() {
        // Synthetic M=63 equiprobable chance node (modeling 9 holes x 7 previews).
        // Verify that Chance-Node Progressive Widening with least-visited re-selection:
        // 1. Is strictly unbiased (mean across trials == exact E[V] within 3*SE) at N=10, 50, 200, 1000.
        // 2. Converges monotonically (RMSE strictly decreases with N).
        // 3. Achieves much deeper visits per expanded child than unwidened sampling at N=50.
        const M: usize = 63;
        let mut true_v = [0.0f64; M];
        for i in 0..M {
            true_v[i] = 1000.0 + ((i * 37) % M) as f64 * 10.0;
        }
        let exact_mean: f64 = true_v.iter().sum::<f64>() / (M as f64);

        let c_pw = 1.0f64;
        let alpha_pw = 0.5f64;
        let trials = 400usize;
        let budgets = [10usize, 50, 200, 1000];
        let mut prev_rmse = f64::INFINITY;
        let mut mean_visits_per_child_n50 = 0.0f64;

        for &n_sims in &budgets {
            let mut est_sum = 0.0f64;
            let mut sq_err_sum = 0.0f64;
            let mut vpc_sum = 0.0f64;

            for tr in 0..trials {
                let mut rng: u64 = 0x517cc1b727220a95
                    ^ ((tr as u64 + 1).wrapping_mul(0x9E3779B97F4A7C15))
                    ^ ((n_sims as u64).wrapping_mul(0xBF58476D1CE4E5B9));
                // Expanded children: (outcome_id, edge_visits)
                let mut children: Vec<(usize, u32)> = Vec::new();
                for sim in 0..n_sims {
                    let k = children.len();
                    let k_max = (c_pw * ((sim + 1) as f64).powf(alpha_pw))
                        .ceil()
                        .max(1.0) as usize;
                    if k >= k_max && k > 0 {
                        // Least-visited re-selection (exact same rule as ismcts.rs)
                        let start_off = sim % k;
                        let mut best_ci = start_off;
                        let mut best_en = u32::MAX;
                        for off in 0..k {
                            let ci = (start_off + off) % k;
                            if children[ci].1 < best_en {
                                best_en = children[ci].1;
                                best_ci = ci;
                            }
                        }
                        children[best_ci].1 += 1;
                    } else {
                        let o = ((IsMctsSearcher::next_rng(&mut rng) >> 32) as usize) % M;
                        if let Some(slot) = children.iter().position(|&(oid, _)| oid == o) {
                            children[slot].1 += 1;
                        } else {
                            children.push((o, 1));
                        }
                    }
                }
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for &(oid, en) in &children {
                    num += (en as f64) * true_v[oid];
                    den += en as f64;
                }
                let q_hat = num / den;
                est_sum += q_hat;
                sq_err_sum += (q_hat - exact_mean).powi(2);
                vpc_sum += (n_sims as f64) / (children.len() as f64);
            }

            let mean_est = est_sum / (trials as f64);
            let rmse = (sq_err_sum / (trials as f64)).sqrt();
            let se = rmse / (trials as f64).sqrt();
            assert!(
                (mean_est - exact_mean).abs() <= 3.5 * se + 1.0,
                "CPW estimator biased at N={}: mean_est={:.2} vs exact={:.2} (SE={:.2})",
                n_sims,
                mean_est,
                exact_mean,
                se
            );
            assert!(
                rmse < prev_rmse,
                "CPW RMSE did not decrease monotonically at N={}: {:.2} >= {:.2}",
                n_sims,
                rmse,
                prev_rmse
            );
            prev_rmse = rmse;
            if n_sims == 50 {
                mean_visits_per_child_n50 = vpc_sum / (trials as f64);
            }
        }
        assert!(
            mean_visits_per_child_n50 >= 5.0,
            "expected >= 5 visits/child at N=50 on M=63 with CPW, got {:.2}",
            mean_visits_per_child_n50
        );
    }
}
