//! Zero-Strategy-Fusion Single-Observer Information-Set MCTS (SO-ISMCTS / Expecti-MCTS)
//! for Legal `p=5` Jstris 100L Cheese Race.
//!
//! Design guarantees:
//! 1. **Strict Legal `p=5` Observability**: Never reads beyond `snapshot.queue[..6]` (cur + 5 previews)
//!    or `ctx.next_bag_mask`. All future pieces and unspawned garbage holes are sampled legally.
//! 2. **Zero Strategy Fusion**: Every decision node in the tree represents an observable state
//!    reached *before* unspawned garbage holes or unrevealed future pieces occur. All determinized
//!    simulations reaching the same decision node share the exact same PUCT action statistics
//!    `(N, Q, P)` and cannot condition their move on unrevealed future outcomes. Branching into
//!    distinct child decision nodes happens strictly *after* a chance event (garbage spawn or new
//!    active piece draw) becomes observable.
//! 3. **Depth-Consistent Cost-to-Go**: Uses Frozen BC-4 FULL-MLP (`SpatialValueModel::evaluate_board_with_bag`)
//!    for leaf values (`J(v) = depth * 100 + V_bc4(v)`, centipieces, lower is better) and Frozen BC-4
//!    fast linear evaluation for 1-ply action priors `P(u, a)`.

use std::sync::Arc;

use super::{
    beam::{BagTracker, GameContext},
    fast::{color_to_idx, idx_to_color, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
    spatial_val::SpatialValueModel,
    Snapshot,
};
use crate::{finesse::FinesseFinder, shape::ShapeTable, Color, Input};

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
    /// Optional strictly-observable deterministic greedy lookahead plies at leaf expansion
    /// (uses ONLY `root_q[qc..q_len]` and stops immediately before any unknown garbage spawn).
    pub leaf_obs_lookahead: usize,
    /// Per-ply step cost in centipieces added to leaf `V_bc4` (`depth * step_cost_cp + V_bc4`).
    /// Calibrating this to `V_bc4`'s empirical per-move drop (~89 cp) eliminates variable-depth bias.
    pub step_cost_cp: f64,
    /// Frozen BC-4 spatial value model (required for full-MLP leaf evaluation + linear prior).
    pub spatial_model: Option<Arc<SpatialValueModel>>,
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
            backup_mode: 0,
            leaf_obs_lookahead: 0,
            step_cost_cp: 100.0,
            spatial_model: None,
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
    pub root_candidates: Vec<(FastPlace, u32, f64, f64)>, // (place, visits, q_mean, prior)
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
    /// Chance outcome key -> child decision node index.
    /// When the transition is deterministic (no garbage spawn & no newly revealed active piece),
    /// there is at most 1 entry with `chance_key == 0`.
    children: Vec<(u64, u32)>,
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
                1 => self.q_bellman,
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
    qc: u8,
    cur_piece: Option<u8>,
    next1_piece: Option<u8>,
    bag_mask: u8,
    /// Cached `depth * 100 + V_bc4(node)` in centipieces.
    leaf_val: f64,
    v_bellman: f64,
    visits: u32,
    v_sum: f64,
    actions: Vec<ActionEdge>,
    expanded: bool,
}

#[derive(Clone, Copy)]
struct SampledWorld {
    q: [u8; 32],
    bags: [u8; 32],
    holes: [u16; 48],
}

/// Reusable Single-Observer IS-MCTS searcher.
pub struct IsMctsSearcher {
    shtb: ShapeTable,
    fst: FastShapeTable,
    pgen: FastPlaceGen,
    ff: FinesseFinder,
    bag_tracker: BagTracker,
    nodes: Vec<DecisionNode>,
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

    /// Legal observable `avail_mask` at depth `d`: uses ONLY pieces observable at `(hold, qc)`
    /// within the initial `q_len_root` window + `bag_mask` (NEVER peeks at `world_q[>= q_len_root]`
    /// beyond `cur_piece` / `next1_piece`).
    #[inline]
    fn observable_avail_mask(
        hold: Option<u8>,
        qc: usize,
        root_q: &[u8],
        cur_piece: Option<u8>,
        next1_piece: Option<u8>,
        bag_mask: u8,
    ) -> u8 {
        let mut mask = 0u8;
        if let Some(h) = hold {
            mask |= 1u8 << h;
        }
        if let Some(c) = cur_piece {
            mask |= 1u8 << c;
        }
        if let Some(n1) = next1_piece {
            mask |= 1u8 << n1;
        }
        // Any remaining pieces already visible from the root preview window (<= 5 previews)
        let mut cnt = 0usize;
        for i in qc..root_q.len() {
            mask |= 1u8 << root_q[i];
            cnt += 1;
            if cnt >= 6 {
                break;
            }
        }
        if cnt < 3 {
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
            q: [0u8; 32],
            bags: [0x7Fu8; 32],
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

        // Stratified first unknown piece from `bag_mask`
        let mut bag_bits = [0u8; 7];
        let mut bag_n = 0usize;
        for p in 0..7u8 {
            if (bag_mask & (1u8 << p)) != 0 {
                bag_bits[bag_n] = p;
                bag_n += 1;
            }
        }
        let mut bag = bag_mask;
        if q_len < 32 {
            if bag_n > 0 {
                let p0 = bag_bits[sim_idx % bag_n];
                w.q[q_len] = p0;
                bag &= !(1u8 << p0);
                if bag == 0 {
                    bag = 0x7F;
                }
                w.bags[q_len] = bag;
                for step in (q_len + 1)..32 {
                    w.q[step] = Self::draw_from_bag(&mut bag, &mut rng);
                    w.bags[step] = bag;
                }
            } else {
                for step in q_len..32 {
                    w.q[step] = Self::draw_from_bag(&mut bag, &mut rng);
                    w.bags[step] = bag;
                }
            }
        }

        // Stratified first unknown garbage hole offset (1..=9)
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

    /// Expand a decision node: generate legal placements for `(cur_piece, hold)` and compute
    /// 1-ply linear BC-4 priors `P(u, a)` using ONLY currently observable information.
    fn expand_node(
        &mut self,
        node_idx: usize,
        root_q: &[u8],
        max_actions: usize,
        cfg: &IsMctsConfig,
    ) {
        let (board, hold, rem_g, qc, cur_piece, next1_piece, bag_mask) = {
            let n = &self.nodes[node_idx];
            if n.expanded {
                return;
            }
            (
                n.board,
                n.hold,
                n.rem_g,
                n.qc as usize,
                n.cur_piece,
                n.next1_piece,
                n.bag_mask,
            )
        };

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

        let sm = cfg.spatial_model.as_deref();
        let mut scored: Vec<(FastPlace, f64, bool)> = Vec::with_capacity(self.tmp_places.len());
        for &fp in &self.tmp_places {
            let mut b2 = board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem2 = rem_g.saturating_sub(ds as u16);

            let (nh, nqc) = if hold.is_some() {
                if fp.did_hold {
                    (cur_piece, if cur_piece.is_some() { qc + 1 } else { qc })
                } else {
                    (hold, qc + 1)
                }
            } else if fp.did_hold {
                (cur_piece, qc + 2)
            } else {
                (None, qc + 1)
            };

            let next_cur_obs = if nqc < root_q.len() {
                Some(root_q[nqc])
            } else {
                None
            };
            let in_hand = nh.or(next_cur_obs);
            let avail = Self::observable_avail_mask(nh, nqc, root_q, next_cur_obs, None, bag_mask);
            let s1 = 100.0 + Self::eval_prior_linear(sm, &b2, rem2, in_hand, avail);
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

        // Softmax prior `P(u, a) = exp(-(s1 - min_s1) / temp)`
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
    /// Strictly uses ONLY `self.nodes[node_idx]` statistics — zero dependence on hidden future!
    #[inline]
    fn select_puct(&self, node_idx: usize, cfg: &IsMctsConfig) -> usize {
        let node = &self.nodes[node_idx];
        let parent_n = (node.visits.max(1)) as f64;
        let sqrt_n = parent_n.sqrt();

        // Reference Q for normalization: best visited Q or node's own leaf_val.
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
        let mut root_q_arr = [0u8; 6];
        for i in 0..q_len {
            root_q_arr[i] = color_to_idx(snapshot.queue[i]) as u8;
        }
        let root_q = &root_q_arr[..q_len];

        let g_rows = ctx
            .garbage_rows
            .unwrap_or_else(|| FastBoard::infer_garbage_rows(&snapshot.matrix));
        let rem_g0 = ctx.remaining_garbage.unwrap_or(g_rows);
        let root_board = FastBoard::from_basic_matrix(&snapshot.matrix, g_rows);
        let hold0 = snapshot.hold.map(|c| color_to_idx(c) as u8);
        let cur0 = if q_len > 0 { Some(root_q[0]) } else { None };
        let next1_0 = if q_len > 1 { Some(root_q[1]) } else { None };
        let bag_mask = ctx
            .next_bag_mask
            .unwrap_or_else(|| self.bag_tracker.observe(snapshot.hold, &snapshot.queue[..q_len]));

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
        let avail0 = Self::observable_avail_mask(hold0, 0, root_q, cur0, next1_0, bag_mask);
        let in_hand0 = hold0.or(cur0);
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
        self.nodes.push(DecisionNode {
            board: root_board,
            hold: hold0,
            rem_g: rem_g0,
            depth: 0,
            qc: 0,
            cur_piece: cur0,
            next1_piece: next1_0,
            bag_mask,
            leaf_val: root_ref,
            v_bellman: root_ref,
            visits: 0,
            v_sum: 0.0,
            actions: Vec::new(),
            expanded: false,
        });
        self.expand_node(0, root_q, cfg.root_max_actions, cfg);

        let mut mlp_evals = 1usize;
        if self.nodes[0].actions.is_empty() {
            return IsMctsResult {
                inputs: Vec::new(),
                best_q: (root_ref + cfg.topout_penalty) as i64,
                best_visits: 0,
                total_simulations: 0,
                nodes_expanded: 1,
                mlp_evals,
                root_candidates: Vec::new(),
            };
        }

        let init_prev_hole = Self::bottom_hole_col(&root_board);
        let sims = cfg.simulations.max(1);
        let mut path: Vec<(usize, usize)> = Vec::with_capacity(cfg.max_depth + 2);

        for sim in 0..sims {
            let world = Self::sample_world(root_q, bag_mask, init_prev_hole, sim, base_seed);
            let mut node_idx = 0usize;
            let mut hole_cursor = 0usize;
            let mut terminal_on_last_edge = false;
            path.clear();

            let backed_val: f64 = loop {
                if !self.nodes[node_idx].expanded {
                    self.expand_node(node_idx, root_q, cfg.max_actions, cfg);
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

                let (board, hold, rem_g, qc, cur_piece, fp) = {
                    let n = &self.nodes[node_idx];
                    (
                        n.board,
                        n.hold,
                        n.rem_g,
                        n.qc as usize,
                        n.cur_piece,
                        n.actions[act_idx].fp,
                    )
                };

                // 2. Deterministic placement
                let mut b2 = board;
                let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                let (lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                let rem2 = rem_g.saturating_sub(ds as u16);
                let next_depth = depth + 1;

                if rem2 == 0 && b2.rows < 20 {
                    terminal_on_last_edge = true;
                    break (next_depth as f64) * cfg.step_cost_cp + (b2.rows as f64) * 5.0;
                }
                if b2.rows >= 20 {
                    terminal_on_last_edge = true;
                    break root_ref + cfg.topout_penalty;
                }

                // 3. Chance Event 1: Garbage spawn (if triggered)
                let spawn_key = if rem2 > 0 {
                    match Self::spawn_cheese_from_stream(
                        &mut b2,
                        rem2,
                        lines,
                        cfg.steady_g4,
                        &world.holes,
                        &mut hole_cursor,
                    ) {
                        Some((_n_spawn, sk)) => sk,
                        None => {
                            terminal_on_last_edge = true;
                            break root_ref + cfg.topout_penalty;
                        }
                    }
                } else {
                    0u64
                };

                // 4. Advance hold / queue cursor
                let (nh, nqc) = if hold.is_some() {
                    if fp.did_hold {
                        (cur_piece, if cur_piece.is_some() { qc + 1 } else { qc })
                    } else {
                        (hold, qc + 1)
                    }
                } else if fp.did_hold {
                    (cur_piece, qc + 2)
                } else {
                    (None, qc + 1)
                };

                // 5. Chance Event 2: Newly revealed playable piece(s) beyond initial root window
                let cur_next = if nqc < 32 { Some(world.q[nqc]) } else { None };
                let next1_next = if nh.is_none() && nqc + 1 < 32 {
                    Some(world.q[nqc + 1])
                } else if nqc + 1 < q_len {
                    Some(root_q[nqc + 1])
                } else {
                    None
                };
                let bag_next = if nqc < 32 { world.bags[nqc] } else { 0x7F };

                let mut piece_key: u64 = 0;
                if nqc >= q_len {
                    if let Some(cn) = cur_next {
                        piece_key = (cn as u64) + 1;
                    }
                }
                if nh.is_none() && nqc + 1 >= q_len {
                    if let Some(n1) = next1_next {
                        piece_key = (piece_key << 4) | ((n1 as u64) + 1);
                    }
                }
                let chance_key = (spawn_key << 8) | piece_key;

                // 6. Look up or expand child decision node for `(node_idx, act_idx, chance_key)`
                let existing_child = self.nodes[node_idx].actions[act_idx]
                    .children
                    .iter()
                    .find(|&&(k, _)| k == chance_key)
                    .map(|&(_, cid)| cid as usize);

                if let Some(cid) = existing_child {
                    node_idx = cid;
                } else {
                    let leaf_v = if cfg.leaf_obs_lookahead == 0 {
                        let in_hand_next = nh.or(cur_next);
                        let avail_next = Self::observable_avail_mask(
                            nh,
                            nqc,
                            root_q,
                            cur_next,
                            next1_next,
                            bag_next,
                        );
                        Self::eval_leaf_full(
                            sm,
                            &b2,
                            rem2,
                            in_hand_next,
                            avail_next,
                            next_depth,
                            cfg.step_cost_cp,
                            root_ref,
                            cfg.topout_penalty,
                        )
                    } else {
                        // Strictly observable deterministic greedy lookahead:
                        // Uses ONLY `root_q[..q_len]` (never `world.q` or `world.holes`) and stops
                        // immediately before any unknown garbage spawn. Zero Strategy Fusion!
                        let mut cb = b2;
                        let mut ch = nh;
                        let mut crem = rem2;
                        let mut cqc = nqc;
                        let mut cd = next_depth;
                        let mut exact_term: Option<f64> = None;

                        for _ in 0..cfg.leaf_obs_lookahead {
                            let (c_nh, c_h) = if ch.is_some() {
                                if cqc < q_len {
                                    (Some(root_q[cqc]), ch)
                                } else {
                                    break;
                                }
                            } else if cqc + 1 < q_len {
                                (Some(root_q[cqc]), Some(root_q[cqc + 1]))
                            } else {
                                break;
                            };

                            self.pgen.generate_for_state(
                                &self.fst,
                                &cb,
                                c_nh,
                                c_h,
                                &mut self.tmp_places,
                            );
                            if self.tmp_places.is_empty() {
                                break;
                            }

                            let mut best_sc = f64::INFINITY;
                            let mut best_step: Option<(FastBoard, u8, u16, Option<u8>, usize)> =
                                None;
                            for &fp_s in &self.tmp_places {
                                let mut bs = cb;
                                let os = &self.fst.shapes[fp_s.color_idx as usize].orients
                                    [fp_s.tf.2 as usize];
                                let (ls, dss, _) = bs.place_and_sift(os, fp_s.norm_i, fp_s.norm_j);
                                let rs = crem.saturating_sub(dss as u16);
                                let (nhs, nqcs) = if ch.is_some() {
                                    if fp_s.did_hold {
                                        (c_nh, cqc + 1)
                                    } else {
                                        (ch, cqc + 1)
                                    }
                                } else if fp_s.did_hold {
                                    (c_nh, cqc + 2)
                                } else {
                                    (None, cqc + 1)
                                };
                                let next_c_obs = if nqcs < q_len {
                                    Some(root_q[nqcs])
                                } else {
                                    None
                                };
                                let ihs = nhs.or(next_c_obs);
                                let avs = Self::observable_avail_mask(
                                    nhs,
                                    nqcs,
                                    root_q,
                                    next_c_obs,
                                    None,
                                    bag_next,
                                );
                                let scs = Self::eval_prior_linear(sm, &bs, rs, ihs, avs);
                                if scs < best_sc {
                                    best_sc = scs;
                                    best_step = Some((bs, ls, rs, nhs, nqcs));
                                }
                            }

                            let (bs, ls, rs, nhs, nqcs) = match best_step {
                                Some(t) => t,
                                None => break,
                            };
                            if rs == 0 && bs.rows < 20 {
                                exact_term = Some(
                                    ((cd + 1) as f64) * cfg.step_cost_cp + (bs.rows as f64) * 5.0,
                                );
                                break;
                            }
                            if bs.rows >= 20 {
                                break;
                            }
                            // Stop strictly before any unknown garbage spawn!
                            let target_g: usize = if cfg.steady_g4 || ls > 0 { 4 } else { 9 };
                            let g_after = bs.garbage_rows as usize;
                            let unspawned = (rs as usize).saturating_sub(g_after);
                            if target_g.saturating_sub(g_after).min(unspawned) > 0 {
                                break;
                            }
                            cb = bs;
                            ch = nhs;
                            crem = rs;
                            cqc = nqcs;
                            cd += 1;
                        }

                        if let Some(tv) = exact_term {
                            tv
                        } else {
                            let c_cur_obs = if cqc < q_len {
                                Some(root_q[cqc])
                            } else if cqc == nqc {
                                cur_next
                            } else {
                                None
                            };
                            let c_n1_obs = if cqc + 1 < q_len {
                                Some(root_q[cqc + 1])
                            } else if cqc == nqc {
                                next1_next
                            } else {
                                None
                            };
                            let ih_f = ch.or(c_cur_obs);
                            let av_f = Self::observable_avail_mask(
                                ch,
                                cqc,
                                root_q,
                                c_cur_obs,
                                c_n1_obs,
                                bag_next,
                            );
                            Self::eval_leaf_full(
                                sm,
                                &cb,
                                crem,
                                ih_f,
                                av_f,
                                cd,
                                cfg.step_cost_cp,
                                root_ref,
                                cfg.topout_penalty,
                            )
                        }
                    };
                    mlp_evals += 1;

                    let new_id = self.nodes.len();
                    self.nodes.push(DecisionNode {
                        board: b2,
                        hold: nh,
                        rem_g: rem2,
                        depth: next_depth as u8,
                        qc: nqc as u8,
                        cur_piece: cur_next,
                        next1_piece: next1_next,
                        bag_mask: bag_next,
                        leaf_val: leaf_v,
                        v_bellman: leaf_v,
                        visits: 1,
                        v_sum: leaf_v,
                        actions: Vec::new(),
                        expanded: false,
                    });
                    self.nodes[node_idx].actions[act_idx]
                        .children
                        .push((chance_key, new_id as u32));
                    break leaf_v;
                }
            };

            // 7. Backup along visited `(node_idx, act_idx)` path
            if terminal_on_last_edge {
                if let Some(&(last_n, last_a)) = path.last() {
                    let e = &mut self.nodes[last_n].actions[last_a];
                    e.term_visits += 1;
                    e.term_q_sum += backed_val;
                }
            }
            for &(n_id, a_id) in path.iter().rev() {
                {
                    let n = &mut self.nodes[n_id];
                    n.visits += 1;
                    n.v_sum += backed_val;
                    let e = &mut n.actions[a_id];
                    e.visits += 1;
                    e.q_sum += backed_val;
                }
                if cfg.backup_mode > 0 {
                    // Exact expectation over all chance outcomes of `(n_id, a_id)`
                    let mut sum_w = self.nodes[n_id].actions[a_id].term_q_sum;
                    let mut cnt_w = self.nodes[n_id].actions[a_id].term_visits as f64;
                    let n_ch = self.nodes[n_id].actions[a_id].children.len();
                    for ci in 0..n_ch {
                        let cid = self.nodes[n_id].actions[a_id].children[ci].1 as usize;
                        let cn = &self.nodes[cid];
                        let w = cn.visits as f64;
                        sum_w += w * cn.v_bellman;
                        cnt_w += w;
                    }
                    let q_b = if cnt_w > 0.0 {
                        sum_w / cnt_w
                    } else {
                        backed_val
                    };
                    let n = &mut self.nodes[n_id];
                    n.actions[a_id].q_bellman = q_b;
                    let mut best_v = f64::INFINITY;
                    for e in &n.actions {
                        if e.visits > 0 {
                            let cand = if cfg.backup_mode == 3 {
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
            // Lowest Q among visited actions (preferring well-visited actions when sims >= 32)
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
            root_candidates,
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
    use crate::{shape::srs, BasicMatrix, Color};

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
        let mut s1 = IsMctsSearcher::new(srs());
        let mut s2 = IsMctsSearcher::new(srs());

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
            ..Default::default()
        };

        let r1 = s1.search_with_ctx(&snap1, ctx, &cfg);
        let r2 = s2.search_with_ctx(&snap2, ctx, &cfg);

        assert_eq!(r1.inputs, r2.inputs, "chosen move must not depend on queue[6..]");
        assert_eq!(r1.best_q, r2.best_q, "Q value must not depend on queue[6..]");
        assert_eq!(r1.best_visits, r2.best_visits);
        assert_eq!(r1.nodes_expanded, r2.nodes_expanded);
    }

    #[test]
    fn test_ismcts_zero_strategy_fusion_branches_only_after_spawn() {
        let mut searcher = IsMctsSearcher::new(srs());
        // 4 garbage rows, top hole at col 0; clearing 1 line triggers 1 unknown garbage spawn.
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
            ..Default::default()
        };

        let res = searcher.search_with_ctx(&snap, ctx, &cfg);
        assert!(!res.inputs.is_empty());

        // Inspect root actions:
        // - Actions with ds==0 (no spawn, qc=1 < 6) must have <= 1 deterministic child (key == 0).
        // - Actions with ds==1 (triggers spawn) may have multiple distinct children with non-zero spawn_key,
        //   all sharing the single pre-spawn root decision node `0`.
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
                assert_eq!(
                    e.children.len(),
                    1,
                    "0ds move within p=5 window must have strictly 1 deterministic child"
                );
                assert_eq!(e.children[0].0, 0);
                saw_deterministic = true;
            } else if e.visits >= 2 {
                for &(k, _) in &e.children {
                    assert_ne!(k, 0, "spawn-triggering move must key children by spawned hole");
                }
                if e.children.len() >= 2 {
                    saw_spawn_branching = true;
                }
            }
        }
        assert!(saw_deterministic || saw_spawn_branching);
    }
}
