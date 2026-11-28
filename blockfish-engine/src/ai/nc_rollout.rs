//! True Non-Clairvoyant Common-Random-Numbers Rollout PIMC (`nc_rollout.rs`)
//! for Legal `p=5` Jstris 100L Cheese Race.
//!
//! Design guarantees:
//! 1. **Strict Legal `p=5` Observability**: Truncates `snapshot.queue[..6]` and uses `BagTracker` only.
//! 2. **Literally Zero Strategy Fusion**: Inside every CRN world `w`, the rollout policy `pi_0`
//!    at step `t` receives ONLY the currently observable `(board, hold, world_q[qc..qc+6], bag_mask, rem_g)`
//!    and NEVER sees `world_holes` or `world_q[qc+6..]`. Garbage lines spawn strictly *after* `pi_0`
//!    commits to its placement at step `t`.
//! 3. **Zero `9^H` Chance-Branching Explosion**: Compute scales linearly `O(K * W * H)` in horizon `H`.
//! 4. **Zero Variable-Depth `V_bc4` Drift**: All non-terminal rollouts evaluate Frozen BC-4 FULL-MLP
//!    at the exact same fixed horizon `H`.

use std::sync::Arc;

use super::{
    beam::{BagTracker, GameContext},
    fast::{color_to_idx, idx_to_color, FastBoard, FastPlace, FastPlaceGen, FastShapeTable},
    spatial_val::SpatialValueModel,
    Snapshot,
};
use crate::{finesse::FinesseFinder, shape::ShapeTable, Color, Input};

#[derive(Clone, Debug)]
pub struct NcRolloutConfig {
    /// Number of top root candidates evaluated with CRN non-clairvoyant rollouts.
    pub root_candidates: usize,
    /// Number of CRN worlds (multiples of 9 stratify all 9 legal first garbage holes).
    pub worlds: usize,
    /// Number of non-clairvoyant rollout steps after the root move (fixed horizon `H`).
    pub horizon: usize,
    /// Internal lookahead plies of the non-clairvoyant rollout policy `pi_0` (`1`, `2`, `3`, or `4`).
    pub policy_plies: usize,
    /// Beam width of the non-clairvoyant rollout policy `pi_0` (ignored when `policy_plies == 1`).
    pub policy_width: usize,
    /// If true, use `steady_g4` (`target_g = 4`) garbage spawning.
    pub steady_g4: bool,
    /// If true, use FULL-MLP at the end of `pi_0`'s internal beam (slower, stronger `pi_0`);
    /// if false, `pi_0` uses ultra-fast `evaluate_linear_board_with_bag_no_pe` and FULL-MLP is
    /// called once at the end of each `H`-step rollout.
    pub policy_full_mlp: bool,
    /// Topout penalty (centipieces above root reference).
    pub topout_penalty: f64,
    /// Frozen BC-4 spatial value model.
    pub spatial_model: Option<Arc<SpatialValueModel>>,
}

impl Default for NcRolloutConfig {
    fn default() -> Self {
        Self {
            root_candidates: 12,
            worlds: 18,
            horizon: 10,
            policy_plies: 2,
            policy_width: 6,
            steady_g4: true,
            policy_full_mlp: false,
            topout_penalty: 5000.0,
            spatial_model: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NcRolloutResult {
    pub inputs: Vec<Input>,
    pub best_score: i64,
    pub rollout_steps: usize,
    pub mlp_evals: usize,
}

#[derive(Clone, Copy)]
struct MiniNode {
    board: FastBoard,
    hold: Option<u8>,
    rem_g: u16,
    q_off: u8,
    first_idx: u16,
    score: f64,
}

#[derive(Clone, Copy)]
struct CrnWorld {
    q: [u8; 40],
    bags: [u8; 40],
    holes: [u16; 48],
}

pub struct NcRolloutSearcher {
    shtb: ShapeTable,
    fst: FastShapeTable,
    pgen: FastPlaceGen,
    ff: FinesseFinder,
    bag_tracker: BagTracker,
    tmp_places: Vec<FastPlace>,
    cur_beam: Vec<MiniNode>,
    nxt_beam: Vec<MiniNode>,
    root_places: Vec<FastPlace>,
}

impl NcRolloutSearcher {
    pub fn new(shapes: ShapeTable) -> Self {
        let fst = FastShapeTable::from_shape_table(&shapes);
        Self {
            shtb: shapes,
            fst,
            pgen: FastPlaceGen::new(),
            ff: FinesseFinder::new(),
            bag_tracker: BagTracker::new(),
            tmp_places: Vec::with_capacity(128),
            cur_beam: Vec::with_capacity(64),
            nxt_beam: Vec::with_capacity(256),
            root_places: Vec::with_capacity(128),
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

    #[inline]
    fn spawn_cheese_from_stream(
        b: &mut FastBoard,
        rem_g: u16,
        lines_cleared: u8,
        steady_g4: bool,
        holes: &[u16],
        hole_cursor: &mut usize,
    ) -> bool {
        let target_g: usize = if steady_g4 || lines_cleared > 0 { 4 } else { 9 };
        let g = b.garbage_rows as usize;
        let unspawned = (rem_g as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 {
            return true;
        }
        let old_rows = b.rows as usize;
        if old_rows + num >= 20 || *hole_cursor + num > holes.len() {
            return false;
        }
        for r in (0..old_rows).rev() {
            b.data[r + num] = b.data[r];
        }
        for k in 0..num {
            let hole = holes[*hole_cursor + k];
            let dst_r = num - 1 - k;
            b.data[dst_r] = 0xFC00 | (0x3FF & !(1u16 << hole));
        }
        *hole_cursor += num;
        b.rows = (old_rows + num) as u8;
        b.garbage_rows = (g + num) as u8;
        true
    }

    #[inline]
    fn avail_mask_obs(hold: Option<u8>, vis_q: &[u8], q_off: usize, bag_mask: u8) -> u8 {
        let mut mask = 0u8;
        if let Some(h) = hold {
            mask |= 1u8 << h;
        }
        let mut cnt = 0usize;
        for &p in vis_q.iter().skip(q_off) {
            mask |= 1u8 << p;
            cnt += 1;
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
    fn eval_fast_linear(
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

    #[inline]
    fn eval_full(
        sm: Option<&SpatialValueModel>,
        board: &FastBoard,
        rem_g: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
        root_ref: f64,
        topout_penalty: f64,
    ) -> f64 {
        if rem_g == 0 && board.rows < 20 {
            return (board.rows as f64) * 5.0;
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
                return (board.rows as f64) * 5.0;
            }
            v as f64
        } else {
            (rem_g as f64) * 185.0 + (board.rows as f64) * 20.0
        }
    }

    /// Strictly non-clairvoyant rollout policy `pi_0`.
    /// Takes ONLY currently observable `(board, hold, vis_q[..<=6], bag_mask, rem_g)`.
    /// NEVER sees any unspawned garbage hole or any piece beyond `vis_q`.
    fn choose_non_clairvoyant_move(
        &mut self,
        board: &FastBoard,
        hold: Option<u8>,
        vis_q: &[u8],
        bag_mask: u8,
        rem_g: u16,
        plies: usize,
        width: usize,
        use_full_mlp: bool,
        sm: Option<&SpatialValueModel>,
        mlp_evals: &mut usize,
    ) -> Option<FastPlace> {
        let cur0 = vis_q.first().copied();
        let (c_nh0, c_h0) = if hold.is_some() {
            (cur0, hold)
        } else {
            (cur0, vis_q.get(1).copied())
        };

        self.root_places.clear();
        self.pgen
            .generate_for_state(&self.fst, board, c_nh0, c_h0, &mut self.root_places);
        if self.root_places.is_empty() {
            return None;
        }

        let max_plies = plies.max(1).min(vis_q.len().max(1));
        self.cur_beam.clear();

        for (idx, &fp) in self.root_places.iter().enumerate() {
            let mut b2 = *board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem2 = rem_g.saturating_sub(ds as u16);
            if rem2 == 0 && b2.rows < 20 {
                return Some(fp);
            }
            if b2.rows >= 20 {
                continue;
            }
            let (nh, nq) = if hold.is_some() {
                if fp.did_hold {
                    (cur0, if cur0.is_some() { 1usize } else { 0usize })
                } else {
                    (hold, 1.min(vis_q.len()))
                }
            } else if fp.did_hold {
                (cur0, 2.min(vis_q.len()))
            } else {
                (None, 1.min(vis_q.len()))
            };
            let ih = nh.or_else(|| vis_q.get(nq).copied());
            let av = Self::avail_mask_obs(nh, vis_q, nq, bag_mask);
            let sc = if max_plies == 1 && use_full_mlp {
                *mlp_evals += 1;
                Self::eval_full(sm, &b2, rem2, ih, av, 20_000.0, 5000.0)
            } else {
                Self::eval_fast_linear(sm, &b2, rem2, ih, av)
            };
            self.cur_beam.push(MiniNode {
                board: b2,
                hold: nh,
                rem_g: rem2,
                q_off: nq as u8,
                first_idx: idx as u16,
                score: sc,
            });
        }

        if self.cur_beam.is_empty() {
            return Some(self.root_places[0]);
        }

        if max_plies == 1 {
            let best = self
                .cur_beam
                .iter()
                .min_by(|a, b| a.score.total_cmp(&b.score))
                .unwrap();
            return Some(self.root_places[best.first_idx as usize]);
        }

        let w = width.max(2);
        self.cur_beam.sort_by(|a, b| a.score.total_cmp(&b.score));
        if self.cur_beam.len() > w {
            self.cur_beam.truncate(w);
        }

        for ply in 2..=max_plies {
            self.nxt_beam.clear();
            let is_last = ply == max_plies;
            for i in 0..self.cur_beam.len() {
                let node = self.cur_beam[i];
                let qo = node.q_off as usize;
                let cur = vis_q.get(qo).copied();
                let (c_nh, c_h) = if node.hold.is_some() {
                    (cur, node.hold)
                } else {
                    (cur, vis_q.get(qo + 1).copied())
                };
                if c_nh.is_none() && c_h.is_none() {
                    self.nxt_beam.push(node);
                    continue;
                }
                self.pgen
                    .generate_for_state(&self.fst, &node.board, c_nh, c_h, &mut self.tmp_places);
                for &fp in &self.tmp_places {
                    let mut b2 = node.board;
                    let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
                    let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
                    let rem2 = node.rem_g.saturating_sub(ds as u16);
                    if b2.rows >= 20 {
                        continue;
                    }
                    let (nh, nq) = if node.hold.is_some() {
                        if fp.did_hold {
                            (cur, if cur.is_some() { qo + 1 } else { qo })
                        } else {
                            (node.hold, (qo + 1).min(vis_q.len()))
                        }
                    } else if fp.did_hold {
                        (cur, (qo + 2).min(vis_q.len()))
                    } else {
                        (None, (qo + 1).min(vis_q.len()))
                    };
                    let ih = nh.or_else(|| vis_q.get(nq).copied());
                    let av = Self::avail_mask_obs(nh, vis_q, nq, bag_mask);
                    let sc = Self::eval_fast_linear(sm, &b2, rem2, ih, av);
                    self.nxt_beam.push(MiniNode {
                        board: b2,
                        hold: nh,
                        rem_g: rem2,
                        q_off: nq as u8,
                        first_idx: node.first_idx,
                        score: sc,
                    });
                }
            }
            if self.nxt_beam.is_empty() {
                break;
            }
            self.nxt_beam.sort_by(|a, b| a.score.total_cmp(&b.score));
            if self.nxt_beam.len() > w {
                self.nxt_beam.truncate(w);
            }
            if is_last && use_full_mlp {
                for n in &mut self.nxt_beam {
                    let qo = n.q_off as usize;
                    let ih = n.hold.or_else(|| vis_q.get(qo).copied());
                    let av = Self::avail_mask_obs(n.hold, vis_q, qo, bag_mask);
                    *mlp_evals += 1;
                    n.score = Self::eval_full(sm, &n.board, n.rem_g, ih, av, 20_000.0, 5000.0);
                }
                self.nxt_beam.sort_by(|a, b| a.score.total_cmp(&b.score));
            }
            std::mem::swap(&mut self.cur_beam, &mut self.nxt_beam);
        }

        let best = self
            .cur_beam
            .iter()
            .min_by(|a, b| a.score.total_cmp(&b.score))
            .unwrap();
        Some(self.root_places[best.first_idx as usize])
    }

    fn build_crn_worlds(
        root_q: &[u8],
        bag_mask: u8,
        init_prev_hole: Option<u16>,
        worlds: usize,
        base_seed: u64,
    ) -> Vec<CrnWorld> {
        let q_len = root_q.len().min(6);
        let mut bag_bits = [0u8; 7];
        let mut bag_n = 0usize;
        for p in 0..7u8 {
            if (bag_mask & (1u8 << p)) != 0 {
                bag_bits[bag_n] = p;
                bag_n += 1;
            }
        }

        let mut out = Vec::with_capacity(worlds);
        for w in 0..worlds {
            let mut cw = CrnWorld {
                q: [0u8; 40],
                bags: [0x7Fu8; 40],
                holes: [0u16; 48],
            };
            for i in 0..q_len {
                cw.q[i] = root_q[i];
                cw.bags[i] = bag_mask;
            }
            let mut rng = base_seed
                ^ ((w as u64).wrapping_add(1)).wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0xBF58476D1CE4E5B9;
            let mut bag = bag_mask;
            if q_len < 40 {
                if bag_n > 0 {
                    let p0 = bag_bits[w % bag_n];
                    cw.q[q_len] = p0;
                    bag &= !(1u8 << p0);
                    if bag == 0 {
                        bag = 0x7F;
                    }
                    cw.bags[q_len] = bag;
                    for step in (q_len + 1)..40 {
                        cw.q[step] = Self::draw_from_bag(&mut bag, &mut rng);
                        cw.bags[step] = bag;
                    }
                } else {
                    for step in q_len..40 {
                        cw.q[step] = Self::draw_from_bag(&mut bag, &mut rng);
                        cw.bags[step] = bag;
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
                    ((w % 9) as u16) + 1
                } else {
                    (((Self::next_rng(&mut h_rng) >> 32) as u16) % 9) + 1
                };
                prev_h = (prev_h + off) % 10;
                cw.holes[k] = prev_h;
            }
            out.push(cw);
        }
        out
    }

    pub fn search_with_ctx(
        &mut self,
        snapshot: &Snapshot,
        ctx: GameContext,
        cfg: &NcRolloutConfig,
    ) -> NcRolloutResult {
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
        let bag_mask = ctx
            .next_bag_mask
            .unwrap_or_else(|| self.bag_tracker.observe(snapshot.hold, &snapshot.queue[..q_len]));

        let sm = cfg.spatial_model.as_deref();
        let ih0 = hold0.or(cur0);
        let av0 = Self::avail_mask_obs(hold0, root_q, 0, bag_mask);
        let root_ref = Self::eval_full(sm, &root_board, rem_g0, ih0, av0, 20_000.0, cfg.topout_penalty);
        let mut mlp_evals = 1usize;

        // 1. Generate and prescreen root candidates using 1-ply linear BC-4 (keeping ds>0 and ds==0 diversity)
        let (c_nh0, c_h0) = if hold0.is_some() {
            (cur0, hold0)
        } else {
            (cur0, if q_len > 1 { Some(root_q[1]) } else { None })
        };
        let mut all_roots = Vec::with_capacity(128);
        self.pgen
            .generate_for_state(&self.fst, &root_board, c_nh0, c_h0, &mut all_roots);
        if all_roots.is_empty() {
            return NcRolloutResult {
                inputs: Vec::new(),
                best_score: (root_ref + cfg.topout_penalty) as i64,
                rollout_steps: 0,
                mlp_evals,
            };
        }

        let mut scored: Vec<(FastPlace, f64, bool)> = Vec::with_capacity(all_roots.len());
        for &fp in &all_roots {
            let mut b2 = root_board;
            let orient = &self.fst.shapes[fp.color_idx as usize].orients[fp.tf.2 as usize];
            let (_lines, ds, _) = b2.place_and_sift(orient, fp.norm_i, fp.norm_j);
            let rem2 = rem_g0.saturating_sub(ds as u16);
            if rem2 == 0 && b2.rows < 20 {
                let inputs = self.fp_to_inputs(&snapshot.matrix, fp);
                return NcRolloutResult {
                    inputs,
                    best_score: 100 + (b2.rows as i64) * 5,
                    rollout_steps: 0,
                    mlp_evals,
                };
            }
            if b2.rows >= 20 {
                continue;
            }
            let (nh, nq) = if hold0.is_some() {
                if fp.did_hold {
                    (cur0, if cur0.is_some() { 1usize } else { 0usize })
                } else {
                    (hold0, 1.min(q_len))
                }
            } else if fp.did_hold {
                (cur0, 2.min(q_len))
            } else {
                (None, 1.min(q_len))
            };
            let ih = nh.or_else(|| root_q.get(nq).copied());
            let av = Self::avail_mask_obs(nh, root_q, nq, bag_mask);
            let sc = Self::eval_fast_linear(sm, &b2, rem2, ih, av);
            scored.push((fp, sc, ds > 0));
        }

        if scored.is_empty() {
            let inputs = self.fp_to_inputs(&snapshot.matrix, all_roots[0]);
            return NcRolloutResult {
                inputs,
                best_score: (root_ref + cfg.topout_penalty) as i64,
                rollout_steps: 0,
                mlp_evals,
            };
        }

        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        let keep = cfg.root_candidates.max(2).min(scored.len());
        let mut cands: Vec<FastPlace> = Vec::with_capacity(keep + 1);
        let mut has_ds = false;
        let mut has_0ds = false;
        for &(fp, _, is_ds) in scored.iter().take(keep) {
            if is_ds {
                has_ds = true;
            } else {
                has_0ds = true;
            }
            cands.push(fp);
        }
        if !has_ds {
            if let Some(&(fp, _, _)) = scored.iter().skip(keep).find(|x| x.2) {
                cands.push(fp);
            }
        }
        if !has_0ds {
            if let Some(&(fp, _, _)) = scored.iter().skip(keep).find(|x| !x.2) {
                cands.push(fp);
            }
        }

        // 2. Build deterministic CRN worlds from observable root state
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
        let init_prev_hole = Self::bottom_hole_col(&root_board);
        let worlds = Self::build_crn_worlds(
            root_q,
            bag_mask,
            init_prev_hole,
            cfg.worlds.max(1),
            base_seed,
        );

        // 3. Evaluate each root candidate across all CRN worlds via strictly non-clairvoyant rollouts
        let mut best_fp = cands[0];
        let mut best_mean = f64::INFINITY;
        let mut rollout_steps = 0usize;

        for &fp0 in &cands {
            let mut sum_w = 0.0f64;
            for world in &worlds {
                let mut b = root_board;
                let orient0 = &self.fst.shapes[fp0.color_idx as usize].orients[fp0.tf.2 as usize];
                let (lines0, ds0, _) = b.place_and_sift(orient0, fp0.norm_i, fp0.norm_j);
                let mut rem = rem_g0.saturating_sub(ds0 as u16);
                let mut hole_cursor = 0usize;

                if rem == 0 && b.rows < 20 {
                    sum_w += 100.0 + (b.rows as f64) * 5.0;
                    continue;
                }
                if b.rows >= 20
                    || (rem > 0
                        && !Self::spawn_cheese_from_stream(
                            &mut b,
                            rem,
                            lines0,
                            cfg.steady_g4,
                            &world.holes,
                            &mut hole_cursor,
                        ))
                {
                    sum_w += root_ref + cfg.topout_penalty;
                    continue;
                }

                let (mut hold, mut qc) = if hold0.is_some() {
                    if fp0.did_hold {
                        (cur0, if cur0.is_some() { 1usize } else { 0usize })
                    } else {
                        (hold0, 1.min(q_len))
                    }
                } else if fp0.did_hold {
                    (cur0, 2.min(q_len))
                } else {
                    (None, 1.min(q_len))
                };

                let mut done_val: Option<f64> = None;
                for step in 1..=cfg.horizon {
                    if qc >= world.q.len() {
                        break;
                    }
                    let vis_end = (qc + 6).min(world.q.len());
                    let vis_q = &world.q[qc..vis_end];
                    let bag_now = world.bags[(vis_end - 1).min(world.bags.len() - 1)];

                    // Strictly non-clairvoyant move selection (NEVER sees `world.holes` or `world.q[vis_end..]`)
                    let fp_t = match self.choose_non_clairvoyant_move(
                        &b,
                        hold,
                        vis_q,
                        bag_now,
                        rem,
                        cfg.policy_plies,
                        cfg.policy_width,
                        cfg.policy_full_mlp,
                        sm,
                        &mut mlp_evals,
                    ) {
                        Some(m) => m,
                        None => {
                            done_val = Some(root_ref + cfg.topout_penalty);
                            break;
                        }
                    };
                    rollout_steps += 1;

                    let orient =
                        &self.fst.shapes[fp_t.color_idx as usize].orients[fp_t.tf.2 as usize];
                    let (lines, ds, _) = b.place_and_sift(orient, fp_t.norm_i, fp_t.norm_j);
                    rem = rem.saturating_sub(ds as u16);

                    if rem == 0 && b.rows < 20 {
                        done_val = Some(((step + 1) as f64) * 100.0 + (b.rows as f64) * 5.0);
                        break;
                    }
                    if b.rows >= 20 {
                        done_val = Some(root_ref + cfg.topout_penalty);
                        break;
                    }
                    if rem > 0
                        && !Self::spawn_cheese_from_stream(
                            &mut b,
                            rem,
                            lines,
                            cfg.steady_g4,
                            &world.holes,
                            &mut hole_cursor,
                        )
                    {
                        done_val = Some(root_ref + cfg.topout_penalty);
                        break;
                    }

                    let cur_t = vis_q.first().copied();
                    let (nh, nqc) = if hold.is_some() {
                        if fp_t.did_hold {
                            (cur_t, if cur_t.is_some() { qc + 1 } else { qc })
                        } else {
                            (hold, qc + 1)
                        }
                    } else if fp_t.did_hold {
                        (cur_t, qc + 2)
                    } else {
                        (None, qc + 1)
                    };
                    hold = nh;
                    qc = nqc;
                }

                let w_val = if let Some(dv) = done_val {
                    dv
                } else {
                    let vis_end = (qc + 6).min(world.q.len());
                    let vis_q = if qc < world.q.len() {
                        &world.q[qc..vis_end]
                    } else {
                        &[]
                    };
                    let bag_now = world.bags[(vis_end.saturating_sub(1)).min(world.bags.len() - 1)];
                    let ih = hold.or_else(|| vis_q.first().copied());
                    let av = Self::avail_mask_obs(hold, vis_q, 0, bag_now);
                    mlp_evals += 1;
                    ((cfg.horizon + 1) as f64) * 100.0
                        + Self::eval_full(sm, &b, rem, ih, av, root_ref, cfg.topout_penalty)
                };
                sum_w += w_val;
            }
            let mean_w = sum_w / (worlds.len() as f64);
            if mean_w < best_mean {
                best_mean = mean_w;
                best_fp = fp0;
            }
        }

        let inputs = self.fp_to_inputs(&snapshot.matrix, best_fp);
        NcRolloutResult {
            inputs,
            best_score: best_mean.round() as i64,
            rollout_steps,
            mlp_evals,
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
    fn test_nc_rollout_zero_future_peeking_beyond_p5() {
        let mut s1 = NcRolloutSearcher::new_srs();
        let mut s2 = NcRolloutSearcher::new_srs();

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
                Color::n('O'),
                Color::n('Z'),
            ],
            matrix: mat,
        };
        let ctx = GameContext {
            garbage_rows: Some(4),
            remaining_garbage: Some(50),
            next_bag_mask: Some(0x7F),
        };
        let cfg = NcRolloutConfig {
            root_candidates: 8,
            worlds: 9,
            horizon: 6,
            policy_plies: 2,
            policy_width: 4,
            ..Default::default()
        };

        let r1 = s1.search_with_ctx(&snap1, ctx, &cfg);
        let r2 = s2.search_with_ctx(&snap2, ctx, &cfg);
        assert_eq!(r1.inputs, r2.inputs);
        assert_eq!(r1.best_score, r2.best_score);
        assert_eq!(r1.rollout_steps, r2.rollout_steps);
    }

    #[test]
    fn test_nc_rollout_policy_is_strictly_non_clairvoyant() {
        let mut s = NcRolloutSearcher::new_srs();
        let mat = make_test_board(&[5, 2, 8, 0]);
        let b = FastBoard::from_basic_matrix(&mat, 4);
        let vis_q = [0u8, 2, 6, 5, 3, 4];
        let mut evals = 0usize;
        // `choose_non_clairvoyant_move` does not even accept `world_holes` or `world_q[6..]`
        // as arguments, structurally guaranteeing zero Strategy Fusion.
        let m1 = s.choose_non_clairvoyant_move(&b, Some(0), &vis_q, 0x7F, 40, 3, 6, false, None, &mut evals);
        let m2 = s.choose_non_clairvoyant_move(&b, Some(0), &vis_q, 0x7F, 40, 3, 6, false, None, &mut evals);
        assert!(m1.is_some());
        assert_eq!(m1.unwrap().tf, m2.unwrap().tf);
    }
}
