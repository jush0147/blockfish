use super::{
    fast::{FastBoard, FastPlace},
    spatial_val::{extract_spatial_features_ex, SpatialValueModel, DENSE_DIM, SPARSE_DIM},
};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

pub const POL_SPARSE_DIM: usize = 166; // 8x10 board + 10 overflow + h1/h2/h3 cols + 5x5 h1 + 5x3 h2 + h1->h2 rel
pub const POL_BOARD_DENSE_DIM: usize = 32;
pub const POL_ACT_DIM: usize = 34;
pub const POL_ORDER_DIM: usize = 58; // held(8) + ordered next5(35) + bag_mask(7) + regime(8)
pub const POL_H1: usize = 96;
pub const POL_H2: usize = 48;

/// Strictly legal observable root context (current piece + up to 5 visible previews, hold, BagTracker mask).
/// NEVER contains any piece beyond the 6 visible queue pieces or any hidden environment seed.
#[derive(Clone)]
pub struct ObservableRootState {
    pub board: FastBoard,
    pub hold: Option<u8>,
    /// Strictly `<= 6` visible pieces (`vis_q[0]` = current piece, `vis_q[1..6]` = ordered `next5`).
    pub vis_q: Vec<u8>,
    pub rem_g: u16,
    pub bag_mask: u8,
}

#[derive(Clone)]
pub struct PolicyCandFeatures {
    pub sparse: Vec<u16>,
    pub board_dense: [f32; POL_BOARD_DENSE_DIM],
    pub act_feat: [f32; POL_ACT_DIM],
    pub order_feat: [f32; POL_ORDER_DIM],
    /// Centered frozen BC-4 afterstate delta (`V_bc4(cand) - min_k V_bc4`), used as skip prior.
    pub bc4_delta: f32,
}

#[inline]
fn quick_board_stats(board: &FastBoard) -> ([i32; 10], i32, i32, i32, i32, i32, i32, Option<usize>, Option<usize>) {
    let col_h = board.col_heights();
    let g = board.garbage_rows as usize;
    let n = board.rows as usize;
    let mut stack_h = [0i32; 10];
    let mut max_sh = 0i32;
    for c in 0..10usize {
        let sh = (col_h[c] as i32 - g as i32).max(0);
        stack_h[c] = sh;
        if sh > max_sh {
            max_sh = sh;
        }
    }
    let mut stack_cells = 0i32;
    let mut cov_holes = 0i32;
    let mut blk_over_holes = 0i32;
    let mut above_mask: u16 = 0;
    let mut row_holes = [0u16; 32];
    let mut i = n;
    while i > g {
        i -= 1;
        let bits = board.data[i] & 0x3FF;
        stack_cells += bits.count_ones() as i32;
        let rh = (!bits & 0x3FF) & above_mask;
        row_holes[i - g] = rh;
        cov_holes += rh.count_ones() as i32;
        above_mask |= bits;
    }
    if cov_holes > 0 {
        let mut below_hole = 0u16;
        for r in g..n {
            let bits = board.data[r] & 0x3FF;
            blk_over_holes += (bits & below_hole).count_ones() as i32;
            below_hole |= row_holes[r - g];
        }
    }
    let mut bump = 0i32;
    for c in 0..9usize {
        bump += (stack_h[c] - stack_h[c + 1]).abs();
    }
    let h1_col = if g >= 1 {
        let c = (!board.data[g - 1] & 0x3FF).trailing_zeros() as usize;
        if c < 10 { Some(c) } else { None }
    } else {
        None
    };
    let h2_col = if g >= 2 {
        let c = (!board.data[g - 2] & 0x3FF).trailing_zeros() as usize;
        if c < 10 { Some(c) } else { None }
    } else {
        None
    };
    let mut h1_blk = 0i32;
    if let Some(c1) = h1_col {
        let ch = col_h[c1] as usize;
        for r in g..ch {
            if (board.data[r] >> c1) & 1 != 0 {
                h1_blk += 1;
            }
        }
    }
    let mut h2_blk = 0i32;
    if let Some(c2) = h2_col {
        let ch = col_h[c2] as usize;
        for r in g..ch {
            if (board.data[r] >> c2) & 1 != 0 {
                h2_blk += 1;
            }
        }
    }
    (stack_h, stack_cells, cov_holes, blk_over_holes, max_sh, bump, h1_blk * 100 + h2_blk, h1_col, h2_col)
}

/// Extract strictly legal observable policy features for one candidate root action `(fp, b_after, lines, ds)`.
/// Enforces `vis_q` truncation to `<= 6` visible pieces (`cur + next5`) so zero future information can ever leak.
pub fn extract_policy_cand_features(
    state: &ObservableRootState,
    fp: FastPlace,
    b_after: &FastBoard,
    lines: u8,
    ds: u8,
    bc4_delta: f32,
) -> PolicyCandFeatures {
    // Strictly clamp visible queue to official Jstris p=5 (`cur` + 5 previews = 6 pieces max)
    let vis_len = state.vis_q.len().min(6);
    let vis_q = &state.vis_q[..vis_len];

    let (
        _sh_b,
        cells_b,
        holes_b,
        boh_b,
        maxh_b,
        bump_b,
        h12_b,
        _h1c_b,
        _h2c_b,
    ) = quick_board_stats(&state.board);
    let h1_b = h12_b / 100;
    let h2_b = h12_b % 100;

    let (
        sh_a,
        cells_a,
        holes_a,
        boh_a,
        maxh_a,
        bump_a,
        h12_a,
        h1c_a,
        h2c_a,
    ) = quick_board_stats(b_after);
    let h1_a = h12_a / 100;
    let h2_a = h12_a % 100;

    let rem1 = state.rem_g.saturating_sub(ds as u16);
    let c_nh = vis_q.first().copied();
    let (held1, qc1) = if state.hold.is_some() {
        if fp.did_hold {
            (c_nh, if c_nh.is_some() { 1usize } else { 0usize })
        } else {
            (state.hold, 1usize)
        }
    } else if fp.did_hold {
        (c_nh, 2usize.min(vis_len))
    } else {
        (None, 1usize.min(vis_len))
    };

    // 1. Ordered future sequence + hold + legal bag_mask (`POL_ORDER_DIM = 58`)
    let mut order_feat = [0.0f32; POL_ORDER_DIM];
    // 0..8: held piece after action
    match held1 {
        Some(p) if (p as usize) < 7 => order_feat[p as usize] = 1.0,
        _ => order_feat[7] = 1.0,
    }
    // 8..43: ordered post-action queue slots `q_after[0..5]` (strictly within `vis_q[qc1..vis_len]`)
    for slot in 0..5usize {
        if let Some(&p) = vis_q.get(qc1 + slot) {
            if (p as usize) < 7 {
                order_feat[8 + slot * 7 + p as usize] = 1.0;
            }
        }
    }
    // 43..50: legal BagTracker remaining bag mask
    for p in 0..7usize {
        if ((state.bag_mask >> p) & 1) != 0 {
            order_feat[43 + p] = 1.0;
        }
    }
    // 50..58: regime & queue context
    order_feat[50] = if fp.did_hold { 1.0 } else { 0.0 };
    order_feat[51] = if state.hold.is_some() { 1.0 } else { 0.0 };
    order_feat[52] = (rem1 as f32) / 100.0;
    order_feat[53] = (b_after.garbage_rows as f32) / 9.0;
    order_feat[54] = if b_after.garbage_rows >= 5 { 1.0 } else { 0.0 };
    order_feat[55] = if rem1 <= 9 && (rem1 as u8) == b_after.garbage_rows { 1.0 } else { 0.0 };
    order_feat[56] = (vis_len.saturating_sub(qc1) as f32) / 5.0;
    order_feat[57] = (state.bag_mask.count_ones() as f32) / 7.0;

    // 2. Candidate action & delta geometry (`POL_ACT_DIM = 34`)
    let mut act_feat = [0.0f32; POL_ACT_DIM];
    let c_idx = (fp.color_idx as usize).min(6);
    act_feat[c_idx] = 1.0; // 0..7
    let rot = (fp.tf.2 as usize).min(3);
    act_feat[7 + rot] = 1.0; // 7..11
    act_feat[11] = (fp.norm_j as f32) / 9.0;
    act_feat[12] = (fp.norm_i as f32) / 18.0;
    act_feat[13] = if fp.did_hold { 1.0 } else { 0.0 };
    act_feat[14] = ds as f32;
    act_feat[15] = (lines as f32) / 4.0;
    act_feat[16] = (lines.saturating_sub(ds) as f32) / 4.0;
    act_feat[17] = if ds == 0 { 1.0 } else { 0.0 };
    let target_g = if lines > 0 { 4usize } else { 9usize };
    let cur_g = b_after.garbage_rows as usize;
    let unspawned = (rem1 as usize).saturating_sub(cur_g);
    let exp_spawn = target_g.saturating_sub(cur_g).min(unspawned);
    act_feat[18] = if exp_spawn > 0 { 1.0 } else { 0.0 };
    act_feat[19] = (exp_spawn as f32) / 5.0;
    act_feat[20] = ((cells_a - cells_b) as f32) / 10.0;
    act_feat[21] = ((h1_a - h1_b) as f32) / 4.0;
    act_feat[22] = ((h2_a - h2_b) as f32) / 4.0;
    act_feat[23] = ((holes_a - holes_b) as f32) / 4.0;
    act_feat[24] = ((boh_a - boh_b) as f32) / 4.0;
    act_feat[25] = ((maxh_a - maxh_b) as f32) / 4.0;
    act_feat[26] = ((bump_a - bump_b) as f32) / 8.0;
    let h1_open_a = b_after.garbage_rows >= 1 && h1_a == 0;
    let h2_open_a = b_after.garbage_rows >= 2 && h2_a == 0;
    act_feat[27] = if h1_open_a { 1.0 } else { 0.0 };
    act_feat[28] = if h2_open_a { 1.0 } else { 0.0 };
    act_feat[29] = if h1_open_a && h2_open_a { 1.0 } else { 0.0 };
    // ds_clears_and_covers_h2 (`cov_nxt` proxy!)
    act_feat[30] = if ds > 0 && h2_b == 0 && h1_a > 0 { 1.0 } else { 0.0 };
    act_feat[31] = (bc4_delta / 3.0).clamp(-3.0, 3.0);
    act_feat[32] = h1c_a.map_or(0.0, |c| (sh_a[c] as f32) / 8.0);
    act_feat[33] = h2c_a.map_or(0.0, |c| (sh_a[c] as f32) / 8.0);

    // 3. 2D Board Spatial Features (`POL_SPARSE_DIM = 166`, `POL_BOARD_DENSE_DIM = 32`)
    let mut act_buf = [0u16; SPARSE_DIM];
    let mut dense_buf = [0.0f32; DENSE_DIM];
    let n_act = extract_spatial_features_ex(
        b_after,
        rem1,
        held1,
        state.bag_mask,
        false,
        &mut act_buf,
        &mut dense_buf,
    );
    let mut sparse = Vec::with_capacity(64);
    for &bit in &act_buf[..n_act] {
        if (bit as usize) < POL_SPARSE_DIM {
            sparse.push(bit);
        }
    }

    let mut board_dense = [0.0f32; POL_BOARD_DENSE_DIM];
    for c in 0..10usize {
        board_dense[c] = (sh_a[c] as f32) / 8.0;
    }
    for c in 0..9usize {
        board_dense[10 + c] = ((sh_a[c] - sh_a[c + 1]) as f32) / 4.0;
    }
    board_dense[19] = (cells_a as f32) / 30.0;
    board_dense[20] = (holes_a as f32) / 6.0;
    board_dense[21] = (boh_a as f32) / 8.0;
    board_dense[22] = (h1_a as f32) / 4.0;
    board_dense[23] = (h2_a as f32) / 4.0;
    board_dense[24] = (maxh_a as f32) / 10.0;
    board_dense[25] = (bump_a as f32) / 16.0;
    board_dense[26] = h1c_a.map_or(-0.5, |c| (c as f32 - 4.5) / 4.5);
    board_dense[27] = h2c_a.map_or(-0.5, |c| (c as f32 - 4.5) / 4.5);
    board_dense[28] = (dense_buf[18] / 20.0).clamp(-3.0, 3.0); // row_transitions
    board_dense[29] = (dense_buf[19] / 20.0).clamp(-3.0, 3.0); // col_transitions
    board_dense[30] = (dense_buf[20] / 4.0).clamp(-3.0, 3.0);  // near_full_9
    board_dense[31] = (dense_buf[21] / 4.0).clamp(-3.0, 3.0);  // near_full_8

    PolicyCandFeatures {
        sparse,
        board_dense,
        act_feat,
        order_feat,
        bc4_delta,
    }
}

/// Pure-Rust FiLM Cross-Interaction Ordered-`next5` Root Policy / Ranking Network.
/// Completely decoupled from frozen BC-4 leaf value evaluator (`SpatialValueModel`).
#[derive(Clone)]
pub struct OrderedRootPolicy {
    pub b_board: [f32; POL_H1],
    pub w_sparse: Box<[[f32; POL_H1]; POL_SPARSE_DIM]>,
    pub w_bdense: Box<[[f32; POL_H1]; POL_BOARD_DENSE_DIM]>,
    pub w_act: Box<[[f32; POL_H1]; POL_ACT_DIM]>,

    pub b_queue: [f32; POL_H1],
    pub w_order: Box<[[f32; POL_H1]; POL_ORDER_DIM]>,
    /// FiLM multiplicative gate weight `gamma = tanh(w_film * h_queue + b_film)` modulating `h_board`.
    pub b_film: [f32; POL_H1],
    pub w_film: [f32; POL_H1],

    pub b2: [f32; POL_H2],
    pub w2: Box<[[f32; POL_H2]; POL_H1]>,
    pub b3: f32,
    pub w3: [f32; POL_H2],
    /// Learnable skip weight on frozen BC-4 prior (`-bc4_delta`).
    pub w_bc4_skip: f32,
}

pub struct PolicyForwardCache {
    pub h_board: [f32; POL_H1],
    pub h_queue: [f32; POL_H1],
    pub film_tanh: [f32; POL_H1],
    pub z1_pre: [f32; POL_H1],
    pub z1: [f32; POL_H1],
    pub a2: [f32; POL_H2],
}

impl Default for PolicyForwardCache {
    fn default() -> Self {
        Self {
            h_board: [0.0; POL_H1],
            h_queue: [0.0; POL_H1],
            film_tanh: [0.0; POL_H1],
            z1_pre: [0.0; POL_H1],
            z1: [0.0; POL_H1],
            a2: [0.0; POL_H2],
        }
    }
}

pub struct PolicyGradBuf {
    pub b_board: [f32; POL_H1],
    pub w_sparse: Box<[[f32; POL_H1]; POL_SPARSE_DIM]>,
    pub w_bdense: Box<[[f32; POL_H1]; POL_BOARD_DENSE_DIM]>,
    pub w_act: Box<[[f32; POL_H1]; POL_ACT_DIM]>,
    pub b_queue: [f32; POL_H1],
    pub w_order: Box<[[f32; POL_H1]; POL_ORDER_DIM]>,
    pub b_film: [f32; POL_H1],
    pub w_film: [f32; POL_H1],
    pub b2: [f32; POL_H2],
    pub w2: Box<[[f32; POL_H2]; POL_H1]>,
    pub b3: f32,
    pub w3: [f32; POL_H2],
    pub w_bc4_skip: f32,
}

impl PolicyGradBuf {
    pub fn new() -> Self {
        Self {
            b_board: [0.0; POL_H1],
            w_sparse: Box::new([[0.0; POL_H1]; POL_SPARSE_DIM]),
            w_bdense: Box::new([[0.0; POL_H1]; POL_BOARD_DENSE_DIM]),
            w_act: Box::new([[0.0; POL_H1]; POL_ACT_DIM]),
            b_queue: [0.0; POL_H1],
            w_order: Box::new([[0.0; POL_H1]; POL_ORDER_DIM]),
            b_film: [0.0; POL_H1],
            w_film: [0.0; POL_H1],
            b2: [0.0; POL_H2],
            w2: Box::new([[0.0; POL_H2]; POL_H1]),
            b3: 0.0,
            w3: [0.0; POL_H2],
            w_bc4_skip: 0.0,
        }
    }

    pub fn zero(&mut self) {
        self.b_board.fill(0.0);
        for row in self.w_sparse.iter_mut() { row.fill(0.0); }
        for row in self.w_bdense.iter_mut() { row.fill(0.0); }
        for row in self.w_act.iter_mut() { row.fill(0.0); }
        self.b_queue.fill(0.0);
        for row in self.w_order.iter_mut() { row.fill(0.0); }
        self.b_film.fill(0.0);
        self.w_film.fill(0.0);
        self.b2.fill(0.0);
        for row in self.w2.iter_mut() { row.fill(0.0); }
        self.b3 = 0.0;
        self.w3.fill(0.0);
        self.w_bc4_skip = 0.0;
    }

    pub fn scale(&mut self, s: f32) {
        for x in &mut self.b_board { *x *= s; }
        for row in self.w_sparse.iter_mut() { for x in row.iter_mut() { *x *= s; } }
        for row in self.w_bdense.iter_mut() { for x in row.iter_mut() { *x *= s; } }
        for row in self.w_act.iter_mut() { for x in row.iter_mut() { *x *= s; } }
        for x in &mut self.b_queue { *x *= s; }
        for row in self.w_order.iter_mut() { for x in row.iter_mut() { *x *= s; } }
        for x in &mut self.b_film { *x *= s; }
        for x in &mut self.w_film { *x *= s; }
        for x in &mut self.b2 { *x *= s; }
        for row in self.w2.iter_mut() { for x in row.iter_mut() { *x *= s; } }
        self.b3 *= s;
        for x in &mut self.w3 { *x *= s; }
        self.w_bc4_skip *= s;
    }
}

impl OrderedRootPolicy {
    pub fn new_random(seed: u64) -> Self {
        let mut rng = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1337);
        let mut next_u = |scale: f32| -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((rng >> 40) as f32) * (1.0 / ((1u32 << 24) as f32));
            (u * 2.0 - 1.0) * scale
        };

        let mut w_sparse = Box::new([[0.0f32; POL_H1]; POL_SPARSE_DIM]);
        for row in w_sparse.iter_mut() {
            for x in row.iter_mut() { *x = next_u(0.05); }
        }
        let mut w_bdense = Box::new([[0.0f32; POL_H1]; POL_BOARD_DENSE_DIM]);
        for row in w_bdense.iter_mut() {
            for x in row.iter_mut() { *x = next_u(0.12); }
        }
        let mut w_act = Box::new([[0.0f32; POL_H1]; POL_ACT_DIM]);
        for row in w_act.iter_mut() {
            for x in row.iter_mut() { *x = next_u(0.12); }
        }
        let mut w_order = Box::new([[0.0f32; POL_H1]; POL_ORDER_DIM]);
        for row in w_order.iter_mut() {
            for x in row.iter_mut() { *x = next_u(0.12); }
        }
        let mut w_film = [0.0f32; POL_H1];
        for x in &mut w_film { *x = next_u(0.25); }
        let mut w2 = Box::new([[0.0f32; POL_H2]; POL_H1]);
        for row in w2.iter_mut() {
            for x in row.iter_mut() { *x = next_u(0.14); }
        }
        let mut w3 = [0.0f32; POL_H2];
        for x in &mut w3 { *x = next_u(0.08); }

        Self {
            b_board: [0.0; POL_H1],
            w_sparse,
            w_bdense,
            w_act,
            b_queue: [0.0; POL_H1],
            w_order,
            b_film: [0.0; POL_H1],
            w_film,
            b2: [0.0; POL_H2],
            w2,
            b3: 0.0,
            w3,
            w_bc4_skip: 1.0,
        }
    }

    /// Forward pass returning policy preference logit (higher = preferred action).
    #[inline]
    pub fn forward(&self, f: &PolicyCandFeatures, cache: &mut PolicyForwardCache) -> f32 {
        let mut hb = self.b_board;
        for &bit in &f.sparse {
            let row = &self.w_sparse[bit as usize];
            for h in 0..POL_H1 {
                hb[h] += row[h];
            }
        }
        for j in 0..POL_BOARD_DENSE_DIM {
            let x = f.board_dense[j];
            if x != 0.0 {
                let row = &self.w_bdense[j];
                for h in 0..POL_H1 {
                    hb[h] += x * row[h];
                }
            }
        }
        for j in 0..POL_ACT_DIM {
            let x = f.act_feat[j];
            if x != 0.0 {
                let row = &self.w_act[j];
                for h in 0..POL_H1 {
                    hb[h] += x * row[h];
                }
            }
        }

        let mut hq = self.b_queue;
        for j in 0..POL_ORDER_DIM {
            let x = f.order_feat[j];
            if x != 0.0 {
                let row = &self.w_order[j];
                for h in 0..POL_H1 {
                    hq[h] += x * row[h];
                }
            }
        }

        let mut h2 = self.b2;
        for h in 0..POL_H1 {
            let ab = hb[h].clamp(0.0, 6.0);
            let aq = hq[h].clamp(0.0, 6.0);
            let ft = (self.w_film[h] * aq + self.b_film[h]).tanh();
            let z_pre = ab * (1.0 + ft) + aq;
            let z = z_pre.clamp(0.0, 6.0);
            cache.h_board[h] = ab;
            cache.h_queue[h] = aq;
            cache.film_tanh[h] = ft;
            cache.z1_pre[h] = z_pre;
            cache.z1[h] = z;
            if z > 0.0 {
                let row = &self.w2[h];
                for k in 0..POL_H2 {
                    h2[k] += z * row[k];
                }
            }
        }

        let mut out = self.b3 - self.w_bc4_skip * f.bc4_delta;
        for k in 0..POL_H2 {
            let a2 = h2[k].clamp(0.0, 6.0);
            cache.a2[k] = a2;
            if a2 > 0.0 {
                out += a2 * self.w3[k];
            }
        }
        out
    }

    /// Exact analytical backward pass accumulating `dL / d theta` given `d_out = dL / d logit`.
    #[inline]
    pub fn backward(
        &self,
        f: &PolicyCandFeatures,
        cache: &PolicyForwardCache,
        d_out: f32,
        g: &mut PolicyGradBuf,
    ) {
        if d_out == 0.0 {
            return;
        }
        g.b3 += d_out;
        g.w_bc4_skip += d_out * (-f.bc4_delta);

        let mut d_h2 = [0.0f32; POL_H2];
        for k in 0..POL_H2 {
            g.w3[k] += d_out * cache.a2[k];
            if cache.a2[k] > 0.0 && cache.a2[k] < 6.0 {
                d_h2[k] = d_out * self.w3[k];
                g.b2[k] += d_h2[k];
            }
        }

        let mut d_hb = [0.0f32; POL_H1];
        let mut d_hq = [0.0f32; POL_H1];
        for h in 0..POL_H1 {
            let z = cache.z1[h];
            let mut dz = 0.0f32;
            for k in 0..POL_H2 {
                let dk = d_h2[k];
                if dk != 0.0 {
                    if z > 0.0 {
                        g.w2[h][k] += z * dk;
                    }
                    dz += self.w2[h][k] * dk;
                }
            }
            if cache.z1_pre[h] > 0.0 && cache.z1_pre[h] < 6.0 && dz != 0.0 {
                let ab = cache.h_board[h];
                let aq = cache.h_queue[h];
                let ft = cache.film_tanh[h];
                // z_pre = ab * (1 + ft) + aq
                let d_ab = dz * (1.0 + ft);
                let d_ft = dz * ab;
                let d_ft_pre = d_ft * (1.0 - ft * ft);
                g.b_film[h] += d_ft_pre;
                g.w_film[h] += d_ft_pre * aq;
                let d_aq = dz + d_ft_pre * self.w_film[h];

                if ab > 0.0 && ab < 6.0 {
                    d_hb[h] = d_ab;
                    g.b_board[h] += d_ab;
                }
                if aq > 0.0 && aq < 6.0 {
                    d_hq[h] = d_aq;
                    g.b_queue[h] += d_aq;
                }
            }
        }

        for &bit in &f.sparse {
            let row = &mut g.w_sparse[bit as usize];
            for h in 0..POL_H1 {
                row[h] += d_hb[h];
            }
        }
        for j in 0..POL_BOARD_DENSE_DIM {
            let x = f.board_dense[j];
            if x != 0.0 {
                let row = &mut g.w_bdense[j];
                for h in 0..POL_H1 {
                    row[h] += x * d_hb[h];
                }
            }
        }
        for j in 0..POL_ACT_DIM {
            let x = f.act_feat[j];
            if x != 0.0 {
                let row = &mut g.w_act[j];
                for h in 0..POL_H1 {
                    row[h] += x * d_hb[h];
                }
            }
        }
        for j in 0..POL_ORDER_DIM {
            let x = f.order_feat[j];
            if x != 0.0 {
                let row = &mut g.w_order[j];
                for h in 0..POL_H1 {
                    row[h] += x * d_hq[h];
                }
            }
        }
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let mut w = BufWriter::new(File::create(path)?);
        writeln!(w, "ORDERED_POLICY_V1 {} {} {} {} {} {}",
            POL_SPARSE_DIM, POL_BOARD_DENSE_DIM, POL_ACT_DIM, POL_ORDER_DIM, POL_H1, POL_H2)?;
        writeln!(w, "{:.7} {:.7}", self.b3, self.w_bc4_skip)?;
        for x in &self.b_board { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        for x in &self.b_queue { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        for x in &self.b_film { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        for x in &self.w_film { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        for x in &self.b2 { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        for x in &self.w3 { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        for row in self.w_sparse.iter() {
            for x in row { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        }
        for row in self.w_bdense.iter() {
            for x in row { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        }
        for row in self.w_act.iter() {
            for x in row { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        }
        for row in self.w_order.iter() {
            for x in row { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        }
        for row in self.w2.iter() {
            for x in row { write!(w, "{:.6} ", x)?; } writeln!(w)?;
        }
        w.flush()
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let r = BufReader::new(File::open(path)?);
        let mut lines = r.lines();
        let header = lines.next().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty file"))??;
        if !header.starts_with("ORDERED_POLICY_V1") {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header"));
        }
        let parse_vec = |line: String| -> Vec<f32> {
            line.split_whitespace().map(|s| s.parse::<f32>().unwrap_or(0.0)).collect()
        };
        let mut mdl = Self::new_random(1);
        let sc = parse_vec(lines.next().unwrap()?);
        mdl.b3 = sc[0];
        mdl.w_bc4_skip = sc[1];
        mdl.b_board.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        mdl.b_queue.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        mdl.b_film.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        mdl.w_film.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        mdl.b2.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H2]);
        mdl.w3.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H2]);
        for row in mdl.w_sparse.iter_mut() {
            row.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        }
        for row in mdl.w_bdense.iter_mut() {
            row.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        }
        for row in mdl.w_act.iter_mut() {
            row.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        }
        for row in mdl.w_order.iter_mut() {
            row.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H1]);
        }
        for row in mdl.w2.iter_mut() {
            row.copy_from_slice(&parse_vec(lines.next().unwrap()?)[..POL_H2]);
        }
        Ok(mdl)
    }

    /// Compute frozen BC-4 afterstate score for a candidate (helper for Policy/Value separation).
    #[inline]
    pub fn frozen_bc4_score(
        bc4: &SpatialValueModel,
        b_after: &FastBoard,
        rem1: u16,
        held1: Option<u8>,
        avail1: u8,
    ) -> f32 {
        (bc4.evaluate_board_with_bag(b_after, rem1, held1, avail1) as f32) * 0.01
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::fast::{FastShapeTable, EMPTY_ROW};

    fn dummy_state(q: Vec<u8>) -> ObservableRootState {
        let mut b = FastBoard::empty();
        b.data[0] = EMPTY_ROW | (0x3FF & !(1u16 << 3));
        b.data[1] = EMPTY_ROW | (0x3FF & !(1u16 << 7));
        b.rows = 2;
        b.garbage_rows = 2;
        ObservableRootState {
            board: b,
            hold: Some(1),
            vis_q: q,
            rem_g: 50,
            bag_mask: 0x3F,
        }
    }

    #[test]
    fn test_ordered_policy_zero_future_peeking_beyond_p5() {
        let s6 = dummy_state(vec![0, 1, 2, 3, 4, 5]);
        let s10 = dummy_state(vec![0, 1, 2, 3, 4, 5, 6, 6, 0, 1]); // extra hidden pieces MUST be ignored
        let fp = FastPlace {
            color_idx: 0,
            tf: (2, 4, crate::Orientation::R0),
            norm_i: 2,
            norm_j: 4,
            did_hold: false,
        };
        let f6 = extract_policy_cand_features(&s6, fp, &s6.board, 0, 0, 0.2);
        let f10 = extract_policy_cand_features(&s10, fp, &s10.board, 0, 0, 0.2);
        assert_eq!(f6.order_feat, f10.order_feat, "must strictly ignore pieces beyond cur+next5");
        assert_eq!(f6.act_feat, f10.act_feat);
        assert_eq!(f6.board_dense, f10.board_dense);
    }

    #[test]
    fn test_ordered_policy_sequence_sensitivity() {
        let s_a = dummy_state(vec![0, 1, 2, 3, 4, 5]);
        let s_b = dummy_state(vec![0, 5, 4, 3, 2, 1]); // same multiset, reversed next5 order!
        let fp = FastPlace {
            color_idx: 0,
            tf: (2, 4, crate::Orientation::R0),
            norm_i: 2,
            norm_j: 4,
            did_hold: false,
        };
        let fa = extract_policy_cand_features(&s_a, fp, &s_a.board, 0, 0, 0.2);
        let fb = extract_policy_cand_features(&s_b, fp, &s_b.board, 0, 0, 0.2);
        assert_ne!(fa.order_feat, fb.order_feat, "reordering next5 MUST change order_feat");

        let pol = OrderedRootPolicy::new_random(42);
        let mut ca = PolicyForwardCache::default();
        let mut cb = PolicyForwardCache::default();
        let la = pol.forward(&fa, &mut ca);
        let lb = pol.forward(&fb, &mut cb);
        assert!((la - lb).abs() > 1e-5, "reordering next5 MUST change policy logit (got {} vs {})", la, lb);
    }

    #[test]
    fn test_ordered_policy_analytical_gradient_matches_finite_diff() {
        let s = dummy_state(vec![0, 2, 4, 1, 6, 3]);
        let fst = FastShapeTable::srs();
        let fp = FastPlace {
            color_idx: 0,
            tf: (2, 4, crate::Orientation::R0),
            norm_i: 2,
            norm_j: 4,
            did_hold: false,
        };
        let mut b2 = s.board;
        let o = &fst.shapes[0].orients[0];
        let (l, ds, _) = b2.place_and_sift(o, fp.norm_i, fp.norm_j);
        let feat = extract_policy_cand_features(&s, fp, &b2, l, ds, 0.35);

        let mut pol = OrderedRootPolicy::new_random(7);
        let mut cache = PolicyForwardCache::default();
        let _ = pol.forward(&feat, &mut cache);
        let mut g = PolicyGradBuf::new();
        pol.backward(&feat, &cache, 1.0, &mut g);

        let eps = 1e-3f32;
        // Check w_film[0], w_order[8][0], w_act[0][0], w3[0], w_bc4_skip
        let check = |pol: &mut OrderedRootPolicy, getter: fn(&mut OrderedRootPolicy) -> &mut f32, ana: f32, name: &str| {
            let orig = *getter(pol);
            let mut c1 = PolicyForwardCache::default();
            let mut c2 = PolicyForwardCache::default();
            *getter(pol) = orig + eps;
            let p1 = pol.forward(&feat, &mut c1);
            *getter(pol) = orig - eps;
            let p2 = pol.forward(&feat, &mut c2);
            *getter(pol) = orig;
            let num = (p1 - p2) / (2.0 * eps);
            assert!(
                (num - ana).abs() < 1e-2,
                "grad mismatch on {}: ana={} num={}",
                name,
                ana,
                num
            );
        };

        check(&mut pol, |p| &mut p.w_bc4_skip, g.w_bc4_skip, "w_bc4_skip");
        check(&mut pol, |p| &mut p.b3, g.b3, "b3");
        check(&mut pol, |p| &mut p.w3[0], g.w3[0], "w3[0]");
        check(&mut pol, |p| &mut p.w_film[0], g.w_film[0], "w_film[0]");
        check(&mut pol, |p| &mut p.w_order[8 + 2][0], g.w_order[8 + 2][0], "w_order");
        check(&mut pol, |p| &mut p.w_act[0][0], g.w_act[0][0], "w_act");
    }
}
