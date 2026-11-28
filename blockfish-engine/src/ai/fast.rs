use crate::{
    place::PlaceFinder,
    shape::{ShapeRef, ShapeTable},
    BasicMatrix, Color, Input, Orientation,
};

pub const BOARD_COLS: u16 = 10;
pub const MAX_ROWS: usize = 32;
pub const EMPTY_ROW: u16 = 0xFC00; // bits 10..16 set, bits 0..10 clear
pub const FULL_ROW: u16 = 0xFFFF;
/// Opaque unspawned garbage row: all 10 columns occupied (`0x3FF`), bit 10 clear so
/// `!= FULL_ROW` (never auto-cleared) and `(!row & 0x3FF).trailing_zeros() >= 10` (hole unknown).
pub const OPAQUE_GARBAGE_ROW: u16 = 0xFBFF;

#[inline(always)]
pub fn color_to_idx(c: Color) -> usize {
    match c.as_char() {
        'I' => 0,
        'O' => 1,
        'T' => 2,
        'S' => 3,
        'Z' => 4,
        'J' => 5,
        'L' => 6,
        _ => panic!("unknown color {:?}", c),
    }
}

#[inline(always)]
pub fn idx_to_color(idx: usize) -> Color {
    const CHARS: [char; 7] = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
    use std::convert::TryInto;
    CHARS[idx].try_into().unwrap()
}

/// Compact, stack-allocated, `Copy` 10-column Tetris board.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct FastBoard {
    pub rows: u8,
    pub garbage_rows: u8,
    pub data: [u16; MAX_ROWS],
}

impl std::hash::Hash for FastBoard {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.rows.hash(state);
        self.garbage_rows.hash(state);
        self.data[..self.rows as usize].hash(state);
    }
}

impl FastBoard {
    pub fn empty() -> Self {
        Self {
            rows: 0,
            garbage_rows: 0,
            data: [EMPTY_ROW; MAX_ROWS],
        }
    }

    pub fn from_basic_matrix(mat: &BasicMatrix, garbage_rows: u16) -> Self {
        assert_eq!(mat.cols(), BOARD_COLS);
        let mut b = Self::empty();
        let n = (mat.rows() as usize).min(MAX_ROWS);
        b.rows = n as u8;
        b.garbage_rows = (garbage_rows as usize).min(n) as u8;
        for i in 0..n {
            let mut row_bits = EMPTY_ROW;
            for j in 0..BOARD_COLS {
                if mat.get((i as u16, j)) {
                    row_bits |= 1 << j;
                }
            }
            b.data[i] = row_bits;
        }
        b
    }

    /// Infer garbage rows if not explicitly provided: contiguous bottom rows with exactly 9 filled cells.
    pub fn infer_garbage_rows(mat: &BasicMatrix) -> u16 {
        let mut g = 0u16;
        for i in 0..mat.rows() {
            let mut cnt = 0;
            for j in 0..mat.cols() {
                if mat.get((i, j)) {
                    cnt += 1;
                }
            }
            if cnt == mat.cols() - 1 {
                g += 1;
            } else {
                break;
            }
        }
        g
    }

    #[inline(always)]
    pub fn row(&self, i: usize) -> u16 {
        if i < self.rows as usize {
            self.data[i]
        } else {
            EMPTY_ROW
        }
    }

    #[inline(always)]
    pub fn col_heights(&self) -> [u8; 10] {
        let mut h = [0u8; 10];
        let mut remaining = 0x3FFu16;
        let mut i = self.rows as usize;
        while i > 0 && remaining != 0 {
            i -= 1;
            let row = self.data[i] & 0x3FF;
            let newly_hit = row & remaining;
            if newly_hit != 0 {
                let mut bits = newly_hit;
                while bits != 0 {
                    let c = bits.trailing_zeros() as usize;
                    h[c] = (i + 1) as u8;
                    bits &= bits - 1;
                }
                remaining &= !newly_hit;
            }
        }
        h
    }

    /// Places a normalized shape and sifts full/empty rows.
    /// Returns `(lines_cleared, garbage_cleared, bottom_removed)`.
    #[inline]
    pub fn place_and_sift(
        &mut self,
        orient: &FastOrient,
        norm_i: i16,
        norm_j: i16,
    ) -> (u8, u8, bool) {
        let i0 = norm_i as usize;
        let j0 = norm_j as u16;
        let new_rows = (i0 + orient.mat_rows as usize).min(MAX_ROWS);
        if new_rows > self.rows as usize {
            for r in (self.rows as usize)..new_rows {
                self.data[r] = EMPTY_ROW;
            }
            self.rows = new_rows as u8;
        }
        for r in 0..(orient.mat_rows as usize) {
            if i0 + r < MAX_ROWS {
                self.data[i0 + r] |= orient.row_masks[r] << j0;
            }
        }

        // Sift rows
        let mut dst = 0usize;
        let mut lines_cleared = 0u8;
        let mut garbage_cleared = 0u8;
        let mut bottom_removed = false;
        let old_garbage = self.garbage_rows as usize;
        let old_rows = self.rows as usize;

        for src in 0..old_rows {
            let row = self.data[src];
            if row == FULL_ROW {
                lines_cleared += 1;
                if src < old_garbage {
                    garbage_cleared += 1;
                }
                if src == 0 {
                    bottom_removed = true;
                }
            } else if row == EMPTY_ROW {
                if src < old_garbage {
                    garbage_cleared += 1;
                }
                if src == 0 {
                    bottom_removed = true;
                }
            } else {
                self.data[dst] = row;
                dst += 1;
            }
        }
        for r in dst..old_rows {
            self.data[r] = EMPTY_ROW;
        }
        self.rows = dst as u8;
        self.garbage_rows = (old_garbage - garbage_cleared as usize) as u8;
        (lines_cleared, garbage_cleared, bottom_removed)
    }

    /// Simulates Cheese garbage spawn at bottom (`min_height=4, max_height=9`).
    /// Returns `false` if board overflows `MAX_ROWS` (topout).
    #[inline]
    pub fn spawn_cheese_rows(&mut self, rem_garbage: u16, lines_cleared: u8, rng: &mut u64) -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        self.spawn_cheese_to(rem_garbage, target_g, rng)
    }

    /// Spawns cheese rows up to `target_g` (capped by remaining unspawned garbage).
    #[inline]
    pub fn spawn_cheese_to(&mut self, rem_garbage: u16, target_g: usize, rng: &mut u64) -> bool {
        let g = self.garbage_rows as usize;
        let unspawned = (rem_garbage as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 {
            return true;
        }
        let old_rows = self.rows as usize;
        if old_rows + num >= MAX_ROWS {
            return false;
        }
        let mut prev_hole = if g > 0 {
            (!self.data[0] & 0x3FF).trailing_zeros() as u16
        } else {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*rng >> 32) as u16) % 10
        };
        if prev_hole >= 10 {
            prev_hole = 0;
        }
        for r in (0..old_rows).rev() {
            self.data[r + num] = self.data[r];
        }
        for r in (0..num).rev() {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let off = (((*rng >> 32) as u16) % 9) + 1;
            let hole = (prev_hole + off) % 10;
            prev_hole = hole;
            self.data[r] = EMPTY_ROW | (0x3FF & !(1u16 << hole));
        }
        self.rows = (old_rows + num) as u8;
        self.garbage_rows = (g + num) as u8;
        true
    }

    /// Returns `true` if the topmost garbage row (`h1`) is an opaque unspawned row.
    #[inline(always)]
    pub fn has_opaque_top_garbage(&self) -> bool {
        let g = self.garbage_rows as usize;
        g > 0 && self.data[g - 1] == OPAQUE_GARBAGE_ROW
    }

    /// Spawns opaque (hole-unknown) unspawned garbage rows up to Cheese `min_height=4 / max_height=9`
    /// without peeking or fabricating fake concrete holes inside beam search.
    #[inline]
    pub fn spawn_fog_rows(&mut self, rem_garbage: u16, lines_cleared: u8) -> bool {
        let target_g: usize = if lines_cleared > 0 { 4 } else { 9 };
        let g = self.garbage_rows as usize;
        let unspawned = (rem_garbage as usize).saturating_sub(g);
        let num = target_g.saturating_sub(g).min(unspawned);
        if num == 0 {
            return true;
        }
        let old_rows = self.rows as usize;
        if old_rows + num >= MAX_ROWS {
            return false;
        }
        for r in (0..old_rows).rev() {
            self.data[r + num] = self.data[r];
        }
        for r in 0..num {
            self.data[r] = OPAQUE_GARBAGE_ROW;
        }
        self.rows = (old_rows + num) as u8;
        self.garbage_rows = (g + num) as u8;
        true
    }

    /// Replaces any `OPAQUE_GARBAGE_ROW` entries in `0..garbage_rows` with valid CRN
    /// consecutive holes (`offset in 1..=9`) before greedy rollouts.
    #[inline]
    pub fn materialize_fog_holes(&mut self, rng: &mut u64) {
        let g = self.garbage_rows as usize;
        if g == 0 {
            return;
        }
        let mut prev_hole: Option<u16> = None;
        for r in (0..g).rev() {
            let row = self.data[r];
            if row == OPAQUE_GARBAGE_ROW {
                let hole = match prev_hole {
                    Some(ph) => {
                        *rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        let off = (((*rng >> 32) as u16) % 9) + 1;
                        (ph + off) % 10
                    }
                    None => {
                        *rng = rng
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((*rng >> 32) as u16) % 10
                    }
                };
                self.data[r] = EMPTY_ROW | (0x3FF & !(1u16 << hole));
                prev_hole = Some(hole);
            } else {
                let h = (!row & 0x3FF).trailing_zeros() as u16;
                if h < 10 {
                    prev_hole = Some(h);
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct FastOrient {
    pub offset: (i16, i16),
    pub mat_idx: u8,
    pub local_mat_idx: u8,
    pub mat_rows: u8,
    pub mat_cols: u16,
    pub row_masks: [u16; 4],
    pub col_bottom: [i16; 4],
    pub col_min: i16,
    pub col_max: i16,
    pub kicks_cw: [(i16, i16); 5],
    pub n_kicks_cw: u8,
    pub kicks_ccw: [(i16, i16); 5],
    pub n_kicks_ccw: u8,
}

pub struct FastShape {
    pub color: Color,
    pub spawn_col: i16,
    pub orients: [FastOrient; 4],
}

pub struct FastShapeTable {
    pub shapes: [FastShape; 7],
}

impl FastShapeTable {
    pub fn new_srs() -> Self {
        Self::from_shape_table(&crate::shape::srs())
    }

    pub fn from_shape_table(shtb: &ShapeTable) -> Self {
        let mut shapes_opt: [Option<FastShape>; 7] = [None, None, None, None, None, None, None];
        for idx in 0..7 {
            let color = idx_to_color(idx);
            let sr: ShapeRef = shtb.shape(color).expect("missing shape");
            let mut local_mats: Vec<usize> = Vec::with_capacity(4);
            let mut orients_opt: [Option<FastOrient>; 4] = [None, None, None, None];

            for (ri, r) in Orientation::iter_all().enumerate() {
                let norm0 = sr.normalize((0, 0, r));
                // Extract mat_idx & offset by inspecting normalized transform
                // Use public ShapeRef methods:
                let vc = sr.valid_cols(r, BOARD_COLS);
                // valid_cols is -(j_off)..=((10 - mat_cols) - j_off)
                // We also can discover mat_rows, row_masks, offset via intersects on an empty/1-cell matrix!
                let mut probe = BasicMatrix::with_cols(BOARD_COLS);
                // Find j_off and mat_cols from valid_cols and intersects:
                // Actually let's probe the 4x4 bounding box by placing a 1-cell block!
                // At tf = (10, 10, r)? Wait: let's place a block at (5, j) for j=0..10 and check intersects((5, j0, r))?
                // Even simpler: let's add a tiny getter in shape.rs OR compute from ShapeRef!
                // Actually, let's check: we are inside the `blockfish` crate (`crate::shape`), so we can check if `shape.rs` fields or helper can be `pub(crate)`.
                let _ = (norm0, vc, &mut probe, &mut local_mats, &mut orients_opt, ri);
            }
            let _ = shapes_opt[idx];
        }
        // Let's use `pub(crate)` helper on ShapeRef in shape.rs so there is zero guesswork!
        Self::build_direct(shtb)
    }

    pub fn srs() -> Self {
        Self::build_direct(&crate::shape::srs())
    }

    fn build_direct(shtb: &ShapeTable) -> Self {
        let mut shapes: [Option<FastShape>; 7] = [None, None, None, None, None, None, None];
        for idx in 0..7 {
            let color = idx_to_color(idx);
            let sr = shtb.shape(color).expect("missing shape");
            let mut local_mats: Vec<usize> = Vec::with_capacity(4);
            let mut orients: [Option<FastOrient>; 4] = [None, None, None, None];

            for (ri, r) in Orientation::iter_all().enumerate() {
                let (i_off, j_off, mat_idx, mat) = sr.raw_orient(r);
                let local_mat_idx = match local_mats.iter().position(|&m| m == mat_idx) {
                    Some(p) => p as u8,
                    None => {
                        local_mats.push(mat_idx);
                        (local_mats.len() - 1) as u8
                    }
                };
                let mat_rows = mat.rows() as u8;
                let mat_cols = mat.cols();
                let mut row_masks = [0u16; 4];
                for row in 0..mat.rows() {
                    let mut mask = 0u16;
                    for col in 0..mat_cols {
                        if mat.get((row, col)) {
                            mask |= 1 << col;
                        }
                    }
                    row_masks[row as usize] = mask;
                }
                let mut col_bottom = [0i16; 4];
                for col in 0..mat_cols {
                    let b = (0..mat.rows())
                        .find(|&row| mat.get((row, col)))
                        .expect("normalized matrix column empty");
                    col_bottom[col as usize] = b as i16;
                }
                let col_min = -j_off;
                let col_max = (BOARD_COLS as i16 - mat_cols as i16) - j_off;

                let mut kicks_cw = [(0i16, 0i16); 5];
                let kcw = sr.kicks(r, r.cw());
                for (k, &off) in kcw.iter().take(5).enumerate() {
                    kicks_cw[k] = off;
                }
                let mut kicks_ccw = [(0i16, 0i16); 5];
                let kccw = sr.kicks(r, r.ccw());
                for (k, &off) in kccw.iter().take(5).enumerate() {
                    kicks_ccw[k] = off;
                }

                orients[ri] = Some(FastOrient {
                    offset: (i_off, j_off),
                    mat_idx: mat_idx as u8,
                    local_mat_idx,
                    mat_rows,
                    mat_cols,
                    row_masks,
                    col_bottom,
                    col_min,
                    col_max,
                    kicks_cw,
                    n_kicks_cw: kcw.len().min(5) as u8,
                    kicks_ccw,
                    n_kicks_ccw: kccw.len().min(5) as u8,
                });
            }

            shapes[idx] = Some(FastShape {
                color,
                spawn_col: sr.spawn_col(),
                orients: [
                    orients[0].unwrap(),
                    orients[1].unwrap(),
                    orients[2].unwrap(),
                    orients[3].unwrap(),
                ],
            });
        }
        Self {
            shapes: [
                shapes[0].take().unwrap(),
                shapes[1].take().unwrap(),
                shapes[2].take().unwrap(),
                shapes[3].take().unwrap(),
                shapes[4].take().unwrap(),
                shapes[5].take().unwrap(),
                shapes[6].take().unwrap(),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FastPlace {
    pub color_idx: u8,
    pub tf: (i16, i16, Orientation),
    pub norm_i: i16,
    pub norm_j: i16,
    pub did_hold: bool,
}

impl FastShape {
    #[inline(always)]
    pub fn intersects(&self, board: &FastBoard, i0: i16, j0: i16, r: usize) -> bool {
        let o = &self.orients[r];
        let i = i0 + o.offset.0;
        let j = j0 + o.offset.1;
        if i < 0 || j < 0 || (j as u16) + o.mat_cols > BOARD_COLS {
            return true;
        }
        let iu = i as usize;
        let brows = board.rows as usize;
        if iu >= brows {
            return false;
        }
        let n = (o.mat_rows as usize).min(brows - iu);
        let ju = j as u16;
        for row in 0..n {
            if (board.data[iu + row] & (o.row_masks[row] << ju)) != 0 {
                return true;
            }
        }
        false
    }

    #[inline(always)]
    pub fn sonic_drop(&self, board: &FastBoard, mut i0: i16, j0: i16, r: usize) -> i16 {
        let o = &self.orients[r];
        let max_i0 = (board.rows as i16) - o.offset.0;
        if i0 > max_i0 {
            i0 = max_i0;
        }
        while !self.intersects(board, i0 - 1, j0, r) {
            i0 -= 1;
        }
        i0
    }

    #[inline(always)]
    pub fn peak_o1(&self, col_heights: &[u8; 10], j0: i16, r: usize) -> i16 {
        let o = &self.orients[r];
        let j = (j0 + o.offset.1) as usize;
        let mut norm_i = 0i16;
        for c in 0..(o.mat_cols as usize) {
            let h = col_heights[j + c] as i16 - o.col_bottom[c];
            if h > norm_i {
                norm_i = h;
            }
        }
        norm_i - o.offset.0
    }
}

pub struct FastPlaceGen {
    queue: Vec<(i16, i16, u8)>,
    places_seen: [[u16; 36]; 4],
    normals_seen: [u64; 36],
}

impl FastPlaceGen {
    pub fn new() -> Self {
        Self {
            queue: Vec::with_capacity(128),
            places_seen: [[0; 36]; 4],
            normals_seen: [0; 36],
        }
    }

    pub fn generate_for_state(
        &mut self,
        fst: &FastShapeTable,
        board: &FastBoard,
        color_nh: Option<u8>,
        color_h: Option<u8>,
        out: &mut Vec<FastPlace>,
    ) {
        out.clear();
        let col_heights = board.col_heights();
        let clear_rows = (board.rows as usize + 6).min(36);
        if let Some(c) = color_nh {
            self.push_shape(fst, board, &col_heights, c, false, clear_rows, out);
        }
        if let Some(c) = color_h {
            if Some(c) != color_nh {
                self.push_shape(fst, board, &col_heights, c, true, clear_rows, out);
            }
        }
    }

    #[inline(always)]
    fn try_push(&mut self, i0: i16, j0: i16, r: usize) {
        let row_key = (i0 + 4) as usize;
        let col_bit = 1u16 << ((j0 + 4) as u32);
        if row_key < 36 {
            if (self.places_seen[r][row_key] & col_bit) != 0 {
                return;
            }
            self.places_seen[r][row_key] |= col_bit;
        }
        self.queue.push((i0, j0, r as u8));
    }

    fn push_shape(
        &mut self,
        fst: &FastShapeTable,
        board: &FastBoard,
        col_heights: &[u8; 10],
        color_idx: u8,
        did_hold: bool,
        clear_rows: usize,
        out: &mut Vec<FastPlace>,
    ) {
        let shape = &fst.shapes[color_idx as usize];
        self.queue.clear();
        for r in 0..4 {
            self.places_seen[r][..clear_rows].fill(0);
        }
        self.normals_seen[..clear_rows].fill(0);

        for r in 0..4usize {
            let o = &shape.orients[r];
            for j0 in o.col_min..=o.col_max {
                let i0 = shape.peak_o1(col_heights, j0, r);
                self.try_push(i0, j0, r);
            }
        }

        while let Some((i0, j0, ru)) = self.queue.pop() {
            let r = ru as usize;

            // Expand Left, Right, CW, CCW
            // Left
            if !shape.intersects(board, i0, j0 - 1, r) {
                let ni = shape.sonic_drop(board, i0, j0 - 1, r);
                self.try_push(ni, j0 - 1, r);
            }
            // Right
            if !shape.intersects(board, i0, j0 + 1, r) {
                let ni = shape.sonic_drop(board, i0, j0 + 1, r);
                self.try_push(ni, j0 + 1, r);
            }
            // CW
            let o = &shape.orients[r];
            let r_cw = (r + 1) & 3;
            for k in 0..(o.n_kicks_cw as usize) {
                let (ioff, joff) = o.kicks_cw[k];
                if !shape.intersects(board, i0 + ioff, j0 + joff, r_cw) {
                    let ni = shape.sonic_drop(board, i0 + ioff, j0 + joff, r_cw);
                    self.try_push(ni, j0 + joff, r_cw);
                    break;
                }
            }
            // CCW
            let r_ccw = (r + 3) & 3;
            for k in 0..(o.n_kicks_ccw as usize) {
                let (ioff, joff) = o.kicks_ccw[k];
                if !shape.intersects(board, i0 + ioff, j0 + joff, r_ccw) {
                    let ni = shape.sonic_drop(board, i0 + ioff, j0 + joff, r_ccw);
                    self.try_push(ni, j0 + joff, r_ccw);
                    break;
                }
            }

            // Check normal deduplication
            let norm_i = i0 + o.offset.0;
            let norm_j = j0 + o.offset.1;
            let niu = norm_i as usize;
            let nbit = 1u64 << (o.local_mat_idx as u32 * 10 + norm_j as u32);
            if niu < 36 {
                if (self.normals_seen[niu] & nbit) != 0 {
                    continue;
                }
                self.normals_seen[niu] |= nbit;
            }

            out.push(FastPlace {
                color_idx,
                tf: (i0, j0, Orientation::from_u8(ru).unwrap()),
                norm_i,
                norm_j,
                did_hold,
            });
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::shape::srs;
    use std::collections::HashSet;

    #[test]
    fn test_fast_places_match_place_finder() {
        let shtb = srs();
        let fst = FastShapeTable::from_shape_table(&shtb);
        let mut pfind = PlaceFinder::new(&shtb);
        let mut fgen = FastPlaceGen::new();
        let mut fplaces = Vec::new();

        // Deterministic LCG to build tricky cheese + overhang boards
        let mut rng: u64 = 123456789;
        let mut next_u64 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng
        };

        for trial in 0..300 {
            let mut mat = BasicMatrix::with_cols(10);
            let h = 4 + (next_u64() % 10) as u16;
            for r in 0..h {
                if r < h - 3 {
                    let hole = (next_u64() % 10) as u16;
                    for c in 0..10 {
                        if c != hole {
                            mat.set((r, c));
                        }
                    }
                } else {
                    for c in 0..10 {
                        if (next_u64() % 10) < 6 {
                            mat.set((r, c));
                        }
                    }
                }
            }
            mat.sift_rows();

            let c1 = idx_to_color((next_u64() % 7) as usize);
            let c2 = idx_to_color((next_u64() % 7) as usize);

            pfind.reset_matrix(&mat);
            pfind.push_shape(c1, false);
            if c2 != c1 {
                pfind.push_shape(c2, true);
            }
            let expected: HashSet<_> = pfind.by_ref().map(|pl| (pl.normal(), pl.did_hold)).collect();

            let fb = FastBoard::from_basic_matrix(&mat, 0);
            fgen.generate_for_state(
                &fst,
                &fb,
                Some(color_to_idx(c1) as u8),
                Some(color_to_idx(c2) as u8),
                &mut fplaces,
            );
            let actual: HashSet<_> = fplaces
                .iter()
                .map(|fp| {
                    let sr = shtb.shape(idx_to_color(fp.color_idx as usize)).unwrap();
                    (sr.normalize(fp.tf), fp.did_hold)
                })
                .collect();

            assert_eq!(actual, expected, "mismatch on trial {}", trial);
            assert_eq!(fplaces.len(), actual.len(), "duplicate normals on trial {}", trial);
        }
    }
}
