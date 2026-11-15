use super::fast::FastBoard;

/// Weights for the fast bitwise evaluator.
/// Scores are roughly in centipieces (100 = 1 piece), lower is better.
#[derive(Clone, Debug)]
pub struct EvalWeights {
    pub garbage_row: i64,
    pub piece_estimate: i64,
    pub pe_net_mode: bool,
    pub pe_nosub: bool,
    pub stack_cell: i64,
    pub upper_cell: i64,
    pub stack_h_sq: i64,
    pub stack_over2_sq: i64,
    pub max_stack_h_excess: i64,
    pub covered_hole: i64,
    pub hole_row: i64,
    pub blocks_over_hole: i64,
    pub h1_block: i64,
    pub h1_empty_blocking: i64,
    pub h1_height: i64,
    pub h2_block: i64,
    pub h2_extra_empty: i64,
    pub h3_block: i64,
    pub h3_extra_empty: i64,
    pub h4_block: i64,
    pub h1_narrow_1: i64,
    pub h1_narrow_2plus: i64,
    pub h1_fit_scarcity: i64,
    pub h1_unfit_hand: i64,
    pub zero_o_spot_pen: i64,
    pub bag_no_i_chute_pen: i64,
    pub g0_row0_cell: i64,
    pub open_h1_bonus: i64,
    pub combo2_bonus: i64,
    pub h1_h2_adj: i64,
    pub bumpiness: i64,
    pub sharp_diff: i64,
    pub i_dep: i64,
    pub two_well: i64,
    pub i_dep_multi: i64,
    pub i_dep_chute: i64,
    pub row_transition: i64,
    pub near_full_9_bonus: i64,
    pub near_full_8_bonus: i64,
    pub row_g_aligned_bonus: i64,
    pub height_limit: i64,
    pub height_excess: i64,
    pub center_height_excess: i64,
    pub hold_bonus: [i64; 7], // I, O, T, S, Z, J, L
}

impl Default for EvalWeights {
    fn default() -> Self {
        Self {
            garbage_row: 260,
            piece_estimate: 0,
            pe_net_mode: false,
            pe_nosub: false,
            stack_cell: 4,
            upper_cell: 4,
            stack_h_sq: 1,
            stack_over2_sq: 10,
            max_stack_h_excess: 18,
            covered_hole: 95,
            hole_row: 45,
            blocks_over_hole: 30,
            h1_block: 60,
            h1_empty_blocking: 18,
            h1_height: 15,
            h2_block: 36,
            h2_extra_empty: 14,
            h3_block: 18,
            h3_extra_empty: 7,
            h4_block: 8,
            h1_narrow_1: 18,
            h1_narrow_2plus: 24,
            h1_fit_scarcity: 0,
            h1_unfit_hand: 0,
            zero_o_spot_pen: 0,
            bag_no_i_chute_pen: 0,
            g0_row0_cell: 0,
            open_h1_bonus: 30,
            combo2_bonus: 28,
            h1_h2_adj: 0,
            bumpiness: 4,
            sharp_diff: 6,
            i_dep: 50,
            two_well: 14,
            i_dep_multi: 70,
            i_dep_chute: 15,
            row_transition: 3,
            near_full_9_bonus: 16,
            near_full_8_bonus: 5,
            row_g_aligned_bonus: 0,
            height_limit: 13,
            height_excess: 70,
            center_height_excess: 140,
            // Prefer keeping versatile pieces (I, T, J, L) over O/S/Z in hold/hand
            hold_bonus: [-16, 10, -10, 2, 2, -4, -4],
        }
    }
}

const fn build_row_pieces_lut() -> [u8; 1024] {
    let mut lut = [0u8; 1024];
    let mut mask = 0usize;
    while mask < 1024 {
        let mut total = 0u8;
        let mut run = 0u8;
        let mut c = 0usize;
        while c < 10 {
            if ((mask >> c) & 1) != 0 {
                run += 1;
            } else if run > 0 {
                total += (run + 3) / 4;
                run = 0;
            }
            c += 1;
        }
        if run > 0 {
            total += (run + 3) / 4;
        }
        lut[mask] = total;
        mask += 1;
    }
    lut
}

static ROW_PIECES_LUT: [u8; 1024] = build_row_pieces_lut();

/// Zero-allocation bitwise equivalent of Mystery's residue-based `piece_estimate`.
#[inline]
pub fn fast_piece_estimate(board: &FastBoard) -> u16 {
    fast_piece_estimate_ex(board, false)
}

#[inline]
pub fn fast_piece_estimate_ex(board: &FastBoard, nosub: bool) -> u16 {
    let mut rows = [0u16; 32];
    let mut n = board.rows as usize;
    for i in 0..n {
        rows[i] = board.data[i] & 0x3FF;
    }

    let mut pieces: u16 = 0;
    let mut depth: u16 = 0;

    loop {
        let mut res_end = [0u8; 10];
        let mut seen_filled: u16 = 0;
        let mut gap_above: u16 = 0;
        let mut found: Option<(usize, usize)> = None;

        let mut i = n;
        while i > 0 {
            i -= 1;
            let bits = rows[i];
            let covered = bits & gap_above;
            if covered != 0 {
                let j = covered.trailing_zeros() as usize;
                found = Some((i + 1, res_end[j] as usize));
                break;
            }
            let mut new_cols = bits & !seen_filled;
            if new_cols != 0 {
                seen_filled |= new_cols;
                let end_r = (i + 1) as u8;
                while new_cols != 0 {
                    let j = new_cols.trailing_zeros() as usize;
                    res_end[j] = end_r;
                    new_cols &= new_cols - 1;
                }
            }
            gap_above |= seen_filled & !bits;
        }
        if found.is_none() && gap_above != 0 {
            let j = gap_above.trailing_zeros() as usize;
            found = Some((0, res_end[j] as usize));
        }

        let (r_lo, r_hi) = match found {
            Some(pair) => pair,
            None => break,
        };

        let start_row = r_lo + 1;

        let mut pieces_to_fill: u16 = 0;
        if start_row < r_hi {
            let len = r_hi - start_row;
            if len == 1 {
                let e = (!rows[start_row] & 0x3FF) as usize;
                pieces_to_fill = ROW_PIECES_LUT[e] as u16;
            } else {
                let mut empty = [0u16; 32];
                let mut any_empty = false;
                let mut v_adj = false;
                let mut prev_e = 0u16;
                for k in 0..len {
                    let e = !rows[start_row + k] & 0x3FF;
                    empty[k] = e;
                    if e != 0 {
                        any_empty = true;
                        if (e & prev_e) != 0 {
                            v_adj = true;
                        }
                    }
                    prev_e = e;
                }
                if any_empty {
                    if !v_adj {
                        for k in 0..len {
                            pieces_to_fill += ROW_PIECES_LUT[empty[k] as usize] as u16;
                        }
                    } else {
                        for k in 0..len {
                            while empty[k] != 0 {
                                let seed = empty[k] & empty[k].wrapping_neg();
                                let mut comp = [0u16; 32];
                                comp[k] = seed;
                                let mut r_min = k;
                                let mut r_max = k;
                                loop {
                                    let mut changed = false;
                                    let lo = r_min;
                                    let hi = r_max;
                                    for r in lo..=hi {
                                        let cur = comp[r];
                                        if cur != 0 {
                                            let mut expanded = cur;
                                            loop {
                                                let next = (expanded | (expanded << 1) | (expanded >> 1))
                                                    & empty[r];
                                                if next == expanded {
                                                    break;
                                                }
                                                expanded = next;
                                            }
                                            if expanded != comp[r] {
                                                comp[r] = expanded;
                                                changed = true;
                                            }
                                            if r + 1 < len {
                                                let down = expanded & empty[r + 1];
                                                if (comp[r + 1] | down) != comp[r + 1] {
                                                    comp[r + 1] |= down;
                                                    if r + 1 > r_max {
                                                        r_max = r + 1;
                                                    }
                                                    changed = true;
                                                }
                                            }
                                            if r > 0 {
                                                let up = expanded & empty[r - 1];
                                                if (comp[r - 1] | up) != comp[r - 1] {
                                                    comp[r - 1] |= up;
                                                    if r - 1 < r_min {
                                                        r_min = r - 1;
                                                    }
                                                    changed = true;
                                                }
                                            }
                                        }
                                    }
                                    if !changed {
                                        break;
                                    }
                                }
                                let mut area: u16 = 0;
                                for r in r_min..=r_max {
                                    if comp[r] != 0 {
                                        area += comp[r].count_ones() as u16;
                                        empty[r] &= !comp[r];
                                    }
                                }
                                pieces_to_fill += (area + 3) / 4;
                            }
                        }
                    }
                }
            }
        }

        // Remove rows start_row..r_hi
        if r_hi > start_row {
            let del = r_hi - start_row;
            rows.copy_within(r_hi..n, start_row);
            n -= del;
        }

        depth += 1;
        let sub = if nosub { 1 } else { depth };
        pieces += pieces_to_fill.saturating_sub(sub);
        pieces += 1;
    }

    pieces
}

/// Evaluates `board` with `rem_garbage` total garbage lines left in the race.
/// Lower score is better. Returns a very negative value if `rem_garbage == 0`.
#[inline]
pub fn eval_fast(
    board: &FastBoard,
    rem_garbage: u16,
    in_hand: Option<u8>,
    w: &EvalWeights,
) -> i64 {
    eval_fast_with_bag(board, rem_garbage, in_hand, 0x7F, w)
}

#[inline]
pub fn eval_fast_with_bag(
    board: &FastBoard,
    rem_garbage: u16,
    in_hand: Option<u8>,
    avail_mask: u8,
    w: &EvalWeights,
) -> i64 {
    let col_h = board.col_heights();
    let max_h = *col_h.iter().max().unwrap() as i64;
    let center_max_h = col_h[3..=6].iter().copied().max().unwrap() as i64;
    if max_h >= 20 || center_max_h >= 19 {
        return 5_000_000 + max_h * 10_000 + center_max_h * 10_000;
    }

    if rem_garbage == 0 {
        return -10_000_000 + (board.rows as i64) * 5;
    }

    let g = board.garbage_rows as usize;
    let n = board.rows as usize;
    let pe = if w.piece_estimate != 0 {
        let raw = fast_piece_estimate_ex(board, w.pe_nosub) as i64;
        if w.pe_net_mode {
            (raw - (g as i64 - 1).max(0)).max(0)
        } else {
            raw
        }
    } else {
        0
    };

    let mut stack_cells: i64 = 0;
    let mut upper_cells: i64 = 0;
    let mut covered_holes: i64 = 0;
    let mut hole_rows: i64 = 0;
    let mut blocks_over_holes: i64 = 0;
    let mut row_transitions: i64 = 0;
    let mut near_full_bonus: i64 = 0;

    let mut row_empty_cnt = [0i64; 32];
    let mut row_holes_buf = [0u16; 32];
    let mut above_mask: u16 = 0;

    let mut i = n;
    while i > g {
        i -= 1;
        let bits = board.data[i] & 0x3FF;
        let cnt = bits.count_ones() as i64;
        let rel = i - g;
        stack_cells += cnt;
        upper_cells += (rel as i64) * cnt;
        row_empty_cnt[rel] = 10 - cnt;

        let rholes = (!bits & 0x3FF) & above_mask;
        row_holes_buf[rel] = rholes;
        above_mask |= bits;

        if rholes != 0 {
            hole_rows += 1;
            covered_holes += rholes.count_ones() as i64;
        }

        // Row transitions including left/right walls
        let padded = (bits << 1) | 1 | (1 << 11);
        row_transitions += ((padded ^ (padded >> 1)) & 0x7FF).count_ones() as i64;

        if cnt == 9 {
            near_full_bonus += w.near_full_9_bonus;
        } else if cnt == 8 {
            near_full_bonus += w.near_full_8_bonus;
        }
    }

    if hole_rows > 0 {
        let mut below_hole: u16 = 0;
        for r in g..n {
            let bits = board.data[r] & 0x3FF;
            blocks_over_holes += (bits & below_hole).count_ones() as i64;
            below_hole |= row_holes_buf[r - g];
        }
    }

    // Analyze top 1..4 garbage holes & non-overlapping empty blocking cells
    let mut h1_col: Option<usize> = None;
    let mut h2_col: Option<usize> = None;
    let mut h1_blocks: i64 = 0;
    let mut h1_empty_blocking: i64 = 0;
    let mut h1_height: i64 = 0;
    let mut h2_blocks: i64 = 0;
    let mut h2_extra_empty: i64 = 0;
    let mut h3_blocks: i64 = 0;
    let mut h3_extra_empty: i64 = 0;
    let mut h4_blocks: i64 = 0;

    let mut req1: usize = 0;
    let mut req2: usize = 0;

    if g >= 1 {
        let c1 = (!board.data[g - 1] & 0x3FF).trailing_zeros() as usize;
        if c1 < 10 {
            h1_col = Some(c1);
            let ch = col_h[c1] as usize;
            if ch > g {
                req1 = ch - g;
                h1_height = req1 as i64;
                for r in g..ch {
                    if (board.data[r] >> c1) & 1 != 0 {
                        h1_blocks += 1;
                    }
                    h1_empty_blocking += row_empty_cnt[r - g];
                }
            }
        }
    }
    if g >= 2 {
        let c2 = (!board.data[g - 2] & 0x3FF).trailing_zeros() as usize;
        if c2 < 10 {
            h2_col = Some(c2);
            let ch = col_h[c2] as usize;
            if ch > g {
                req2 = ch - g;
                for r in g..ch {
                    if (board.data[r] >> c2) & 1 != 0 {
                        h2_blocks += 1;
                    }
                }
                if req2 > req1 {
                    for rel in req1..req2 {
                        h2_extra_empty += row_empty_cnt[rel];
                    }
                }
            }
        }
    }
    if g >= 3 {
        let c3 = (!board.data[g - 3] & 0x3FF).trailing_zeros() as usize;
        if c3 < 10 {
            let ch = col_h[c3] as usize;
            if ch > g {
                let req3 = ch - g;
                for r in g..ch {
                    if (board.data[r] >> c3) & 1 != 0 {
                        h3_blocks += 1;
                    }
                }
                let req12 = req1.max(req2);
                if req3 > req12 {
                    for rel in req12..req3 {
                        h3_extra_empty += row_empty_cnt[rel];
                    }
                }
            }
        }
    }
    if g >= 4 {
        let c4 = (!board.data[g - 4] & 0x3FF).trailing_zeros() as usize;
        if c4 < 10 {
            let ch = col_h[c4] as usize;
            for r in g..ch {
                if (board.data[r] >> c4) & 1 != 0 {
                    h4_blocks += 1;
                }
            }
        }
    }

    // Surface heights above garbage
    let mut stack_h = [0i64; 10];
    let mut stack_h_sq: i64 = 0;
    let mut stack_over2_sq: i64 = 0;
    let mut max_stack_h: i64 = 0;
    for c in 0..10 {
        let sh = (col_h[c] as i64 - g as i64).max(0);
        stack_h[c] = sh;
        stack_h_sq += sh * sh;
        if sh > 2 {
            let ex = sh - 2;
            stack_over2_sq += ex * ex;
        }
        if sh > max_stack_h {
            max_stack_h = sh;
        }
    }

    let mut bumpiness: i64 = 0;
    let mut sharp_diff: i64 = 0;
    for c in 0..9 {
        let d = (stack_h[c] - stack_h[c + 1]).abs();
        bumpiness += d;
        if d >= 2 {
            let ex = d - 1;
            sharp_diff += ex * ex;
        }
    }

    // Wells / I-dependencies + h1 open narrow-chute penalty
    let mut i_dep_depth: i64 = 0;
    let mut two_well_cnt: i64 = 0;
    let mut i_dep_cols: i64 = 0;
    let mut chute_excess: i64 = 0;
    let mut h1_narrow_1: i64 = 0;
    let mut h1_narrow_2plus: i64 = 0;

    for c in 0..10 {
        let left = if c == 0 { 99 } else { stack_h[c - 1] };
        let right = if c == 9 { 99 } else { stack_h[c + 1] };
        let wall_min = left.min(right);
        let well = wall_min - stack_h[c];
        if Some(c) == h1_col && stack_h[c] == 0 {
            if wall_min == 1 {
                h1_narrow_1 = 1;
            } else if wall_min >= 2 {
                // Slightly softer if I-piece is immediately in hand
                let base = wall_min - 1;
                h1_narrow_2plus = if in_hand == Some(0) { base.max(1) - 1 } else { base };
            }
            if well >= 4 {
                chute_excess += well - 3;
            }
        } else if well >= 3 {
            i_dep_depth += well - 2;
            i_dep_cols += 1;
        } else if well == 2 {
            two_well_cnt += 1;
        }
    }
    let multi_i_dep = if i_dep_cols > 1 { i_dep_cols - 1 } else { 0 };

    let max_stack_pen = if max_stack_h > 4 {
        let ex = max_stack_h - 4;
        ex * ex * w.max_stack_h_excess
    } else {
        0
    };

    let height_pen = if max_h > w.height_limit {
        let ex = max_h - w.height_limit;
        ex * ex * w.height_excess
    } else {
        0
    };
    let center_limit = w.height_limit + 1;
    let center_pen = if center_max_h > center_limit {
        let ex = center_max_h - center_limit;
        ex * ex * w.center_height_excess
    } else {
        0
    };

    let open_bonus = if g >= 1 && h1_height == 0 {
        let mut b = w.open_h1_bonus;
        if g >= 2 && h2_blocks == 0 {
            b += w.combo2_bonus;
        }
        b
    } else {
        0
    };

    let row_g_aligned = if w.row_g_aligned_bonus != 0 && g >= 1 && n > g {
        if let Some(c1) = h1_col {
            let rg = board.data[g] & 0x3FF;
            let cnt = rg.count_ones();
            if cnt >= 8 && ((rg >> c1) & 1) == 0 {
                if cnt == 9 {
                    w.row_g_aligned_bonus * 2
                } else {
                    w.row_g_aligned_bonus
                }
            } else {
                0
            }
        } else {
            0
        }
    } else {
        0
    };

    let hold_adj = match in_hand {
        Some(idx) if (idx as usize) < 7 => w.hold_bonus[idx as usize],
        _ => 0,
    };

    let h1_h2_adj_pen = match (h1_col, h2_col) {
        (Some(c1), Some(c2)) if c1.abs_diff(c2) == 1 => w.h1_h2_adj,
        _ => 0,
    };

    let (h1_scarcity_pen, h1_unfit_pen) = if (w.h1_fit_scarcity != 0 || w.h1_unfit_hand != 0)
        && g >= 1
        && h1_height == 0
    {
        if let Some(c1) = h1_col {
            let l1 = if c1 == 0 { 99 } else { stack_h[c1 - 1] };
            let r1 = if c1 == 9 { 99 } else { stack_h[c1 + 1] };
            let l2 = if c1 <= 1 { 99 } else { stack_h[c1 - 2] };
            let r2 = if c1 >= 8 { 99 } else { stack_h[c1 + 2] };
            // Exact SRS direct-fit mask over bits: 0=I, 1=O, 2=T, 3=S, 4=Z, 5=J, 6=L
            let mut fits: u8 = 1; // I always fits open h1
            if l1 == 0 || r1 == 0 {
                fits |= 1 << 2; // T
            }
            if l1 == 0 {
                fits |= 1 << 3; // S vertical hangs left
            }
            if r1 == 0 {
                fits |= 1 << 4; // Z vertical hangs right
            }
            if r1 <= 1 || (l1 == 0 && l2 == 0) {
                fits |= 1 << 5; // J (CW hangs right at g+1; 180 hangs 2 left at g)
            }
            if l1 <= 1 || (r1 == 0 && r2 == 0) {
                fits |= 1 << 6; // L (CCW hangs left at g+1; 180 hangs 2 right at g)
            }
            let fit_cnt = fits.count_ones() as i64;
            let scarcity = (6 - fit_cnt).max(0) * w.h1_fit_scarcity;
            let unfit = match in_hand {
                Some(p) if (p as usize) < 7 && p != 1 && ((fits >> p) & 1) == 0 => {
                    // Exempt if row g already has >= 9 cells (one placement finishes row g)
                    let row_g_filled = if n > g {
                        (board.data[g] & 0x3FF).count_ones()
                    } else {
                        0
                    };
                    if row_g_filled >= 9 { 0 } else { w.h1_unfit_hand }
                }
                _ => 0,
            };
            (scarcity, unfit)
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    let bag_no_i_pen = if w.bag_no_i_chute_pen != 0 && h1_narrow_2plus > 0 {
        let has_i = in_hand == Some(0) || (avail_mask & 1) != 0;
        if !has_i { h1_narrow_2plus * w.bag_no_i_chute_pen } else { 0 }
    } else {
        0
    };

    let zero_o_pen = if w.zero_o_spot_pen != 0 {
        let mut has_o_spot = false;
        for c in 0..9 {
            if stack_h[c] == stack_h[c + 1]
                && stack_h[c] <= 4
                && Some(c) != h1_col
                && Some(c + 1) != h1_col
                && (g < 2 || h2_blocks > 0 || (Some(c) != h2_col && Some(c + 1) != h2_col))
            {
                has_o_spot = true;
                break;
            }
        }
        if !has_o_spot {
            let o_soon = in_hand == Some(1) || (avail_mask & 2) != 0;
            if o_soon { w.zero_o_spot_pen * 2 } else { w.zero_o_spot_pen }
        } else {
            0
        }
    } else {
        0
    };

    let g0_row0_pen = if w.g0_row0_cell != 0 && g == 0 && rem_garbage > 0 && n > 0 {
        ((board.data[0] & 0x3FF).count_ones() as i64) * w.g0_row0_cell
    } else {
        0
    };

    w.garbage_row * (rem_garbage as i64)
        + w.piece_estimate * pe
        + w.stack_cell * stack_cells
        + w.upper_cell * upper_cells
        + w.stack_h_sq * stack_h_sq
        + w.stack_over2_sq * stack_over2_sq
        + max_stack_pen
        + w.covered_hole * covered_holes
        + w.hole_row * hole_rows
        + w.blocks_over_hole * blocks_over_holes
        + w.h1_block * h1_blocks
        + w.h1_empty_blocking * h1_empty_blocking
        + w.h1_height * h1_height
        + w.h2_block * h2_blocks
        + w.h2_extra_empty * h2_extra_empty
        + w.h3_block * h3_blocks
        + w.h3_extra_empty * h3_extra_empty
        + w.h4_block * h4_blocks
        + w.h1_narrow_1 * h1_narrow_1
        + w.h1_narrow_2plus * h1_narrow_2plus
        + h1_scarcity_pen
        + h1_unfit_pen
        + bag_no_i_pen
        + zero_o_pen
        + g0_row0_pen
        + h1_h2_adj_pen
        + w.bumpiness * bumpiness
        + w.sharp_diff * sharp_diff
        + w.i_dep * i_dep_depth
        + w.two_well * two_well_cnt
        + w.i_dep_multi * multi_i_dep
        + w.i_dep_chute * chute_excess
        + w.row_transition * row_transitions
        + height_pen
        + center_pen
        + hold_adj
        - near_full_bonus
        - open_bonus
        - row_g_aligned
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{ai::eval::eval as legacy_eval, BasicMatrix};

    #[test]
    fn test_fast_piece_estimate_matches_legacy() {
        let mut rng: u64 = 987654321;
        let mut next_u64 = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            rng
        };

        for trial in 0..400 {
            let mut mat = BasicMatrix::with_cols(10);
            let h = 2 + (next_u64() % 12) as u16;
            for r in 0..h {
                if r < h.saturating_sub(3) {
                    let hole = (next_u64() % 10) as u16;
                    for c in 0..10 {
                        if c != hole {
                            mat.set((r, c));
                        }
                    }
                } else {
                    for c in 0..10 {
                        if (next_u64() % 10) < 5 {
                            mat.set((r, c));
                        }
                    }
                }
            }
            mat.sift_rows();
            let fb = FastBoard::from_basic_matrix(&mat, 0);
            let expected = legacy_eval(&mat).piece_estimate;
            let actual = fast_piece_estimate(&fb);
            assert_eq!(actual, expected, "mismatch on trial {}", trial);
        }
    }
}
