use super::{
    fast::{color_to_idx, FastBoard},
    fast_eval::fast_piece_estimate_ex,
};
use crate::{BasicMatrix, Color};

pub const FEAT_DIM: usize = 98;

pub const FEAT_NAMES: [&str; FEAT_DIM] = [
    "rem_g",
    "g_rows",
    "g_zero_nonend",
    "g_le_2",
    "g0_row0_cells",
    "endgame_low",
    "pe_raw",
    "pe_net",
    "pe_nosub_net",
    "stack_cells",
    "stack_cells_row_g",
    "stack_cells_above1",
    "stack_cells_above2",
    "upper_cells",
    "max_stack_h",
    "min_stack_h",
    "mean_stack_h",
    "stack_h_sq",
    "stack_over2_sq",
    "max_h_over13_sq",
    "covered_holes",
    "hole_rows",
    "blocks_over_holes",
    "shallow_holes",
    "h1_blocks",
    "h1_height",
    "h1_empty_blocking",
    "h1_open",
    "h1_chute_1",
    "h1_chute_2plus",
    "h1_edge",
    "h1_one_wall",
    "h1_fit_cnt",
    "hand_fits_h1",
    "bag_fit_frac",
    "no_i_when_i_only",
    "h2_blocks",
    "h2_empty_blocking",
    "h2_extra_empty",
    "h2_open",
    "h1_h2_open_both",
    "h1_h2_adj",
    "h3_blocks",
    "h3_extra_empty",
    "h4_blocks",
    "bumpiness",
    "sharp_diff",
    "max_bump",
    "row_transitions",
    "col_transitions",
    "i_dep_depth",
    "i_dep_cols",
    "i_dep_no_i_avail",
    "two_wells",
    "clean_o_spots",
    "zero_o_spots",
    "o_soon_zero_spot",
    "near_full_9",
    "near_full_8",
    "row_g_9_aligned",
    "row_g_8_aligned",
    "parity_imbalance",
    "hand_I",
    "hand_O",
    "hand_T",
    "hand_S",
    "hand_Z",
    "hand_J",
    "hand_L",
    "bag_size",
    "avail_has_I",
    "avail_has_O",
    "avail_has_T",
    "h1_open_x_stack",
    "h1_cov_x_empty",
    "endgame_x_stack",
    "cells_to_uncover_h1",
    "cells_to_uncover_h12",
    "cells_to_uncover_h123",
    "h2_blocked_row_g",
    "h3_blocked_row_g",
    "row_g_clearable_with_h1",
    "total_o_spots",
    "bottom_covered_holes",
    "stack_excess_cells",
    "open_h2_clean_h1",
    "h2_narrow_1",
    "h2_narrow_2plus",
    "h2_only_row_g_blocked",
    "h2_fit_cnt",
    "h1_ds_clears_rg_open_h2",
    "h12_adj_overhangs",
    "clean_skim_rows",
    "stack_cells_low",
    "max_cliff_around_h1",
    "o_danger",
    "pillars_2plus",
    "stack_above1_excess",
];

pub fn extract_snapshot_features(
    mat: &BasicMatrix,
    g_rows: u16,
    rem_garbage: u16,
    in_hand: Option<Color>,
    out: &mut [f32; FEAT_DIM],
) {
    let fb = FastBoard::from_basic_matrix(mat, g_rows);
    let ih = in_hand.map(|c| color_to_idx(c) as u8);
    extract_features_with_bag(&fb, rem_garbage, ih, 0x7F, out);
}

#[inline]
pub fn extract_features(
    board: &FastBoard,
    rem_garbage: u16,
    in_hand: Option<u8>,
    out: &mut [f32; FEAT_DIM],
) {
    extract_features_with_bag(board, rem_garbage, in_hand, 0x7F, out);
}

#[inline]
pub fn extract_features_with_bag(
    board: &FastBoard,
    rem_garbage: u16,
    in_hand: Option<u8>,
    avail_mask: u8,
    out: &mut [f32; FEAT_DIM],
) {
    extract_features_opt(board, rem_garbage, in_hand, avail_mask, true, out);
}

#[inline]
pub fn extract_features_opt(
    board: &FastBoard,
    rem_garbage: u16,
    in_hand: Option<u8>,
    avail_mask: u8,
    need_nosub: bool,
    out: &mut [f32; FEAT_DIM],
) {
    out.fill(0.0);
    let col_h = board.col_heights();
    let max_h = *col_h.iter().max().unwrap() as i32;
    let g = board.garbage_rows as usize;
    let n = board.rows as usize;

    let pe = fast_piece_estimate_ex(board, false) as i32;
    let pe_nosub = if need_nosub {
        fast_piece_estimate_ex(board, true) as i32
    } else {
        pe
    };
    let g_off = (g as i32 - 1).max(0);
    let pe_net = (pe - g_off).max(0);
    let pe_nosub_net = (pe_nosub - g_off).max(0);

    out[0] = rem_garbage as f32;
    out[1] = g as f32;
    out[2] = if g == 0 && rem_garbage > 0 { 1.0 } else { 0.0 };
    out[3] = if g <= 2 && rem_garbage > 2 { 1.0 } else { 0.0 };
    out[4] = if g == 0 && rem_garbage > 0 && n > 0 {
        (board.data[0] & 0x3FF).count_ones() as f32
    } else {
        0.0
    };
    let endgame_low = if rem_garbage <= 6 { 1.0 } else { 0.0 };
    out[5] = endgame_low;
    out[6] = pe as f32;
    out[7] = pe_net as f32;
    out[8] = pe_nosub_net as f32;

    let mut stack_cells: i32 = 0;
    let stack_cells_row_g: i32 = if n > g {
        (board.data[g] & 0x3FF).count_ones() as i32
    } else {
        0
    };
    let mut stack_cells_above1: i32 = 0;
    let mut stack_cells_above2: i32 = 0;
    let mut upper_cells: i32 = 0;
    let mut covered_holes: i32 = 0;
    let mut hole_rows: i32 = 0;
    let mut blocks_over_holes: i32 = 0;
    let mut shallow_holes: i32 = 0;
    let mut row_transitions: i32 = 0;
    let mut col_transitions: i32 = 0;
    let mut near_full_9: i32 = 0;
    let mut near_full_8: i32 = 0;
    let mut even_parity: i32 = 0;
    let mut odd_parity: i32 = 0;

    let mut bottom_covered_holes: i32 = 0;
    let mut row_empty_cnt = [0i32; 32];
    let mut row_holes_buf = [0u16; 32];
    let mut above_mask: u16 = 0;
    let mut prev_bits: u16 = 0;

    let mut i = n;
    while i > g {
        i -= 1;
        let bits = board.data[i] & 0x3FF;
        let cnt = bits.count_ones() as i32;
        let rel = i - g;
        stack_cells += cnt;
        if rel >= 1 {
            stack_cells_above1 += cnt;
        }
        if rel >= 2 {
            stack_cells_above2 += cnt;
        }
        upper_cells += (rel as i32) * cnt;
        row_empty_cnt[rel] = 10 - cnt;

        let rholes = (!bits & 0x3FF) & above_mask;
        row_holes_buf[rel] = rholes;
        above_mask |= bits;

        if rholes != 0 {
            let hc = rholes.count_ones() as i32;
            hole_rows += 1;
            covered_holes += hc;
            if n - 1 - i <= 1 {
                shallow_holes += hc;
            }
            if rel == 0 {
                bottom_covered_holes += hc;
            }
        }

        let padded = (bits << 1) | 1 | (1 << 11);
        row_transitions += ((padded ^ (padded >> 1)) & 0x7FF).count_ones() as i32;
        col_transitions += (bits ^ prev_bits).count_ones() as i32;
        prev_bits = bits;

        let e_cols = (bits & 0x155).count_ones() as i32;
        let o_cols = cnt - e_cols;
        if (rel & 1) == 0 {
            even_parity += e_cols;
            odd_parity += o_cols;
        } else {
            even_parity += o_cols;
            odd_parity += e_cols;
        }

        if cnt == 9 {
            near_full_9 += 1;
        } else if cnt == 8 {
            near_full_8 += 1;
        }
    }
    if n > g {
        let g_bits = if g > 0 { board.data[g - 1] & 0x3FF } else { 0x3FF };
        col_transitions += (prev_bits ^ g_bits).count_ones() as i32;
    }

    if hole_rows > 0 {
        let mut below_hole: u16 = 0;
        for r in g..n {
            let bits = board.data[r] & 0x3FF;
            blocks_over_holes += (bits & below_hole).count_ones() as i32;
            below_hole |= row_holes_buf[r - g];
        }
    }

    let mut h1_col: Option<usize> = None;
    let mut h2_col: Option<usize> = None;
    let mut h3_col: Option<usize> = None;
    let mut h1_blocks: i32 = 0;
    let mut h1_empty_blocking: i32 = 0;
    let mut h1_height: i32 = 0;
    let mut h2_blocks: i32 = 0;
    let mut h2_empty_blocking: i32 = 0;
    let mut h2_extra_empty: i32 = 0;
    let mut h3_blocks: i32 = 0;
    let mut h3_extra_empty: i32 = 0;
    let mut h4_blocks: i32 = 0;
    let mut req1: usize = 0;
    let mut req2: usize = 0;

    if g >= 1 {
        let c1 = (!board.data[g - 1] & 0x3FF).trailing_zeros() as usize;
        if c1 < 10 {
            h1_col = Some(c1);
            let ch = col_h[c1] as usize;
            if ch > g {
                req1 = ch - g;
                h1_height = req1 as i32;
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
                    h2_empty_blocking += row_empty_cnt[r - g];
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
            h3_col = Some(c3);
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

    let mut cells_to_uncover_h1: i32 = 0;
    let mut cells_to_uncover_h12: i32 = 0;
    let mut cells_to_uncover_h123: i32 = 0;
    let m1 = h1_col.map_or(0u16, |c| 1u16 << c);
    let m12 = m1 | h2_col.map_or(0u16, |c| 1u16 << c);
    let m123 = m12 | h3_col.map_or(0u16, |c| 1u16 << c);
    if m123 != 0 && n > g {
        for r in g..n {
            let bits = board.data[r] & 0x3FF;
            let emp = row_empty_cnt[r - g];
            if (bits & m1) != 0 {
                cells_to_uncover_h1 += emp;
            }
            if (bits & m12) != 0 {
                cells_to_uncover_h12 += emp;
            }
            if (bits & m123) != 0 {
                cells_to_uncover_h123 += emp;
            }
        }
    }
    let h2_blocked_row_g = if n > g {
        h2_col.map_or(0.0, |c2| if ((board.data[g] >> c2) & 1) != 0 { 1.0 } else { 0.0 })
    } else {
        0.0
    };
    let h3_blocked_row_g = if n > g {
        h3_col.map_or(0.0, |c3| if ((board.data[g] >> c3) & 1) != 0 { 1.0 } else { 0.0 })
    } else {
        0.0
    };

    let mut stack_h = [0i32; 10];
    let mut stack_h_sq: i32 = 0;
    let mut stack_over2_sq: i32 = 0;
    let mut max_stack_h: i32 = 0;
    let mut min_stack_h: i32 = 99;
    let mut sum_stack_h: i32 = 0;
    for c in 0..10 {
        let sh = (col_h[c] as i32 - g as i32).max(0);
        stack_h[c] = sh;
        sum_stack_h += sh;
        stack_h_sq += sh * sh;
        if sh > 2 {
            let ex = sh - 2;
            stack_over2_sq += ex * ex;
        }
        if sh > max_stack_h {
            max_stack_h = sh;
        }
        if sh < min_stack_h {
            min_stack_h = sh;
        }
    }

    let mut bumpiness: i32 = 0;
    let mut sharp_diff: i32 = 0;
    let mut max_bump: i32 = 0;
    for c in 0..9 {
        let d = (stack_h[c] - stack_h[c + 1]).abs();
        bumpiness += d;
        if d > max_bump {
            max_bump = d;
        }
        if d >= 2 {
            let ex = d - 1;
            sharp_diff += ex * ex;
        }
    }

    let mut i_dep_depth: i32 = 0;
    let mut i_dep_cols: i32 = 0;
    let mut two_well_cnt: i32 = 0;
    let mut h1_narrow_1: i32 = 0;
    let mut h1_narrow_2plus: i32 = 0;

    for c in 0..10 {
        let left = if c == 0 { 99 } else { stack_h[c - 1] };
        let right = if c == 9 { 99 } else { stack_h[c + 1] };
        let wall_min = left.min(right);
        let well = wall_min - stack_h[c];
        if Some(c) == h1_col && stack_h[c] == 0 {
            if wall_min == 1 {
                h1_narrow_1 = 1;
            } else if wall_min >= 2 {
                h1_narrow_2plus = wall_min - 1;
            }
        } else if well >= 3 {
            i_dep_depth += well - 2;
            i_dep_cols += 1;
        } else if well == 2 {
            two_well_cnt += 1;
        }
    }

    let h1_open = g >= 1 && h1_height == 0;
    let h2_open = g >= 2 && h2_blocks == 0;

    // SRS direct-fit mask & bag readiness for open h1
    let mut h1_one_wall = 0.0f32;
    let mut h1_fit_cnt = 0.0f32;
    let mut hand_fits_h1 = 0.0f32;
    let mut bag_fit_frac = 0.0f32;
    if h1_open {
        if let Some(c1) = h1_col {
            let l1 = if c1 == 0 { 99 } else { stack_h[c1 - 1] };
            let r1 = if c1 == 9 { 99 } else { stack_h[c1 + 1] };
            let l2 = if c1 <= 1 { 99 } else { stack_h[c1 - 2] };
            let r2 = if c1 >= 8 { 99 } else { stack_h[c1 + 2] };
            if l1.max(r1) >= 1 {
                h1_one_wall = 1.0;
            }
            let mut fits: u8 = 1; // I always fits open h1
            if l1 == 0 || r1 == 0 {
                fits |= 1 << 2; // T
            }
            if l1 == 0 {
                fits |= 1 << 3; // S
            }
            if r1 == 0 {
                fits |= 1 << 4; // Z
            }
            if r1 <= 1 || (l1 == 0 && l2 == 0) {
                fits |= 1 << 5; // J
            }
            if l1 <= 1 || (r1 == 0 && r2 == 0) {
                fits |= 1 << 6; // L
            }
            h1_fit_cnt = fits.count_ones() as f32;
            if let Some(p) = in_hand {
                if (p as usize) < 7 && ((fits >> p) & 1) != 0 {
                    hand_fits_h1 = 1.0;
                }
            }
            let denom = avail_mask.count_ones().max(1) as f32;
            bag_fit_frac = (fits & avail_mask).count_ones() as f32 / denom;
        }
    }

    let has_i_avail = in_hand == Some(0) || (avail_mask & 1) != 0;
    let has_o_avail = in_hand == Some(1) || (avail_mask & 2) != 0;
    let no_i_when_i_only = if h1_narrow_2plus > 0 && !has_i_avail { 1.0 } else { 0.0 };

    // Clean flat 2-wide O landing spots (avoiding h1 and uncovered h2)
    let mut clean_o_spots = 0i32;
    let mut total_o_spots = 0i32;
    for c in 0..9 {
        if stack_h[c] == stack_h[c + 1] {
            if Some(c) != h1_col && Some(c + 1) != h1_col {
                total_o_spots += 1;
            }
            if stack_h[c] <= 4
                && Some(c) != h1_col
                && Some(c + 1) != h1_col
                && (g < 2 || h2_blocks > 0 || (Some(c) != h2_col && Some(c + 1) != h2_col))
            {
                clean_o_spots += 1;
            }
        }
    }
    let clean_o_cap = clean_o_spots.min(3) as f32;
    let total_o_cap = total_o_spots.min(4) as f32;
    let zero_o_spots = if clean_o_spots == 0 { 1.0 } else { 0.0 };
    let o_soon_zero_spot = if clean_o_spots == 0 && has_o_avail { 1.0 } else { 0.0 };

    let (row_g_9_aligned, row_g_8_aligned, row_g_clearable_with_h1) = if g >= 1 && n > g {
        if let Some(c1) = h1_col {
            let rg = board.data[g] & 0x3FF;
            let cnt = rg.count_ones();
            let empty_at_c1 = ((rg >> c1) & 1) == 0;
            (
                if empty_at_c1 && cnt == 9 { 1.0 } else { 0.0 },
                if empty_at_c1 && cnt == 8 { 1.0 } else { 0.0 },
                if h1_open && empty_at_c1 && cnt >= 7 { 1.0 } else { 0.0 },
            )
        } else {
            (0.0, 0.0, 0.0)
        }
    } else {
        (0.0, 0.0, 0.0)
    };

    out[9] = stack_cells as f32;
    out[10] = stack_cells_row_g as f32;
    out[11] = stack_cells_above1 as f32;
    out[12] = stack_cells_above2 as f32;
    out[13] = upper_cells as f32;
    out[14] = max_stack_h as f32;
    out[15] = min_stack_h as f32;
    out[16] = sum_stack_h as f32 * 0.1;
    out[17] = stack_h_sq as f32;
    out[18] = stack_over2_sq as f32;
    let ex13 = (max_h - 13).max(0);
    out[19] = (ex13 * ex13) as f32;

    out[20] = covered_holes as f32;
    out[21] = hole_rows as f32;
    out[22] = blocks_over_holes as f32;
    out[23] = shallow_holes as f32;

    out[24] = h1_blocks as f32;
    out[25] = h1_height as f32;
    out[26] = h1_empty_blocking as f32;
    out[27] = if h1_open { 1.0 } else { 0.0 };
    out[28] = h1_narrow_1 as f32;
    out[29] = h1_narrow_2plus as f32;
    out[30] = match h1_col {
        Some(c) if c == 0 || c == 9 => 1.0,
        _ => 0.0,
    };
    out[31] = h1_one_wall;
    out[32] = h1_fit_cnt;
    out[33] = hand_fits_h1;
    out[34] = bag_fit_frac;
    out[35] = no_i_when_i_only;

    out[36] = h2_blocks as f32;
    out[37] = h2_empty_blocking as f32;
    out[38] = h2_extra_empty as f32;
    out[39] = if h2_open { 1.0 } else { 0.0 };
    out[40] = if h1_open && h2_open { 1.0 } else { 0.0 };
    out[41] = match (h1_col, h2_col) {
        (Some(c1), Some(c2)) if (c1 as i32 - c2 as i32).abs() == 1 => 1.0,
        _ => 0.0,
    };
    out[42] = h3_blocks as f32;
    out[43] = h3_extra_empty as f32;
    out[44] = h4_blocks as f32;

    out[45] = bumpiness as f32;
    out[46] = sharp_diff as f32;
    out[47] = max_bump as f32;
    out[48] = row_transitions as f32;
    out[49] = col_transitions as f32;
    out[50] = i_dep_depth as f32;
    out[51] = i_dep_cols as f32;
    out[52] = if !has_i_avail { i_dep_cols as f32 } else { 0.0 };
    out[53] = two_well_cnt as f32;
    out[54] = clean_o_cap;
    out[55] = zero_o_spots;
    out[56] = o_soon_zero_spot;
    out[57] = near_full_9 as f32;
    out[58] = near_full_8 as f32;
    out[59] = row_g_9_aligned;
    out[60] = row_g_8_aligned;
    out[61] = (even_parity - odd_parity).abs() as f32;

    if let Some(idx) = in_hand {
        if (idx as usize) < 7 {
            out[62 + idx as usize] = 1.0;
        }
    }
    out[69] = avail_mask.count_ones() as f32;
    out[70] = if (avail_mask & 1) != 0 { 1.0 } else { 0.0 };
    out[71] = if (avail_mask & 2) != 0 { 1.0 } else { 0.0 };
    out[72] = if (avail_mask & 4) != 0 { 1.0 } else { 0.0 };
    out[73] = if h1_open { stack_cells as f32 } else { 0.0 };
    out[74] = if !h1_open && g >= 1 { h1_empty_blocking as f32 } else { 0.0 };
    out[75] = endgame_low * (stack_cells as f32);
    out[76] = cells_to_uncover_h1 as f32;
    out[77] = cells_to_uncover_h12 as f32;
    out[78] = cells_to_uncover_h123 as f32;
    out[79] = h2_blocked_row_g;
    out[80] = h3_blocked_row_g;
    out[81] = row_g_clearable_with_h1;
    out[82] = total_o_cap;
    out[83] = bottom_covered_holes as f32;
    out[84] = (stack_cells - 12).max(0) as f32;
    out[85] = if h1_open && h2_open && stack_cells_above1 <= 4 { 1.0 } else { 0.0 };

    // Features 86..=97: h2 chute/fit, row-g synergy, cheese-adjacent overhangs, skim cleanliness, cliffs & pillars
    let mut h2_narrow_1 = 0.0f32;
    let mut h2_narrow_2plus = 0.0f32;
    let mut h2_fit_cnt = 0.0f32;
    if h2_open {
        if let Some(c2) = h2_col {
            let l1 = if c2 == 0 { 99 } else { stack_h[c2 - 1] };
            let r1 = if c2 == 9 { 99 } else { stack_h[c2 + 1] };
            let l2 = if c2 <= 1 { 99 } else { stack_h[c2 - 2] };
            let r2 = if c2 >= 8 { 99 } else { stack_h[c2 + 2] };
            let wm = l1.min(r1);
            if wm == 1 {
                h2_narrow_1 = 1.0;
            } else if wm >= 2 {
                h2_narrow_2plus = (wm - 1) as f32;
            }
            let mut fits2: u8 = 1; // I
            if l1 == 0 || r1 == 0 { fits2 |= 1 << 2; }
            if l1 == 0 { fits2 |= 1 << 3; }
            if r1 == 0 { fits2 |= 1 << 4; }
            if r1 <= 1 || (l1 == 0 && l2 == 0) { fits2 |= 1 << 5; }
            if l1 <= 1 || (r1 == 0 && r2 == 0) { fits2 |= 1 << 6; }
            h2_fit_cnt = fits2.count_ones() as f32;
        }
    }
    let h2_only_row_g_blocked = if g >= 2 && h2_blocks == 1 && h2_blocked_row_g > 0.5 { 1.0 } else { 0.0 };
    let h1_ds_clears_rg_open_h2 = if row_g_9_aligned > 0.5 && h2_open { 1.0 } else { 0.0 };

    let mut adj_mask: u16 = 0;
    if let Some(c1) = h1_col {
        adj_mask |= 1u16 << c1;
        if c1 > 0 { adj_mask |= 1u16 << (c1 - 1); }
        if c1 + 1 < 10 { adj_mask |= 1u16 << (c1 + 1); }
    }
    if let Some(c2) = h2_col {
        adj_mask |= 1u16 << c2;
    }
    let mut h12_adj_overhangs: i32 = 0;
    let mut clean_skim_rows: i32 = 0;
    if n > g {
        let mut below_h: u16 = 0;
        for r in g..n {
            let rel = r - g;
            let rh = row_holes_buf[rel];
            if rh != 0 && adj_mask != 0 {
                h12_adj_overhangs += (rh & adj_mask).count_ones() as i32;
            }
            if row_empty_cnt[rel] <= 2 && below_h == 0 {
                clean_skim_rows += 1;
            }
            below_h |= rh;
        }
    }

    let max_cliff_around_h1: i32 = if let Some(c1) = h1_col {
        let ch = stack_h[c1];
        let dl = if c1 > 0 { (stack_h[c1 - 1] - ch).max(0) } else { 0 };
        let dr = if c1 + 1 < 10 { (stack_h[c1 + 1] - ch).max(0) } else { 0 };
        dl.max(dr)
    } else {
        0
    };

    let mut pillars_2plus: i32 = 0;
    for c in 0..10 {
        let dl = if c == 0 { 99 } else { stack_h[c] - stack_h[c - 1] };
        let dr = if c == 9 { 99 } else { stack_h[c] - stack_h[c + 1] };
        if dl >= 2 && dr >= 2 {
            pillars_2plus += 1;
        }
    }

    out[86] = h2_narrow_1;
    out[87] = h2_narrow_2plus;
    out[88] = h2_only_row_g_blocked;
    out[89] = h2_fit_cnt;
    out[90] = h1_ds_clears_rg_open_h2;
    out[91] = h12_adj_overhangs as f32;
    out[92] = clean_skim_rows as f32;
    out[93] = stack_cells.min(8) as f32;
    out[94] = max_cliff_around_h1 as f32;
    out[95] = if has_o_avail && clean_o_spots == 0 && total_o_spots <= 1 { 1.0 } else { 0.0 };
    out[96] = pillars_2plus as f32;
    out[97] = (stack_cells_above1 - 4).max(0) as f32;
}

pub const MLP_HIDDEN: usize = 32;

#[derive(Clone, Debug)]
pub struct LearnedValueModel {
    pub mean: [f32; FEAT_DIM],
    pub inv_std: [f32; FEAT_DIM],
    pub linear_w: [f32; FEAT_DIM],
    pub linear_bias: f32,
    pub w1: [[f32; FEAT_DIM]; MLP_HIDDEN],
    pub b1: [f32; MLP_HIDDEN],
    pub w2: [f32; MLP_HIDDEN],
    pub b2: f32,
    pub mlp_weight: f32,
    // Pre-folded fast inference parameters
    pub fast_w: [f32; FEAT_DIM],
    pub fast_bias: f32,
    pub fast_w1: [[f32; FEAT_DIM]; MLP_HIDDEN],
    pub fast_b1: [f32; MLP_HIDDEN],
    pub fast_ready: bool,
}

impl Default for LearnedValueModel {
    fn default() -> Self {
        Self {
            mean: [0.0; FEAT_DIM],
            inv_std: [1.0; FEAT_DIM],
            linear_w: [0.0; FEAT_DIM],
            linear_bias: 0.0,
            w1: [[0.0; FEAT_DIM]; MLP_HIDDEN],
            b1: [0.0; MLP_HIDDEN],
            w2: [0.0; MLP_HIDDEN],
            b2: 0.0,
            mlp_weight: 0.0,
            fast_w: [0.0; FEAT_DIM],
            fast_bias: 0.0,
            fast_w1: [[0.0; FEAT_DIM]; MLP_HIDDEN],
            fast_b1: [0.0; MLP_HIDDEN],
            fast_ready: false,
        }
    }
}

impl LearnedValueModel {
    /// Pre-folds `(mean, inv_std)` into linear weights and MLP input layer for faster search eval.
    pub fn prepare_fast(&mut self) {
        let mut lb = self.linear_bias;
        for j in 0..FEAT_DIM {
            let fw = self.linear_w[j] * self.inv_std[j];
            self.fast_w[j] = fw;
            lb -= fw * self.mean[j];
        }
        self.fast_bias = lb;

        for h in 0..MLP_HIDDEN {
            let mut hb = self.b1[h];
            for j in 0..FEAT_DIM {
                let fw1 = self.w1[h][j] * self.inv_std[j];
                self.fast_w1[h][j] = fw1;
                hb -= fw1 * self.mean[j];
            }
            self.fast_b1[h] = hb;
        }
        self.fast_ready = true;
    }

    #[inline]
    pub fn predict_features(&self, feats: &[f32; FEAT_DIM]) -> f32 {
        if self.fast_ready {
            let mut pred = self.fast_bias;
            for j in 0..FEAT_DIM {
                pred += self.fast_w[j] * feats[j];
            }
            if self.mlp_weight > 0.0 {
                let mut res = self.b2;
                for h in 0..MLP_HIDDEN {
                    let mut sum = self.fast_b1[h];
                    let row = &self.fast_w1[h];
                    for j in 0..FEAT_DIM {
                        sum += row[j] * feats[j];
                    }
                    if sum > 0.0 {
                        res += self.w2[h] * sum;
                    }
                }
                pred += self.mlp_weight * res;
            }
            return pred;
        }

        let mut x = [0.0f32; FEAT_DIM];
        let mut pred = self.linear_bias;
        for j in 0..FEAT_DIM {
            let z = (feats[j] - self.mean[j]) * self.inv_std[j];
            x[j] = z;
            pred += self.linear_w[j] * z;
        }
        if self.mlp_weight > 0.0 {
            let mut res = self.b2;
            for h in 0..MLP_HIDDEN {
                let mut sum = self.b1[h];
                let row = &self.w1[h];
                for j in 0..FEAT_DIM {
                    sum += row[j] * x[j];
                }
                if sum > 0.0 {
                    res += self.w2[h] * sum;
                }
            }
            pred += self.mlp_weight * res;
        }
        pred
    }

    /// Returns predicted remaining pieces * 100 (centipieces; lower is better).
    #[inline]
    pub fn evaluate_board(&self, board: &FastBoard, rem_garbage: u16, in_hand: Option<u8>) -> i64 {
        self.evaluate_board_with_bag(board, rem_garbage, in_hand, 0x7F)
    }

    #[inline]
    pub fn evaluate_board_with_bag(
        &self,
        board: &FastBoard,
        rem_garbage: u16,
        in_hand: Option<u8>,
        avail_mask: u8,
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

        let need_nosub = self.mlp_weight > 0.0 || !self.fast_ready || self.fast_w[8] != 0.0;
        let mut feats = [0.0f32; FEAT_DIM];
        extract_features_opt(board, rem_garbage, in_hand, avail_mask, need_nosub, &mut feats);
        (self.predict_features(&feats) * 100.0).round() as i64
    }
}
