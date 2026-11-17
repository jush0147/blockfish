use super::{fast::FastBoard, fast_eval::fast_piece_estimate_ex};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

pub const SPARSE_DIM: usize = 186;
pub const DENSE_DIM: usize = 72;
pub const SPATIAL_H1: usize = 64;
pub const SPATIAL_H2: usize = 32;

/// Extracts sparse binary indices (`0..SPARSE_DIM`) and dense tactical scalars (`DENSE_DIM`)
/// from a legal observable `FastBoard` state. Returns the number of active sparse indices written
/// into `active_bits`.
#[inline]
pub fn extract_spatial_features(
    board: &FastBoard,
    rem_garbage: u16,
    in_hand: Option<u8>,
    avail_mask: u8,
    active_bits: &mut [u16; SPARSE_DIM],
    dense: &mut [f32; DENSE_DIM],
) -> usize {
    dense.fill(0.0);
    let mut n_act = 0usize;

    let col_h = board.col_heights();
    let max_h = *col_h.iter().max().unwrap() as i32;
    let g = board.garbage_rows as usize;
    let n = board.rows as usize;

    let pe = fast_piece_estimate_ex(board, false) as i32;
    let g_off = (g as i32 - 1).max(0);
    let pe_net = (pe - g_off).max(0);

    // 1. Absolute 8x10 spatial window above garbage (`0..80`)
    for rel in 0..8usize {
        let r = g + rel;
        if r >= n {
            break;
        }
        let mut bits = board.data[r] & 0x3FF;
        let base = (rel * 10) as u16;
        while bits != 0 {
            let c = bits.trailing_zeros() as u16;
            active_bits[n_act] = base + c;
            n_act += 1;
            bits &= bits - 1;
        }
    }

    let mut stack_h = [0i32; 10];
    let mut stack_h_sq: i32 = 0;
    let mut stack_over2_sq: i32 = 0;
    let mut max_stack_h: i32 = 0;
    let mut min_stack_h: i32 = 99;
    for c in 0..10usize {
        let sh = (col_h[c] as i32 - g as i32).max(0);
        stack_h[c] = sh;
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
        // 2. Column overflow bits (`80..90`)
        if sh > 8 {
            active_bits[n_act] = (80 + c) as u16;
            n_act += 1;
        }
    }

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
    let mut bottom_covered_holes: i32 = 0;
    let mut row_transitions: i32 = 0;
    let mut col_transitions: i32 = 0;
    let mut near_full_9: i32 = 0;
    let mut near_full_8: i32 = 0;
    let mut even_parity: i32 = 0;
    let mut odd_parity: i32 = 0;

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
            active_bits[n_act] = (90 + c1) as u16;
            n_act += 1;
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
            active_bits[n_act] = (100 + c2) as u16;
            n_act += 1;
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
            active_bits[n_act] = (110 + c3) as u16;
            n_act += 1;
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

    // 4. Translation-invariant 5x5 window around h1 (`120..145`) with walls=1
    if let Some(c1) = h1_col {
        for rel in 0..5usize {
            let row_bits = if g + rel < n {
                board.data[g + rel] & 0x3FF
            } else {
                0
            };
            let base = (120 + rel * 5) as u16;
            for dc in 0..5usize {
                let cc = c1 as i32 + (dc as i32 - 2);
                let filled = if cc < 0 || cc >= 10 {
                    true
                } else {
                    ((row_bits >> (cc as usize)) & 1) != 0
                };
                if filled {
                    active_bits[n_act] = base + dc as u16;
                    n_act += 1;
                }
            }
        }
    }

    // 5. Translation-invariant 5x3 window around h2 (`145..160`) with walls=1
    if let Some(c2) = h2_col {
        for rel in 0..5usize {
            let row_bits = if g + rel < n {
                board.data[g + rel] & 0x3FF
            } else {
                0
            };
            let base = (145 + rel * 3) as u16;
            for dc in 0..3usize {
                let cc = c2 as i32 + (dc as i32 - 1);
                let filled = if cc < 0 || cc >= 10 {
                    true
                } else {
                    ((row_bits >> (cc as usize)) & 1) != 0
                };
                if filled {
                    active_bits[n_act] = base + dc as u16;
                    n_act += 1;
                }
            }
        }
    }

    // 6. Relative h1->h2 & edge bits (`160..166`)
    if let Some(c1) = h1_col {
        if c1 == 0 || c1 == 9 {
            active_bits[n_act] = 160;
            n_act += 1;
        }
        if let Some(c2) = h2_col {
            if c2 == 0 || c2 == 9 {
                active_bits[n_act] = 161;
                n_act += 1;
            }
            let d = (c1 as i32 - c2 as i32).abs();
            if d == 1 {
                active_bits[n_act] = 162;
                n_act += 1;
            } else if d == 2 {
                active_bits[n_act] = 163;
                n_act += 1;
            } else {
                active_bits[n_act] = 164;
                n_act += 1;
            }
            if c2 < c1 {
                active_bits[n_act] = 165;
                n_act += 1;
            }
        }
    }

    // 7. Piece in hand (`166..173`) & available bag (`173..180`)
    if let Some(p) = in_hand {
        if (p as usize) < 7 {
            active_bits[n_act] = 166 + p as u16;
            n_act += 1;
        }
    }
    let mut am = avail_mask & 0x7F;
    while am != 0 {
        let p = am.trailing_zeros() as u16;
        active_bits[n_act] = 173 + p;
        n_act += 1;
        am &= am - 1;
    }

    let h1_open = g >= 1 && h1_height == 0;
    let h2_open = g >= 2 && h2_blocks == 0;

    // 8. Regime bits (`180..186`)
    if g == 0 && rem_garbage > 0 {
        active_bits[n_act] = 180;
        n_act += 1;
    }
    if g >= 5 {
        active_bits[n_act] = 181;
        n_act += 1;
    }
    if rem_garbage <= 9 && (rem_garbage as usize) == g {
        active_bits[n_act] = 182;
        n_act += 1;
    }
    if h1_open {
        active_bits[n_act] = 183;
        n_act += 1;
    }
    if h2_open {
        active_bits[n_act] = 184;
        n_act += 1;
    }
    if h1_open && h2_open {
        active_bits[n_act] = 185;
        n_act += 1;
    }

    // Tactical dense features
    let mut cells_to_uncover_h1: i32 = 0;
    let mut cells_to_uncover_h12: i32 = 0;
    let m1 = h1_col.map_or(0u16, |c| 1u16 << c);
    let m12 = m1 | h2_col.map_or(0u16, |c| 1u16 << c);
    if m12 != 0 && n > g {
        for r in g..n {
            let bits = board.data[r] & 0x3FF;
            let emp = row_empty_cnt[r - g];
            if (bits & m1) != 0 {
                cells_to_uncover_h1 += emp;
            }
            if (bits & m12) != 0 {
                cells_to_uncover_h12 += emp;
            }
        }
    }

    let mut bumpiness: i32 = 0;
    let mut sharp_diff: i32 = 0;
    for c in 0..9usize {
        let d = (stack_h[c] - stack_h[c + 1]).abs();
        bumpiness += d;
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

    for c in 0..10usize {
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

    let mut h1_fit_cnt = 0.0f32;
    let mut hand_fits_h1 = 0.0f32;
    let mut bag_fit_frac = 0.0f32;
    if h1_open {
        if let Some(c1) = h1_col {
            let l1 = if c1 == 0 { 99 } else { stack_h[c1 - 1] };
            let r1 = if c1 == 9 { 99 } else { stack_h[c1 + 1] };
            let l2 = if c1 <= 1 { 99 } else { stack_h[c1 - 2] };
            let r2 = if c1 >= 8 { 99 } else { stack_h[c1 + 2] };
            let mut fits: u8 = 1; // I
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
            let mut fits2: u8 = 1;
            if l1 == 0 || r1 == 0 {
                fits2 |= 1 << 2;
            }
            if l1 == 0 {
                fits2 |= 1 << 3;
            }
            if r1 == 0 {
                fits2 |= 1 << 4;
            }
            if r1 <= 1 || (l1 == 0 && l2 == 0) {
                fits2 |= 1 << 5;
            }
            if l1 <= 1 || (r1 == 0 && r2 == 0) {
                fits2 |= 1 << 6;
            }
            h2_fit_cnt = fits2.count_ones() as f32;
        }
    }

    let has_o_avail = in_hand == Some(1) || (avail_mask & 2) != 0;
    let mut clean_o_spots = 0i32;
    let mut total_o_spots = 0i32;
    for c in 0..9usize {
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

    let mut adj_mask: u16 = 0;
    if let Some(c1) = h1_col {
        adj_mask |= 1u16 << c1;
        if c1 > 0 {
            adj_mask |= 1u16 << (c1 - 1);
        }
        if c1 + 1 < 10 {
            adj_mask |= 1u16 << (c1 + 1);
        }
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

    let mut pillars_2plus: i32 = 0;
    for c in 0..10usize {
        let dl = if c == 0 { 99 } else { stack_h[c] - stack_h[c - 1] };
        let dr = if c == 9 { 99 } else { stack_h[c] - stack_h[c + 1] };
        if dl >= 2 && dr >= 2 {
            pillars_2plus += 1;
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

    let ex13 = (max_h - 13).max(0);
    dense[0] = rem_garbage as f32;
    dense[1] = g as f32;
    dense[2] = pe_net as f32;
    dense[3] = stack_cells as f32;
    dense[4] = stack_cells_row_g as f32;
    dense[5] = stack_cells_above1 as f32;
    dense[6] = stack_cells_above2 as f32;
    dense[7] = upper_cells as f32;
    for c in 0..10 {
        dense[8 + c] = stack_h[c] as f32;
    }
    dense[18] = max_stack_h as f32;
    dense[19] = min_stack_h as f32;
    dense[20] = stack_h_sq as f32;
    dense[21] = stack_over2_sq as f32;
    dense[22] = (ex13 * ex13) as f32;
    dense[23] = covered_holes as f32;
    dense[24] = hole_rows as f32;
    dense[25] = blocks_over_holes as f32;
    dense[26] = shallow_holes as f32;
    dense[27] = bottom_covered_holes as f32;
    dense[28] = h1_blocks as f32;
    dense[29] = h1_height as f32;
    dense[30] = h1_empty_blocking as f32;
    dense[31] = h1_narrow_1 as f32;
    dense[32] = h1_narrow_2plus as f32;
    dense[33] = h1_fit_cnt;
    dense[34] = hand_fits_h1;
    dense[35] = bag_fit_frac;
    dense[36] = h2_blocks as f32;
    dense[37] = h2_empty_blocking as f32;
    dense[38] = h2_extra_empty as f32;
    dense[39] = h2_narrow_1;
    dense[40] = h2_narrow_2plus;
    dense[41] = h2_fit_cnt;
    dense[42] = h3_blocks as f32;
    dense[43] = h3_extra_empty as f32;
    dense[44] = h4_blocks as f32;
    dense[45] = cells_to_uncover_h1 as f32;
    dense[46] = cells_to_uncover_h12 as f32;
    dense[47] = bumpiness as f32;
    dense[48] = sharp_diff as f32;
    dense[49] = row_transitions as f32;
    dense[50] = col_transitions as f32;
    dense[51] = i_dep_depth as f32;
    dense[52] = i_dep_cols as f32;
    dense[53] = two_well_cnt as f32;
    dense[54] = clean_o_spots.min(3) as f32;
    dense[55] = total_o_spots.min(4) as f32;
    dense[56] = near_full_9 as f32;
    dense[57] = near_full_8 as f32;
    dense[58] = row_g_9_aligned;
    dense[59] = row_g_8_aligned;
    dense[60] = row_g_clearable_with_h1;
    dense[61] = clean_skim_rows as f32;
    dense[62] = (even_parity - odd_parity).abs() as f32;
    dense[63] = pillars_2plus as f32;
    dense[64] = h12_adj_overhangs as f32;
    dense[65] = max_cliff_around_h1 as f32;
    dense[66] = if has_o_avail && clean_o_spots == 0 && total_o_spots <= 1 { 1.0 } else { 0.0 };
    dense[67] = avail_mask.count_ones() as f32;
    let has_i_avail = in_hand == Some(0) || (avail_mask & 1) != 0;
    let no_i_when_i_only = if h1_narrow_2plus > 0 && !has_i_avail { 1.0 } else { 0.0 };
    let i_dep_cols_no_i = if !has_i_avail { i_dep_cols as f32 } else { 0.0 };
    let h2_blocked_row_g = if n > g {
        h2_col.map_or(0.0, |c2| if ((board.data[g] >> c2) & 1) != 0 { 1.0 } else { 0.0 })
    } else {
        0.0
    };
    let h2_only_row_g_blocked = if g >= 2 && h2_blocks == 1 && h2_blocked_row_g > 0.5 { 1.0 } else { 0.0 };
    let h1_ds_clears_rg_open_h2 = if row_g_9_aligned > 0.5 && h2_open { 1.0 } else { 0.0 };

    dense[68] = no_i_when_i_only;
    dense[69] = i_dep_cols_no_i;
    dense[70] = h2_only_row_g_blocked;
    dense[71] = h1_ds_clears_rg_open_h2;

    n_act
}

/// Fast NNUE-style spatial + tactical value network with linear skip connection:
/// `(Sparse(186) + Dense(72)) -> H1(64, ReLU) -> H2(32, ReLU) -> 1 + Linear(Sparse + Dense)`.
/// Output is predicted expected remaining pieces until finishing all garbage (lower is better).
#[derive(Clone, Debug)]
pub struct SpatialValueModel {
    pub dense_mean: [f32; DENSE_DIM],
    pub dense_inv_std: [f32; DENSE_DIM],
    pub lin_bias: f32,
    pub lin_sparse: [f32; SPARSE_DIM],
    pub lin_dense: [f32; DENSE_DIM],
    pub b1: [f32; SPATIAL_H1],
    pub w_sparse: Box<[[f32; SPATIAL_H1]; SPARSE_DIM]>,
    pub w_dense: Box<[[f32; SPATIAL_H1]; DENSE_DIM]>,
    pub b2: [f32; SPATIAL_H2],
    pub w2: Box<[[f32; SPATIAL_H2]; SPATIAL_H1]>,
    pub b3: f32,
    pub w3: [f32; SPATIAL_H2],
    // Pre-folded inference parameters (mean/inv_std folded into dense weights & biases)
    pub fast_lin_bias: f32,
    pub fast_lin_dense: [f32; DENSE_DIM],
    pub fast_b1: [f32; SPATIAL_H1],
    pub fast_w_dense: Box<[[f32; SPATIAL_H1]; DENSE_DIM]>,
    pub fast_ready: bool,
}

impl Default for SpatialValueModel {
    fn default() -> Self {
        Self {
            dense_mean: [0.0; DENSE_DIM],
            dense_inv_std: [1.0; DENSE_DIM],
            lin_bias: 0.0,
            lin_sparse: [0.0; SPARSE_DIM],
            lin_dense: [0.0; DENSE_DIM],
            b1: [0.0; SPATIAL_H1],
            w_sparse: Box::new([[0.0; SPATIAL_H1]; SPARSE_DIM]),
            w_dense: Box::new([[0.0; SPATIAL_H1]; DENSE_DIM]),
            b2: [0.0; SPATIAL_H2],
            w2: Box::new([[0.0; SPATIAL_H2]; SPATIAL_H1]),
            b3: 0.0,
            w3: [0.0; SPATIAL_H2],
            fast_lin_bias: 0.0,
            fast_lin_dense: [0.0; DENSE_DIM],
            fast_b1: [0.0; SPATIAL_H1],
            fast_w_dense: Box::new([[0.0; SPATIAL_H1]; DENSE_DIM]),
            fast_ready: false,
        }
    }
}

impl SpatialValueModel {
    pub fn prepare_fast(&mut self) {
        let mut lb = self.lin_bias;
        for j in 0..DENSE_DIM {
            let fl = self.lin_dense[j] * self.dense_inv_std[j];
            self.fast_lin_dense[j] = fl;
            lb -= fl * self.dense_mean[j];
        }
        self.fast_lin_bias = lb;

        for h in 0..SPATIAL_H1 {
            self.fast_b1[h] = self.b1[h];
        }
        for j in 0..DENSE_DIM {
            let inv = self.dense_inv_std[j];
            let m = self.dense_mean[j];
            for h in 0..SPATIAL_H1 {
                let fw = self.w_dense[j][h] * inv;
                self.fast_w_dense[j][h] = fw;
                self.fast_b1[h] -= fw * m;
            }
        }
        self.fast_ready = true;
    }

    #[inline]
    pub fn predict_extracted(
        &self,
        active_bits: &[u16],
        dense: &[f32; DENSE_DIM],
    ) -> f32 {
        let mut lin = if self.fast_ready { self.fast_lin_bias } else { self.lin_bias };
        let mut h1 = if self.fast_ready { self.fast_b1 } else { self.b1 };

        for &bit in active_bits {
            let idx = bit as usize;
            lin += self.lin_sparse[idx];
            let row = &self.w_sparse[idx];
            for h in 0..SPATIAL_H1 {
                h1[h] += row[h];
            }
        }

        if self.fast_ready {
            for j in 0..DENSE_DIM {
                let x = dense[j];
                if x != 0.0 {
                    lin += self.fast_lin_dense[j] * x;
                    let row = &self.fast_w_dense[j];
                    for h in 0..SPATIAL_H1 {
                        h1[h] += x * row[h];
                    }
                }
            }
        } else {
            for j in 0..DENSE_DIM {
                let z = (dense[j] - self.dense_mean[j]) * self.dense_inv_std[j];
                lin += self.lin_dense[j] * z;
                if z != 0.0 {
                    let row = &self.w_dense[j];
                    for h in 0..SPATIAL_H1 {
                        h1[h] += z * row[h];
                    }
                }
            }
        }

        let mut h2 = self.b2;
        for i in 0..SPATIAL_H1 {
            let a1 = h1[i];
            if a1 > 0.0 {
                let c1 = a1.min(6.0);
                let row = &self.w2[i];
                for k in 0..SPATIAL_H2 {
                    h2[k] += c1 * row[k];
                }
            }
        }

        let mut mlp = self.b3;
        for k in 0..SPATIAL_H2 {
            let a2 = h2[k];
            if a2 > 0.0 {
                mlp += a2.min(6.0) * self.w3[k];
            }
        }
        let mlp_sat = mlp / (1.0 + mlp.abs() / 5.0);
        lin + mlp_sat
    }

    #[inline]
    pub fn predict_linear_extracted(
        &self,
        active_bits: &[u16],
        dense: &[f32; DENSE_DIM],
    ) -> f32 {
        let mut lin = if self.fast_ready { self.fast_lin_bias } else { self.lin_bias };
        for &bit in active_bits {
            lin += self.lin_sparse[bit as usize];
        }
        if self.fast_ready {
            for j in 0..DENSE_DIM {
                let x = dense[j];
                if x != 0.0 {
                    lin += self.fast_lin_dense[j] * x;
                }
            }
        } else {
            for j in 0..DENSE_DIM {
                let z = (dense[j] - self.dense_mean[j]) * self.dense_inv_std[j];
                lin += self.lin_dense[j] * z;
            }
        }
        lin
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
        let mut active_bits = [0u16; SPARSE_DIM];
        let mut dense = [0.0f32; DENSE_DIM];
        let n_act = extract_spatial_features(
            board,
            rem_garbage,
            in_hand,
            avail_mask,
            &mut active_bits,
            &mut dense,
        );
        (self.predict_extracted(&active_bits[..n_act], &dense) * 100.0).round() as i64
    }

    #[inline]
    pub fn evaluate_linear_board_with_bag(
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
        let mut active_bits = [0u16; SPARSE_DIM];
        let mut dense = [0.0f32; DENSE_DIM];
        let n_act = extract_spatial_features(
            board,
            rem_garbage,
            in_hand,
            avail_mask,
            &mut active_bits,
            &mut dense,
        );
        (self.predict_linear_extracted(&active_bits[..n_act], &dense) * 100.0).round() as i64
    }

    pub fn save_to_file(&self, path: &str) -> std::io::Result<()> {
        let f = File::create(path)?;
        let mut w = BufWriter::new(f);
        writeln!(
            w,
            "SPATIAL_VAL_V2 {} {} {} {}",
            SPARSE_DIM, DENSE_DIM, SPATIAL_H1, SPATIAL_H2
        )?;
        for j in 0..DENSE_DIM {
            write!(
                w,
                "{}{:.7}",
                if j == 0 { "" } else { " " },
                self.dense_mean[j]
            )?;
        }
        writeln!(w)?;
        for j in 0..DENSE_DIM {
            write!(
                w,
                "{}{:.7}",
                if j == 0 { "" } else { " " },
                self.dense_inv_std[j]
            )?;
        }
        writeln!(w)?;
        writeln!(w, "{:.7}", self.lin_bias)?;
        for i in 0..SPARSE_DIM {
            write!(
                w,
                "{}{:.7}",
                if i == 0 { "" } else { " " },
                self.lin_sparse[i]
            )?;
        }
        writeln!(w)?;
        for j in 0..DENSE_DIM {
            write!(
                w,
                "{}{:.7}",
                if j == 0 { "" } else { " " },
                self.lin_dense[j]
            )?;
        }
        writeln!(w)?;
        for h in 0..SPATIAL_H1 {
            write!(w, "{}{:.7}", if h == 0 { "" } else { " " }, self.b1[h])?;
        }
        writeln!(w)?;
        for i in 0..SPARSE_DIM {
            for h in 0..SPATIAL_H1 {
                write!(
                    w,
                    "{}{:.7}",
                    if h == 0 { "" } else { " " },
                    self.w_sparse[i][h]
                )?;
            }
            writeln!(w)?;
        }
        for j in 0..DENSE_DIM {
            for h in 0..SPATIAL_H1 {
                write!(
                    w,
                    "{}{:.7}",
                    if h == 0 { "" } else { " " },
                    self.w_dense[j][h]
                )?;
            }
            writeln!(w)?;
        }
        for k in 0..SPATIAL_H2 {
            write!(w, "{}{:.7}", if k == 0 { "" } else { " " }, self.b2[k])?;
        }
        writeln!(w)?;
        for i in 0..SPATIAL_H1 {
            for k in 0..SPATIAL_H2 {
                write!(w, "{}{:.7}", if k == 0 { "" } else { " " }, self.w2[i][k])?;
            }
            writeln!(w)?;
        }
        writeln!(w, "{:.7}", self.b3)?;
        for k in 0..SPATIAL_H2 {
            write!(w, "{}{:.7}", if k == 0 { "" } else { " " }, self.w3[k])?;
        }
        writeln!(w)?;
        w.flush()
    }

    pub fn load_from_file(path: &str) -> std::io::Result<Self> {
        let f = File::open(path)?;
        let mut lines = BufReader::new(f).lines();
        let header = lines.next().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "missing header")
        })??;
        let mut next_floats = || -> std::io::Result<Vec<f32>> {
            let line = lines.next().ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated spatial model")
            })??;
            line.split_whitespace()
                .map(|s| {
                    s.parse::<f32>().map_err(|e| {
                        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
                    })
                })
                .collect()
        };
        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 5 || (parts[0] != "SPATIAL_VAL_V1" && parts[0] != "SPATIAL_VAL_V2") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "bad SPATIAL_VAL header",
            ));
        }
        let sd: usize = parts[1].parse().unwrap_or(0);
        let dd: usize = parts[2].parse().unwrap_or(0);
        let h1: usize = parts[3].parse().unwrap_or(0);
        let h2: usize = parts[4].parse().unwrap_or(0);
        let dd_ok = (parts[0] == "SPATIAL_VAL_V1" && (dd == 68 || dd == DENSE_DIM))
            || (parts[0] == "SPATIAL_VAL_V2" && dd == DENSE_DIM);
        if sd != SPARSE_DIM || !dd_ok || h1 != SPATIAL_H1 || h2 != SPATIAL_H2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "dimension mismatch in spatial model",
            ));
        }

        let mut m = Self::default();
        let dm = next_floats()?;
        let dis = next_floats()?;
        m.dense_mean[..dd].copy_from_slice(&dm[..dd]);
        m.dense_inv_std[..dd].copy_from_slice(&dis[..dd]);
        m.lin_bias = next_floats()?[0];
        m.lin_sparse.copy_from_slice(&next_floats()?[..SPARSE_DIM]);
        m.lin_dense[..dd].copy_from_slice(&next_floats()?[..dd]);
        m.b1.copy_from_slice(&next_floats()?[..SPATIAL_H1]);
        for i in 0..SPARSE_DIM {
            m.w_sparse[i].copy_from_slice(&next_floats()?[..SPATIAL_H1]);
        }
        for j in 0..dd {
            m.w_dense[j].copy_from_slice(&next_floats()?[..SPATIAL_H1]);
        }
        m.b2.copy_from_slice(&next_floats()?[..SPATIAL_H2]);
        for i in 0..SPATIAL_H1 {
            m.w2[i].copy_from_slice(&next_floats()?[..SPATIAL_H2]);
        }
        m.b3 = next_floats()?[0];
        m.w3.copy_from_slice(&next_floats()?[..SPATIAL_H2]);
        m.prepare_fast();
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::fast::EMPTY_ROW;

    #[test]
    fn test_spatial_feature_invariants_and_fast_folding() {
        let mut fb = FastBoard::empty();
        fb.data[0] = EMPTY_ROW | (0x3FF & !(1u16 << 3)); // h2 col = 3
        fb.data[1] = EMPTY_ROW | (0x3FF & !(1u16 << 7)); // h1 col = 7
        fb.data[2] = EMPTY_ROW | ((0x3FF & !(1u16 << 7)) & !(1u16 << 8)); // 8 filled, empty at 7 & 8
        fb.garbage_rows = 2;
        fb.rows = 3;

        let mut active = [0u16; SPARSE_DIM];
        let mut dense = [0.0f32; DENSE_DIM];
        let n_act = extract_spatial_features(&fb, 50, Some(0), 0x7F, &mut active, &mut dense);
        assert!(n_act > 10 && n_act <= SPARSE_DIM);
        for &bit in &active[..n_act] {
            assert!((bit as usize) < SPARSE_DIM);
        }
        // Check no duplicate active indices
        let mut seen = [false; SPARSE_DIM];
        for &bit in &active[..n_act] {
            assert!(!seen[bit as usize], "duplicate bit {}", bit);
            seen[bit as usize] = true;
        }
        assert_eq!(dense[0], 50.0); // rem_g
        assert_eq!(dense[1], 2.0); // g
        assert_eq!(dense[3], 8.0); // stack_cells
        assert_eq!(dense[58], 0.0);
        assert_eq!(dense[59], 1.0); // row_g_8_aligned (empty at 7 and 8; col 3 is filled!)
        assert_eq!(dense[70], 1.0); // h2_only_row_g_blocked (h2=3 has 1 block on row g)
        assert_eq!(dense[71], 0.0); // h1_ds_clears_rg_open_h2

        // Check row_g_9_aligned + h2_only_row_g_blocked + h1_ds_clears_rg_open_h2
        fb.data[2] = EMPTY_ROW | (0x3FF & !(1u16 << 7)); // row g has 9 filled, col 3 filled, col 7 empty
        let _ = extract_spatial_features(&fb, 50, Some(1), 0x7E, &mut active, &mut dense);
        assert_eq!(dense[58], 1.0); // row_g_9_aligned
        assert_eq!(dense[70], 1.0); // h2_only_row_g_blocked
        assert_eq!(dense[71], 0.0); // h2 is blocked

        // Now check h2 open + row_g_9_aligned (col 3 = h1, col 7 = h2, or make h2=col 3 open and row g filled except 3? Wait: if h1=7 and h2=3, row g empty at 3 & 7 is 8 filled; test h1_ds_clears_rg_open_h2 with h2=7 and h1=7 same col)
        fb.data[0] = EMPTY_ROW | (0x3FF & !(1u16 << 7)); // h2 col = 7 (same as h1)
        let n_act = extract_spatial_features(&fb, 50, Some(1), 0x7E, &mut active, &mut dense);
        assert_eq!(dense[58], 1.0); // row_g_9_aligned
        assert_eq!(dense[70], 0.0); // h2 is open
        assert_eq!(dense[71], 1.0); // h1_ds_clears_rg_open_h2

        let mut m = SpatialValueModel::default();
        for j in 0..DENSE_DIM {
            m.dense_mean[j] = (j as f32) * 0.5;
            m.dense_inv_std[j] = 0.25 + (j as f32) * 0.01;
            m.lin_dense[j] = ((j as i32 % 5) - 2) as f32 * 0.1;
            for h in 0..SPATIAL_H1 {
                m.w_dense[j][h] = (((j * 7 + h) % 11) as f32 - 5.0) * 0.02;
            }
        }
        for i in 0..SPARSE_DIM {
            m.lin_sparse[i] = ((i as i32 % 7) - 3) as f32 * 0.05;
            for h in 0..SPATIAL_H1 {
                m.w_sparse[i][h] = (((i * 3 + h) % 9) as f32 - 4.0) * 0.03;
            }
        }
        for h in 0..SPATIAL_H1 {
            m.b1[h] = (h as f32) * 0.01 - 0.3;
            for k in 0..SPATIAL_H2 {
                m.w2[h][k] = (((h * 5 + k) % 7) as f32 - 3.0) * 0.04;
            }
        }
        for k in 0..SPATIAL_H2 {
            m.b2[k] = 0.05;
            m.w3[k] = 0.1 * ((k as i32 % 3) - 1) as f32;
        }
        m.lin_bias = 1.25;
        m.b3 = -0.2;

        let raw_pred = m.predict_extracted(&active[..n_act], &dense);
        let raw_lin = m.predict_linear_extracted(&active[..n_act], &dense);
        m.prepare_fast();
        let fast_pred = m.predict_extracted(&active[..n_act], &dense);
        let fast_lin = m.predict_linear_extracted(&active[..n_act], &dense);
        assert!((raw_pred - fast_pred).abs() < 1e-4, "{} vs {}", raw_pred, fast_pred);
        assert!((raw_lin - fast_lin).abs() < 1e-4, "lin {} vs {}", raw_lin, fast_lin);
        assert_eq!(
            m.evaluate_board_with_bag(&fb, 50, Some(1), 0x7E),
            (fast_pred * 100.0).round() as i64
        );
        assert_eq!(
            m.evaluate_linear_board_with_bag(&fb, 50, Some(1), 0x7E),
            (fast_lin * 100.0).round() as i64
        );

        let tmp = std::env::temp_dir().join("test_spatial_v2_roundtrip.txt");
        let tmp_str = tmp.to_str().unwrap();
        m.save_to_file(tmp_str).unwrap();
        let m2 = SpatialValueModel::load_from_file(tmp_str).unwrap();
        let _ = std::fs::remove_file(&tmp);
        let reload_pred = m2.predict_extracted(&active[..n_act], &dense);
        let reload_lin = m2.predict_linear_extracted(&active[..n_act], &dense);
        assert!((fast_pred - reload_pred).abs() < 1e-3, "{} vs {}", fast_pred, reload_pred);
        assert!((fast_lin - reload_lin).abs() < 1e-3, "lin {} vs {}", fast_lin, reload_lin);
    }
}
