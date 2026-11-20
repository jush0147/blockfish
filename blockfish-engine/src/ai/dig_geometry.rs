//! Geometry around the highest opening in a contiguous, almost-full bottom stack.
use crate::BasicMatrix;
include!("dig_weights.rs");

fn features(board: &BasicMatrix) -> [u16; 6] {
    let mut depth = 0;
    let mut target = None;
    let mut previous = None;
    for row in 0..board.rows() {
        let gap = board.vacant_bits(row);
        if gap.count_ones() != 1 { break; }
        previous = target;
        target = Some(gap.trailing_zeros() as u16);
        depth += 1;
    }
    let column = match target { Some(c) => c, None => return [0; 6] };
    let mut cover = 0;
    let mut span = 0;
    let mut left = if column == 0 { 4 } else { 0 };
    let mut right = if column + 1 == board.cols() { 4 } else { 0 };
    for row in depth..board.rows() {
        let height = row - depth + 1;
        if board.get((row,column)) { cover += 1; span = height; }
        if column > 0 && board.get((row,column-1)) { left = height.min(4); }
        if column + 1 < board.cols() && board.get((row,column+1)) { right = height.min(4); }
    }
    [cover,span,left.min(right),left.max(right),u16::from(column==0 || column+1==board.cols()),
     column.abs_diff(previous.unwrap_or(column))]
}

pub(super) fn correction(board: &BasicMatrix) -> f64 {
    features(board).iter().zip(DIG.iter()).map(|(&v,&w)|f64::from(v)*w).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bottom_prefix_and_capped_wall() {
        let mut board = BasicMatrix::with_cols(10);
        assert_eq!(features(&board), [0;6]);
        for c in 0..10 {
            if c!=3 { board.set((0,c)); }
            if c!=0 { board.set((1,c)); }
        }
        board.set((4,0));
        board.set((2,1));
        assert_eq!(features(&board), [1,3,1,4,1,3]);
    }
}
