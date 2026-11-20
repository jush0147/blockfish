use crate::{config::Parameters, matrix::BasicMatrix};
#[cfg(any(test, not(feature = "flood-eval")))]
use red_union_find::UF;
use std::ops::Range;

// Evaluations

#[derive(Default, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Eval {
    pub rows: u16,
    pub piece_estimate: u16,
    pub i_dependencies: u16,
}

impl Eval {
    /// Computes the "score" based on this evaluation. Lower is better.
    ///
    /// Note: used by A* to compute "h" value (remaining cost heuristic).
    pub fn score(&self, params: &Parameters) -> i64 {
        params.row_factor * (self.rows as i64)
            + params.piece_estimate_factor * (self.piece_estimate as i64)
            + params.i_dependency_factor * (self.i_dependencies as i64)
    }
}

/// Computes the "penalty" for placing the given number of pieces.
///
/// Note: used in A* to compute "g" value (path cost).
pub fn penalty(params: &Parameters, depth: usize) -> i64 {
    (depth as i64) * params.piece_penalty
}

/// Evaluates a matrix, returning the different heuristic values.
pub fn eval(matrix: &BasicMatrix) -> Eval {
    Eval {
        rows: matrix.rows(),
        piece_estimate: piece_estimate(matrix.clone()),
        i_dependencies: i_dependencies(&matrix, 0..matrix.rows()).count() as _,
    }
}

pub(super) fn search_score(matrix: &BasicMatrix, params: &Parameters) -> i64 {
    let evaluation = eval(matrix);
    let score = evaluation.score(params);
    #[cfg(feature = "learned-value")]
    {
        let learned = super::value_features::learned_score(matrix, &evaluation);
        if cfg!(feature = "learned-blend") {
            (score + learned) / 2
        } else {
            learned
        }
    }
    #[cfg(all(not(feature = "learned-value"), feature = "surface-cost"))]
    {
        let heights: Vec<_> = (0..matrix.cols())
            .map(|j| matrix.col_height(j) as i64)
            .collect();
        score + 2 * heights.windows(2).map(|h| (h[1] - h[0]).abs()).sum::<i64>()
    }
    #[cfg(all(not(feature = "learned-value"), not(feature = "surface-cost")))]
    {
        score
    }
}

#[cfg(feature = "hold-credit")]
pub(super) fn hold_credit(matrix: &BasicMatrix, params: &Parameters) -> i64 {
    if i_dependencies(matrix, 0..matrix.rows()).next().is_some() {
        params.i_dependency_factor
    } else {
        0
    }
}

/// Mystery's residue-based minimum piece estimate algorithm.
fn piece_estimate(mut matrix: BasicMatrix) -> u16 {
    let mut pieces = 0;
    let mut depth = 0;

    let mut residue_buf = ResidueBuf::new();
    while let Some((i, res)) = covered_hole(&matrix, &mut residue_buf) {
        let rows = (i + 1)..res.end;
        let pieces_to_fill: u16 = negative_spaces(&matrix, rows.clone())
            .map(|area| (area + 3) / 4)
            .sum();

        matrix.remove_rows(rows);

        depth += 1;
        let discount = if cfg!(feature = "shallow-discount") {
            (depth + 1) / 2
        } else {
            depth
        };
        pieces += pieces_to_fill.saturating_sub(discount);
        pieces += 1;
    }

    pieces
}

/// Returns the area of each disjoint contiguous negative space in the given matrix.
#[cfg(not(feature = "flood-eval"))]
fn negative_spaces<'a>(
    matrix: &'a BasicMatrix,
    row_range: Range<u16>,
) -> impl Iterator<Item = u16> + 'a {
    gaps_contiguous_areas(row_range.map(move |i| matrix.gaps(i)))
}

#[cfg(feature = "flood-eval")]
fn negative_spaces(matrix: &BasicMatrix, row_range: Range<u16>) -> impl Iterator<Item = u16> {
    let mut vacant: Vec<u16> = row_range.map(|r| matrix.vacant_bits(r)).collect();
    let mut areas = Vec::new();
    let mut pending = Vec::new();
    for start in 0..vacant.len() {
        while vacant[start] != 0 {
            pending.push((start, 1u16 << vacant[start].trailing_zeros()));
            let mut area = 0;
            while let Some((row, incoming)) = pending.pop() {
                let mut fill = incoming & vacant[row];
                if fill == 0 {
                    continue;
                }
                loop {
                    let expanded = (fill | (fill << 1) | (fill >> 1)) & vacant[row];
                    if expanded == fill {
                        break;
                    }
                    fill = expanded;
                }
                vacant[row] &= !fill;
                area += fill.count_ones() as u16;
                if row > 0 && vacant[row - 1] & fill != 0 {
                    pending.push((row - 1, fill));
                }
                if row + 1 < vacant.len() && vacant[row + 1] & fill != 0 {
                    pending.push((row + 1, fill));
                }
            }
            areas.push(area);
        }
    }
    areas.into_iter()
}

/// Returns the columns containing I dependencies.
#[cfg(not(feature = "bit-eval"))]
fn i_dependencies<'a>(
    matrix: &'a BasicMatrix,
    row_range: Range<u16>,
) -> impl Iterator<Item = u16> + 'a {
    let cols = matrix.cols();
    (0..cols).filter(move |&j| {
        let mut count = None;
        row_range.clone().any(|i| {
            let c0 = (j == 0) || matrix.get((i, j - 1));
            let c1 = matrix.get((i, j));
            let c2 = (j == cols - 1) || matrix.get((i, j + 1));
            if c0 && !c1 && c2 {
                let count = count.get_or_insert(0);
                *count += 1;
                *count >= 3
            } else {
                count = None;
                false
            }
        })
    })
}

#[cfg(feature = "bit-eval")]
fn i_dependencies<'a>(
    matrix: &'a BasicMatrix,
    row_range: Range<u16>,
) -> impl Iterator<Item = u16> + 'a {
    let (mut previous, mut pair, mut found) = (0u16, 0u16, 0u16);
    for row in row_range {
        let gaps = matrix.narrow_gaps(row);
        found |= pair & gaps;
        pair = previous & gaps;
        previous = gaps;
    }
    (0..matrix.cols()).filter(move |&col| found & (1 << col) != 0)
}

/// Returns the size of each contiguous area given by the overlapping, neighboring ranges
/// in `iter`.
///
/// # Example
///
/// If `iter` is `[[0..5, 7..10], [0..4], [2..3]]` returns `[10, 3]`. `10` is given by the
/// set of intervals `{0..5,0..4,2..3}` which all overlap; `3` is given by the last
/// interval `7..10`.
#[cfg(any(test, not(feature = "flood-eval")))]
fn gaps_contiguous_areas<I>(iter: I) -> impl Iterator<Item = u16> + 'static
where
    I: IntoIterator,
    I::Item: IntoIterator<Item = Range<u16>>,
{
    let iter = iter.into_iter();
    let size_hint = iter.size_hint().1.unwrap_or(0);
    let mut gaps = Vec::with_capacity(size_hint * 2);
    let mut row_end_idxs = Vec::with_capacity(size_hint);
    for row_gaps in iter {
        gaps.extend(row_gaps);
        row_end_idxs.push(gaps.len());
    }

    let mut uf = UF::new_reflexive(gaps.len());
    let mut idx0 = std::usize::MAX;
    let mut idx1 = 0;
    for idx2 in row_end_idxs {
        if idx0 < std::usize::MAX {
            let row1 = &gaps[idx0..idx1];
            let row2 = &gaps[idx1..idx2];
            for (i0, i1) in intersecting_ranges(row1, row2) {
                uf.union(idx0 + i0, idx1 + i1);
            }
        }
        idx0 = idx1;
        idx1 = idx2;
    }

    let mut areas = vec![0; gaps.len()];
    for (i, gap) in gaps.into_iter().enumerate() {
        areas[uf.find(i)] += gap.end - gap.start;
    }

    areas.into_iter().filter(|&a| a > 0)
}

/// Given `xs` and `ys` both ordered lists of non-overlapping ranges, returns every pair
/// of indices `(i, j)` such that `xs[i]` intersects with `ys[j]`.
#[cfg(any(test, not(feature = "flood-eval")))]
fn intersecting_ranges<'a, T: Ord>(
    xs: &'a [Range<T>],
    ys: &'a [Range<T>],
) -> impl Iterator<Item = (usize, usize)> + 'a {
    let (mut i1, mut i2) = (0, 0);
    std::iter::from_fn(move || loop {
        let r1 = xs.get(i1)?;
        let r2 = ys.get(i2)?;
        if r2.start >= r1.end {
            i1 += 1;
        } else if r1.start >= r2.end {
            i2 += 1;
        } else if r2.end >= r1.end {
            i1 += 1;
            return Some((i1 - 1, i2));
        } else {
            i2 += 1;
            return Some((i1, i2 - 1));
        }
    })
}

type ResidueBuf = Vec<Range<u16>>;

/// Searches for a hole covered by residue. If any is found returns `Some((i, r))` where
/// `i` is the row containing the hole and `r` is the (half-open) range of rows above `i`
/// containing the residue.
///
/// `buf` is used for internal bookkeeping to perform the algorithm, and ought to be
/// reused to save on allocations.
///
/// Example:
///
/// 5 | . . . . .
/// 4 | . x . . .  <-
/// 3 | x x x . .  <- r
/// 2 | x _ x x x
/// 1 | x _ x x x  <- i
/// 0 | x x x _ x         <= note: this is a valid hole, but the one above is found first.
///
/// Returns `Some((1, 3..5))`.
#[allow(unused)]
fn covered_hole(mat: &BasicMatrix, buf: &mut ResidueBuf) -> Option<(u16, Range<u16>)> {
    let (n_rows, n_cols) = (mat.rows(), mat.cols());
    // `buf[j]` holds the known range of residue cells in column `j`
    buf.clear();
    buf.resize(n_cols as usize, 0..0);
    for i in (0..n_rows).rev() {
        for j in 0..n_cols {
            // do nothing for empty cells
            if !mat.get((i, j)) {
                continue;
            }
            let res = &mut buf[j as usize];
            if res.start > i + 1 {
                // prev cell was empty, so we finished finding a hole
                return Some((i + 1, res.clone()));
            } else {
                // update the residue since the cell above was also filled
                *res = i..std::cmp::max(i + 1, res.end);
            }
        }
    }
    // run a simplified version of the inner loop for underneath the bottom row, i.e. i=-1
    for j in 0..n_cols {
        let res = &buf[j as usize];
        if res.start > 0 {
            return Some((0, res.clone()));
        }
    }
    None
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::basic_matrix;

    #[cfg(feature = "bit-eval")]
    #[test]
    fn bit_evaluation_matches_reference_on_random_matrices() {
        let mut rng = 173u64;
        for case in 0..5000 {
            // Upstream row masks reserve a sentinel bit: widths 1..15.
            let cols = 1 + case % 15;
            let rows = 1 + case % 32;
            let mut matrix = BasicMatrix::with_cols(cols);
            for _ in 0..rows {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                matrix.push_row((0..cols).map(|c| rng & (1 << (c + 24)) != 0));
            }
            let range = (case % rows)..rows;
            let expected: Vec<_> = (0..cols)
                .filter(|&j| {
                    let mut run = 0;
                    range.clone().any(|i| {
                        if !matrix.get((i, j))
                            && (j == 0 || matrix.get((i, j - 1)))
                            && (j + 1 == cols || matrix.get((i, j + 1)))
                        {
                            run += 1;
                        } else {
                            run = 0;
                        }
                        run >= 3
                    })
                })
                .collect();
            assert_eq!(
                i_dependencies(&matrix, range.clone()).collect::<Vec<_>>(),
                expected
            );
            #[cfg(feature = "flood-eval")]
            {
                let mut expected: Vec<_> =
                    gaps_contiguous_areas(range.clone().map(|i| matrix.gaps(i))).collect();
                let mut actual: Vec<_> = negative_spaces(&matrix, range).collect();
                expected.sort_unstable();
                actual.sort_unstable();
                assert_eq!(actual, expected, "case {}", case);
            }
        }
    }

    #[test]
    fn test_intersecting_ranges() {
        let irs = |xs: &[Range<i32>], ys| intersecting_ranges(xs, ys).collect::<Vec<_>>();
        // 0 1 2 3 4 5 6 7 8 9 10 11 12 13
        // {-----}     {-----------}    {----
        //     {-----} {-}   {--------}
        //     {#}     {#}   {#####}
        let xs = [0..3, 6..11, 13..20];
        let ys = [2..5, 6..7, 9..12];
        assert_eq!(irs(&xs, &ys), [(0, 0), (1, 1), (1, 2)]);
        assert_eq!(irs(&ys, &xs), [(0, 0), (1, 1), (2, 1)]);
        assert_eq!(irs(&xs, &[]), []);
        assert_eq!(irs(&[], &xs), []);
        assert_eq!(irs(&xs, &[10..15]), [(1, 0), (2, 0)]);
        assert_eq!(irs(&xs, &[11..15]), [(2, 0)]);
    }

    fn neg_space(mat: BasicMatrix) -> Vec<u16> {
        let mut nss = negative_spaces(&mat, 0..mat.rows()).collect::<Vec<_>>();
        nss.sort();
        nss
    }

    #[test]
    fn test_negative_spaces_1() {
        let (xx, __) = (true, false);
        assert_eq!(
            neg_space(basic_matrix![
                [xx, xx, xx, xx, xx, __],
                [xx, __, xx, xx, xx, xx],
                [xx, xx, __, xx, xx, xx],
            ]),
            [1, 1, 1],
            "cheese"
        );
        assert_eq!(
            neg_space(basic_matrix![
                [xx, __, xx, xx, xx, xx],
                [xx, xx, __, __, xx, xx],
            ]),
            [1, 2],
            "2w gap"
        );
    }

    #[test]
    fn test_negative_spaces_2() {
        let (xx, __) = (true, false);
        assert_eq!(
            neg_space(basic_matrix![
                [xx, __, __, __, xx, xx],
                [xx, __, xx, xx, xx, xx],
                [xx, xx, xx, __, __, xx],
            ]),
            [2, 4],
            "overlap (depth=1)"
        );
        assert_eq!(
            neg_space(basic_matrix![
                [xx, xx, xx, __, __, xx],
                [xx, __, xx, xx, xx, xx],
                [xx, __, __, __, xx, xx],
            ]),
            [2, 4],
            "overlap (depth=1, inverted)"
        );
        assert_eq!(
            neg_space(basic_matrix![
                [xx, __, __, __, xx, xx],
                [xx, __, xx, __, __, xx],
                [__, xx, __, xx, xx, xx],
            ]),
            [1, 1, 6],
            "overlap (depth=2)"
        );
        assert_eq!(
            neg_space(basic_matrix![
                [__, xx, __, xx, xx, xx],
                [xx, __, xx, __, __, xx],
                [xx, __, __, __, xx, xx],
            ]),
            [1, 1, 6],
            "overlap (depth=2, inverted)"
        );
    }

    #[test]
    fn test_negative_spaces_3() {
        let (xx, __) = (true, false);
        assert_eq!(
            neg_space(basic_matrix![
                [xx, __, __, __, xx, __],
                [__, __, xx, __, xx, xx],
                [xx, __, xx, __, __, xx],
            ]),
            [1, 9],
            "overlap (depth=3)"
        );
        assert_eq!(
            neg_space(basic_matrix![
                [xx, __, xx, __, __, xx],
                [__, __, xx, __, xx, xx],
                [xx, __, __, __, xx, __],
            ]),
            [1, 9],
            "overlap (depth=3, ivnerted)"
        );
    }

    #[test]
    fn test_negative_spaces_all_clear() {
        assert_eq!(neg_space(BasicMatrix::with_cols(5)), [0u16; 0]);
    }

    #[test]
    fn test_covered_hole_0() {
        let (xx, __) = (true, false);
        assert_eq!(
            covered_hole(&BasicMatrix::with_cols(5), &mut Default::default()),
            None,
            "zero"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![[xx, xx, xx, xx, xx, xx], [xx, xx, xx, xx, xx, xx],],
                &mut Default::default()
            ),
            None,
            "full",
        );
    }

    #[test]
    fn test_covered_hole_1() {
        let (xx, rr, __) = (true, true, false);
        let mut rbuf = ResidueBuf::default();
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, xx, xx, xx],
                    [__, xx, xx, xx, xx, xx],
                    [rr, xx, xx, xx, xx, xx],
                ],
                &mut rbuf
            ),
            Some((1, 2..3)),
            "cheese h=1"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [__, xx, xx, xx, xx, xx],
                    [rr, xx, __, xx, xx, xx],
                    [rr, xx, rr, xx, __, xx],
                    [rr, xx, rr, xx, rr, xx],
                    [rr, xx, rr, xx, rr, xx],
                ],
                &mut rbuf
            ),
            Some((2, 3..5)),
            "cheese h=2"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [__, xx, xx, xx, xx, xx],
                    [rr, xx, xx, __, xx, xx],
                    [rr, xx, xx, __, xx, xx],
                    [rr, xx, xx, rr, xx, xx],
                    [rr, xx, xx, rr, xx, xx],
                ],
                &mut rbuf
            ),
            Some((1, 3..5)),
            "cheese d=2"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [__, xx, xx, xx, xx, xx],
                    [rr, xx, xx, __, xx, xx],
                    [rr, xx, xx, rr, xx, __],
                    [rr, xx, xx, rr, xx, xx],
                    [rr, xx, xx, rr, xx, __],
                ],
                &mut rbuf
            ),
            Some((2, 3..4)),
            "reuse rbuf"
        );
    }

    #[test]
    fn test_covered_hole_2() {
        let (xx, rr, __) = (true, true, false);
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, __, xx, xx],
                    [xx, xx, __, __, xx, xx],
                    [xx, __, __, __, __, xx],
                ],
                &mut Default::default()
            ),
            None,
            "funnel"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, __, xx, xx],
                    [__, __, __, rr, __, __],
                    [__, __, __, rr, __, __],
                    [__, __, __, rr, __, __],
                ],
                &mut Default::default()
            ),
            Some((0, 1..4)),
            "spike"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, __, xx, xx],
                    [__, __, __, rr, __, __],
                    [__, __, __, rr, __, __],
                    [__, __, rr, rr, __, __],
                ],
                &mut Default::default()
            ),
            Some((1, 3..4)),
            "overhang"
        );
    }

    #[test]
    fn test_covered_holes() {
        let (xx, rr, __) = (true, true, false);
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, xx, xx, xx],
                    [xx, __, xx, __, xx, xx],
                    [xx, rr, xx, rr, xx, xx],
                ],
                &mut Default::default()
            ),
            Some((1, 2..3)),
            "double"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, xx, xx, xx],
                    [xx, __, xx, __, xx, xx],
                    [xx, rr, xx, rr, xx, xx],
                    [__, __, __, rr, __, __],
                ],
                &mut Default::default()
            ),
            Some((1, 2..3)),
            //Some((1, 2..4)),  <- would this be better?
            "double, right has greater residue"
        );
        assert_eq!(
            covered_hole(
                &basic_matrix![
                    [xx, xx, xx, __, xx, xx],
                    [xx, __, xx, __, xx, xx],
                    [xx, rr, xx, rr, xx, xx],
                ],
                &mut Default::default()
            ),
            Some((1, 2..3)),
            // Some((0, 2..3)),  <- would this be better?
            "double, right has deeper residue"
        );
    }

    #[test]
    fn test_i_deps_1() {
        let (xx, __) = (true, false);
        let i_deps = |mat: BasicMatrix| i_dependencies(&mat, 0..mat.rows()).collect::<Vec<_>>();
        assert_eq!(i_deps(basic_matrix![[__, __, __]]), &[] as &[u16]);
        assert_eq!(i_deps(basic_matrix![[xx, xx, xx]]), &[] as &[u16]);

        assert_eq!(
            i_deps(basic_matrix![
                [__, xx, xx, __],
                [__, xx, xx, __],
                [__, xx, xx, xx],
            ]),
            [0]
        );

        assert_eq!(
            i_deps(basic_matrix![
                [xx, xx, xx, xx, __],
                [xx, __, xx, xx, __],
                [xx, __, xx, xx, __],
                [xx, __, xx, xx, xx],
                [xx, xx, xx, xx, xx],
            ]),
            [1, 4]
        );

        assert_eq!(
            i_deps(basic_matrix![
                [xx, xx, xx, xx, __],
                [xx, __, xx, xx, __],
                [xx, __, xx, xx, __],
                [xx, __, xx, xx, xx],
                [xx, xx, xx, __, xx],
                [xx, __, xx, __, xx],
                [xx, __, xx, xx, xx],
                [xx, __, xx, xx, xx],
                [xx, __, xx, xx, xx],
            ]),
            [1, 4]
        );

        assert_eq!(
            i_deps(basic_matrix![
                [xx, xx, xx, __, xx],
                [__, __, xx, __, xx],
                [xx, __, xx, xx, xx],
                [xx, __, xx, xx, xx],
                [xx, xx, xx, xx, __],
                [xx, __, xx, xx, __],
                [xx, __, xx, xx, __],
                [xx, __, __, __, xx],
                [xx, __, xx, xx, xx],
            ]),
            [4]
        );
    }
}
