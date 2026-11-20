//! Observable board features for offline policy evaluation; no seed/future inputs.
use super::eval::{eval, Eval};
use crate::BasicMatrix;

pub const FEATURE_COUNT: usize = 14;

pub fn board_features(matrix: &BasicMatrix) -> [f64; FEATURE_COUNT] {
    features_with_eval(matrix, &eval(matrix))
}

pub(super) fn features_with_eval(matrix: &BasicMatrix, base: &Eval) -> [f64; FEATURE_COUNT] {
    #[cfg(feature = "fast-value")]
    {
        features_bitwise(matrix, base)
    }
    #[cfg(not(feature = "fast-value"))]
    {
        features_reference(matrix, base)
    }
}

#[cfg(any(test, not(feature = "fast-value")))]
fn features_reference(matrix: &BasicMatrix, base: &Eval) -> [f64; FEATURE_COUNT] {
    let rows = matrix.rows();
    let cols = matrix.cols();
    let heights: Vec<_> = (0..cols).map(|c| matrix.col_height(c)).collect();
    let sum_heights: u32 = heights.iter().map(|&h| h as u32).sum();
    let bump: u32 = heights
        .windows(2)
        .map(|h| (h[1] as i32 - h[0] as i32).abs() as u32)
        .sum();
    let mut holes = 0u32;
    let mut depth = 0u32;
    let mut covered_rows = 0u32;
    let mut row_transitions = 0u32;
    let mut column_transitions = 0u32;
    for r in 0..rows {
        let mut covered = false;
        let mut previous = true;
        for c in 0..cols {
            let filled = matrix.get((r, c));
            row_transitions += (filled != previous) as u32;
            previous = filled;
            if !filled && r < heights[c as usize] {
                holes += 1;
                depth += (heights[c as usize] - r) as u32;
                covered = true;
            }
        }
        row_transitions += (!previous) as u32;
        covered_rows += covered as u32;
    }
    for c in 0..cols {
        let mut previous = true;
        for r in 0..rows {
            let filled = matrix.get((r, c));
            column_transitions += (filled != previous) as u32;
            previous = filled;
        }
        column_transitions += previous as u32;
    }
    let mut wells = 0u32;
    for c in 0..cols as usize {
        let left = if c == 0 { rows } else { heights[c - 1] };
        let right = if c + 1 == cols as usize {
            rows
        } else {
            heights[c + 1]
        };
        let d = left.min(right).saturating_sub(heights[c]) as u32;
        wells += d * (d + 1) / 2;
    }
    let spread = rows.saturating_sub(heights.iter().copied().min().unwrap_or(0));
    [
        1.0,
        rows as f64,
        base.piece_estimate as f64,
        base.i_dependencies as f64,
        sum_heights as f64,
        bump as f64,
        holes as f64,
        depth as f64,
        covered_rows as f64,
        row_transitions as f64,
        column_transitions as f64,
        wells as f64,
        spread as f64,
        (base.piece_estimate as f64).powi(2),
    ]
}

#[cfg(feature = "fast-value")]
fn features_bitwise(matrix: &BasicMatrix, base: &Eval) -> [f64; FEATURE_COUNT] {
    let rows = matrix.rows();
    let cols = matrix.cols() as usize;
    let mask = ((1u32 << cols) - 1) as u16;
    let mut heights = [0u16; 16];
    let (mut seen, mut previous) = (0u16, 0u16);
    let (mut holes, mut depth, mut covered_rows, mut rt, mut ct) = (0u32, 0u32, 0u32, 0u32, 0u32);
    for r in (0..rows).rev() {
        let vacant = matrix.vacant_bits(r);
        let occupied = !vacant & mask;
        let mut fresh = occupied & !seen;
        while fresh != 0 {
            let c = fresh.trailing_zeros() as usize;
            heights[c] = r + 1;
            fresh &= fresh - 1;
        }
        seen |= occupied;
        let mut covered = vacant & seen;
        holes += covered.count_ones();
        covered_rows += (covered != 0) as u32;
        while covered != 0 {
            let c = covered.trailing_zeros() as usize;
            depth += (heights[c] - r) as u32;
            covered &= covered - 1;
        }
        rt += 2 * (vacant & !(vacant << 1)).count_ones();
        ct += (occupied ^ previous).count_ones();
        previous = occupied;
    }
    ct += (mask ^ previous).count_ones();
    let heights = &heights[..cols];
    let sum_heights: u32 = heights.iter().map(|&h| h as u32).sum();
    let bump: u32 = heights
        .windows(2)
        .map(|h| (h[1] as i32 - h[0] as i32).abs() as u32)
        .sum();
    let mut wells = 0u32;
    for c in 0..cols {
        let left = if c == 0 { rows } else { heights[c - 1] };
        let right = if c + 1 == cols { rows } else { heights[c + 1] };
        let d = left.min(right).saturating_sub(heights[c]) as u32;
        wells += d * (d + 1) / 2;
    }
    let spread = rows.saturating_sub(heights.iter().copied().min().unwrap_or(0));
    [
        1.0,
        rows as f64,
        base.piece_estimate as f64,
        base.i_dependencies as f64,
        sum_heights as f64,
        bump as f64,
        holes as f64,
        depth as f64,
        covered_rows as f64,
        rt as f64,
        ct as f64,
        wells as f64,
        spread as f64,
        (base.piece_estimate as f64).powi(2),
    ]
}

#[cfg(all(test, feature = "fast-value"))]
mod tests {
    use super::*;
    #[test]
    fn bitwise_features_match_training_features() {
        let mut rng = 911u64;
        for case in 0..5000 {
            let cols = 1 + case % 15;
            let mut matrix = BasicMatrix::with_cols(cols);
            for _ in 0..case % 32 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                matrix.push_row((0..cols).map(|c| rng & (1 << (c + 24)) != 0));
            }
            let base = eval(&matrix);
            assert_eq!(
                features_reference(&matrix, &base),
                features_bitwise(&matrix, &base)
            );
        }
    }
}

#[cfg(feature = "learned-value")]
pub(super) fn learned_score(matrix: &BasicMatrix, base: &Eval) -> i64 {
    let features = features_with_eval(matrix, base);
    #[cfg(feature = "tunable-value")]
    let weights = runtime_weights();
    #[cfg(not(feature = "tunable-value"))]
    let weights = &WEIGHTS;
    let score: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(x, w)| x * w)
        .sum();
    #[cfg(feature = "rank-blend")]
    let score = 0.25 * score + 0.75 * rank_prior::score(&features);
    #[cfg(feature="dig-geometry")]
    let score = score + super::dig_geometry::correction(matrix);
    (score.max(0.0) * 10.0).round() as i64
}

#[cfg(feature = "tunable-value")]
fn runtime_weights() -> &'static [f64; 14] {
    static WEIGHT_OVERRIDE: std::sync::OnceLock<[f64; 14]> = std::sync::OnceLock::new();
    WEIGHT_OVERRIDE.get_or_init(|| {
        if let Ok(json) = std::env::var("BLOCKFISH_VALUE_WEIGHTS") {
            let weights: [f64; 14] = serde_json::from_str(&json).expect("14 numeric value weights");
            assert!(
                weights.iter().all(|x| x.is_finite()),
                "value weights must be finite"
            );
            weights
        } else {
            WEIGHTS
        }
    })
}

// Replaced by research/fit_value.py using training splits only.
#[cfg(all(
    feature = "learned-value",
    not(any(
        feature = "policy2-value",
        feature = "td-value",
        feature = "rank-value",
        feature = "direct-value"
    ))
))]
include!("learned_weights.rs");

#[cfg(all(
    feature = "policy2-value",
    not(any(feature = "td-value", feature = "rank-value", feature = "direct-value"))
))]
include!("policy2_weights.rs");

#[cfg(all(
    feature = "td-value",
    not(any(feature = "rank-value", feature = "direct-value"))
))]
include!("td_weights.rs");

#[cfg(all(feature = "rank-value", not(feature = "direct-value")))]
include!("rank_weights.rs");

#[cfg(feature = "rank-blend")]
mod rank_prior {
    include!("learned_weights.rs");
    pub(super) fn score(features: &[f64; 14]) -> f64 {
        features
            .iter()
            .zip(WEIGHTS.iter())
            .map(|(x, w)| x * w)
            .sum()
    }
}

#[cfg(feature = "direct-value")]
include!("direct_weights.rs");
