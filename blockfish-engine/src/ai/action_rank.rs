//! Root-only residual correction; this is not a replacement leaf value.
include!("rerank_weights.rs");

pub fn score(rating: i64, projected_features: &[f64; 14]) -> f64 {
    WEIGHTS[0] * rating as f64 / 20.0
        + projected_features[1..]
            .iter()
            .zip(WEIGHTS[1..].iter())
            .map(|(x, w)| x * w)
            .sum::<f64>()
}
