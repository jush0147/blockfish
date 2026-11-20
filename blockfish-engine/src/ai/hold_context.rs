//! Experimental S21 residual. Only public held type and board are inputs.
use super::{state::State, value_features::board_features};
use crate::Color;
#[cfg(feature = "hold-context-cache")]
use std::convert::TryFrom;
include!("hold_weights.rs");
const TYPES: &str = "IJLOSTZ";
const INDICES: [usize; 5] = [0, 1, 2, 3, 11];

fn correction(features: &[f64; 14], hold: Option<Color>) -> f64 {
    match hold.and_then(|color| TYPES.chars().position(|c| c == color.as_char())) {
        None => 0.0,
        Some(i) => INDICES.iter().zip(HOLD_WEIGHTS[i].iter())
            .map(|(&j, &w)| features[j] * w).sum(),
    }
}

pub(super) fn adjust(score: i64, state: &State) -> i64 {
    let hold = state.held_color();
    if hold.is_none() { return score; }
    let delta = correction(&board_features(state.matrix()), hold);
    (score as f64 + delta * 10.0).max(0.0).round() as i64
}

/// The cached object is a vector of all type corrections, not a Hold-specific
/// final score. Different held types on the same board remain distinguishable.
#[cfg(feature = "hold-context-cache")]
#[derive(Default)]
pub(super) struct Cache(std::collections::HashMap<crate::BasicMatrix, [f64; 7]>);

#[cfg(feature = "hold-context-cache")]
impl Cache {
    pub(super) fn adjust(&mut self, score: i64, state: &State) -> i64 {
        let index = match state.held_color()
            .and_then(|c| TYPES.chars().position(|t| t == c.as_char())) {
            Some(i) => i,
            None => return score,
        };
        let deltas = self.0.entry(state.matrix().clone()).or_insert_with(|| {
            let features = board_features(state.matrix());
            let mut values = [0.0; 7];
            for (i,c) in TYPES.chars().enumerate() {
                values[i] = correction(&features, Some(Color::try_from(c).unwrap()));
            }
            values
        });
        (score as f64 + deltas[index] * 10.0).max(0.0).round() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ai::Snapshot, BasicMatrix};
    #[test]
    fn empty_hold_is_not_the_second_preview() {
        let state: State = Snapshot { matrix: BasicMatrix::with_cols(10), hold: None,
            queue: "IJ".chars().map(Color::n).collect() }.into();
        assert_eq!(state.next().1, Some(Color::n('J')));
        assert_eq!(state.held_color(), None);
        assert_eq!(adjust(120, &state), 120);
    }
    #[test]
    fn held_piece_remains_available_when_preview_is_exhausted() {
        let state: State = Snapshot { matrix: BasicMatrix::with_cols(10),
            hold: Some(Color::n('I')), queue: vec![] }.into();
        assert_eq!(state.is_terminal(), cfg!(feature = "preserve-hold"));
        assert_eq!(state.held_color(), Some(Color::n('I')));
    }
    #[test]
    fn type_corrections_are_centered_for_every_feature() {
        for j in 0..14 {
            let mut features = [0.;14]; features[j] = 1.;
            let total: f64 = TYPES.chars().map(|c| correction(&features, Some(Color::n(c)))).sum();
            assert!(total.abs() < 1e-10);
            assert_eq!(correction(&features, None), 0.0);
        }
    }
    #[cfg(feature = "hold-context-cache")]
    #[test]
    fn cache_matches_direct_for_all_held_types_and_reused_boards() {
        let mut cache = Cache::default();
        for height in 0..8 {
            let mut matrix = BasicMatrix::with_cols(10);
            for row in 0..height {
                for col in 0..10 {
                    if col != row % 10 { matrix.set((row,col)); }
                }
            }
            for _ in 0..2 {
                for hold in std::iter::once(None).chain(TYPES.chars().map(|c|Some(Color::n(c)))) {
                    let state: State = Snapshot { matrix: matrix.clone(), hold,
                        queue: "IOT".chars().map(Color::n).collect() }.into();
                    for score in [0,1,20,100,1000] {
                        assert_eq!(cache.adjust(score,&state),adjust(score,&state));
                    }
                }
            }
        }
        assert_eq!(cache.0.len(),8);
    }
}
