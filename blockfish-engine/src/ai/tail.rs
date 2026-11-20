//! One-step uniform future approximation, using no hidden queue or simulator RNG.
use super::{
    b_star::{Evaluator, Node},
    state::State,
    Snapshot,
};
use crate::{config::Parameters, place::PlaceFinder, Color};
use std::convert::TryFrom;

const COLORS: [char; 7] = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
const NO_PLACEMENT: i64 = 1_000_000;

pub(super) enum Progress {
    Generated,
    Boundary,
    Finished,
}

pub(super) struct Tail {
    node: Node,
    state: Option<State>,
    color: usize,
    best: i64,
    sum: i64,
}

impl Tail {
    pub fn new(node: Node) -> Self {
        Self {
            node,
            state: None,
            color: 0,
            best: NO_PLACEMENT,
            sum: 0,
        }
    }
    // At most one generated state per call, allowing the caller's strict budget
    // to interrupt without publishing a partially averaged future.
    pub fn step(
        &mut self,
        finder: &mut PlaceFinder<'_>,
        evaluator: &mut Evaluator,
        params: &Parameters,
    ) -> Progress {
        if self.color == COLORS.len() {
            return Progress::Finished;
        }
        if self.state.is_none() {
            let state: State = Snapshot {
                matrix: self.node.state.matrix().clone(),
                hold: if cfg!(feature = "tail-hold") {
                    self.node.state.next().1
                } else {
                    None
                },
                queue: vec![Color::try_from(COLORS[self.color]).unwrap()],
            }
            .into();
            state.placements(finder);
            self.state = Some(state);
            self.best = NO_PLACEMENT;
            return Progress::Generated;
        }
        if let Some(place) = finder.next() {
            let mut successor = self.state.as_ref().unwrap().clone();
            successor.place(&place);
            self.best = self.best.min(evaluator.score(&successor, params));
            Progress::Generated
        } else {
            self.sum += self.best;
            self.color += 1;
            self.state = None;
            Progress::Boundary
        }
    }
    pub fn finish(mut self, params: &Parameters) -> Node {
        assert_eq!(self.color, COLORS.len());
        if !cfg!(feature = "tail-placebo") {
            self.node
                .extend_value(params, self.sum / COLORS.len() as i64);
        }
        self.node
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shape::srs, BasicMatrix};
    #[test]
    fn future_average_matches_explicit_seven_piece_enumeration() {
        let params = Parameters::default();
        let shapes = srs();
        let state: State = Snapshot {
            matrix: BasicMatrix::with_cols(10),
            hold: None,
            queue: vec![],
        }
        .into();
        let mut evaluator = Evaluator::default();
        let mut tail = Tail::new(Node::root(&params, state, &mut evaluator));
        let mut finder = PlaceFinder::new(&shapes);
        let mut charged = 0;
        loop {
            match tail.step(&mut finder, &mut evaluator, &params) {
                Progress::Generated => charged += 1,
                Progress::Boundary => (),
                Progress::Finished => break,
            }
        }
        let mut sum = 0;
        let mut expected_nodes = 7;
        for color in COLORS {
            let state: State = Snapshot {
                matrix: BasicMatrix::with_cols(10),
                hold: None,
                queue: vec![Color::try_from(color).unwrap()],
            }
            .into();
            let mut best = NO_PLACEMENT;
            let places: Vec<_> = state.placements(&mut finder).collect();
            for place in places {
                expected_nodes += 1;
                let mut next = state.clone();
                next.place(&place);
                best = best.min(evaluator.score(&next, &params));
            }
            sum += best;
        }
        assert_eq!(charged, expected_nodes);
        assert_eq!(tail.sum, sum);
        assert!(charged > 7);
    }
    #[test]
    #[cfg(feature = "tail-hold")]
    fn known_held_i_remains_available_for_every_hypothetical_current() {
        let params = Parameters::default();
        let shapes = srs();
        let mut board = BasicMatrix::with_cols(10);
        for row in 0..4 {
            for col in 0..10 {
                if col != 4 {
                    board.set((row, col));
                }
            }
        }
        let root: State = Snapshot {
            matrix: board.clone(),
            hold: Some(Color::n('I')),
            queue: vec![],
        }
        .into();
        let mut evaluator = Evaluator::default();
        let empty: State = Snapshot {
            matrix: BasicMatrix::with_cols(10),
            hold: None,
            queue: vec![],
        }
        .into();
        let empty_score = evaluator.score(&empty, &params);
        let mut tail = Tail::new(Node::root(&params, root, &mut evaluator));
        let mut finder = PlaceFinder::new(&shapes);
        while !matches!(
            tail.step(&mut finder, &mut evaluator, &params),
            Progress::Finished
        ) {}
        assert_eq!(tail.sum, 7 * empty_score);
        let no_hold: State = Snapshot {
            matrix: board,
            hold: None,
            queue: vec![Color::n('O')],
        }
        .into();
        let places: Vec<_> = no_hold.placements(&mut finder).collect();
        let best_without = places
            .iter()
            .map(|place| {
                let mut next = no_hold.clone();
                next.place(place);
                evaluator.score(&next, &params)
            })
            .min()
            .unwrap();
        assert!(
            best_without > empty_score,
            "an O alone cannot clear the I well"
        );
    }
}
