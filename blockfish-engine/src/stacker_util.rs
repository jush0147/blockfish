use crate::{
    ai::Snapshot,
    common::{Color, Input},
    matrix::BasicMatrix,
};
use std::convert::TryInto;

#[cfg(test)]
mod projection_tests {
    use super::*;
    #[test]
    fn projections_do_not_mutate_game_or_random_stream() {
        let mut cfg = block_stacker::Config::default();
        cfg.prng_seed = Some(42);
        cfg.garbage.total_lines = Some(0);
        let mut game = block_stacker::Stacker::new(block_stacker::Ruleset::guideline().into(), cfg);
        let mut twin = game.clone();
        let snapshot = game.snapshot();
        let features = game.projected_features(&[]);
        for _ in 0..20 {
            assert_eq!(features, game.projected_features(&[]));
        }
        assert_eq!(snapshot, game.snapshot());
        assert_eq!(game.hard_drop(), twin.hard_drop());
        assert_eq!(game.snapshot(), twin.snapshot());
        assert_eq!(
            features,
            crate::ai::board_features(&twin.snapshot().unwrap().matrix)
        );
    }
}

pub trait StackerExt {
    /// Returns a snapshot of the game state. Returns `None` if the snapshot would
    /// be invalid (e.g., there is no current piece).
    fn snapshot(&self) -> Option<Snapshot>;

    /// Features after a proposed lock, excluding future garbage; no hard drop in inputs.
    fn projected_features(&self, inputs: &[Input]) -> [f64; 14];

    /// Simulates input `inp` on the game state.
    fn input(&mut self, inp: Input);

    /// Simulates all of the inputs in `iter` on the game state.
    fn run<I: IntoIterator<Item = Input>>(&mut self, iter: I) {
        for inp in iter {
            self.input(inp);
        }
    }
}

impl StackerExt for block_stacker::Stacker {
    fn projected_features(&self, inputs: &[Input]) -> [f64; 14] {
        assert!(!inputs.contains(&Input::HD));
        let mut position = self.clone();
        position.freeze();
        position.run(inputs.iter().copied());
        position.hard_drop();
        let mut matrix = BasicMatrix::with_cols(position.ruleset().cols as u16);
        for (coord, _) in position.matrix() {
            matrix.set(coord);
        }
        matrix.sift_rows();
        crate::ai::board_features(&matrix)
    }

    fn snapshot(&self) -> Option<Snapshot> {
        fn to_color(typ: block_stacker::PieceType) -> Color {
            typ.try_into().expect("bug: invalid color")
        }
        let cur = to_color(self.current_piece_type()?);
        let next = self.next().iter().map(|&typ| to_color(typ));
        let queue = std::iter::once(cur).chain(next).collect();
        let hold = self.held().map(to_color);
        let cols = self.ruleset().cols;
        let mut matrix = BasicMatrix::with_cols(cols as u16);
        for (coord, _) in self.matrix() {
            matrix.set(coord);
        }
        Some(Snapshot {
            queue,
            hold,
            matrix,
        })
    }

    fn input(&mut self, inp: Input) {
        match inp {
            Input::Left => {
                self.move_horizontal(-1);
            }
            Input::Right => {
                self.move_horizontal(1);
            }
            Input::CCW => {
                self.rotate(-1);
            }
            Input::CW => {
                self.rotate(1);
            }
            Input::Hold => {
                self.hold();
            }
            Input::HD => {
                self.hard_drop();
            }
            Input::SD => {
                self.sonic_drop();
            }
        }
    }
}
