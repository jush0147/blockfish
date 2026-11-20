use super::Snapshot;
use crate::{
    common::Color,
    matrix::BasicMatrix,
    place::{Place, PlaceFinder},
};

/// A game state: matrix plus queue/hold.
///
/// Represention should prioritize efficiently taking pieces from the queue/hold slot.
#[derive(Clone, Eq, PartialEq, Hash)]
pub struct State {
    #[cfg(feature="reveal-boundary")]
    reveal: Option<(u16,u16,u16)>,
    #[cfg(feature="reveal-boundary")]
    needs_reveal: bool,
    matrix: BasicMatrix,
    queue_rev: Vec<Color>,
    has_held: bool,
    reached_goal: bool,
    final_goal: bool,
    #[cfg(feature = "endgame-solver")]
    goal_rows: Option<u16>,
}

impl State {
    #[cfg(feature = "hold-context")]
    pub(super) fn held_color(&self) -> Option<Color> {
        if self.has_held { self.next().1 } else { None }
    }
    #[cfg(feature = "endgame-queue-bound")]
    pub fn useful_piece_bound(
        &self,
        required: usize,
        memo: &mut std::collections::HashMap<(Vec<Color>, bool, usize), usize>,
    ) -> usize {
        fn visit(
            queue: &[Color],
            held: bool,
            required: usize,
            memo: &mut std::collections::HashMap<(Vec<Color>, bool, usize), usize>,
        ) -> usize {
            if required == 0 {
                return 0;
            }
            let key = (queue.to_vec(), held, required);
            if let Some(&cost) = memo.get(&key) {
                return cost;
            }
            let mut best = usize::MAX / 4;
            for hold in [false, true] {
                let pos = if held == hold { 1 } else { 2 };
                if queue.len() < pos {
                    continue;
                }
                let mut next = queue.to_vec();
                let color = next.remove(queue.len() - pos);
                let remaining = required.saturating_sub((color.as_char() != 'O') as usize);
                best = best.min(1 + visit(&next, held || hold, remaining, memo));
            }
            memo.insert(key, best);
            best
        }
        visit(&self.queue_rev, self.has_held, required, memo)
    }
    #[cfg(feature = "endgame-solver")]
    pub fn with_goal_rows(mut self, rows: u16) -> Self {
        assert!(rows <= self.matrix.rows());
        self.goal_rows = Some(rows);
        self.reached_goal = rows == 0;
        self
    }

    #[cfg(feature = "endgame-solver")]
    pub fn goal_lower_bound(&self) -> usize {
        (self
            .matrix
            .empty_cells_below(self.goal_rows.expect("endgame state"))
            + 3)
            / 4
    }

    #[cfg(feature = "endgame-color-bound")]
    pub fn goal_hole_pattern(&self) -> Option<Vec<u16>> {
        self.matrix
            .single_hole_columns(self.goal_rows.expect("endgame state"))
    }
    pub fn with_final_goal(mut self, final_goal: bool) -> Self {
        self.final_goal = final_goal;
        self
    }

    pub fn goal_is_final(&self) -> bool {
        self.final_goal
    }

    pub fn matrix(&self) -> &BasicMatrix {
        &self.matrix
    }

    pub fn reached_goal(&self) -> bool {
        self.reached_goal
    }

    #[cfg(feature="reveal-boundary")]
    pub fn until_reveal(mut self, visible:u16, minimum:u16, maximum:u16, unseen:bool) -> Self {
        self.final_goal = !unseen;
        self.reveal = unseen.then_some((visible,minimum.min(maximum),maximum));
        self
    }
    pub fn is_terminal(&self) -> bool {
        #[cfg(feature="reveal-boundary")]
        if self.needs_reveal {return true;}
        self.reached_goal
            || self.queue_rev.is_empty()
            || (cfg!(feature = "preserve-hold") && self.has_held && self.queue_rev.len() == 1)
    }

    /// Returns the next available piece colors, as pair `(next_piece, hold_piece)`.
    /// Either may be `None` if not available (e.g. the next queue is empty).
    ///
    /// Note: `hold_piece` is not exactly the current piece in hold, rather the piece you
    /// will get if you press hold, i.e. if hold is currently empty then it refers to the
    /// 2nd piece in the queue.
    pub fn next(&self) -> (Option<Color>, Option<Color>) {
        let from_top = |i| {
            self.queue_rev
                .len()
                .checked_sub(i)
                .and_then(|i| self.queue_rev.get(i))
                .cloned()
        };
        let c1 = from_top(1);
        let c2 = from_top(2);
        if self.has_held {
            (c2, c1)
        } else {
            (c1, c2)
        }
    }

    /// Configures the placements iterator `pls` to produce the set of valid placements
    /// for this state.
    pub fn placements<'p, 's>(&self, pfind: &'p mut PlaceFinder<'s>) -> &'p mut PlaceFinder<'s> {
        pfind.reset_matrix(&self.matrix);
        let (color_nh, color_h) = self.next();
        if let Some(c) = color_nh {
            pfind.push_shape(c, false);
        }
        if let Some(c) = color_h {
            // don't use hold piece if its identical to current piece
            if color_h != color_nh {
                pfind.push_shape(c, true);
            }
        }
        pfind
    }

    /// Applies the given placement to this state, modifying the queue and matrix.
    pub fn place(&mut self, pl: &Place) {
        pl.shape.blit_to(&mut self.matrix, pl.tf);
        #[cfg(feature = "endgame-solver")]
        if let Some(rows) = self.goal_rows {
            let remaining = rows - self.matrix.full_rows_below(rows);
            self.goal_rows = Some(remaining);
            self.matrix.sift_rows();
            self.reached_goal = remaining == 0;
            self.pop(pl.did_hold);
            return;
        }
        #[cfg(feature="reveal-boundary")]
        if let Some((visible, minimum, maximum)) = self.reveal {
            let remaining = visible - self.matrix.full_rows_below(visible);
            let any_clear = self.matrix.full_rows_below(self.matrix.rows()) > 0;
            self.needs_reveal = remaining < if any_clear { minimum } else { maximum };
            self.reveal = Some((remaining,minimum,maximum));
        }
        self.reached_goal |= self.matrix.sift_rows();
        self.pop(pl.did_hold);
    }

    /// Removes a piece from the next queue, or hold slot if `hold` is `true`.
    fn pop(&mut self, hold: bool) {
        //  | has_held | hold  | pos
        // -+----------+-------+-----
        //  | true     | false | 2
        //  | true     | true  | 1
        //  | false    | false | 1
        //  | false    | true  | 2
        let pos = if self.has_held == hold { 1 } else { 2 };
        self.queue_rev.remove(self.queue_rev.len() - pos);
        self.has_held |= hold;
    }
}

impl From<Snapshot> for State {
    fn from(snapshot: Snapshot) -> Self {
        let matrix = snapshot.matrix;
        // the queue is represented in reverse order, so the next item can easily be
        // removed. the hold piece (if any) is stored on top, after the previews.
        let mut queue_rev = snapshot.queue;
        queue_rev.reverse();
        let mut has_held = false;
        if let Some(hold_color) = snapshot.hold {
            has_held = true;
            queue_rev.push(hold_color);
        }
        Self {
            #[cfg(feature="reveal-boundary")]
            reveal:None,
            #[cfg(feature="reveal-boundary")]
            needs_reveal:false,
            matrix,
            queue_rev,
            has_held,
            reached_goal: false,
            final_goal: true,
            #[cfg(feature = "endgame-solver")]
            goal_rows: None,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{shape::srs, Orientation::*};

    #[test]
    fn test_state_operations() {
        let queue = || "LTJI".chars().map(Color::n);

        let mut s: State = Snapshot {
            hold: None,
            queue: queue().collect(),
            matrix: BasicMatrix::with_cols(10),
        }
        .into();
        assert!(!s.is_terminal());
        assert_eq!(s.matrix.rows(), 0);
        assert_eq!(s.matrix.cols(), 10);
        assert_eq!(s.next(), (Some(Color::n('L')), Some(Color::n('T'))));

        let srs = srs();
        for (i, color) in queue().enumerate() {
            let shape = srs.shape(color).unwrap();
            let tf = ((i * 2) as i16 - 1, 0, R0);
            s.place(&Place::new(shape, tf, false));
        }
        assert!(s.is_terminal());
        assert_eq!(s.matrix.rows(), 7);
        assert_eq!(s.next(), (None, None));
    }

    #[test]
    fn test_state_use_hold() {
        // something already in hold
        let mut s: State = Snapshot {
            hold: Some(Color::n('S')),
            queue: "LTJI".chars().map(Color::n).collect(),
            matrix: BasicMatrix::with_cols(10),
        }
        .into();
        assert_eq!(s.next(), (Some(Color::n('L')), Some(Color::n('S'))));
        s.pop(true);
        assert_eq!(s.next(), (Some(Color::n('T')), Some(Color::n('L'))));
        s.pop(false);
        assert_eq!(s.next(), (Some(Color::n('J')), Some(Color::n('L'))));
        // nothing previously in hold
        s = Snapshot {
            hold: None,
            queue: "LTJI".chars().map(Color::n).collect(),
            matrix: BasicMatrix::with_cols(10),
        }
        .into();
        assert_eq!(s.next(), (Some(Color::n('L')), Some(Color::n('T'))));
        s.pop(true);
        assert_eq!(s.next(), (Some(Color::n('J')), Some(Color::n('L'))));
    }

    #[test]
    fn test_state_nearly_empty_queue() {
        let mut s: State = Snapshot {
            hold: None,
            queue: vec![Color::n('I')],
            matrix: BasicMatrix::with_cols(10),
        }
        .into();
        assert_eq!(s.next(), (Some(Color::n('I')), None));
        s = Snapshot {
            hold: Some(Color::n('O')),
            queue: vec![],
            matrix: BasicMatrix::with_cols(10),
        }
        .into();
        assert_eq!(s.next(), (None, Some(Color::n('O'))));
    }
}

#[cfg(all(test,feature="reveal-boundary"))]
mod reveal_tests {
    use super::*;
    use crate::shape::srs;

    // A four-cell gap admits a horizontal I clear and other nonclearing placements.
    #[test]
    fn clearing_and_nonclearing_locks_use_different_refill_heights() {
        let mut board = BasicMatrix::with_cols(10);
        for r in 0..5 {
            for c in 0..10 {
                if (r==4 && c>=4) || (r<4 && c!=r) {board.set((r,c));}
            }
        }
        let snapshot=Snapshot {matrix:board,hold:None,queue:"IOTLSZ".chars().map(Color::n).collect()};
        let initial=State::from(snapshot).until_reveal(5,4,9,true);
        let shapes=srs();
        let mut finder=PlaceFinder::new(&shapes);
        let mut observed=[false;2];
        for p in initial.placements(&mut finder) {
            let mut child=initial.clone();
            child.place(&p);
            let remaining=child.reveal.unwrap().0;
            if remaining==4 {
                assert!(!child.needs_reveal);
                observed[0]=true;
            } else if remaining==5 {
                assert!(child.needs_reveal && child.is_terminal());
                observed[1]=true;
            }
        }
        assert_eq!(observed,[true,true]);
    }
}
