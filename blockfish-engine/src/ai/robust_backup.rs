//! Compare roots by several distinct plausible continuations, not one optimistic leaf.
use super::state::State;

#[derive(Default)]
struct LeafSet(Vec<(State, i64, Vec<usize>, bool)>);

#[derive(Default)]
pub(super) struct Leaves {
    global: LeafSet,
    #[cfg(feature = "hierarchical-backup")]
    branches: std::collections::HashMap<usize, LeafSet>,
}

impl LeafSet {
    #[cfg(feature = "support-probe")]
    pub(super) fn support(&self) -> (usize, usize) {
        let mut actions = std::collections::HashSet::new();
        for row in &self.0 {
            actions.insert(row.2.get(1).copied());
        }
        (self.0.len(), actions.len())
    }

    pub fn record(
        &mut self,
        state: State,
        rating: i64,
        trace: Vec<usize>,
        goal: bool,
    ) -> (i64, Vec<usize>) {
        if let Some(old) = self.0.iter_mut().find(|x| {
            if cfg!(feature = "branch-backup") {
                x.2.get(1) == trace.get(1)
            } else {
                x.0 == state
            }
        }) {
            if rating < old.1 {
                *old = (state, rating, trace, goal);
            }
        } else {
            self.0.push((state, rating, trace, goal));
        }
        self.0.sort_by_key(|x| x.1);
        self.0.truncate(4);
        let score = if self.0[0].3 {
            self.0[0].1
        } else {
            // Fixed-point quarter units avoid losing mean-score distinctions.
            self.0.iter().map(|x| x.1).sum::<i64>() * 4 / self.0.len() as i64
        };
        #[cfg(feature = "robust-placebo")]
        let score = if self.0[0].3 {
            self.0[0].1
        } else {
            self.0[0].1 * 4
        };
        (score, self.0[0].2.clone())
    }
}

impl Leaves {
    #[cfg(feature = "support-probe")]
    pub(super) fn support(&self) -> (usize, usize) {
        self.global.support()
    }

    pub fn record(
        &mut self,
        state: State,
        rating: i64,
        trace: Vec<usize>,
        goal: bool,
    ) -> (i64, Vec<usize>) {
        #[cfg(not(feature = "hierarchical-backup"))]
        {
            self.global.record(state, rating, trace, goal)
        }
        #[cfg(feature = "hierarchical-backup")]
        {
            let fallback = self
                .global
                .record(state.clone(), rating, trace.clone(), goal);
            if let Some(&action) = trace.get(1) {
                self.branches
                    .entry(action)
                    .or_default()
                    .record(state, rating, trace, goal);
            }
            if self.global.0[0].3 {
                return fallback;
            }
            // A known second action is controllable, not a chance outcome.
            // Require four supports before comparing its robust value; a lone
            // optimistic leaf must not beat a fully supported branch for free.
            self.branches
                .values()
                .filter(|b| b.0.len() == 4)
                .map(|b| (b.0.iter().map(|x| x.1).sum::<i64>(), b.0[0].2.clone()))
                .min()
                .unwrap_or(fallback)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ai::Snapshot, BasicMatrix, Color};
    fn state(c: char) -> State {
        Snapshot {
            matrix: BasicMatrix::with_cols(10),
            hold: None,
            queue: vec![Color::n(c)],
        }
        .into()
    }
    #[test]
    #[cfg(not(any(feature = "robust-placebo", feature = "branch-backup")))]
    fn duplicate_paths_cannot_fake_diverse_support() {
        let mut leaves = Leaves::default();
        assert_eq!(leaves.record(state('I'), 10, vec![0, 1], false).0, 40);
        assert_eq!(leaves.record(state('O'), 30, vec![0, 2], false).0, 80);
        for _ in 0..10 {
            assert_eq!(
                leaves.record(state('I'), 10, vec![0, 3], false),
                (80, vec![0, 1])
            );
        }
        assert_eq!(leaves.record(state('T'), 50, vec![0, 4], false).0, 120);
        assert_eq!(leaves.record(state('L'), 70, vec![0, 5], false).0, 160);
        assert_eq!(leaves.record(state('S'), 90, vec![0, 6], false).0, 160);
        assert_eq!(
            leaves.record(state('O'), 6, vec![0, 7], false),
            (136, vec![0, 7])
        );
    }
    #[test]
    fn exact_completion_retains_shortest_goal_preference() {
        let mut leaves = Leaves::default();
        leaves.record(state('I'), 100, vec![0, 1], false);
        assert_eq!(
            leaves.record(state('O'), 3, vec![0, 2], true),
            (3, vec![0, 2])
        );
    }
    #[test]
    #[cfg(all(feature = "branch-backup", not(feature = "robust-placebo")))]
    fn one_second_action_cannot_supply_multiple_supports() {
        let mut leaves = Leaves::default();
        assert_eq!(leaves.record(state('I'), 10, vec![0, 1, 0], false).0, 40);
        assert_eq!(leaves.record(state('O'), 20, vec![0, 1, 1], false).0, 40);
        assert_eq!(leaves.record(state('T'), 30, vec![0, 2, 0], false).0, 80);
        assert_eq!(leaves.record(state('J'), 50, vec![0, 3, 0], false).0, 120);
        assert_eq!(leaves.record(state('L'), 70, vec![0, 4, 0], false).0, 160);
        assert_eq!(leaves.record(state('S'), 90, vec![0, 5, 0], false).0, 160);
        assert_eq!(
            leaves.record(state('Z'), 6, vec![0, 2, 1], false),
            (136, vec![0, 2, 1])
        );
    }
    #[test]
    #[cfg(feature = "hierarchical-backup")]
    fn sparse_lucky_branch_does_not_replace_supported_known_choice() {
        let mut leaves = Leaves::default();
        for c in ['I', 'O', 'T', 'L'] {
            leaves.record(state(c), 10, vec![0, 1, c as usize], false);
        }
        assert_eq!(leaves.record(state('I'), 1, vec![0, 2, 0], false).0, 40);
        leaves.record(state('O'), 2, vec![0, 2, 1], false);
        leaves.record(state('T'), 3, vec![0, 2, 2], false);
        assert_eq!(
            leaves.record(state('L'), 4, vec![0, 2, 3], false),
            (10, vec![0, 2, 0])
        );
        // Unattractive alternatives cannot worsen the chosen supported branch.
        for c in ['I', 'O', 'T', 'L'] {
            assert_eq!(
                leaves
                    .record(state(c), 100, vec![0, 3, c as usize], false)
                    .0,
                10
            );
        }
    }
}
