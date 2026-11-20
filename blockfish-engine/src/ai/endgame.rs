//! Bounded A*: minimize placements to clear the exact remaining garbage prefix.
//! ceil(empty garbage cells / 4) is admissible; learned value only breaks f ties.
use super::{analysis::reconstruct_inputs, eval::search_score, state::State, Snapshot};
use crate::{config::Parameters, place::PlaceFinder, shape::ShapeTable, Input};
use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashSet},
};

#[cfg(feature = "endgame-color-bound")]
fn pattern_bound(pattern: &[u16], memo: &mut std::collections::HashMap<Vec<u16>, usize>) -> usize {
    if pattern.is_empty() {
        return 0;
    }
    if let Some(&cost) = memo.get(pattern) {
        return cost;
    }
    let mut best = pattern.len();
    for start in 0..pattern.len() {
        for len in 1..=4.min(pattern.len() - start) {
            if pattern[start..start + len]
                .iter()
                .any(|&c| c != pattern[start])
            {
                break;
            }
            let mut next = pattern[..start].to_vec();
            next.extend_from_slice(&pattern[start + len..]);
            best = best.min(1 + pattern_bound(&next, memo));
        }
    }
    memo.insert(pattern.to_vec(), best);
    best
}

#[derive(Default)]
struct BoundMemo {
    patterns: std::collections::HashMap<Vec<u16>, usize>,
    #[cfg(feature = "endgame-queue-bound")]
    queues: std::collections::HashMap<(Vec<crate::Color>, bool, usize), usize>,
}

fn bound(state: &State, _memo: &mut BoundMemo) -> usize {
    let weak = state.goal_lower_bound();
    #[cfg(feature = "endgame-color-bound")]
    if let Some(pattern) = state.goal_hole_pattern() {
        let count = pattern_bound(&pattern, &mut _memo.patterns);
        #[cfg(feature = "endgame-queue-bound")]
        let count = state.useful_piece_bound(count, &mut _memo.queues);
        return weak.max(count);
    }
    weak
}

pub struct EndgameResult {
    pub plan: Option<Vec<Vec<Input>>>,
    pub generated_nodes: usize,
}

struct Node {
    state: State,
    trace: Vec<usize>,
    bound: usize,
    tie: i64,
    serial: usize,
}
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.serial == other.serial
    }
}
impl Eq for Node {}
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .bound
            .cmp(&self.bound)
            .then_with(|| other.tie.cmp(&self.tie))
            .then_with(|| self.trace.len().cmp(&other.trace.len()))
            .then_with(|| other.serial.cmp(&self.serial))
    }
}

pub(super) fn solve(
    shapes: &ShapeTable,
    params: &Parameters,
    snapshot: Snapshot,
    rows: u16,
    budget: usize,
) -> EndgameResult {
    if budget == 0 || rows == 0 {
        return EndgameResult {
            plan: None,
            generated_nodes: 0,
        };
    }
    let root = State::from(snapshot).with_goal_rows(rows);
    let mut bound_memo = BoundMemo::default();
    let mut heap = BinaryHeap::new();
    let mut seen = HashSet::new();
    seen.insert(root.clone());
    #[cfg(feature = "endgame-wide")]
    if bound(&root, &mut bound_memo) >= usize::MAX / 4 {
        return EndgameResult {
            plan: None,
            generated_nodes: 1,
        };
    }
    heap.push(Node {
        bound: bound(&root, &mut bound_memo),
        tie: 0,
        state: root.clone(),
        trace: vec![],
        serial: 0,
    });
    let mut finder = PlaceFinder::new(shapes);
    let mut generated = 1;
    while let Some(node) = heap.pop() {
        if node.state.is_terminal() {
            continue;
        }
        node.state.placements(&mut finder);
        for place in finder.by_ref() {
            if generated >= budget {
                return EndgameResult {
                    plan: None,
                    generated_nodes: generated,
                };
            }
            generated += 1;
            let mut state = node.state.clone();
            state.place(&place);
            let mut trace = node.trace.clone();
            trace.push(place.idx);
            if state.reached_goal() {
                // A goal child can fill at most four empty target cells: its
                // parent has h=1, so this goal reaches the lowest open f bound.
                debug_assert_eq!(trace.len(), node.bound);
                let inputs = reconstruct_inputs(shapes, root, &trace);
                let plan = inputs
                    .split(|i| *i == Input::HD)
                    .take(trace.len())
                    .map(|p| p.to_vec())
                    .collect();
                return EndgameResult {
                    plan: Some(plan),
                    generated_nodes: generated,
                };
            }
            if state.is_terminal() || !seen.insert(state.clone()) {
                continue;
            }
            let bound = trace.len() + bound(&state, &mut bound_memo);
            if bound >= usize::MAX / 4 {
                continue;
            }
            let tie = search_score(state.matrix(), params);
            heap.push(Node {
                state,
                trace,
                bound,
                tie,
                serial: generated,
            });
        }
    }
    EndgameResult {
        plan: None,
        generated_nodes: generated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{shape::srs, BasicMatrix, Color};

    #[cfg(feature = "endgame-queue-bound")]
    #[test]
    fn queue_bound_accounts_for_forced_o_and_empty_hold() {
        let mut memo = BoundMemo::default();
        assert_eq!(
            bound(&State::from(snapshot(1, "OI")).with_goal_rows(1), &mut memo),
            1
        );
        assert_eq!(
            bound(
                &State::from(snapshot(1, "OOI")).with_goal_rows(1),
                &mut memo
            ),
            2
        );
        assert!(
            bound(&State::from(snapshot(1, "OO")).with_goal_rows(1), &mut memo) >= usize::MAX / 4
        );
    }

    #[cfg(feature = "endgame-color-bound")]
    #[test]
    fn relaxed_garbage_removal_allows_merging_and_four_cell_limit() {
        let mut memo = std::collections::HashMap::new();
        assert_eq!(pattern_bound(&[0, 1, 2, 3], &mut memo), 4);
        assert_eq!(pattern_bound(&[0, 1, 0], &mut memo), 2);
        assert_eq!(pattern_bound(&[1, 1, 1, 1, 1], &mut memo), 2);
        assert_eq!(pattern_bound(&[], &mut memo), 0);
    }
    fn snapshot(rows: usize, queue: &str) -> Snapshot {
        let mut matrix = BasicMatrix::with_cols(10);
        for _ in 0..rows {
            matrix.push_row((0..10).map(|c| c != 0));
        }
        Snapshot {
            matrix,
            queue: queue.chars().map(Color::n).collect(),
            hold: None,
        }
    }
    #[test]
    fn bottom_clear_is_not_enough_when_target_garbage_remains() {
        let shapes = srs();
        let params = Parameters::default();
        let one = solve(&shapes, &params, snapshot(4, "I"), 4, 10_000);
        assert_eq!(one.plan.unwrap().len(), 1);
        let two = solve(&shapes, &params, snapshot(5, "II"), 5, 10_000);
        assert_eq!(two.plan.unwrap().len(), 2);
        assert!(solve(&shapes, &params, snapshot(5, "I"), 5, 10_000)
            .plan
            .is_none());
    }
    #[test]
    fn exact_solver_never_exceeds_generated_budget() {
        for budget in [0, 1, 2, 10, 100] {
            let result = solve(&srs(), &Parameters::default(), snapshot(5, "II"), 5, budget);
            assert!(result.generated_nodes <= budget);
        }
    }
}
