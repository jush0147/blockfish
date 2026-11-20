//! Breadth beam ablation with a B* warmup to guarantee a completed fallback.
//! Uses the same states, legal placements, evaluations and terminal ratings.
use super::{
    b_star::{self, Evaluator, MoveId, Node, SearchTerminated, Step},
    state::State,
};
use crate::{config::Parameters, place::PlaceFinder, shape::ShapeTable};
use std::collections::{BinaryHeap, HashMap, VecDeque};

pub(super) struct Search<'s> {
    warmup: Option<b_star::Search<'s>>,
    params: Parameters,
    evaluator: Evaluator,
    finder: PlaceFinder<'s>,
    current: Option<Node>,
    parents: VecDeque<Node>,
    children: BinaryHeap<Node>,
    best: HashMap<MoveId, i64>,
    generated: usize,
    width: usize,
}

impl<'s> Search<'s> {
    pub fn new(shapes: &'s ShapeTable, params: Parameters) -> Self {
        Self {
            warmup: Some(b_star::Search::new(shapes, params.clone())),
            params,
            evaluator: Evaluator::default(),
            finder: PlaceFinder::new(shapes),
            current: None,
            parents: VecDeque::new(),
            children: BinaryHeap::new(),
            best: HashMap::new(),
            generated: 1,
            width: 1,
        }
    }

    pub fn set_budget(&mut self, limit: usize) {
        self.width = (limit
            / if cfg!(feature = "beam-widest") {
                128
            } else if cfg!(feature = "beam-wider") {
                192
            } else if cfg!(feature = "wide-beam") {
                256
            } else {
                512
            })
        .max(1);
    }

    pub fn start(&mut self, state: State) {
        self.warmup.as_mut().unwrap().start(state.clone());
        state.placements(&mut self.finder);
        self.current = Some(Node::root(&self.params, state, &mut self.evaluator));
    }

    pub fn generated_nodes(&self) -> usize {
        self.warmup
            .as_ref()
            .map_or(self.generated, |s| s.generated_nodes())
    }
    pub fn budget_used(&self) -> usize {
        self.generated_nodes()
    }
    pub fn node_count(&self) -> usize {
        self.warmup
            .as_ref()
            .map_or(self.parents.len() + self.children.len(), |s| s.node_count())
    }

    pub fn step(&mut self) -> Result<Step, SearchTerminated> {
        if let Some(warmup) = self.warmup.as_mut() {
            let step = warmup.step()?;
            if let Step::RatingChanged {
                move_id, rating, ..
            } = &step
            {
                self.best.insert(*move_id, *rating);
                self.generated = warmup.generated_nodes() + 1; // own root
                self.warmup = None;
            }
            return Ok(step);
        }
        if let Some(node) = self.current.take() {
            if node.is_terminal() {
                return self.rate(node);
            }
            if let Some(place) = self.finder.next() {
                let child = node.succ(&self.params, &place, &mut self.evaluator);
                self.generated += 1;
                self.current = Some(node);
                // Terminal nodes need no beam capacity. Rate every generated
                // leaf using the actual terminal rating, rather than discard
                // leaves by their different intermediate priority f.
                if cfg!(feature = "beam-leaves") && child.is_terminal() {
                    return self.rate(child);
                }
                self.children.push(child);
            }
            return Ok(Step::Other);
        }
        if self.parents.is_empty() {
            #[cfg(feature = "beam-dedup")]
            let mut seen = std::collections::HashSet::new();
            while self.parents.len() < self.width {
                match self.children.pop() {
                    Some(node) => {
                        #[cfg(feature = "beam-dedup")]
                        if !seen.insert(node.state.clone()) {
                            continue;
                        }
                        self.parents.push_back(node)
                    }
                    None => break,
                }
            }
            self.children.clear();
        }
        let node = self.parents.pop_front().ok_or(SearchTerminated)?;
        node.state.placements(&mut self.finder);
        self.current = Some(node);
        Ok(Step::Other)
    }

    fn rate(&mut self, node: Node) -> Result<Step, SearchTerminated> {
        let rating = node.rating();
        let trace: Vec<_> = node.trace().collect();
        let move_id = MoveId(*node.trace.first().ok_or(SearchTerminated)?);
        let best = self.best.entry(move_id).or_insert(i64::MAX);
        Ok(if rating < *best {
            *best = rating;
            Step::RatingChanged {
                move_id,
                rating,
                trace,
            }
        } else {
            Step::SequenceRejected { rating, trace }
        })
    }
}
