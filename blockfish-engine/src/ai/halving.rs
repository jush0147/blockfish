//! Root-action sequential halving, retaining the original B* inside each arm.
//! All root enumeration, scouting and refinement share one generated-node cap.
use super::b_star::{self, Evaluator, MoveId, Node, SearchTerminated, Step};
use super::state::State;
use crate::{config::Parameters, place::PlaceFinder, shape::ShapeTable};
use std::collections::VecDeque;

struct Arm<'s> {
    id: MoveId,
    scout_score: i64,
    search: Option<b_star::Search<'s>>,
    rating: Option<i64>,
    exhausted: bool,
    target: usize,
}

pub(super) struct Search<'s> {
    shapes: &'s ShapeTable,
    params: Parameters,
    finder: PlaceFinder<'s>,
    evaluator: Evaluator,
    root: Option<Node>,
    arms: Vec<Arm<'s>>,
    active: Vec<usize>,
    removed: VecDeque<MoveId>,
    cursor: usize,
    scouting: bool,
    budget: usize,
    generated: usize,
    scout_nodes: Option<usize>,
}

impl<'s> Search<'s> {
    pub fn new(shapes: &'s ShapeTable, params: Parameters) -> Self {
        Self { shapes, params, finder: PlaceFinder::new(shapes),
            evaluator: Evaluator::default(), root: None, arms: vec![],
            active: vec![], removed: VecDeque::new(), cursor: 0, scouting: true,
            budget: 0, generated: 0, scout_nodes: None }
    }
    pub fn set_budget(&mut self, budget: usize) { self.budget = budget; }
    pub fn start(&mut self, state: State) {
        self.arms.clear(); self.active.clear(); self.removed.clear();
        self.cursor = 0; self.scouting = true; self.generated = 1;
        self.scout_nodes = None;
        self.evaluator = Evaluator::default();
        state.placements(&mut self.finder);
        self.root = Some(Node::root(&self.params, state, &mut self.evaluator));
    }
    pub fn generated_nodes(&self) -> usize { self.generated }
    pub fn budget_used(&self) -> usize { self.generated }
    pub fn node_count(&self) -> usize {
        usize::from(self.root.is_some()) + self.arms.iter()
            .filter_map(|a| a.search.as_ref()).map(|s| s.node_count()).sum::<usize>()
    }
    pub fn report_allocation(&self) {
        eprintln!("ROOT_ALLOCATION {}",serde_json::json!({
            "generated":self.generated,"roots":self.arms.len(),
            "rated_roots":self.arms.iter().filter(|a|a.rating.is_some()).count(),
            "survivors":self.active.len(),"scout_complete":!self.scouting,
            "scout_nodes":self.scout_nodes.unwrap_or(self.generated)
        }));
    }
    fn begin_round(&mut self) {
        let arms = &self.arms;
        self.active.sort_by_key(|&i| (arms[i].rating.unwrap_or(i64::MAX), arms[i].id));
        let n = self.active.len();
        if n == 0 { return; }
        let rounds = if n == 1 { 1 } else { usize::BITS - (n-1).leading_zeros() } as usize;
        let quota = (self.budget.saturating_sub(self.generated) / rounds / n).max(1);
        for &i in &self.active {
            let arm = &mut self.arms[i];
            arm.target = arm.search.as_ref().unwrap().generated_nodes().saturating_add(quota);
        }
        self.cursor = 0;
    }
    fn advance(&mut self, index: usize) -> Result<Step, SearchTerminated> {
        let arm = &mut self.arms[index];
        let search = arm.search.as_mut().unwrap();
        let before = search.generated_nodes();
        let result = search.step();
        self.generated += search.generated_nodes() - before;
        match result {
            Ok(Step::RatingChanged { rating, .. }) => arm.rating = Some(rating),
            Err(_) => arm.exhausted = true,
            _ => {}
        }
        // Exhausting one arm does not terminate the whole root search.
        Ok(result.unwrap_or(Step::Other))
    }
    pub fn step(&mut self) -> Result<Step, SearchTerminated> {
        if let Some(id) = self.removed.pop_front() { return Ok(Step::RemoveMove(id)); }
        if let Some(root) = self.root.as_ref() {
            if let Some(place) = self.finder.next() {
                let node = root.succ(&self.params, &place, &mut self.evaluator);
                let id = MoveId(node.trace[0]);
                let scout_score = node.f;
                self.generated += 1;
                self.arms.push(Arm { id, scout_score, search: Some(b_star::Search::from_node(
                    self.shapes, self.params.clone(), node)), rating: None,
                    exhausted: false, target: 0 });
                return Ok(Step::Other);
            }
            self.root = None;
            self.active = (0..self.arms.len()).collect();
            let arms = &self.arms;
            self.active.sort_by_key(|&i| (arms[i].scout_score,arms[i].id));
        }
        if self.scouting {
            while self.cursor < self.active.len() {
                let i = self.active[self.cursor];
                if self.arms[i].rating.is_some() || self.arms[i].exhausted {
                    self.cursor += 1;
                } else { return self.advance(i); }
            }
            self.scouting = false;
            self.scout_nodes = Some(self.generated);
            let arms = &self.arms;
            self.active.retain(|&i| arms[i].rating.is_some());
            self.begin_round();
        }
        while self.cursor < self.active.len() {
            let i = self.active[self.cursor];
            let arm = &self.arms[i];
            if arm.exhausted || arm.search.as_ref().unwrap().generated_nodes() >= arm.target {
                self.cursor += 1;
            } else { return self.advance(i); }
        }
        if self.active.len() <= 1 { return Err(SearchTerminated); }
        let arms = &self.arms;
        self.active.sort_by_key(|&i| (arms[i].rating.unwrap(),arms[i].id));
        let keep = (self.active.len()+1)/2;
        for i in self.active.drain(keep..) {
            self.removed.push_back(self.arms[i].id);
            self.arms[i].search = None;
        }
        self.begin_round();
        Ok(self.removed.pop_front().map(Step::RemoveMove).unwrap_or(Step::Other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ai::Snapshot, shape::srs, BasicMatrix, Color};
    #[test]
    fn roots_are_scouted_and_budget_counts_all_arms() {
        let shapes=srs();
        let state:State=Snapshot { matrix:BasicMatrix::with_cols(10),hold:None,
            queue:"IJLOST".chars().map(Color::n).collect() }.into();
        let mut search=Search::new(&shapes,Parameters::default());
        search.set_budget(50000);search.start(state.clone());
        let mut moves=std::collections::HashMap::new();
        let mut retired=std::collections::HashSet::new();
        while search.budget_used()<50000 {
            let before=search.generated_nodes();
            match search.step() {
                Ok(Step::RatingChanged {move_id,rating,trace}) => {
                    assert!(!retired.contains(&move_id));
                    assert_eq!(trace[0],move_id.0 as usize);
                    // Every reported sequence reconstructs through the original legal generator.
                    let mut next=state.clone();let mut finder=PlaceFinder::new(&shapes);
                    for index in &trace {
                        let place=next.placements(&mut finder).find(|p|p.idx==*index).unwrap();
                        next.place(&place);
                    }
                    assert!(next.is_terminal());moves.insert(move_id,(rating,trace));
                }
                Ok(Step::RemoveMove(id)) => { assert!(moves.remove(&id).is_some());retired.insert(id); }
                Ok(_) => {}, Err(_) => break,
            }
            assert!(search.generated_nodes()<=before+1);
        }
        assert!(search.generated_nodes()<=50000);
        assert!(!search.scouting);
        assert!(search.arms.iter().all(|a|a.rating.is_some()||a.exhausted));
        assert!(!retired.is_empty());assert!(!moves.is_empty());
    }
}
