use super::{
    eval::{penalty, search_score},
    state::State,
};
use crate::{
    config::Parameters,
    place::{Place, PlaceFinder},
    shape::ShapeTable,
};
use std::collections::{BinaryHeap, HashMap};

// Search algorithm

/// An instance of the "B*" search algorithm.
///
/// B* is a slight modification of A* to better fit the needs of blockfish. The primary
/// motivation for a new search algorithm, is that one must always be placing pieces; that
/// is, a good looking game state is not actually good if the following placements lead it
/// to being worse. Qhile A* has been good at finding game states with good evaluations,
/// it is not good at ruling them out if the subsequent placements are bad.
///
/// The gist of B* is that it does a best-first search to find full sequences (every piece
/// in the queue is placed), then tries to improve sequences by modifying mid-sequence
/// moves. Crucially, evaluations are only ever propogated back once the best-first search
/// has finished. This means a placement is only actually considered good if the full
/// sequence is good.
///
/// While A* contains a single fringe set (priority queue ordered by `f(n)` value), B*
/// works by keeping separate fringe sets for each depth level. That is, when a node at
/// level `i` is expanded, its successors are put into the fringe set for level `i+1`. The
/// algorithm starts with best-first search at level `0` with just the root node in the
/// fringe set. This phase proceeds by picking a node from depth level `i`, expanded it,
/// then repeating at depth level `i+1`, until a terminal node is found. This terminal
/// node's rating is backed up, then the best-first search begins again starting at the
/// node with lowest `f(n)` (among all depth levels).
///
/// NOTE: the final "rating" given to a terminal node† is not actually its `f(n)` value,
/// but rather `f(n) + f(n.parent)`. The reasoning for this, is that the final node in a
/// sequence is not actually the only placement available. Because of "hold", there may
/// actually be some other type of piece immediately beyond the queue that could be placed
/// instead. Adding the two f values rounds off the rating for sequences that were pretty
/// good until the last piece, since this last piece won't be the only option. This edge
/// condition will hopefully be removed once predicting-pieces-after-the-previews is
/// implemented.
///
/// † unless this node clears the bottom line of the matrix, i.e. "reaches the goal" -- in
///   this case the node will be given a very low rating proportional to the number of
///   placed pieces, in order to prioritize short sequences at the very end of the cheese
///   race.
pub struct Search<'s> {
    #[cfg(feature="plan-reuse")]
    plan:PlanReuse,
    #[cfg(feature = "root-dag")]
    expanded: std::collections::HashSet<(NodeState, usize, u8)>,
    #[cfg(feature = "root-dag")]
    duplicate_expansions: usize,
    #[cfg(feature = "root-dag")]
    merge_transpositions: bool,
    #[cfg(feature = "root-focus")]
    focus: Focus,
    // parameters to the heuristic function
    params: Parameters,
    // holds the best rating for each move
    move_best: HashMap<MoveId, i64>,
    #[cfg(feature = "robust-backup")]
    root_leaves: HashMap<MoveId, super::robust_backup::Leaves>,
    // fringe set for each depth level
    lvls: Vec<BinaryHeap<Node>>,
    // index of current depth level either being selected from or expanded into
    lvl_idx: usize,
    // current node being expanded
    node: Option<Node>,
    // placement generator; only used when `node` is not `None`
    pfind: PlaceFinder<'s>,
    // number of nodes retained in the frontier (legacy budget)
    node_count: usize,
    evaluator: Evaluator,
    #[cfg(feature = "tail-expectation")]
    tail: Option<super::tail::Tail>,
    #[cfg(feature = "tail-expectation")]
    tail_finder: PlaceFinder<'s>,
    generated: usize,
    #[cfg(feature = "root-complete")]
    covering_root: Option<u8>,
    #[cfg(feature = "deduplicate")]
    seen: std::collections::HashSet<(NodeState, usize, u8, i64)>,
}

#[cfg(feature = "root-focus")]
#[derive(Default)]
struct Focus {
    threshold: usize,
    allowed: Option<std::collections::HashSet<u8>>,
    removed: std::collections::VecDeque<MoveId>,
    trigger_nodes: usize,
    rated_before: usize,
    pruned_nodes: usize,
}

#[cfg(feature="plan-reuse")]
#[derive(Default)]
struct PlanReuse {
    hint:super::PlanHint,active:bool,prefix:Vec<u8>,target:Option<Vec<u8>>,
    supplied:usize,matched:usize,followed:usize,fallback:Option<usize>,
}

// Cache only static matrix scores. Queue, hold and path cost are deliberately
// excluded: they are not inputs to eval(). A cache lives for one analysis.
#[derive(Default)]
pub(super) struct Evaluator {
    #[cfg(feature = "eval-cache")]
    scores: HashMap<crate::matrix::BasicMatrix, i64>,
    #[cfg(feature = "hold-context-cache")]
    hold_scores: super::hold_context::Cache,
}

impl Evaluator {
    pub(super) fn score(&mut self, state: &State, params: &Parameters) -> i64 {
        let score = self.static_score(state, params);
        #[cfg(all(feature = "hold-context", not(feature = "hold-context-cache")))]
        let score = super::hold_context::adjust(score, state);
        #[cfg(feature = "hold-context-cache")]
        let score = self.hold_scores.adjust(score, state);
        #[cfg(feature = "hold-credit")]
        if state.next().1.map(|c| c.as_char()) == Some('I') {
            return score - super::eval::hold_credit(state.matrix(), params);
        }
        score
    }

    fn static_score(&mut self, state: &State, params: &Parameters) -> i64 {
        #[cfg(feature = "eval-cache")]
        {
            if let Some(&score) = self.scores.get(state.matrix()) {
                return score;
            }
            let score = search_score(state.matrix(), params);
            self.scores.insert(state.matrix().clone(), score);
            score
        }
        #[cfg(not(feature = "eval-cache"))]
        {
            search_score(state.matrix(), params)
        }
    }
}

/// Opaque identifier that indicates a "move" -- the next placement one make after the
/// initial state.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct MoveId(pub(super) u8);

/// Indicates what happened as a result of a step of the algorithm. Returned by
/// `Search::step()`.
#[derive(Clone, Debug)]
pub enum Step {
    /// Indicates that the rating for a particular move has changed since the last iteration
    /// of the analysis.
    RatingChanged {
        /// Move whos rating has changed.
        move_id: MoveId,
        /// Placement trace for the new best-sequence for this move.
        trace: Vec<usize>,
        /// The new rating for this move.
        rating: i64,
    },

    /// Indicates a node was discovered but was rejected since it is not better than the
    /// current best node for the move.
    SequenceRejected {
        /// Placement trace for this sequence.
        trace: Vec<usize>,
        /// The sequence's rating.
        rating: i64,
    },

    Other,
    RemoveMove(MoveId),
}

/// Indicates that the search is over since there are no more placements left to analyze.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct SearchTerminated;

impl<'s> Search<'s> {
    #[cfg(feature = "root-focus")]
    pub fn set_budget(&mut self, budget: usize) {
        self.focus.threshold = (budget / 4).max(1);
    }
    #[cfg(feature = "root-focus")]
    fn focus_roots(&mut self) {
        if self.focus.allowed.is_some() || self.generated < self.focus.threshold
            || self.move_best.is_empty() { return; }
        let mut ranked: Vec<_> = self.move_best.iter().map(|(&id,&rating)|(rating,id)).collect();
        ranked.sort();
        let keep=(ranked.len()+1)/2;
        let allowed: std::collections::HashSet<u8> = ranked[..keep].iter().map(|p|p.1.0).collect();
        self.focus.rated_before=ranked.len();
        self.focus.trigger_nodes=self.generated;
        for &(_,id) in &ranked[keep..] {
            self.focus.removed.push_back(id);
            self.move_best.remove(&id);
            self.root_leaves.remove(&id);
        }
        let old=self.lvls.iter().map(|h|h.len()).sum::<usize>()+usize::from(self.node.is_some());
        for heap in &mut self.lvls {
            heap.retain(|n|n.trace.first().map_or(false,|id|allowed.contains(id)));
        }
        if self.node.as_ref().map_or(false,|n|n.trace.first().map_or(true,|id|!allowed.contains(id))) {
            self.node=None;
        }
        self.node_count=self.lvls.iter().map(|h|h.len()).sum::<usize>()+usize::from(self.node.is_some());
        self.focus.pruned_nodes=old-self.node_count;
        self.focus.allowed=Some(allowed);
    }
    #[cfg(feature = "root-focus")]
    pub fn report_focus(&self) {
        eprintln!("ROOT_FOCUS {}",serde_json::json!({"generated":self.generated,
            "activated":self.focus.allowed.is_some(),"trigger_nodes":self.focus.trigger_nodes,
            "rated_before":self.focus.rated_before,"retained_roots":self.focus.allowed.as_ref().map_or(0,|s|s.len()),
            "pruned_frontier":self.focus.pruned_nodes}));
    }
    #[cfg(feature = "root-halving")]
    pub(super) fn from_node(shape_table: &'s ShapeTable, params: Parameters, node: Node) -> Self {
        let mut search = Self::new(shape_table, params);
        node.state.placements(&mut search.pfind);
        search.node_count = 1;
        // Outer scheduler already counted this first-placement node.
        search.generated = 0;
        search.node = Some(node);
        search
    }

    /// Constructs a new instance of the "B*" search algorithm.
    pub fn new(shape_table: &'s ShapeTable, params: Parameters) -> Self {
        Self {
            #[cfg(feature="plan-reuse")]
            plan:PlanReuse::default(),
            #[cfg(feature = "root-dag")]
            expanded: Default::default(),
            #[cfg(feature = "root-dag")]
            duplicate_expansions: 0,
            #[cfg(feature = "root-dag")]
            merge_transpositions: !cfg!(feature = "dag-observer"),
            #[cfg(feature = "root-focus")]
            focus: Focus::default(),
            params,
            move_best: HashMap::with_capacity(64),
            #[cfg(feature = "robust-backup")]
            root_leaves: HashMap::new(),
            lvls: Vec::with_capacity(8),
            lvl_idx: 0,
            node: None,
            pfind: PlaceFinder::new(shape_table),
            node_count: 0,
            evaluator: Evaluator::default(),
            #[cfg(feature = "tail-expectation")]
            tail: None,
            #[cfg(feature = "tail-expectation")]
            tail_finder: PlaceFinder::new(shape_table),
            generated: 0,
            #[cfg(feature = "root-complete")]
            covering_root: None,
            #[cfg(feature = "deduplicate")]
            seen: Default::default(),
        }
    }

    /// Starts the search at `root_state`.
    pub fn start(&mut self, root_state: State) {
        #[cfg(feature="plan-reuse")]
        {self.plan=PlanReuse::default();}
        #[cfg(feature = "root-dag")]
        { self.expanded.clear(); self.duplicate_expansions=0; }
        #[cfg(feature = "root-focus")]
        { self.focus=Focus {threshold:self.focus.threshold,..Focus::default()}; }
        self.move_best.clear();
        #[cfg(feature = "tail-expectation")]
        {
            self.tail = None;
        }
        #[cfg(feature = "robust-backup")]
        self.root_leaves.clear();
        self.evaluator = Evaluator::default();
        self.generated = 1;
        #[cfg(feature = "root-complete")]
        {
            self.covering_root = None;
        }
        #[cfg(feature = "deduplicate")]
        self.seen.clear();
        for lvl in self.lvls.iter_mut() {
            lvl.clear();
        }
        self.lvl_idx = 0;
        root_state.placements(&mut self.pfind);

        self.node = Some(Node::root(&self.params, root_state, &mut self.evaluator));
        self.node_count = 1;
    }

    /// Returns the total number of generated nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Work budget for experiments that prune nodes. Counting attempted successors
    /// avoids giving deduplication an unbounded advantage under a heap-size limit.
    #[cfg(not(feature = "beam-search"))]
    pub fn budget_used(&self) -> usize {
        if cfg!(feature = "generated-budget") {
            self.generated
        } else {
            self.node_count
        }
    }

    #[cfg(feature = "support-probe")]
    pub(super) fn report_support(&self) {
        let mut full = 0;
        let mut unique = 0;
        let mut single = 0;
        for leaves in self.root_leaves.values() {
            let (count, actions) = leaves.support();
            if count == 4 {
                full += 1;
                unique += actions;
                single += (actions == 1) as usize;
            }
        }
        eprintln!(
            "SUPPORT_PROBE {}",
            serde_json::json!({
                "roots":self.root_leaves.len(), "full_roots":full,
                "full_unique_second_actions":unique, "full_single_second_roots":single
            })
        );
    }

    #[cfg(feature = "root-dag")]
    pub fn report_dag(&self) {
        eprintln!("ROOT_DAG {}",serde_json::json!({"generated":self.generated,
            "unique_expanded":self.expanded.len(),"duplicate_expansions":self.duplicate_expansions,
            "merging":self.merge_transpositions}));
    }
    #[cfg(feature="plan-reuse")]
    pub fn set_hint(&mut self,hint:super::PlanHint) {
        self.plan.supplied=hint.0.len();
        self.plan.active=hint.0.first().map_or(false,|h|self.node.as_ref().map_or(false,|n|n.state.matrix()==&h.before));
        self.plan.hint=hint;
    }
    #[cfg(feature="plan-reuse")]
    pub fn fallback_move(&self)->Option<usize> {self.plan.fallback}
    #[cfg(feature="plan-reuse")]
    pub fn report_plan(&self) {
        eprintln!("PLAN_REUSE {}",serde_json::json!({"generated":self.generated,
            "supplied":self.plan.supplied,"matched":self.plan.matched,"followed":self.plan.followed,
            "observer":cfg!(feature="plan-observer"),"fallback_available":self.plan.fallback.is_some()}));
    }
    pub fn generated_nodes(&self) -> usize {
        self.generated
    }

    /// Runs one iteration of the algorithm. Returns `Ok(Some(rc))` it move rating was
    /// modified, `Ok(None)` if work was performed but no ratings were modified yet, or
    /// `Err(SearchTerminated)` if there are no more nodes remaining to be processed.
    #[cfg(feature = "tail-expectation")]
    fn step_tail(&mut self) -> Result<Step, SearchTerminated> {
        use super::tail::Progress;
        match self.tail.as_mut().unwrap().step(
            &mut self.tail_finder,
            &mut self.evaluator,
            &self.params,
        ) {
            Progress::Generated => {
                self.generated += 1;
                Ok(Step::Other)
            }
            Progress::Boundary => Ok(Step::Other),
            Progress::Finished => {
                let node = self.tail.take().unwrap().finish(&self.params);
                Ok(match self.back_up(node) {
                    (rating, trace, Some(move_id)) => Step::RatingChanged {
                        move_id,
                        rating,
                        trace,
                    },
                    (rating, trace, None) => Step::SequenceRejected { rating, trace },
                })
            }
        }
    }

    pub fn step(&mut self) -> Result<Step, SearchTerminated> {
        #[cfg(feature = "root-focus")]
        {
            self.focus_roots();
            if let Some(id)=self.focus.removed.pop_front() { return Ok(Step::RemoveMove(id)); }
        }
        #[cfg(feature = "tail-expectation")]
        if self.tail.is_some() {
            return self.step_tail();
        }
        if let Some(node) = self.node.take() {
            // best-first iteration phase
            if node.is_terminal() {
                #[cfg(feature = "tail-expectation")]
                if !node.state.reached_goal() {
                    self.tail = Some(super::tail::Tail::new(node));
                    return self.step_tail();
                }
                // stop at terminal nodes
                return Ok(match self.back_up(node) {
                    (rating, trace, Some(move_id)) => Step::RatingChanged {
                        move_id,
                        rating,
                        trace,
                    },
                    (rating, trace, None) => Step::SequenceRejected { rating, trace },
                });
            }
            #[cfg(feature = "root-dag")]
            let mut node = node;
            #[cfg(feature = "root-dag")]
            if !node.expansion_started {
                node.expansion_started=true;
                // Nonterminal parent_f cannot affect descendants: succ replaces it
                // with this node's f, fixed by full State and depth. Preserve roots.
                let key=(node.state.clone(),node.trace.len(),node.trace.first().copied().unwrap_or(255));
                if !self.expanded.insert(key) {
                    self.duplicate_expansions+=1;
                    if self.merge_transpositions { return Ok(Step::Other); }
                }
            }
            // expansion
            if let Some(pl) = self.pfind.next() {
                let successor = node.succ(&self.params, &pl, &mut self.evaluator);
                #[cfg(feature="plan-reuse")]
                if self.plan.active && node.trace==self.plan.prefix {
                    if let Some(h)=self.plan.hint.0.get(node.trace.len()) {
                        if node.state.matrix()==&h.before && pl.shape.color()==h.color && pl.did_hold==h.hold && pl.normal()==h.normal {
                            self.plan.target=Some(successor.trace.clone());self.plan.matched+=1;
                        }
                    }
                }
                #[cfg(feature = "eager-leaves")]
                if successor.is_terminal() {
                    self.generated += 1;
                    self.node = Some(node);
                    return Ok(match self.back_up(successor) {
                        (rating, trace, Some(move_id)) => Step::RatingChanged {
                            move_id,
                            rating,
                            trace,
                        },
                        (rating, trace, None) => Step::SequenceRejected { rating, trace },
                    });
                }
                self.push(successor);
                self.node = Some(node);
            } else {
                #[cfg(feature = "eager-leaves")]
                {
                    if self.pop().is_err() {
                        self.select();
                        self.pop()?;
                    }
                }
                #[cfg(not(feature = "eager-leaves"))]
                self.pop()?;
            }
        } else {
            // reselection phase
            self.select();
            self.pop()?;
        }
        Ok(Step::Other)
    }

    /// Adds `node` to the fringe set at the current level index.
    fn push(&mut self, node: Node) {
        self.generated += 1;
        #[cfg(feature="plan-reuse")]
        if node.trace.len()==1 && self.plan.fallback.is_none() {self.plan.fallback=Some(node.trace[0] as usize);}
        #[cfg(feature = "deduplicate")]
        {
            // Depth fixes path cost; parent_f is needed by smoothed terminal
            // ratings. The global ablation merges equivalent paths across root
            // moves; it preserves the best-action objective, not an independent
            // rating for every possible root action. Nonterminal parent_f cannot
            // affect any descendant's score and is omitted in that ablation.
            let key = (
                node.state.clone(),
                node.trace.len(),
                if cfg!(feature = "global-dedup") {
                    0
                } else {
                    node.trace[0]
                },
                if cfg!(feature = "global-dedup") && !node.is_terminal() {
                    0
                } else {
                    node.parent_f
                },
            );
            if !self.seen.insert(key) {
                return;
            }
        }
        let lvl = match self.lvls.get_mut(self.lvl_idx) {
            Some(lvl) => lvl,
            None => {
                self.lvls.resize_with(self.lvl_idx + 1, default_level);
                &mut self.lvls[self.lvl_idx]
            }
        };
        lvl.push(node);
        self.node_count += 1;
    }

    /// Removes the best node at the current level, initializes `self.node` and
    /// `self.pfind` to that node (for generating successors), then advances the level
    /// index.
    fn pop(&mut self) -> Result<(), SearchTerminated> {
        let lvl = self.lvls.get_mut(self.lvl_idx).ok_or(SearchTerminated)?;
        #[cfg(feature = "root-complete")]
        let node = if self.lvl_idx > 0 && self.covering_root.is_some() {
            // Shared level heaps contain siblings from earlier roots. Keep the
            // first rollout on its selected root, restoring every skipped node.
            let root = self.covering_root.unwrap();
            let mut skipped = Vec::new();
            let found = loop {
                match lvl.pop() {
                    Some(node) if node.trace.first() == Some(&root) => break Some(node),
                    Some(node) => skipped.push(node),
                    None => break None,
                }
            };
            lvl.extend(skipped);
            found.ok_or(SearchTerminated)?
        } else {
            lvl.pop().ok_or(SearchTerminated)?
        };
        #[cfg(all(not(feature = "root-complete"),not(feature="plan-reuse")))]
        let node = lvl.pop().ok_or(SearchTerminated)?;
        #[cfg(all(not(feature="root-complete"),feature="plan-reuse"))]
        let node = if self.plan.active && self.plan.target.is_some() && !cfg!(feature="plan-observer") {
            let mut skipped=vec![];
            let found=loop {match lvl.pop(){Some(n) if Some(&n.trace)==self.plan.target.as_ref()=>break Some(n),
                Some(n)=>skipped.push(n),None=>break None}};
            lvl.extend(skipped);found.ok_or(SearchTerminated)?
        } else {lvl.pop().ok_or(SearchTerminated)?};
        #[cfg(feature="plan-reuse")]
        if self.plan.active {
            if Some(&node.trace)==self.plan.target.as_ref() {
                self.plan.followed+=1;self.plan.prefix=node.trace.clone();
                if self.plan.prefix.len()>=self.plan.hint.0.len(){self.plan.active=false;}
            }else{self.plan.active=false;}
            self.plan.target=None;
        }
        #[cfg(feature = "root-complete")]
        if self.lvl_idx == 0 {
            self.covering_root = node.trace.first().copied();
        }
        self.node_count -= 1;
        node.state.placements(&mut self.pfind);
        self.node = Some(node);
        self.lvl_idx += 1;
        Ok(())
    }

    /// Selects the level index corresponding to the node with best evaluation.
    fn select(&mut self) {
        // Expand every root before global refinement. The root-complete
        // ablation additionally keeps each initial rollout on that root.
        if cfg!(feature = "root-coverage") && self.lvls.first().map_or(false, |lvl| !lvl.is_empty())
        {
            self.lvl_idx = 0;
            return;
        }
        self.lvl_idx = (0..self.lvls.len())
            .min_by_key(|&i| self.lvls[i].peek().map_or(std::i64::MAX, Node::priority))
            .unwrap_or(0);
    }

    /// Propogates `node`'s rating back to the move at the root of this node.
    fn back_up(&mut self, node: Node) -> (i64, Vec<usize>, Option<MoveId>) {
        #[cfg(feature="plan-reuse")]
        {self.plan.active=false;self.plan.target=None;}
        #[cfg(feature = "root-complete")]
        {
            self.covering_root = None;
        }
        let rating = node.rating();
        let trace = node.trace().collect();
        #[cfg(feature = "robust-backup")]
        if let Some(&idx) = node.trace.first() {
            let id = MoveId(idx);
            let (rating, trace) = self.root_leaves.entry(id).or_default().record(
                (*std::borrow::Borrow::<State>::borrow(&node.state)).clone(),
                rating,
                trace,
                node.state.reached_goal()
                    && node.state.goal_is_final()
                    && !cfg!(feature = "goal-residue"),
            );
            let changed = self.move_best.get(&id).copied() != Some(rating);
            self.move_best.insert(id, rating);
            return (rating, trace, if changed { Some(id) } else { None });
        }
        let move_id = match node.trace.get(0) {
            Some(&idx) => {
                let m_id = MoveId(idx as _);
                let best = self.move_best.entry(m_id).or_insert(std::i64::MAX);
                if rating < *best {
                    *best = rating;
                    Some(m_id)
                } else {
                    None
                }
            }
            None => None,
        };
        (rating, trace, move_id)
    }
}

fn default_level() -> BinaryHeap<Node> {
    BinaryHeap::with_capacity(1024)
}

impl MoveId {
    #[cfg(test)]
    pub fn n(x: i32) -> Self {
        Self(x as u8)
    }
}

// Nodes

#[cfg(feature = "shared-states")]
type NodeState = std::rc::Rc<State>;
#[cfg(not(feature = "shared-states"))]
type NodeState = State;

pub(super) struct Node {
    #[cfg(feature = "root-dag")]
    expansion_started: bool,
    pub(super) state: NodeState,
    pub(super) trace: Vec<u8>,
    pub(super) f: i64,
    #[cfg(feature = "classic-guide")]
    guide: i64,
    parent_f: i64,
}

impl Node {
    fn priority(&self) -> i64 {
        #[cfg(feature = "classic-guide")]
        { self.guide }
        #[cfg(not(feature = "classic-guide"))]
        { self.f }
    }

    #[cfg(feature = "tail-expectation")]
    pub(super) fn extend_value(&mut self, params: &Parameters, future_value: i64) {
        #[cfg(feature = "tail-hold")]
        {
            self.parent_f = self.f;
        }
        self.f = penalty(params, self.trace.len() + 1) + future_value;
    }
    pub(super) fn root(params: &Parameters, state: State, evaluator: &mut Evaluator) -> Self {
        let h = evaluator.score(&state, params);
        Self {
            #[cfg(feature = "root-dag")]
            expansion_started: false,
            #[cfg(feature = "classic-guide")]
            guide: super::eval::eval(state.matrix()).score(params),
            state: state.into(),
            trace: vec![],
            f: h,
            parent_f: h,
        }
    }

    /// Generates a successor node from this node, by placing `pl`. Uses `params` to
    /// compute the new evaluation.
    pub(super) fn succ(&self, params: &Parameters, pl: &Place, evaluator: &mut Evaluator) -> Self {
        #[cfg(feature = "shared-states")]
        let mut state = (*self.state).clone();
        #[cfg(not(feature = "shared-states"))]
        let mut state = self.state.clone();
        state.place(pl);
        let mut trace = self.trace.clone();
        trace.push(pl.idx as u8);
        let g = penalty(params, trace.len());
        let h = evaluator.score(&state, params);
        Self {
            #[cfg(feature = "root-dag")]
            expansion_started: false,
            #[cfg(feature = "classic-guide")]
            guide: g + super::eval::eval(state.matrix()).score(params),
            state: state.into(),
            trace,
            f: g + h,
            parent_f: self.f,
        }
    }

    /// Returns `true` if this node is a terminal node (aka leaf node).
    pub(super) fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Returns the rating value for this node. Lower is always better.
    pub(super) fn rating(&self) -> i64 {
        if self.state.reached_goal()
            && self.state.goal_is_final()
            && !cfg!(feature = "goal-residue")
        {
            // just use number-of-pieces as rating
            self.trace.len() as i64
        } else if cfg!(feature = "terminal-only") {
            self.f.saturating_mul(2)
        } else {
            self.f.saturating_add(self.parent_f)
        }
    }

    pub(super) fn trace<'a>(&'a self) -> impl Iterator<Item = usize> + 'a {
        self.trace.iter().map(|&i| i as usize)
    }
}

impl PartialEq for Node {
    fn eq(&self, rhs: &Self) -> bool {
        self.priority() == rhs.priority()
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}

impl Eq for Node {}

impl Ord for Node {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&rhs.priority()).reverse()
    }
}

#[cfg(all(test, not(feature = "goal-residue")))]
mod completion_tests {
    use super::*;
    use crate::{ai::Snapshot, shape::srs, BasicMatrix, Color};

    #[cfg(feature = "root-focus")]
    #[test]
    fn focus_preserves_ratings_paths_and_cumulative_budget() {
        let shapes=srs();
        let mut matrix=BasicMatrix::with_cols(10);
        for row in 0..9 { for col in 0..10 {
            if col!=row%10 { matrix.set((row,col)); }
        }}
        let state:State=Snapshot {matrix,hold:None,queue:"IJLOST".chars().map(Color::n).collect()}.into();
        let mut search=Search::new(&shapes,Parameters::default());
        search.set_budget(50000);search.start(state.clone());
        while search.generated_nodes()<12500 { search.step().unwrap(); }
        let old=search.move_best.clone();assert!(!old.is_empty());
        let nodes=search.generated_nodes();search.focus_roots();
        assert_eq!(search.generated_nodes(),nodes);
        assert_eq!(search.focus.trigger_nodes,12500);
        let allowed=search.focus.allowed.clone().unwrap();
        assert_eq!(allowed.len(),(old.len()+1)/2);
        assert_eq!(search.root_leaves.len(),allowed.len());
        for &id in &allowed { assert_eq!(search.move_best[&MoveId(id)],old[&MoveId(id)]); }
        assert!(search.lvls.iter().all(|h|h.iter().all(|n|allowed.contains(&n.trace[0]))));
        while search.generated_nodes()<50000 {
            let before=search.generated_nodes();
            match search.step() {
                Ok(Step::RatingChanged{move_id,trace,..}) => {
                    assert!(allowed.contains(&move_id.0));
                    assert_eq!(trace[0],move_id.0 as usize);
                    let mut next=state.clone();let mut finder=PlaceFinder::new(&shapes);
                    for index in trace {
                        let place=next.placements(&mut finder).find(|p|p.idx==index).unwrap();
                        next.place(&place);
                    }
                    assert!(next.is_terminal());
                }
                Err(_) => break,
                _ => {},
            }
            assert!(search.generated_nodes()<=before+1);
        }
        assert!(search.generated_nodes()<=50000);
    }

    #[cfg(all(
        feature = "root-complete",
        not(any(
            feature = "deduplicate",
            feature = "eager-leaves",
            feature = "preserve-hold"
        ))
    ))]
    #[test]
    fn root_completion_covers_all_first_moves_within_budget() {
        let state: State = Snapshot {
            matrix: BasicMatrix::with_cols(10),
            hold: None,
            queue: "IOTLJS".chars().map(Color::n).collect(),
        }
        .into();
        let shapes = srs();
        let mut finder = PlaceFinder::new(&shapes);
        let expected = state.placements(&mut finder).count();
        let mut search = Search::new(&shapes, Parameters::default());
        search.start(state);
        let mut roots = std::collections::HashSet::new();
        while search.generated_nodes() < 50_000 && roots.len() < expected {
            match search
                .step()
                .expect("preview tree should not terminate early")
            {
                Step::RatingChanged { move_id, .. } => {
                    roots.insert(move_id);
                }
                _ => {}
            }
        }
        assert_eq!(roots.len(), expected);
    }

    #[cfg(all(feature = "eager-leaves", not(feature = "deduplicate")))]
    #[test]
    fn eager_search_covers_every_root_after_a_terminal_level_is_drained() {
        let state: State = Snapshot {
            matrix: BasicMatrix::with_cols(10),
            hold: None,
            queue: "IO".chars().map(Color::n).collect(),
        }
        .into();
        let shapes = srs();
        let mut finder = PlaceFinder::new(&shapes);
        let expected_roots = state.placements(&mut finder).count();
        assert!(expected_roots > 1);
        let mut search = Search::new(&shapes, Parameters::default());
        search.start(state);
        let mut roots = std::collections::HashSet::new();
        let mut ended = false;
        for _ in 0..100_000 {
            match search.step() {
                Ok(Step::RatingChanged { move_id, .. }) => {
                    roots.insert(move_id);
                }
                Ok(_) => {}
                Err(_) => {
                    ended = true;
                    break;
                }
            }
        }
        assert!(ended, "finite two-piece tree should be exhausted");
        assert_eq!(roots.len(), expected_roots);
        assert!(search.generated_nodes() > expected_roots);
    }

    #[test]
    fn visible_bottom_clear_only_gets_finish_reward_at_race_end() {
        let mut matrix = BasicMatrix::with_cols(10);
        matrix.push_row((0..10).map(|c| c >= 4));
        matrix.push_row((0..10).map(|c| c == 9));
        let state: State = Snapshot {
            matrix,
            hold: None,
            queue: "ILJ".chars().map(Color::n).collect(),
        }
        .into();
        let params = Parameters::default();
        let mut evaluator = Evaluator::default();
        let final_root = Node::root(&params, state.clone(), &mut evaluator);
        let mid_root = Node::root(&params, state.with_final_goal(false), &mut evaluator);
        let shapes = srs();
        let mut finder = PlaceFinder::new(&shapes);
        final_root.state.placements(&mut finder);
        let mut checked = false;
        for place in finder {
            let final_node = final_root.succ(&params, &place, &mut evaluator);
            if final_node.state.reached_goal() {
                let mid_node = mid_root.succ(&params, &place, &mut evaluator);
                assert!(mid_node.is_terminal());
                assert_eq!(mid_node.state.matrix(), final_node.state.matrix());
                assert_eq!(final_node.rating(), 1);
                assert!(mid_node.rating() > final_node.rating());
                checked = true;
                break;
            }
        }
        assert!(checked, "test board must admit a legal bottom clear");
    }
}

#[cfg(all(test, feature = "tail-expectation", not(feature = "beam-search")))]
mod tail_budget_tests {
    use super::*;
    use crate::{ai::Snapshot, shape::srs, BasicMatrix, Color};
    #[test]
    fn hypothetical_work_is_charged_and_never_enters_action_trace() {
        let shapes = srs();
        for cap in [1, 2, 20, 500, 5000] {
            let root: State = Snapshot {
                matrix: BasicMatrix::with_cols(10),
                hold: None,
                queue: vec![Color::n('I')],
            }
            .into();
            let mut search = Search::new(&shapes, Parameters::default());
            search.start(root);
            let mut sequences = 0;
            while search.budget_used() < cap {
                let before = search.generated_nodes();
                let step = search.step();
                assert!(search.generated_nodes() <= before + 1);
                match step {
                    Ok(Step::RatingChanged { trace, .. })
                    | Ok(Step::SequenceRejected { trace, .. }) => {
                        assert_eq!(trace.len(), 1, "unknown pieces must not be executable");
                        sequences += 1;
                    }
                    Err(_) => break,
                    _ => (),
                }
            }
            assert!(search.generated_nodes() <= cap);
            if cap == 5000 {
                assert!(sequences > 0);
            }
        }
    }
}

#[cfg(all(test, feature="root-dag"))]
mod dag_tests {
    use super::*;
    use crate::{ai::Snapshot,shape::srs,BasicMatrix,Color};
    #[test]
    fn exhaustive_root_ratings_preserved_with_transpositions() {
        let shapes=srs();
        let mut eliminated=0;
        for queue in ["IIOO","TSTZ"] {
            let state:State=Snapshot{matrix:BasicMatrix::with_cols(4),hold:None,
                queue:queue.chars().map(Color::n).collect()}.into();
            let mut scores=Vec::new();
            for merging in [false,true] {
                let mut search=Search::new(&shapes,Parameters::default());
                search.merge_transpositions=merging;search.start(state.clone());
                let mut finished=false;
                for _ in 0..5_000_000 {
                    let before=search.generated_nodes();
                    match search.step() {
                        Ok(Step::RatingChanged{move_id,trace,..}) => {
                            assert_eq!(trace[0],move_id.0 as usize);
                            let mut replay=state.clone();let mut finder=PlaceFinder::new(&shapes);
                            for idx in trace {
                                let place=replay.placements(&mut finder).find(|p|p.idx==idx).unwrap();
                                replay.place(&place);
                            }
                            assert!(replay.is_terminal());
                        },
                        Err(_) => {finished=true;break;},
                        _ => (),
                    }
                    assert!(search.generated_nodes()<=before+1);
                }
                assert!(finished,"small exhaustive tree must finish");
                assert!(!search.move_best.is_empty());
                if merging { eliminated+=search.duplicate_expansions; }
                scores.push(search.move_best);
            }
            assert_eq!(scores[0],scores[1],"all first-action exhaustive robust scores must match");
        }
        assert!(eliminated>0,"test must exercise actual transpositions");
    }
    #[test]
    fn bounded_paths_legal_and_table_resets() {
        let shapes=srs();
        let state:State=Snapshot{matrix:BasicMatrix::with_cols(10),hold:None,
            queue:"IOTLSZ".chars().map(Color::n).collect()}.into();
        let mut search=Search::new(&shapes,Parameters::default());search.start(state.clone());
        while search.generated_nodes()<5000 {
            let before=search.generated_nodes();
            if search.step().is_err(){break;}
            assert!(search.generated_nodes()<=before+1);
        }
        assert!(search.generated_nodes()<=5000&&!search.expanded.is_empty());
        search.start(state);assert!(search.expanded.is_empty());assert_eq!(search.duplicate_expansions,0);
        assert_eq!(search.generated_nodes(),1);
    }
}

#[cfg(all(test,feature="plan-reuse"))]
mod plan_tests {
    use super::*;
    use crate::{ai::{Snapshot,PlanHint,PlanStep},shape::srs,BasicMatrix,Color};
    fn initial()->State {
        let mut matrix=BasicMatrix::with_cols(10);
        for r in 0..9 {for c in 0..10 {if c!=r%10 {matrix.set((r,c));}}}
        Snapshot{matrix,hold:None,queue:"IOTLSZ".chars().map(Color::n).collect()}.into()
    }
    #[test]
    fn hint_path_is_freshly_legal_and_charged() {
        let shapes=srs();let root=initial();let mut state=root.clone();let mut finder=PlaceFinder::new(&shapes);
        let mut expected=vec![];let mut hint=vec![];
        for _ in 0..2 {
            let p=state.placements(&mut finder).last().unwrap();expected.push(p.idx);
            hint.push(PlanStep{before:state.matrix().clone(),color:p.shape.color(),normal:p.normal(),hold:p.did_hold});
            state.place(&p);
        }
        let mut search=Search::new(&shapes,Parameters::default());search.start(root.clone());search.set_hint(PlanHint(hint));
        let mut first=None;
        while search.generated_nodes()<50000 {
            let before=search.generated_nodes();
            match search.step(){
                Ok(Step::RatingChanged{trace,..})|Ok(Step::SequenceRejected{trace,..})=>{
                    let mut state=root.clone();
                    for &idx in &trace {let p=state.placements(&mut finder).find(|p|p.idx==idx).unwrap();state.place(&p);}
                    assert!(state.is_terminal());if first.is_none(){first=Some(trace);}
                },Err(_)=>break,_=>()}
            assert!(search.generated_nodes()<=before+1);
        }
        assert!(search.generated_nodes()<=50000&&first.is_some());
        if !cfg!(feature="plan-observer") {assert_eq!(&first.unwrap()[..2],expected.as_slice());assert_eq!(search.plan.followed,2);}
        assert!(search.fallback_move().is_some());
    }
    #[test]
    fn changed_board_rejects_hint_and_reset_clears_history() {
        let shapes=srs();let root=initial();let mut finder=PlaceFinder::new(&shapes);
        let p=root.placements(&mut finder).next().unwrap();let mut before=root.matrix().clone();before.set((20,0));
        let hint=PlanHint(vec![PlanStep{before,color:p.shape.color(),normal:p.normal(),hold:p.did_hold}]);
        let mut search=Search::new(&shapes,Parameters::default());search.start(root.clone());search.set_hint(hint);
        assert!(!search.plan.active);
        while search.generated_nodes()<1000 {if search.step().is_err(){break;}}
        assert_eq!(search.plan.followed,0);assert!(search.fallback_move().is_some());
        search.start(root);assert_eq!(search.plan.supplied,0);assert!(search.fallback_move().is_none());
    }
}
