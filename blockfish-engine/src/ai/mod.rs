use crate::{
    config::Config,
    shape::{srs, ShapeTable},
    BasicMatrix, Color, Input,
};

mod analysis;
#[cfg(feature = "bag-inference")]
mod bag_observer;
#[cfg(feature = "bag-inference")]
pub use bag_observer::BagObserver;
mod b_star;
#[cfg(feature="global-regret")]
mod global;
#[cfg(feature = "root-halving")]
mod halving;
#[cfg(feature = "beam-search")]
mod beam;
#[cfg(feature = "endgame-solver")]
mod endgame;
mod eval;
#[cfg(feature="mirror-fusion")]
mod mirror;
#[cfg(feature="dig-geometry")]
mod dig_geometry;
#[cfg(feature = "hold-context")]
mod hold_context;
mod state;
#[cfg(feature = "tail-expectation")]
mod tail;
#[cfg(feature = "robust-backup")]
mod robust_backup;
#[cfg(feature = "endgame-solver")]
pub use endgame::EndgameResult;
#[cfg(feature = "root-rerank")]
pub mod action_rank;
mod value_features;
pub use value_features::board_features;
#[cfg(feature = "public-rollout")]
pub fn rollout_board_cost(matrix: &BasicMatrix, config: &Config) -> f64 {
    eval::search_score(matrix, &config.parameters) as f64 / 10.0
}

// Input / output types

/// A game state snapshot to begin an analysis from.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Snapshot {
    pub hold: Option<Color>,
    pub queue: Vec<Color>,
    pub matrix: BasicMatrix,
}

/// A suggested sequence and its rating.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Suggestion {
    /// List of inputs to perform the suggested move.
    pub inputs: Vec<Input>,
    // The "rating" is an abstract measurement for how good a move is (lower is better).
    pub rating: i64,
}

/// Statistics about the analysis after it has finished.
#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct Stats {
    /// Number of iterations of the search algorithm.
    pub iterations: usize,
    /// Number of nodes retained in the frontier (legacy budget).
    pub nodes: usize,
    /// Number of attempted nodes, including the root and pruned duplicates.
    pub generated_nodes: usize,
    /// Total time taken to do the analysis.
    pub time_taken: std::time::Duration,
}

// Evaluation function interface

pub use eval::Eval;

/// Performs the static analysis function on a snapshot.
pub fn static_eval(snapshot: &Snapshot) -> Eval {
    eval::eval(&snapshot.matrix)
}

// AI interface

// Re-export
pub use analysis::{Analysis, AnalysisDone, MoveId};

/// An instance of the Blockfish AI. Holds engine configuration and can be used to spawn
/// an analysis.
///
/// `AI` can currently be seen as just a sort of builder-pattern type for creating
/// `Analysis`'s.  However, in the future it could be extended to handle more things such
/// as a reusable thread pool and/or the place to call `static_eval`.
#[cfg(feature="plan-reuse")]
#[derive(Clone,Default)]
pub struct PlanHint(pub(super) Vec<PlanStep>);
#[cfg(feature="plan-reuse")]
#[derive(Clone)]
pub(super) struct PlanStep {
    pub before:BasicMatrix,
    pub color:Color,
    pub normal:crate::shape::NormalizedShapeTransform,
    pub hold:bool,
}
pub struct AI {
    #[cfg(feature="plan-reuse")]
    hint:PlanHint,
    config: Config,
    shape_table: std::sync::Arc<ShapeTable>,
    all_tx: Option<std::sync::mpsc::Sender<Suggestion>>,
}

impl AI {
    /// Solve a fully visible remaining garbage prefix using only known pieces.
    /// The caller must ensure no more garbage will be generated during the plan.
    #[cfg(feature = "endgame-solver")]
    pub fn solve_endgame(
        &self,
        snapshot: Snapshot,
        garbage_rows: u16,
        budget: usize,
    ) -> EndgameResult {
        endgame::solve(
            &self.shape_table,
            &self.config.parameters,
            snapshot,
            garbage_rows,
            budget,
        )
    }
    /// Constructs a new Blockfish AI instance with the given engine configuration.
    pub fn new(config: Config) -> Self {
        Self {
            #[cfg(feature="plan-reuse")]
            hint:PlanHint::default(),
            config,
            shape_table: std::sync::Arc::new(srs()),
            all_tx: None,
        }
    }

    #[cfg(feature="plan-reuse")]
    pub fn set_plan_hint(&mut self,hint:PlanHint) {self.hint=hint;}
    pub fn config(&self) -> Config {
        self.config.clone()
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// Begins a new analysis of `snapshot`, returning a handle to it.
    pub fn analyze(&mut self, snapshot: Snapshot) -> Analysis {
        self.analyze_with_limit(snapshot, None)
    }

    /// Analyze with a wall-time budget measured inside the search worker.
    /// Always finishes at least one sequence so callers receive a legal move;
    /// budget checks occur every 64 search steps. This is a soft deadline.
    pub fn analyze_for(&mut self, snapshot: Snapshot, limit: std::time::Duration) -> Analysis {
        self.analyze_with_limit(snapshot, Some(limit))
    }

    fn analyze_with_limit(
        &mut self,
        snapshot: Snapshot,
        limit: Option<std::time::Duration>,
    ) -> Analysis {
        self.analyze_with_completion(snapshot, true, limit)
    }

    /// Analyze a race position with public completion information. When more
    /// garbage remains unseen, clearing the visible bottom still ends this
    /// search horizon, but retains the remaining board's estimated cost.
    pub fn analyze_with_completion(
        &mut self,
        snapshot: Snapshot,
        all_garbage_visible: bool,
        limit: Option<std::time::Duration>,
    ) -> Analysis {
        analysis::spawn(
            self.shape_table.clone(),
            self.config.clone(),
            state::State::from(snapshot).with_final_goal(all_garbage_visible),
            self.all_tx.take(),
            limit,
            #[cfg(feature="plan-reuse")]
            std::mem::take(&mut self.hint),
        )
    }

    #[cfg(feature="reveal-boundary")]
    pub fn analyze_until_reveal(&mut self, snapshot:Snapshot, visible:u16, minimum:u16, maximum:u16, unseen:bool) -> Analysis {
        let root=state::State::from(snapshot).until_reveal(visible,minimum,maximum,unseen);
        analysis::spawn(self.shape_table.clone(),self.config.clone(),root,self.all_tx.take(),None,
            std::mem::take(&mut self.hint))
    }

    /// Configures the next analysis (via `analyze()`) to send every suggestion it
    /// encounters to a non-blocking channel. Returns the rx end of that channel.
    ///
    /// NOTE: Even rejected suggestions are sent to this channel; this channel should just
    /// be used for diagnostic purposes.
    pub fn listen_all(&mut self) -> std::sync::mpsc::Receiver<Suggestion> {
        let (tx, rx) = std::sync::mpsc::channel();
        self.all_tx = Some(tx);
        rx
    }
}

#[cfg(test)]
mod timed_tests {
    use super::*;

    #[test]
    fn zero_deadline_still_returns_a_legal_sequence() {
        let mut matrix = BasicMatrix::with_cols(10);
        for col in 1..10 {
            matrix.set((0, col));
        }
        let snapshot = Snapshot {
            matrix,
            hold: None,
            queue: "TSZLIO".chars().map(Color::n).collect(),
        };
        let mut cfg = Config::default();
        cfg.search_limit = 1_000_000;
        let mut ai = AI::new(cfg);
        let mut result = ai.analyze_for(snapshot, std::time::Duration::from_millis(0));
        result.wait();
        let best = result
            .all_moves()
            .min_by(|&a, &b| result.cmp(a, b))
            .unwrap();
        assert_eq!(result.suggestion(best, 1).inputs.last(), Some(&Input::HD));
        let stats = result.stats().unwrap();
        assert!(stats.iterations > 0);
        assert!(stats.generated_nodes < 1_000_000);
    }
}
