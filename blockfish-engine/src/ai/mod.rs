use crate::{
    config::Config,
    shape::{srs, ShapeTable},
    BasicMatrix, Color, Input,
};

mod analysis;
mod b_star;
pub mod beam;
pub mod cheese_val;
mod eval;
pub mod fast;
pub mod fast_eval;
pub mod ordered_policy;
pub mod spatial_val;
mod state;

pub use beam::{BagTracker, BeamConfig, BeamResult, BeamSearcher, GameContext};
pub use cheese_val::LearnedValueModel;
pub use fast_eval::EvalWeights;
pub use ordered_policy::OrderedRootPolicy;
pub use spatial_val::SpatialValueModel;

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
    /// Number of nodes generated.
    pub nodes: usize,
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
pub struct AI {
    config: Config,
    shape_table: std::sync::Arc<ShapeTable>,
    all_tx: Option<std::sync::mpsc::Sender<Suggestion>>,
}

impl AI {
    /// Constructs a new Blockfish AI instance with the given engine configuration.
    pub fn new(config: Config) -> Self {
        Self {
            config,
            shape_table: std::sync::Arc::new(srs()),
            all_tx: None,
        }
    }

    pub fn config(&self) -> Config {
        self.config.clone()
    }

    pub fn config_mut(&mut self) -> &mut Config {
        &mut self.config
    }

    /// Begins a new analysis of `snapshot`, returning a handle to it.
    pub fn analyze(&mut self, snapshot: Snapshot) -> Analysis {
        analysis::spawn(
            self.shape_table.clone(),
            self.config.clone(),
            snapshot.into(),
            self.all_tx.take(),
        )
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

/// Stateful, zero-allocation Cheese Race AI using bitboard beam search + 7-bag rollout.
pub struct FastAI {
    shape_table: ShapeTable,
    searcher: BeamSearcher,
    bag_tracker: BagTracker,
    pub beam_config: BeamConfig,
}

impl FastAI {
    pub fn new(beam_config: BeamConfig) -> Self {
        let shape_table = srs();
        let searcher = BeamSearcher::new(&shape_table);
        Self {
            shape_table,
            searcher,
            bag_tracker: BagTracker::new(),
            beam_config,
        }
    }

    pub fn reset_bag(&mut self) {
        self.bag_tracker = BagTracker::new();
    }

    pub fn last_top_candidates(&self) -> &[(char, i16, i16, u8, bool, i64)] {
        self.searcher.last_top_candidates()
    }

    pub fn last_top_leaf_info(&self) -> &[(u16, u16, i64, u8, u8)] {
        self.searcher.last_top_leaf_info()
    }

    pub fn last_pimc_reranked(&self) -> &[(fast::FastPlace, i64)] {
        self.searcher.last_pimc_reranked()
    }

    pub fn diverse_beam_leaves(
        &self,
        max_siblings: usize,
    ) -> Vec<(fast::FastBoard, Option<u8>, u16, u8)> {
        self.searcher.diverse_beam_leaves(max_siblings)
    }

    pub fn best_move(
        &mut self,
        snapshot: &Snapshot,
        garbage_rows: Option<u16>,
        remaining_garbage: Option<u16>,
    ) -> Option<BeamResult> {
        let mask = self.bag_tracker.observe(snapshot.hold, &snapshot.queue);
        let ctx = GameContext {
            garbage_rows,
            remaining_garbage,
            next_bag_mask: Some(mask),
        };
        self.searcher
            .search(&self.shape_table, snapshot, ctx, &self.beam_config)
    }

    pub fn best_move_with_ctx(
        &mut self,
        snapshot: &Snapshot,
        ctx: GameContext,
    ) -> Option<BeamResult> {
        self.searcher
            .search(&self.shape_table, snapshot, ctx, &self.beam_config)
    }

    pub fn inputs_for_place(
        &mut self,
        matrix: &BasicMatrix,
        fp: fast::FastPlace,
    ) -> Option<Vec<Input>> {
        self.searcher.inputs_for_place(&self.shape_table, matrix, fp)
    }

    pub fn search_with_ctx(
        &mut self,
        snapshot: &Snapshot,
        ctx: GameContext,
    ) -> Option<BeamResult> {
        self.searcher
            .search(&self.shape_table, snapshot, ctx, &self.beam_config)
    }
}
