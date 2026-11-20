//! Offline counterfactual labels. Hidden RNG is used only by completed training
//! rollouts; candidate selection and features use the current public position.
use argh::FromArgs;
use block_stacker::{Config as GameConfig, Ruleset, Stacker};
use blockfish::{ai::AI, Config, Input, StackerExt};
use serde::Serialize;
use std::path::PathBuf;

#[derive(FromArgs)]
/// Collect paired action-cost labels from a single training game.
struct Args {
    /// training game seed
    #[argh(option)]
    seed: u64,
    /// output JSON file
    #[argh(option)]
    out: PathBuf,
}

#[derive(Clone, Serialize)]
struct Trace {
    seed: u64,
    garbage_min: usize,
    garbage_max: usize,
    inputs: Vec<Vec<Input>>,
    ds: Vec<usize>,
}
impl Trace {
    fn cleared(&self) -> usize {
        self.ds.last().copied().unwrap_or(0)
    }
    fn complete(&self) -> bool {
        self.cleared() == 100
    }
    fn play(&mut self, game: &mut Stacker, inputs: Vec<Input>) {
        game.run(inputs.clone());
        let cleared = game.hard_drop().1;
        self.inputs.push(inputs);
        self.ds.push(self.cleared() + cleared);
    }
}

#[derive(Serialize)]
struct Candidate {
    rating: i64,
    features: [f64; 14],
    pieces_remaining: usize,
    complete: bool,
    trace: Trace,
}
#[derive(Serialize)]
struct Probe {
    step: usize,
    candidates: Vec<Candidate>,
}
#[derive(Serialize)]
struct Data {
    seed: u64,
    source_config: &'static str,
    probe_config: &'static str,
    rollout_config: &'static str,
    failure_penalty: usize,
    source: Trace,
    probes: Vec<Probe>,
}

fn choices(ai: &mut AI, game: &Stacker, count: usize) -> Vec<(i64, Vec<Input>)> {
    let mut search = ai.analyze(game.snapshot().expect("live position"));
    search.wait();
    let mut moves: Vec<_> = search.all_moves().collect();
    moves.sort_by(|a, b| search.cmp(*a, *b));
    moves
        .into_iter()
        .take(count)
        .map(|m| {
            let suggestion = search.suggestion(m, 1);
            let mut inputs = suggestion.inputs;
            assert_eq!(inputs.pop(), Some(Input::HD));
            (suggestion.rating, inputs)
        })
        .collect()
}

fn projected_features(game: &Stacker, inputs: &[Input]) -> [f64; 14] {
    game.projected_features(inputs)
}

fn rollout(mut game: Stacker, mut trace: Trace, inputs: Vec<Input>, rating: i64) -> Candidate {
    let start = trace.inputs.len();
    let features = projected_features(&game, &inputs);
    trace.play(&mut game, inputs);
    let mut ai = AI::new("5/5,10,10,10".parse::<Config>().unwrap());
    while !trace.complete()
        && game.current_piece_type().is_some()
        && trace.inputs.len() - start < 500
    {
        let next = choices(&mut ai, &game, 1)
            .pop()
            .expect("legal rollout move")
            .1;
        trace.play(&mut game, next);
    }
    let complete = trace.complete();
    Candidate {
        rating,
        features,
        pieces_remaining: if complete {
            trace.inputs.len() - start
        } else {
            500
        },
        complete,
        trace,
    }
}

fn main() {
    let args: Args = argh::from_env();
    let mut cfg = GameConfig::default();
    cfg.prng_seed = Some(args.seed);
    cfg.garbage.total_lines = Some(100);
    cfg.garbage.min_height = 4;
    cfg.garbage.max_height = 9;
    let mut game = Stacker::new(Ruleset::guideline().into(), cfg);
    let mut source = Trace {
        seed: args.seed,
        garbage_min: 4,
        garbage_max: 9,
        inputs: Vec::new(),
        ds: Vec::new(),
    };
    let mut probes = Vec::new();
    let mut source_ai = AI::new("5/5,10,10,10".parse().unwrap());
    let mut probe_ai = AI::new("50/5,10,10,10".parse().unwrap());
    while !source.complete() && game.current_piece_type().is_some() && source.inputs.len() < 500 {
        let step = source.inputs.len();
        if [20, 60, 100, 140].contains(&step) {
            let candidates = choices(&mut probe_ai, &game, 3)
                .into_iter()
                .map(|(rating, inputs)| rollout(game.clone(), source.clone(), inputs, rating))
                .collect();
            probes.push(Probe { step, candidates });
        }
        let inputs = choices(&mut source_ai, &game, 1)
            .pop()
            .expect("source move")
            .1;
        source.play(&mut game, inputs);
    }
    let data = Data {
        seed: args.seed,
        source_config: "5/5,10,10,10",
        probe_config: "50/5,10,10,10",
        rollout_config: "5/5,10,10,10",
        failure_penalty: 500,
        source,
        probes,
    };
    serde_json::to_writer(std::fs::File::create(args.out).unwrap(), &data).unwrap();
}
