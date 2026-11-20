#[cfg(feature = "public-rollout")]
mod world_rollout;
use argh::FromArgs;
use block_stacker::{Config as BSConfig, Ruleset, Stacker};
use blockfish::{ai::AI, Config as BFConfig, StackerExt as _};
use serde::{Deserialize, Serialize};
#[cfg(feature = "bag-inference")]
use std::convert::TryInto;
use std::io::Write;
use std::path::PathBuf;
use std::sync::{atomic, Arc};
use std::time::{Duration, Instant};

#[derive(FromArgs)]
/// Headless cheese race simulator for Blockfish.
struct Args {
    /// audit public hypothetical worlds and actual observed history
    #[cfg(feature = "public-rollout")]
    #[argh(option)]
    world_audit: Option<PathBuf>,
    #[argh(positional)]
    goal: Option<usize>,
    /// don't display stats until race finishes
    #[argh(switch, short = 'q')]
    quiet: bool,
    /// garbage level, defaults to 9
    #[argh(option, short = 'g')]
    garbage: Option<usize>,
    /// minimum garbage level, defaults to 4
    #[argh(option, short = 'G')]
    min_garbage: Option<usize>,
    /// parameters to AI, defaults to "50/5,10,10,10"
    #[argh(option, short = 'A')]
    ai_params: Option<BFConfig>,
    /// integer used to seed the random number generator
    #[argh(option, short = 's')]
    seed: Option<u64>,
    /// display ASCII rendering of game state at the end
    #[argh(switch)]
    ascii: bool,
    /// write a trace of the race to the given file
    #[argh(option)]
    trace_file: Option<PathBuf>,
    /// soft search deadline per placement in milliseconds (default: no deadline)
    #[argh(option)]
    move_ms: Option<u64>,
    /// verify a saved operation trace in the original simulator, without searching
    #[argh(option)]
    replay_file: Option<PathBuf>,
    /// export observed board features during a verified replay (requires replay-file)
    #[argh(option)]
    replay_features_out: Option<PathBuf>,
    /// include observable board features in the trace for offline value learning
    #[argh(switch)]
    collect_features: bool,
}

impl Args {
    fn to_game_config(&self) -> BSConfig {
        let mut cfg = BSConfig::default();
        if let Some(h) = self.garbage {
            cfg.garbage.max_height = h;
        }
        if let Some(h) = self.min_garbage {
            cfg.garbage.min_height = h;
        }
        cfg.garbage.total_lines = self.goal;
        cfg.prng_seed = self.seed;
        cfg
    }

    fn to_ai_config(&self) -> BFConfig {
        self.ai_params.clone().unwrap_or_default()
    }
}

struct Race {
    #[cfg(feature = "public-rollout")]
    world_garbage: block_stacker::GarbageConfig,
    #[cfg(feature = "public-rollout")]
    last_garbage_hole: Option<u16>,
    #[cfg(feature = "public-rollout")]
    world_events: Vec<world_rollout::Decision>,
    ai: AI,
    stacker: Stacker,
    ds_goal: Option<usize>,
    start_time: Instant,
    trace: Vec<usize>,
    search: Vec<SearchStats>,
    move_limit: Option<Duration>,
    inputs: Vec<Vec<blockfish::Input>>,
    collect_features: bool,
    samples: Vec<BoardSample>,
    #[cfg(feature = "bag-inference")]
    bag: blockfish::ai::BagObserver,
    #[cfg(feature = "bag-inference")]
    bag_events: Vec<BagEvent>,
    #[cfg(feature = "endgame-solver")]
    endgame_plan: std::collections::VecDeque<Vec<blockfish::Input>>,
}

#[cfg(feature = "bag-inference")]
#[derive(Serialize)]
struct BagEvent {
    visible_queue: String,
    hold: Option<char>,
    revealed: usize,
    remaining_mask: u8,
    inferred: Option<char>,
    applied: bool,
}

#[derive(Serialize)]
struct BoardSample {
    hold: Option<char>,
    features: [f64; 14],
    cleared: usize,
    visible_garbage: usize,
}

fn board_sample(stacker: &Stacker, cleared: usize) -> BoardSample {
    let snapshot = stacker.snapshot().expect("no snapshot");
    let visible_garbage = stacker
        .matrix()
        .filter(|(_, color)| *color == 'G')
        .map(|((row, _), _)| row as usize + 1)
        .max()
        .unwrap_or(0);
    BoardSample {
        hold: snapshot.hold.map(|c| c.as_char()),
        features: blockfish::ai::board_features(&snapshot.matrix),
        cleared,
        visible_garbage,
    }
}

#[derive(Serialize)]
struct SearchStats {
    frontier_nodes: usize,
    generated_nodes: usize,
    sequences: usize,
    seconds: f64,
    root_moves: usize,
    rerank_nodes: usize,
    reranked: bool,
    endgame_nodes: usize,
    endgame_plan_pieces: usize,
}

#[derive(Serialize)]
struct Trace<'a> {
    #[cfg(feature = "public-rollout")]
    world_events: &'a [world_rollout::Decision],
    time: f64,
    seed: u64,
    ds: &'a [usize],
    search: &'a [SearchStats],
    inputs: &'a [Vec<blockfish::Input>],
    ai_config: String,
    move_ms: Option<u128>,
    garbage_min: usize,
    garbage_max: usize,
    samples: &'a [BoardSample],
    #[cfg(feature = "bag-inference")]
    bag_events: &'a [BagEvent],
    #[cfg(feature = "bag-inference")]
    bag_valid: bool,
    #[cfg(feature = "bag-inference")]
    bag_checks: usize,
}

impl Race {
    fn new(ai_cfg: BFConfig, game_cfg: BSConfig, rules: Ruleset) -> Self {
        let ds_goal = game_cfg.garbage.total_lines;
        #[cfg(feature = "public-rollout")]
        let world_garbage = game_cfg.garbage.clone();
        let stacker = Stacker::new(rules.into(), game_cfg);
        #[cfg(feature = "public-rollout")]
        let last_garbage_hole = world_rollout::bottom_hole(&stacker);
        let ai = AI::new(ai_cfg);
        #[cfg(feature = "bag-inference")]
        let bag = {
            let mut types: Vec<_> = stacker.ruleset().types().collect();
            types.sort_unstable();
            assert_eq!(types, vec!['I', 'J', 'L', 'O', 'S', 'T', 'Z']);
            assert_eq!(stacker.next().len(), 5);
            blockfish::ai::BagObserver::from_visible(&stacker.snapshot().unwrap().queue)
        };
        Self {
            #[cfg(feature = "public-rollout")]
            world_garbage,
            #[cfg(feature = "public-rollout")]
            last_garbage_hole,
            #[cfg(feature = "public-rollout")]
            world_events: Vec::new(),
            ai,
            stacker,
            ds_goal,
            start_time: Instant::now(),
            trace: Vec::with_capacity(ds_goal.unwrap_or(100) * 5),
            search: Vec::new(),
            move_limit: None,
            inputs: Vec::new(),
            collect_features: false,
            samples: Vec::new(),
            #[cfg(feature = "bag-inference")]
            bag,
            #[cfg(feature = "bag-inference")]
            bag_events: Vec::new(),
            #[cfg(feature = "endgame-solver")]
            endgame_plan: Default::default(),
        }
    }

    fn ds(&self) -> usize {
        self.trace.last().cloned().unwrap_or(0)
    }

    fn pc(&self) -> usize {
        self.trace.len()
    }

    fn won(&self) -> bool {
        if let Some(goal) = self.ds_goal {
            self.ds() >= goal
        } else {
            false
        }
    }

    fn lost(&self) -> bool {
        // no current piece indicates lose
        self.stacker.current_piece_type().is_none()
    }

    fn done(&self) -> bool {
        self.won() || self.lost()
    }

    fn next_inputs(&mut self) -> Vec<blockfish::Input> {
        let snapshot = self.stacker.snapshot().expect("no snapshot");
        #[cfg(feature = "bag-inference")]
        let snapshot = {
            let mut snapshot = snapshot;
            let inferred = self.bag.inferred();
            let applied = inferred.is_some() && !cfg!(feature = "bag-placebo");
            self.bag_events.push(BagEvent {
                visible_queue: snapshot.queue.iter().map(|c| c.as_char()).collect(),
                hold: snapshot.hold.map(|c| c.as_char()),
                revealed: self.bag.revealed(),
                remaining_mask: self.bag.remaining_mask(),
                inferred: inferred.map(|c| c.as_char()),
                applied,
            });
            if applied {
                snapshot.queue.push(inferred.unwrap());
            }
            snapshot
        };
        if self.collect_features {
            self.samples.push(board_sample(&self.stacker, self.ds()));
        }
        #[cfg(feature="mirror-fusion")]
        {
            assert!(self.move_limit.is_none());
            let start=Instant::now();
            let (inputs,stats)=self.ai.mirror_decision(snapshot.clone());
            eprintln!("MIRROR {}",serde_json::to_string(&stats).unwrap());
            self.search.push(SearchStats{frontier_nodes:0,generated_nodes:stats.generated,sequences:stats.sequences,
                seconds:start.elapsed().as_secs_f64(),root_moves:stats.native_roots,
                rerank_nodes:stats.mapping+stats.mirrored_nodes,reranked:stats.changed,endgame_nodes:0,endgame_plan_pieces:0});
            return inputs;
        }
        #[cfg(feature = "public-rollout")]
        return self.world_inputs();
        #[cfg(not(feature = "endgame-solver"))]
        let (endgame_nodes, endgame_seconds) = (0, 0.0);
        #[cfg(feature = "endgame-solver")]
        let original_budget = self.ai.config().search_limit;
        #[cfg(feature = "endgame-solver")]
        let (endgame_nodes, endgame_seconds) = {
            if let Some(inputs) = self.endgame_plan.pop_front() {
                self.search.push(SearchStats {
                    frontier_nodes: 0,
                    generated_nodes: 0,
                    sequences: 0,
                    seconds: 0.0,
                    root_moves: 0,
                    rerank_nodes: 0,
                    reranked: false,
                    endgame_nodes: 0,
                    endgame_plan_pieces: self.endgame_plan.len() + 1,
                });
                return inputs;
            }
            let remaining = self.ds_goal.unwrap_or(usize::MAX).saturating_sub(self.ds());
            let visible = self
                .stacker
                .matrix()
                .filter(|(_, c)| *c == 'G')
                .map(|((row, _), _)| row as usize + 1)
                .max()
                .unwrap_or(0);
            let endgame_rows = if cfg!(feature = "endgame-wide") { 7 } else { 4 };
            if remaining > 0
                && remaining <= endgame_rows
                && visible == remaining
                && self.move_limit.is_none()
                && original_budget >= 5
            {
                let start = Instant::now();
                let result =
                    self.ai
                        .solve_endgame(snapshot.clone(), remaining as u16, original_budget / 5);
                let seconds = start.elapsed().as_secs_f64();
                let nodes = result.generated_nodes;
                if !cfg!(feature = "endgame-placebo") {
                    if let Some(plan) = result.plan {
                        let pieces = plan.len();
                        self.endgame_plan = plan.into();
                        self.search.push(SearchStats {
                            frontier_nodes: 0,
                            generated_nodes: nodes,
                            sequences: 1,
                            seconds,
                            root_moves: 0,
                            rerank_nodes: 0,
                            reranked: false,
                            endgame_nodes: nodes,
                            endgame_plan_pieces: pieces,
                        });
                        return self.endgame_plan.pop_front().expect("nonempty finish plan");
                    }
                }
                self.ai.config_mut().search_limit = original_budget - nodes;
                (nodes, seconds)
            } else {
                (0, 0.0)
            }
        };
        #[cfg(feature = "root-rerank")]
        let full_budget = self.ai.config().search_limit;
        #[cfg(feature = "root-rerank")]
        let allow_rerank = self.ds() < 80 && full_budget > 3;
        #[cfg(feature = "root-rerank")]
        if allow_rerank {
            self.ai.config_mut().search_limit = full_budget - 3;
        }
        #[cfg(feature = "race-goal")]
        let mut analysis = {
            let visible = self
                .stacker
                .matrix()
                .filter(|(_, c)| *c == 'G')
                .map(|((row, _), _)| row as usize + 1)
                .max()
                .unwrap_or(0);
            let final_goal = self
                .ds_goal
                .map_or(false, |goal| self.ds() + visible >= goal);
            self.ai
                .analyze_with_completion(snapshot, final_goal, self.move_limit)
        };
        #[cfg(feature="reveal-boundary")]
        let mut analysis = {
            assert!(self.move_limit.is_none());
            let visible = self.stacker.matrix().filter(|(_,color)|*color=='G')
                .map(|((row,_),_)|row+1).max().unwrap_or(0);
            let garbage=self.stacker.config().garbage;
            self.ai.analyze_until_reveal(snapshot,visible,garbage.min_height as u16,
                garbage.max_height as u16,self.ds()+usize::from(visible)<self.ds_goal.unwrap())
        };
        #[cfg(not(any(feature = "race-goal",feature="reveal-boundary")))]
        let mut analysis = match self.move_limit {
            Some(limit) => self.ai.analyze_for(snapshot, limit),
            None => self.ai.analyze(snapshot),
        };
        #[cfg(feature = "root-rerank")]
        {
            self.ai.config_mut().search_limit = full_budget;
        }
        analysis.wait();
        #[cfg(feature = "endgame-solver")]
        {
            self.ai.config_mut().search_limit = original_budget;
        }
        let mut moves: Vec<_> = analysis.all_moves().collect();
        moves.sort_by(|a, b| analysis.cmp(*a, *b));
        #[cfg(feature="plan-reuse")]
        if moves.is_empty() {
            let mut inputs=analysis.fallback_inputs().expect("no legal first placement");
            assert_eq!(inputs.pop(),Some(blockfish::Input::HD));
            self.ai.set_plan_hint(Default::default());
            let stats=analysis.stats().unwrap();
            self.search.push(SearchStats{frontier_nodes:stats.nodes,generated_nodes:stats.generated_nodes,
                sequences:stats.iterations,seconds:stats.time_taken.as_secs_f64(),root_moves:0,
                rerank_nodes:0,reranked:false,endgame_nodes:0,endgame_plan_pieces:0});
            eprintln!("LEGAL_FALLBACK used");return inputs;
        }
        let move_id = *moves.first().expect("no suggestions");
        #[cfg(feature="plan-reuse")]
        self.ai.set_plan_hint(analysis.continuation(move_id));
        #[cfg(not(feature = "root-rerank"))]
        let (rerank_nodes, reranked) = (0, false);
        #[cfg(feature = "root-rerank")]
        let (move_id, rerank_nodes, reranked) = {
            let mut chosen = move_id;
            let mut count = 0;
            if allow_rerank && moves.len() > 1 {
                let candidates: Vec<_> = moves
                    .iter()
                    .take(3)
                    .map(|&m| (m, analysis.suggestion(m, 1)))
                    .collect();
                if candidates.iter().all(|(_, s)| s.rating >= 20) {
                    let scores: Vec<_> = candidates
                        .iter()
                        .map(|(m, s)| {
                            let mut inputs = s.inputs.clone();
                            assert_eq!(inputs.pop(), Some(blockfish::Input::HD));
                            let features = self.stacker.projected_features(&inputs);
                            (*m, blockfish::ai::action_rank::score(s.rating, &features))
                        })
                        .collect();
                    count = scores.len();
                    let mut best = scores[0];
                    for &(m, score) in &scores[1..] {
                        if score < best.1 {
                            best = (m, score);
                        }
                    }
                    if scores[0].1 - best.1 > blockfish::ai::action_rank::MARGIN
                        && !cfg!(feature = "root-rerank-placebo")
                    {
                        chosen = best.0;
                    }
                }
            }
            (chosen, count, chosen != move_id)
        };
        if let Some(stats) = analysis.stats() {
            self.search.push(SearchStats {
                frontier_nodes: stats.nodes,
                generated_nodes: stats.generated_nodes + rerank_nodes + endgame_nodes,
                sequences: stats.iterations,
                seconds: stats.time_taken.as_secs_f64() + endgame_seconds,
                root_moves: moves.len(),
                rerank_nodes,
                reranked,
                endgame_nodes,
                endgame_plan_pieces: 0,
            });
        }
        let mut inputs = analysis.suggestion(move_id, 1).inputs;
        assert_eq!(inputs.pop(), Some(blockfish::Input::HD));
        inputs
    }

    #[cfg(feature = "public-rollout")]
    fn world_inputs(&mut self) -> Vec<blockfish::Input> {
        assert!(self.move_limit.is_none() && self.bag.valid());
        let start = Instant::now();
        let observation = block_stacker::PublicObservation {
            cells: self.stacker.matrix().collect(),
            current: self.stacker.current_piece_type().unwrap(),
            next: self.stacker.next().to_vec(), held: self.stacker.held(),
            remaining_bag: "IJLOSTZ".chars().enumerate()
                .filter(|(i,_)|self.bag.remaining_mask() & (1 << i) != 0).map(|(_,c)|c).collect(),
            garbage_generated: self.ds()+world_rollout::visible(&self.stacker),
            last_garbage_hole: self.last_garbage_hole,
        };
        let cleared = self.ds();
        let d = world_rollout::plan(&mut self.ai, observation, cleared,
            self.stacker.ruleset().clone(), self.world_garbage.clone(), cfg!(feature="rollout-placebo"));
        let inputs=d.candidates[d.chosen].inputs.clone();
        self.search.push(SearchStats {frontier_nodes:0, generated_nodes:d.generated(),
            sequences:d.sequences, seconds:start.elapsed().as_secs_f64(), root_moves:d.base_roots,
            rerank_nodes:d.rollout_nodes+d.state_nodes, reranked:d.chosen!=0,
            endgame_nodes:0,endgame_plan_pieces:0});
        self.world_events.push(d);
        inputs
    }

    fn step(&mut self) {
        let inputs = self.next_inputs();
        #[cfg(feature = "bag-inference")]
        let revealed_count = 1
            + (self.stacker.held().is_none() && inputs.contains(&blockfish::Input::Hold)) as usize;
        self.inputs.push(inputs.clone());
        #[cfg(feature = "public-rollout")]
        let before_visible=world_rollout::visible(&self.stacker);
        self.stacker.run(inputs);
        let (_, garbage_cleared) = self.stacker.hard_drop();
        self.trace.push(self.ds() + garbage_cleared);
        #[cfg(feature = "public-rollout")]
        if world_rollout::visible(&self.stacker)+garbage_cleared>before_visible {
            self.last_garbage_hole=world_rollout::bottom_hole(&self.stacker);
        }
        #[cfg(feature = "bag-inference")]
        {
            let previews = self.stacker.next();
            for &color in &previews[previews.len() - revealed_count..] {
                self.bag.observe(color.try_into().unwrap());
            }
        }
    }

    fn print_stats(&self, w: &mut impl Write, now: Instant, short: bool) -> std::io::Result<()> {
        let elapsed = (now - self.start_time).as_secs_f64();
        let ds = self.ds();
        let pc = self.pc();
        let pps = (pc as f64) / elapsed;
        if short {
            write!(w, "{} p, {}", pc, ds)?;
            if let Some(goal) = self.ds_goal {
                write!(w, "/{}", goal)?;
            }
            write!(w, "L ds, {:.2} pps ", pps)?;
        } else {
            if self.lost() {
                writeln!(w, "topped out early")?;
            }
            writeln!(w, "{} pieces", pc)?;
            writeln!(w, "{}L downstack", ds)?;
            writeln!(w, "total time: {:.2}s ({:.2}pps)", elapsed, pps)?;
            writeln!(w, "PRNG seed: {}", self.stacker.prng_seed())?;
        }
        Ok(())
    }

    fn as_trace(&self) -> Trace {
        let time = (Instant::now() - self.start_time).as_secs_f64();
        let seed = self.stacker.prng_seed();
        Trace {
            #[cfg(feature = "public-rollout")]
            world_events: &self.world_events,
            time,
            seed,
            ds: &self.trace,
            search: &self.search,
            inputs: &self.inputs,
            ai_config: self.ai.config().to_string(),
            move_ms: self.move_limit.map(|d| d.as_millis()),
            garbage_min: self.stacker.config().garbage.min_height,
            garbage_max: self.stacker.config().garbage.max_height,
            samples: &self.samples,
            #[cfg(feature = "bag-inference")]
            bag_events: &self.bag_events,
            #[cfg(feature = "bag-inference")]
            bag_valid: self.bag.valid(),
            #[cfg(feature = "bag-inference")]
            bag_checks: self.bag.checks(),
        }
    }
}

static REFRESH_PERIOD: Duration = Duration::from_millis(500);

#[derive(Deserialize)]
struct Replay {
    seed: u64,
    ds: Vec<usize>,
    inputs: Vec<Vec<blockfish::Input>>,
    garbage_min: usize,
    garbage_max: usize,
}

fn verify_replay(
    path: &PathBuf,
    features_out: Option<&PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let replay: Replay = serde_json::from_reader(std::fs::File::open(path)?)?;
    if replay.ds.len() != replay.inputs.len() {
        return Err("trace lengths differ".into());
    }
    let mut cfg = BSConfig::default();
    cfg.prng_seed = Some(replay.seed);
    cfg.garbage.total_lines = Some(100);
    cfg.garbage.min_height = replay.garbage_min;
    cfg.garbage.max_height = replay.garbage_max;
    let mut stacker = Stacker::new(Ruleset::guideline().into(), cfg);
    let mut ds = 0;
    let mut samples = Vec::new();
    for (step, inputs) in replay.inputs.into_iter().enumerate() {
        if stacker.current_piece_type().is_none() {
            return Err("topped out before trace ends".into());
        }
        if features_out.is_some() {
            samples.push(board_sample(&stacker, ds));
        }
        if inputs.contains(&blockfish::Input::HD) {
            return Err("unexpected extra hard drop".into());
        }
        if inputs
            .iter()
            .filter(|&&i| i == blockfish::Input::Hold)
            .count()
            > 1
        {
            return Err("multiple holds in a placement".into());
        }
        stacker.run(inputs);
        ds += stacker.hard_drop().1;
        if ds != replay.ds[step] {
            return Err(format!("clear mismatch at piece {}", step + 1).into());
        }
        if ds >= 100 && step + 1 != replay.ds.len() {
            return Err("inputs after completion".into());
        }
    }
    if ds != 100 {
        return Err("replay did not clear 100 garbage rows".into());
    }
    if let Some(out) = features_out {
        serde_json::to_writer(std::fs::File::create(out)?, &samples)?;
    }
    println!(
        "Verified {} pieces, 100 garbage rows, seed {}",
        replay.ds.len(),
        replay.seed
    );
    Ok(())
}

fn main() {
    let args = argh::from_env::<Args>();
    #[cfg(feature = "public-rollout")]
    if let Some(path)=&args.world_audit {
        verify_replay(path,None).expect("actual replay failed");
        println!("Verified {} hypothetical paths",world_rollout::audit(path).expect("world audit failed"));
        return;
    }
    if let Some(path) = &args.replay_file {
        if let Err(error) = verify_replay(path, args.replay_features_out.as_ref()) {
            eprintln!("Replay verification failed: {}", error);
            std::process::exit(1);
        }
        return;
    }
    if args.replay_features_out.is_some() {
        eprintln!("--replay-features-out requires --replay-file");
        std::process::exit(1);
    }

    // cheese race bot
    let mut race = Race::new(
        args.to_ai_config(),
        args.to_game_config(),
        Ruleset::guideline(),
    );
    race.move_limit = args.move_ms.map(Duration::from_millis);
    race.collect_features = args.collect_features;

    // periodic stats updates
    let stderr = std::io::stderr();
    let mut stderr = if args.quiet {
        None
    } else {
        Some(stderr.lock())
    };
    let mut prev_refresh = Instant::now();

    // install ctrl-c signal handler
    let sigint = Arc::new(atomic::AtomicBool::new(false));
    let sigint_handler = {
        let sigint = sigint.clone();
        move || sigint.store(true, atomic::Ordering::Relaxed)
    };
    let sigint_handler_id = unsafe { signal_hook::register(signal_hook::SIGINT, sigint_handler) }
        .expect("failed to register interrupt signal handler");

    // loop until goal is met
    while !race.done() {
        race.step();

        if let Some(mut out) = stderr.as_mut() {
            let now = Instant::now();
            if now >= prev_refresh + REFRESH_PERIOD {
                let _ = write!(out, "\r");
                let _ = race.print_stats(&mut out, now, true);
                let _ = out.flush();
                prev_refresh = now;
            }
        }

        if sigint.load(atomic::Ordering::Relaxed) {
            signal_hook::unregister(sigint_handler_id);
            stderr = None;
            eprint!("\r\n--- caught ctrl-C ---\n\n");
            break;
        }
    }

    // finish stats updates
    if let Some(mut out) = stderr {
        let _ = write!(out, "\r                                \r");
    }

    // print report
    let stdout = std::io::stdout();
    let mut stdout = stdout.lock();
    race.print_stats(&mut stdout, Instant::now(), false)
        .unwrap();
    if args.ascii {
        writeln!(stdout, "\n{:?}", race.stacker).unwrap();
    }

    // render JSON if requested
    if let Some(path) = args.trace_file {
        let file = match std::fs::File::create(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error writing trace file:\n{}", e);
                std::process::exit(1);
            }
        };
        serde_json::to_writer(file, &race.as_trace()).unwrap();
    }

    std::process::exit(if race.won() {
        0
    } else if race.lost() {
        1
    } else {
        // ctrl-c
        130
    });
}
