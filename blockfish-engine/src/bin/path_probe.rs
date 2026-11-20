//! Audit returned paths in an independent game simulator after garbage supply ends.
use block_stacker::{GarbageConfig, PublicObservation, Ruleset, Stacker};
use blockfish::{ai::AI, Config, Input, StackerExt};
use serde_json::json;
use std::io::{self, BufRead};
use std::time::Instant;

fn replay(mut game: Stacker, inputs: &[Input], remaining: usize) -> (Vec<usize>, bool) {
    let mut progress = Vec::new();
    let mut cleared = 0;
    for input in inputs {
        let ok = match input {
            Input::Left => game.move_horizontal(-1),
            Input::Right => game.move_horizontal(1),
            Input::CW => game.rotate(1),
            Input::CCW => game.rotate(-1),
            Input::Hold => game.hold(),
            Input::SD => game.sonic_drop(),
            Input::HD => {
                if game.current_piece_type().is_none() { return (progress,false); }
                cleared += game.hard_drop().1;
                progress.push(cleared);
                if progress.len() == 6 || cleared == remaining { break; }
                true
            }
        };
        if !ok { return (progress,false); }
    }
    (progress,true)
}

fn main() {
    let mut ai = AI::new(Config::default());
    for line in io::stdin().lock().lines() {
        let obs: PublicObservation = serde_json::from_str(&line.unwrap()).unwrap();
        assert_eq!(obs.garbage_generated,100);
        assert_eq!(obs.next.len(),5);
        assert!(obs.held.is_some()); // Six locks can use only publicly known pieces.
        let remaining = obs.cells.iter().filter(|(_,c)| *c == 'G')
            .map(|((r,_),_)| *r).collect::<std::collections::HashSet<_>>().len();
        let world = |seed| Stacker::from_public(Ruleset::guideline().into(),
            GarbageConfig {total_lines:Some(100), ..Default::default()},&obs,seed);
        let a = world(37);
        let b = world(137);
        let start = Instant::now();
        let mut analysis = ai.analyze(a.snapshot().unwrap());
        analysis.wait();
        let seconds = start.elapsed().as_secs_f64();
        let mut moves: Vec<_> = analysis.all_moves().collect();
        moves.sort_by(|x,y|analysis.cmp(*x,*y));
        assert!(!moves.is_empty());
        let mut paths = vec![];
        for id in moves {
            let inputs = analysis.suggestion(id,usize::MAX).inputs;
            let (progress,valid) = replay(a.clone(),&inputs,remaining);
            assert_eq!((progress.clone(),valid),replay(b.clone(),&inputs,remaining));
            paths.push(json!({"inputs":inputs,"progress":progress,"valid":valid,
                "solved":valid && progress.last()==Some(&remaining)}));
        }
        let nodes = analysis.stats().unwrap().generated_nodes;
        assert!(nodes<=50000);
        println!("{}",json!({"remaining":remaining,"nodes":nodes,"seconds":seconds,"paths":paths}));
    }
}
