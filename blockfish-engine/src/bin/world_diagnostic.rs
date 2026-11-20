//! Offline diagnostic only: no search, model fitting, or new game seeds.
#[path = "world_rollout/mod.rs"]
mod world_rollout;
use block_stacker::{Config,Ruleset,Stacker};
use blockfish::{Input,StackerExt};
use serde::Deserialize;
#[derive(Deserialize)]
struct Saved {world_events:Vec<world_rollout::Decision>,inputs:Vec<Vec<Input>>,garbage_min:usize,garbage_max:usize}
fn state(s:&Stacker)->(Vec<((u16,u16),char)>,Option<char>,Option<char>,Vec<char>) {
    (s.matrix().collect(),s.current_piece_type(),s.held(),s.next().to_vec())
}
fn main() {
    let path=std::env::args().nth(1).expect("trace path");
    let t:Saved=serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    let mut cfg=Config::default();cfg.garbage.min_height=t.garbage_min;
    cfg.garbage.max_height=t.garbage_max;cfg.garbage.total_lines=Some(100);
    let rules=std::rc::Rc::new(Ruleset::guideline());
    let(mut examined,mut matched,mut different_inputs,mut different_outcomes)=(0,0,0,0);
    for (i,d) in t.world_events.iter().enumerate() {
        assert!(d.placebo);assert_eq!(d.chosen,0);
        if i+1==t.world_events.len() {continue;}
        for sample in &d.candidates[0].samples {
            if sample.actions.len()<2 {continue;}
            examined+=1;
            let mut world=Stacker::from_public(rules.clone(),cfg.garbage.clone(),&d.observation,sample.seed);
            world.run(sample.actions[0].clone());world.hard_drop();
            let actual=&t.world_events[i+1].observation;
            if state(&world)!=(actual.cells.clone(),Some(actual.current),actual.held,actual.next.clone()) {continue;}
            matched+=1;
            different_inputs+=usize::from(sample.actions[1]!=t.inputs[i+1]);
            let mut strong=world.clone();strong.run(t.inputs[i+1].clone());let strong_clear=strong.hard_drop();
            world.run(sample.actions[1].clone());let weak_clear=world.hard_drop();
            different_outcomes+=usize::from(state(&world)!=state(&strong)||weak_clear!=strong_clear);
        }
    }
    println!("{}",serde_json::json!({"examined":examined,"matched_public_states":matched,
        "different_inputs":different_inputs,"different_outcomes":different_outcomes}));
}
