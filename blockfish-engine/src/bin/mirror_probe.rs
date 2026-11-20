//! Read public states only. No source game RNG or future inputs.
use blockfish::{ai::{AI,Snapshot}, BasicMatrix, Color,Config};
use serde::Deserialize;
use std::io::{self,BufRead};
#[derive(Deserialize)]
struct Observation {cells:Vec<((u16,u16),char)>,current:Color,next:Vec<Color>,held:Option<Color>}
fn main() {
    let mut ai=AI::new(Config::default());
    for line in io::stdin().lock().lines() {
        let obs:Observation=serde_json::from_str(&line.unwrap()).unwrap();
        let mut matrix=BasicMatrix::with_cols(10);
        for (coord,_) in obs.cells {matrix.set(coord);}
        let mut queue=vec![obs.current];queue.extend(obs.next);
        let (_,stats)=ai.mirror_decision(Snapshot{matrix,hold:obs.held,queue});
        println!("{}",serde_json::to_string(&stats).unwrap());
    }
}
