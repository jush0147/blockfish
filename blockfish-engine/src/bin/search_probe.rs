//! Fixed public observation, variable legal preview and node limit.
use blockfish::{ai::{AI, Snapshot}, BasicMatrix, Color, Config, Input};
use serde::Deserialize;
use std::io::{self, BufRead};

#[derive(Deserialize)]
struct Observation {
    cells: Vec<((u16,u16), char)>,
    current: Color,
    next: Vec<Color>,
    held: Option<Color>,
}

fn main() {
    let mut ai = AI::new(Config::default());
    for line in io::stdin().lock().lines() {
        let obs: Observation = serde_json::from_str(&line.unwrap()).unwrap();
        let mut matrix = BasicMatrix::with_cols(10);
        for (coord, _) in obs.cells { matrix.set(coord); }
        assert_eq!(obs.next.len(), 5);
        for preview in [4, 5].iter().copied() {
            for nodes in [5000, 50000].iter().copied() {
                ai.config_mut().search_limit = nodes;
                let mut queue = vec![obs.current];
                queue.extend_from_slice(&obs.next[..preview]);
                let mut a = ai.analyze(Snapshot { matrix: matrix.clone(), hold: obs.held, queue });
                a.wait();
                let mut moves: Vec<_> = a.all_moves().collect();
                moves.sort_by(|x,y| a.cmp(*x,*y));
                let actions: Vec<_> = moves.iter().map(|id| {
                    let mut inputs = a.suggestion(*id,1).inputs;
                    assert_eq!(inputs.pop(),Some(Input::HD));
                    inputs
                }).collect();
                let generated = a.stats().unwrap().generated_nodes;
                assert!(generated <= nodes);
                println!("{}", serde_json::json!({"nodes":nodes,"generated":generated,"preview":preview,"actions":actions}));
            }
        }
    }
}
