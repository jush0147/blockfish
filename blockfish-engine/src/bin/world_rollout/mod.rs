//! Research planner: this module never receives the actual Stacker or its seed.
use block_stacker::{GarbageConfig, PublicObservation, Ruleset, Stacker};
use blockfish::{ai::{AI, Snapshot}, BasicMatrix, Config, Input, StackerExt};
use serde::{Deserialize, Serialize};
use std::{convert::TryInto, rc::Rc};

const PRICE: f64 = 2.083984375; // Frozen original training garbage price.
const WORLDS: usize = 2;
const HORIZON: usize = 4;

pub fn visible(game: &Stacker) -> usize {
    game.matrix().filter(|(_,c)|*c=='G').map(|((r,_),_)|r as usize+1).max().unwrap_or(0)
}
pub fn bottom_hole(game: &Stacker) -> Option<u16> {
    if visible(game)==0 { return None; }
    (0..game.ruleset().cols as u16).find(|&c|!game.matrix().any(|((r,j),_)|r==0&&j==c))
}
fn snapshot(obs: &PublicObservation) -> Snapshot {
    let mut matrix=BasicMatrix::with_cols(10);
    for &(coord,_) in &obs.cells { matrix.set(coord); }
    Snapshot {matrix,hold:obs.held.map(|c|c.try_into().unwrap()),
        queue:std::iter::once(obs.current).chain(obs.next.iter().copied()).map(|c|c.try_into().unwrap()).collect()}
}
pub fn public_hash(obs: &PublicObservation) -> u64 {
    serde_json::to_vec(obs).unwrap().into_iter().fold(0xcbf29ce484222325u64,
        |h,b|(h^b as u64).wrapping_mul(0x100000001b3))
}

#[derive(Serialize,Deserialize)]
pub struct Sample {
    pub seed: u64,
    pub actions: Vec<Vec<Input>>,
    pub cleared: usize,
    pub visible_garbage: usize,
    pub cost: f64,
    pub truncated: bool,
    pub topout: bool,
    pub search_nodes: usize,
}
#[derive(Serialize,Deserialize)]
pub struct Candidate {
    pub inputs: Vec<Input>,
    pub base_rating: i64,
    pub samples: Vec<Sample>,
    pub cost: f64,
}
#[derive(Serialize,Deserialize)]
pub struct Decision {
    pub observation: PublicObservation,
    pub cleared_before: usize,
    pub public_hash: u64,
    pub budget: usize,
    pub base_limit: usize,
    pub rollout_limit: usize,
    pub base_nodes: usize,
    pub rollout_nodes: usize,
    pub state_nodes: usize,
    pub sequences: usize,
    pub base_roots: usize,
    pub chosen: usize,
    pub placebo: bool,
    pub candidates: Vec<Candidate>,
}
impl Decision {
    pub fn generated(&self) -> usize { self.base_nodes+self.rollout_nodes+self.state_nodes }
}
fn cost(game:&Stacker, cleared:usize, locks:usize, config:&Config) -> f64 {
    if cleared>=100 { return locks as f64; }
    if game.current_piece_type().is_none() { return 1000.0+locks as f64; }
    let matrix=game.snapshot().unwrap().matrix;
    locks as f64+blockfish::ai::rollout_board_cost(&matrix,config)
        +PRICE*100usize.saturating_sub(cleared+visible(game)) as f64
}

pub fn plan(ai:&mut AI, obs:PublicObservation, cleared:usize,
            rules:Rc<Ruleset>, garbage:GarbageConfig, placebo:bool) -> Decision {
    assert_eq!(rules.cols,10);assert_eq!(rules.previews,5);
    assert_eq!(garbage.total_lines,Some(100));
    let config=ai.config();let budget=config.search_limit;
    assert!(budget>=1000);
    let base_limit=budget*3/4;
    ai.config_mut().search_limit=base_limit;
    let mut analysis=ai.analyze(snapshot(&obs));analysis.wait();
    let stats=analysis.stats().unwrap();
    let mut moves:Vec<_>=analysis.all_moves().collect();moves.sort_by(|a,b|analysis.cmp(*a,*b));
    assert!(!moves.is_empty(),"base search did not find a legal continuation");
    let base_roots=moves.len();
    let mut candidates:Vec<_>=moves.into_iter().take(2).map(|m| {
        let s=analysis.suggestion(m,1);let mut inputs=s.inputs;
        assert_eq!(inputs.pop(),Some(Input::HD));
        Candidate {inputs,base_rating:s.rating,samples:vec![],cost:0.0}
    }).collect();
    let paths=candidates.len()*WORLDS;
    let overhead=paths*(HORIZON+1);
    let rollout_limit=(budget-base_limit-overhead)/(paths*(HORIZON-1));
    assert!(rollout_limit>0);
    ai.config_mut().search_limit=rollout_limit;
    let hash=public_hash(&obs);
    let mut rollout_nodes=0;let mut state_nodes=0;let mut sequences=stats.iterations;
    for candidate in &mut candidates {
        for index in 0..WORLDS {
            let seed=hash^((index as u64+1).wrapping_mul(0x9e3779b97f4a7c15));
            let mut world=Stacker::from_public(rules.clone(),garbage.clone(),&obs,seed);
            state_nodes+=1;
            let mut ds=cleared;let mut actions=vec![];let mut truncated=false;let mut used=0;
            for step in 0..HORIZON {
                if ds>=100 || world.current_piece_type().is_none() { break; }
                let inputs=if step==0 { candidate.inputs.clone() } else {
                    let mut a=ai.analyze(world.snapshot().unwrap());a.wait();
                    let s=a.stats().unwrap();used+=s.generated_nodes;sequences+=s.iterations;
                    assert!(s.generated_nodes<=rollout_limit);
                    let best=a.all_moves().min_by(|x,y|a.cmp(*x,*y));
                    match best {
                        Some(m) => { let mut inputs=a.suggestion(m,1).inputs;
                            assert_eq!(inputs.pop(),Some(Input::HD));inputs },
                        None => { truncated=true;break; },
                    }
                };
                world.run(inputs.clone());ds+=world.hard_drop().1;state_nodes+=1;actions.push(inputs);
            }
            rollout_nodes+=used;
            candidate.samples.push(Sample {seed,cleared:ds,visible_garbage:visible(&world),
                cost:cost(&world,ds,actions.len(),&config),topout:world.current_piece_type().is_none(),
                actions,truncated,search_nodes:used});
        }
        candidate.cost=candidate.samples.iter().map(|s|s.cost).sum::<f64>()/WORLDS as f64;
    }
    *ai.config_mut()=config;
    let mut chosen=0;
    if !placebo { for i in 1..candidates.len() {
        if candidates[i].cost<candidates[chosen].cost { chosen=i; }
    }}
    let decision=Decision {observation:obs,cleared_before:cleared,public_hash:hash,budget,base_limit,
        rollout_limit,base_nodes:stats.generated_nodes,rollout_nodes,state_nodes,sequences,
        base_roots,chosen,placebo,candidates};
    assert!(decision.generated()<=budget);
    decision
}

pub fn audit(path:&std::path::Path) -> Result<usize,Box<dyn std::error::Error>> {
    #[derive(Deserialize)]
    struct Saved {world_events:Vec<Decision>,garbage_min:usize,garbage_max:usize,ai_config:String,
        seed:u64,inputs:Vec<Vec<Input>>,ds:Vec<usize>,search:Vec<SavedSearch>}
    #[derive(Deserialize)]
    struct SavedSearch {generated_nodes:usize}
    let saved:Saved=serde_json::from_reader(std::fs::File::open(path)?)?;
    let config:Config=saved.ai_config.parse()?;
    let rules:Rc<Ruleset>=Ruleset::guideline().into();
    let garbage=GarbageConfig {min_height:saved.garbage_min,max_height:saved.garbage_max,total_lines:Some(100)};
    // Actual seed is used only by this audit to reconstruct already observed history.
    // The planner above receives exclusively PublicObservation and a public-derived seed.
    let mut cfg=block_stacker::Config::default();cfg.garbage=garbage.clone();cfg.prng_seed=Some(saved.seed);
    let mut actual=Stacker::new(rules.clone(),cfg);
    let mut bag=blockfish::ai::BagObserver::from_visible(&actual.snapshot().unwrap().queue);
    let mut last_hole=bottom_hole(&actual);let mut cleared=0;
    assert_eq!(saved.world_events.len(),saved.inputs.len());
    assert_eq!(saved.ds.len(),saved.inputs.len());assert_eq!(saved.search.len(),saved.inputs.len());
    for (step,d) in saved.world_events.iter().enumerate() {
        let expected=PublicObservation {cells:actual.matrix().collect(),current:actual.current_piece_type().unwrap(),
            next:actual.next().to_vec(),held:actual.held(),remaining_bag:"IJLOSTZ".chars().enumerate()
                .filter(|(i,_)|bag.remaining_mask()&(1<<i)!=0).map(|(_,c)|c).collect(),
            garbage_generated:cleared+visible(&actual),last_garbage_hole:last_hole};
        assert!(bag.valid());assert_eq!(d.observation,expected);assert_eq!(d.cleared_before,cleared);
        assert_eq!(d.candidates[d.chosen].inputs,saved.inputs[step]);
        assert_eq!(d.generated(),saved.search[step].generated_nodes);
        let reveals=1+(actual.held().is_none()&&saved.inputs[step].contains(&Input::Hold)) as usize;
        let before=visible(&actual);actual.run(saved.inputs[step].clone());
        let n=actual.hard_drop().1;cleared+=n;assert_eq!(cleared,saved.ds[step]);
        if visible(&actual)+n>before {last_hole=bottom_hole(&actual);}
        let previews=actual.next();for &c in &previews[previews.len()-reveals..] {bag.observe(c.try_into().unwrap());}
    }
    let mut count=0;
    for d in saved.world_events {
        assert_eq!(public_hash(&d.observation),d.public_hash);
        assert_eq!(d.budget,config.search_limit);assert!(d.generated()<=d.budget);
        assert!(d.candidates.len()>=1&&d.candidates.len()<=2);
        assert_eq!(d.base_limit,d.budget*3/4);assert!(d.base_nodes<=d.base_limit);
        let paths=d.candidates.len()*WORLDS;
        assert_eq!(d.rollout_limit,(d.budget-d.base_limit-paths*(HORIZON+1))/(paths*(HORIZON-1)));
        let mut charged=0;let mut search=0;let mut best=0;
        for (i,c) in d.candidates.iter().enumerate() {
            assert_eq!(c.samples.len(),WORLDS);
            for (j,s) in c.samples.iter().enumerate() {
                assert_eq!(s.seed,d.public_hash^((j as u64+1).wrapping_mul(0x9e3779b97f4a7c15)));
                let mut world=Stacker::from_public(rules.clone(),garbage.clone(),&d.observation,s.seed);
                charged+=1;let mut ds=d.cleared_before;
                assert!(!s.actions.is_empty()&&s.actions.len()<=HORIZON);
                assert_eq!(s.actions[0],c.inputs);
                for inputs in &s.actions {
                    assert!(ds<100 && world.current_piece_type().is_some());
                    assert!(!inputs.contains(&Input::HD));
                    assert!(inputs.iter().filter(|&&x|x==Input::Hold).count()<=1);
                    world.run(inputs.clone());ds+=world.hard_drop().1;charged+=1;
                }
                assert_eq!(ds,s.cleared);assert_eq!(visible(&world),s.visible_garbage);
                assert_eq!(world.current_piece_type().is_none(),s.topout);
                assert!((cost(&world,ds,s.actions.len(),&config)-s.cost).abs()<1e-8);
                assert!(s.search_nodes<=d.rollout_limit*(HORIZON-1));search+=s.search_nodes;count+=1;
            }
            assert!((c.cost-c.samples.iter().map(|s|s.cost).sum::<f64>()/WORLDS as f64).abs()<1e-8);
            if c.cost<d.candidates[best].cost { best=i; }
        }
        assert_eq!(d.state_nodes,charged);assert_eq!(d.rollout_nodes,search);
        assert_eq!(d.chosen,if d.placebo {0}else{best});
    }
    Ok(count)
}
