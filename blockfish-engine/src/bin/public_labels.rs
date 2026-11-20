//! Offline bounded public-world counterfactual labels. Not a deployed policy.
#[path="world_rollout/mod.rs"] mod world_rollout;
use block_stacker::{Stacker,Ruleset,PublicObservation,GarbageConfig,Config as GameConfig};
use blockfish::{ai::{AI,BagObserver},Config,Input,StackerExt};
use serde::{Serialize,Deserialize};
use std::{rc::Rc,convert::TryInto,path::Path};
const STRONG:usize=50000;
const WEAK:usize=1040;
const MAX_LOCKS:usize=400;
#[derive(Deserialize)]
struct Source {seed:u64,inputs:Vec<Vec<Input>>,ds:Vec<usize>,garbage_min:usize,garbage_max:usize}
#[derive(Serialize,Deserialize)]
struct PathRun {actions:Vec<Vec<Input>>,search:Vec<usize>,cleared:usize,complete:bool,topout:bool,truncated:bool,cost:f64,short_cost:f64}
#[derive(Serialize,Deserialize)]
struct Sample {seed:u64,strong:PathRun,weak:PathRun}
#[derive(Serialize,Deserialize)]
struct Root {inputs:Vec<Input>,rating:i64,samples:Vec<Sample>}
#[derive(Serialize,Deserialize)]
struct Saved {source:String,observation:PublicObservation,cleared:usize,garbage:SavedGarbage,public_hash:u64,base_nodes:usize,roots:Vec<Root>}
#[derive(Serialize,Deserialize)]
struct SavedGarbage {min:usize,max:usize}
fn observe(source:&Path)->(PublicObservation,usize,GarbageConfig) {
    let s:Source=serde_json::from_reader(std::fs::File::open(source).unwrap()).unwrap();
    assert!(s.inputs.len()>100&&s.ds.len()==s.inputs.len());assert_eq!(s.ds.last(),Some(&100));
    let mut cfg=GameConfig::default();cfg.prng_seed=Some(s.seed);cfg.garbage.total_lines=Some(100);
    cfg.garbage.min_height=s.garbage_min;cfg.garbage.max_height=s.garbage_max;
    let garbage=cfg.garbage.clone();let mut game=Stacker::new(Ruleset::guideline().into(),cfg);
    let mut bag=BagObserver::from_visible(&game.snapshot().unwrap().queue);
    let mut last=world_rollout::bottom_hole(&game);let mut cleared=0;
    // Actual seed is used only to replay the already observed source prefix.
    for (i,inputs) in s.inputs[..100].iter().enumerate() {
        let reveals=1+usize::from(game.held().is_none()&&inputs.contains(&Input::Hold));
        let before=world_rollout::visible(&game);game.run(inputs.clone());
        let n=game.hard_drop().1;cleared+=n;assert_eq!(cleared,s.ds[i]);
        if world_rollout::visible(&game)+n>before {last=world_rollout::bottom_hole(&game);}
        let previews=game.next();for &c in &previews[previews.len()-reveals..]{bag.observe(c.try_into().unwrap());}
    }
    assert!(bag.valid()&&cleared<100);
    (PublicObservation{cells:game.matrix().collect(),current:game.current_piece_type().unwrap(),next:game.next().to_vec(),held:game.held(),
        remaining_bag:"IJLOSTZ".chars().enumerate().filter(|(i,_)|bag.remaining_mask()&(1<<i)!=0).map(|(_,c)|c).collect(),
        garbage_generated:cleared+world_rollout::visible(&game),last_garbage_hole:last},cleared,garbage)
}
fn endpoint(game:&Stacker,ds:usize,locks:usize,config:&Config)->f64 {
    if ds>=100{return locks as f64;}
    if game.current_piece_type().is_none(){return 1000.0+locks as f64;}
    locks as f64+blockfish::ai::rollout_board_cost(&game.snapshot().unwrap().matrix,config)
        +2.083984375*100usize.saturating_sub(ds+world_rollout::visible(game)) as f64
}
fn best(ai:&mut AI,game:&Stacker)->(Option<Vec<Input>>,usize) {
    let mut a=ai.analyze(game.snapshot().unwrap());a.wait();let n=a.stats().unwrap().generated_nodes;
    assert!(n<=ai.config().search_limit);
    let m=a.all_moves().min_by(|x,y|a.cmp(*x,*y));
    let inputs=m.map(|m|{let mut v=a.suggestion(m,1).inputs;assert_eq!(v.pop(),Some(Input::HD));v});
    (inputs,n)
}
fn rollout(ai:&mut AI,mut game:Stacker,mut ds:usize,first:&[Input],budget:usize,horizon:usize)->PathRun {
    ai.config_mut().search_limit=budget;
    let mut actions=vec![];let mut search=vec![];let mut short=None;let mut truncated=false;
    while ds<100&&game.current_piece_type().is_some()&&actions.len()<horizon {
        let inputs=if actions.is_empty(){first.to_vec()}else{
            let (inputs,n)=best(ai,&game);search.push(n);
            match inputs {Some(v)=>v,None=>{truncated=true;break;}}
        };
        game.run(inputs.clone());ds+=game.hard_drop().1;actions.push(inputs);
        if actions.len()==4 {short=Some(endpoint(&game,ds,actions.len(),&ai.config()));}
    }
    let complete=ds>=100;let topout=game.current_piece_type().is_none();
    let short_cost=short.unwrap_or_else(||endpoint(&game,ds,actions.len(),&ai.config()));
    let cost=if horizon==4 {endpoint(&game,ds,actions.len(),&ai.config())}
        else if complete {actions.len() as f64} else {1000.0+actions.len() as f64};
    PathRun{actions,search,cleared:ds,complete,topout,truncated,cost,short_cost}
}
fn audit_run(mut game:Stacker,start:usize,root:&[Input],run:&PathRun,budget:usize,horizon:usize,config:&Config) {
    assert!(!run.actions.is_empty()&&run.actions.len()<=horizon);assert_eq!(run.actions[0],root);
    assert!(run.search.iter().all(|&n|n<=budget));
    assert_eq!(run.search.len(),run.actions.len()-1+usize::from(run.truncated));
    let mut ds=start;let mut short=None;
    for (i,inputs) in run.actions.iter().enumerate() {
        assert!(ds<100&&game.current_piece_type().is_some());assert!(!inputs.contains(&Input::HD));
        assert!(inputs.iter().filter(|&&x|x==Input::Hold).count()<=1);
        game.run(inputs.clone());ds+=game.hard_drop().1;
        if i==3 {short=Some(endpoint(&game,ds,4,config));}
    }
    assert_eq!(ds,run.cleared);assert_eq!(ds>=100,run.complete);assert_eq!(game.current_piece_type().is_none(),run.topout);
    let end=endpoint(&game,ds,run.actions.len(),config);
    assert!((short.unwrap_or(end)-run.short_cost).abs()<1e-8);
    let cost=if horizon==4 {end}else if ds>=100 {run.actions.len() as f64}else{1000.0+run.actions.len() as f64};
    assert!((run.cost-cost).abs()<1e-8);
    assert!(horizon==4||run.complete||run.topout||run.truncated||run.actions.len()==MAX_LOCKS);
}
fn audit(path:&Path) {
    let s:Saved=serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
    let (obs,ds,garbage)=observe(Path::new(&s.source));assert_eq!(obs,s.observation);assert_eq!(ds,s.cleared);
    assert_eq!(garbage.min_height,s.garbage.min);assert_eq!(garbage.max_height,s.garbage.max);
    assert_eq!(world_rollout::public_hash(&obs),s.public_hash);assert!(s.base_nodes<=STRONG);
    assert!(s.roots.len()>=1&&s.roots.len()<=2);
    let rules:Rc<Ruleset>=Ruleset::guideline().into();let cfg=Config::default();
    for root in &s.roots {
        assert_eq!(root.samples.len(),4);
        for (j,sample) in root.samples.iter().enumerate() {
            assert_eq!(sample.seed,s.public_hash^((j as u64+1).wrapping_mul(0x9e3779b97f4a7c15)));
            let game=Stacker::from_public(rules.clone(),garbage.clone(),&obs,sample.seed);
            audit_run(game.clone(),ds,&root.inputs,&sample.strong,STRONG,MAX_LOCKS,&cfg);
            audit_run(game,ds,&root.inputs,&sample.weak,WEAK,4,&cfg);
        }
    }
    println!("Verified {} public full/short paired paths",s.roots.len()*4);
}
fn main() {
    let args:Vec<_>=std::env::args().collect();
    if args[1]=="--audit" {audit(Path::new(&args[2]));return;}
    let source=&args[1];let (observation,cleared,garbage)=observe(Path::new(source));
    let rules:Rc<Ruleset>=Ruleset::guideline().into();let hash=world_rollout::public_hash(&observation);
    let game=Stacker::from_public(rules.clone(),garbage.clone(),&observation,hash);
    let mut cfg=Config::default();cfg.search_limit=STRONG;let mut ai=AI::new(cfg);
    let mut a=ai.analyze(game.snapshot().unwrap());a.wait();let base_nodes=a.stats().unwrap().generated_nodes;
    assert!(base_nodes<=STRONG);let mut moves:Vec<_>=a.all_moves().collect();moves.sort_by(|x,y|a.cmp(*x,*y));
    assert!(!moves.is_empty());
    let mut roots=vec![];
    for m in moves.into_iter().take(2) {
        let s=a.suggestion(m,1);let mut inputs=s.inputs;assert_eq!(inputs.pop(),Some(Input::HD));
        let mut samples=vec![];
        for j in 0..4 {
            let seed=hash^((j as u64+1).wrapping_mul(0x9e3779b97f4a7c15));
            let world=Stacker::from_public(rules.clone(),garbage.clone(),&observation,seed);
            let strong=rollout(&mut ai,world.clone(),cleared,&inputs,STRONG,MAX_LOCKS);
            let weak=rollout(&mut ai,world,cleared,&inputs,WEAK,4);
            samples.push(Sample{seed,strong,weak});
        }
        roots.push(Root{inputs,rating:s.rating,samples});
    }
    let saved=Saved{source:source.clone(),observation,cleared,garbage:SavedGarbage{min:garbage.min_height,max:garbage.max_height},public_hash:hash,base_nodes,roots};
    serde_json::to_writer(std::fs::File::create(&args[2]).unwrap(),&saved).unwrap();
    audit(Path::new(&args[2]));
}
