//! Global best-first search, with either cumulative sibling-relative heuristic
//! regret or absolute f. Terminal scoring remains the frozen robust backup.
use super::{b_star::{Evaluator,MoveId,Node,SearchTerminated,Step},robust_backup::Leaves,state::State};
use crate::{config::Parameters,place::PlaceFinder,shape::ShapeTable};
use std::{cmp::Ordering,collections::{BinaryHeap,HashMap}};
struct Entry {node:Node,regret:i64,serial:usize}
impl Entry {
    fn key(&self)->(i64,std::cmp::Reverse<usize>,i64,usize) {
        (if cfg!(feature="global-absolute"){self.node.f}else{self.regret},
            std::cmp::Reverse(self.node.trace.len()),self.node.f,self.serial)
    }
}
impl PartialEq for Entry {fn eq(&self,other:&Self)->bool{self.key()==other.key()}}
impl Eq for Entry {}
impl Ord for Entry {fn cmp(&self,other:&Self)->Ordering{other.key().cmp(&self.key())}}
impl PartialOrd for Entry {fn partial_cmp(&self,other:&Self)->Option<Ordering>{Some(self.cmp(other))}}
pub(super) struct Search<'s> {
    params:Parameters,evaluator:Evaluator,finder:PlaceFinder<'s>,
    current:Option<Entry>,children:Vec<Node>,frontier:BinaryHeap<Entry>,
    leaves:HashMap<MoveId,Leaves>,best:HashMap<MoveId,i64>,
    generated:usize,serial:usize,expanded:usize,completed:usize,last_regret:i64,
}
impl<'s> Search<'s> {
    pub fn new(shapes:&'s ShapeTable,params:Parameters)->Self {
        Self{params,evaluator:Evaluator::default(),finder:PlaceFinder::new(shapes),current:None,
            children:vec![],frontier:BinaryHeap::new(),leaves:HashMap::new(),best:HashMap::new(),
            generated:0,serial:0,expanded:0,completed:0,last_regret:0}
    }
    pub fn start(&mut self,state:State) {
        self.current=None;self.children.clear();self.frontier.clear();self.leaves.clear();self.best.clear();
        self.evaluator=Evaluator::default();self.generated=1;self.serial=0;self.expanded=0;self.completed=0;self.last_regret=0;
        let node=Node::root(&self.params,state,&mut self.evaluator);
        self.frontier.push(Entry{node,regret:0,serial:0});
    }
    pub fn generated_nodes(&self)->usize{self.generated}
    pub fn budget_used(&self)->usize{self.generated}
    pub fn node_count(&self)->usize{self.frontier.len()+self.children.len()+usize::from(self.current.is_some())}
    pub fn report_global(&self) {
        eprintln!("GLOBAL_SEARCH {}",serde_json::json!({"generated":self.generated,"expanded":self.expanded,
            "completed":self.completed,"roots":self.best.len(),"absolute":cfg!(feature="global-absolute"),
            "pending_children":self.children.len()}));
    }
    pub fn step(&mut self)->Result<Step,SearchTerminated> {
        if let Some(parent)=self.current.take() {
            if let Some(place)=self.finder.next() {
                self.children.push(parent.node.succ(&self.params,&place,&mut self.evaluator));
                self.generated+=1;self.current=Some(parent);
            } else {
                // Finish sibling enumeration before computing local differences.
                // Every child was charged when generated, including a partial batch.
                if let Some(minimum)=self.children.iter().map(|n|n.f).min() {
                    for node in self.children.drain(..) {
                        let regret=parent.regret.saturating_add(node.f.saturating_sub(minimum));
                        self.serial+=1;
                        self.frontier.push(Entry{node,regret,serial:self.serial});
                    }
                }
            }
            return Ok(Step::Other);
        }
        let entry=self.frontier.pop().ok_or(SearchTerminated)?;
        if !cfg!(feature="global-absolute") {
            assert!(entry.regret>=self.last_regret);self.last_regret=entry.regret;
        }
        if entry.node.is_terminal() {
            let node=entry.node;self.completed+=1;
            let trace:Vec<_>=node.trace().collect();let id=MoveId(*node.trace.first().ok_or(SearchTerminated)?);
            let (rating,trace)=self.leaves.entry(id).or_default().record(node.state.clone(),node.rating(),trace,
                node.state.reached_goal()&&node.state.goal_is_final());
            let changed=self.best.get(&id).copied()!=Some(rating);self.best.insert(id,rating);
            return Ok(if changed {Step::RatingChanged{move_id:id,rating,trace}}
                else{Step::SequenceRejected{rating,trace}});
        }
        self.expanded+=1;entry.node.state.placements(&mut self.finder);self.current=Some(entry);
        Ok(Step::Other)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ai::{Snapshot,b_star},shape::srs,BasicMatrix,Color};
    fn replay(state:&State,trace:Vec<usize>,shapes:&ShapeTable) {
        let mut state=state.clone();let mut finder=PlaceFinder::new(shapes);
        for index in trace {let p=state.placements(&mut finder).find(|p|p.idx==index).unwrap();state.place(&p);}
        assert!(state.is_terminal());
    }
    #[test]
    fn complete_search_matches_all_bstar_root_scores() {
        let shapes=srs();
        for q in ["IIOO","TSTZ"] {
            let state:State=Snapshot{matrix:BasicMatrix::with_cols(4),hold:None,queue:q.chars().map(Color::n).collect()}.into();
            let mut baseline=b_star::Search::new(&shapes,Parameters::default());baseline.start(state.clone());
            let mut expected=HashMap::new();let mut ended=false;
            for _ in 0..5000000 {match baseline.step(){
                Ok(Step::RatingChanged{move_id,rating,..})=>{expected.insert(move_id,rating);},
                Err(_)=>{ended=true;break;},_=>()}}
            assert!(ended&&!expected.is_empty());
            let mut search=Search::new(&shapes,Parameters::default());search.start(state.clone());ended=false;
            for _ in 0..5000000 {
                let before=search.generated_nodes();
                match search.step(){Ok(Step::RatingChanged{trace,..})|Ok(Step::SequenceRejected{trace,..})=>replay(&state,trace,&shapes),
                    Err(_)=>{ended=true;break;},_=>()}
                assert!(search.generated_nodes()<=before+1);
            }
            assert!(ended);assert_eq!(search.best,expected);
        }
    }
    #[test]
    fn fixed_budget_has_legal_suggestions_and_resets() {
        let shapes=srs();
        let mut board=BasicMatrix::with_cols(10);
        for r in 0..9 {for c in 0..10 {if c!=r%10 {board.set((r,c));}}}
        let state:State=Snapshot{matrix:board,hold:None,queue:"IJLOST".chars().map(Color::n).collect()}.into();
        let mut search=Search::new(&shapes,Parameters::default());
        for budget in [1,100,5000,50000] {
            search.start(state.clone());assert_eq!(search.generated_nodes(),1);assert!(search.best.is_empty());
            while search.budget_used()<budget {
                let before=search.generated_nodes();
                match search.step(){Ok(Step::RatingChanged{trace,..})=>replay(&state,trace,&shapes),Err(_)=>break,_=>()}
                assert!(search.generated_nodes()<=before+1);
            }
            assert!(search.generated_nodes()<=budget);
            if budget==50000 {assert!(!search.best.is_empty());}
        }
    }
}
