//! Rank fusion across two search orderings. Match locked cells before line clears.
use super::{analysis::reconstruct_inputs, state::State, Snapshot, AI};
use crate::{place::PlaceFinder, shape::ShapeTable, BasicMatrix, Color, Input};
use std::{collections::HashMap, convert::TryInto};

fn reflect_color(color: Color) -> Color {
    match color.as_char() {
        'J' => 'L', 'L' => 'J', 'S' => 'Z', 'Z' => 'S', c => c,
    }.try_into().unwrap()
}
fn reflected(s: &Snapshot) -> Snapshot {
    let cols = s.matrix.cols();
    let mut matrix = BasicMatrix::with_cols(cols);
    for row in 0..s.matrix.rows() {
        for col in 0..cols {
            if s.matrix.get((row,col)) { matrix.set((row,cols-1-col)); }
        }
    }
    Snapshot {matrix, hold:s.hold.map(reflect_color), queue:s.queue.iter().copied().map(reflect_color).collect()}
}
#[derive(PartialEq,Eq,Hash)]
struct PlacementKey {color:Color, hold:bool, cells:Vec<(u16,u16)>}

fn root_keys(snapshot:&Snapshot, shapes:&ShapeTable, flipped:bool)->Vec<PlacementKey> {
    let state=State::from(snapshot.clone());
    let mut finder=PlaceFinder::new(shapes);
    let cols=snapshot.matrix.cols();
    state.placements(&mut finder).enumerate().map(|(index,p)| {
        assert_eq!(index,p.idx);
        let mut occupied=BasicMatrix::with_cols(cols);
        p.shape.blit_to(&mut occupied,p.tf);
        let mut cells=vec![];
        for r in 0..occupied.rows() {
            for c in 0..cols {
                if occupied.get((r,c)) { cells.push((r,if flipped {cols-1-c} else {c})); }
            }
        }
        cells.sort_unstable();
        assert_eq!(cells.len(),4);
        PlacementKey {color:if flipped {reflect_color(p.shape.color())} else {p.shape.color()},hold:p.did_hold,cells}
    }).collect()
}
#[derive(serde::Serialize)]
pub struct FusionStats {
    pub generated:usize,pub mapping:usize,pub native_nodes:usize,pub mirrored_nodes:usize,
    pub sequences:usize,pub native_roots:usize,pub matched:usize,pub changed:bool,pub fallback:bool,
}
impl AI {
    pub fn mirror_decision(&mut self,snapshot:Snapshot)->(Vec<Input>,FusionStats) {
        let other=reflected(&snapshot);
        let native=root_keys(&snapshot,&self.shape_table,false);
        let mirrored=root_keys(&other,&self.shape_table,true);
        assert!(!native.is_empty());
        let mapping=2+native.len()+mirrored.len();
        let full=self.config.search_limit;
        assert!(full>mapping+2);
        let half=(full-mapping)/2;
        self.config.search_limit=half;
        let mut a=self.analyze(snapshot.clone());a.wait();
        self.config.search_limit=full-mapping-half;
        let mut b=self.analyze(other);b.wait();
        self.config.search_limit=full;
        let sa=a.stats().unwrap();let sb=b.stats().unwrap();
        let mut left:Vec<_>=a.all_moves().collect();
        left.sort_by(|x,y|a.cmp(*x,*y));
        let mut right:Vec<_>=b.all_moves().collect();
        right.sort_by(|x,y|b.cmp(*x,*y));
        let ranks:HashMap<_,_>=right.iter().enumerate().map(|(i,id)|(&mirrored[id.0 as usize],i+1)).collect();
        let mut matched=0;let mut choice=0;let mut best=0.0;
        for (i,id) in left.iter().enumerate() {
            let vote=ranks.get(&native[id.0 as usize]).map_or(0.0,|rank|{matched+=1;1.0/(*rank as f64)});
            let score=1.0/((i+1) as f64)+vote;
            if score>best {best=score;choice=i;}
        }
        let mut inputs=if left.is_empty(){reconstruct_inputs(&self.shape_table,State::from(snapshot),&[0])}
            else {a.suggestion(left[choice],1).inputs};
        assert_eq!(inputs.pop(),Some(Input::HD));
        let generated=mapping+sa.generated_nodes+sb.generated_nodes;
        assert!(generated<=full);
        (inputs,FusionStats{generated,mapping,native_nodes:sa.generated_nodes,mirrored_nodes:sb.generated_nodes,
            sequences:sa.iterations+sb.iterations,native_roots:left.len(),matched,changed:choice!=0,fallback:left.is_empty()})
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reflection_is_involutive_and_matches_real_locks() {
        let mut board=BasicMatrix::with_cols(10);
        for c in 1..10 {board.set((0,c));}
        let s=Snapshot{matrix:board,hold:Some(Color::n('J')),queue:"SLIOTZ".chars().map(Color::n).collect()};
        assert_eq!(reflected(&reflected(&s)),s);
        let table=crate::shape::srs();
        let a=root_keys(&s,&table,false);let b=root_keys(&reflected(&s),&table,true);
        assert!(a.iter().any(|k|k.hold));assert!(a.iter().any(|k|!k.hold));
        assert!(a.iter().all(|k|b.contains(k)));
    }
}
