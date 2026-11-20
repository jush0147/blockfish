use crate::PieceType;
use rand::{Rng as _, RngCore, SeedableRng as _};

// Deterministic PRNG

pub type PRNG = rand_pcg::Pcg64Mcg;

pub fn prng(seed: Option<u64>) -> (u64, PRNG) {
    // obtain the seed from an entropy based RNG
    let seed = seed.unwrap_or_else(|| rand::thread_rng().next_u64());
    let prng = PRNG::seed_from_u64(seed);
    (seed, prng)
}

//////////////////////////////////////////////////////////////////////////////////////////
// Piece generator

#[derive(Clone)]
pub struct Pieces {
    prng: PRNG,
    bag: Vec<PieceType>,
    pos: usize,
}

impl Pieces {
    #[cfg(feature = "public-world")]
    pub fn conditioned(prng: PRNG, types: Vec<PieceType>, remaining: &[PieceType]) -> Self {
        let mut all=types;all.sort_unstable();all.dedup();
        assert!(!all.is_empty());
        let mut rest=remaining.to_vec();rest.sort_unstable();rest.dedup();
        assert_eq!(rest.len(),remaining.len());
        assert!(rest.iter().all(|c|all.contains(c)));
        if rest.is_empty() { return Self::with_random_bag(prng,all); }
        let mut bag:Vec<_>=all.iter().copied().filter(|c|!rest.contains(c)).collect();
        let pos=bag.len();bag.extend(rest);
        Self {prng,bag,pos}
    }
    pub fn with_random_bag(prng: PRNG, bag: impl IntoIterator<Item = PieceType>) -> Self {
        let mut bag = bag.into_iter().collect::<Vec<_>>();
        bag.sort();
        Self { prng, bag, pos: 0 }
    }

    pub fn next(&mut self) -> PieceType {
        let idx = self.prng.gen_range(self.pos, self.bag.len());
        let typ = self.bag[idx];
        self.bag.swap(self.pos, idx);
        self.pos = (self.pos + 1) % self.bag.len();
        typ
    }
}

//////////////////////////////////////////////////////////////////////////////////////////
// Cheese column generator

#[derive(Clone)]
pub struct CheeseCols {
    prng: PRNG,
    cols: u16,
    prev: Option<u16>,
}

impl CheeseCols {
    #[cfg(feature = "public-world")]
    pub fn conditioned(prng: PRNG, cols: u16, prev: Option<u16>) -> Self {
        assert!(prev.map_or(true,|c|c<cols));
        Self {prng,cols,prev}
    }
    pub fn new(prng: PRNG, cols: u16) -> Self {
        Self {
            prng,
            cols,
            prev: None,
        }
    }

    pub fn cols(&self) -> u16 {
        self.cols
    }

    pub fn next_col(&mut self) -> u16 {
        let col = match self.prev {
            None => self.prng.gen_range(0, self.cols),
            Some(prev) => {
                let off = self.prng.gen_range(0, self.cols - 1);
                (prev + 1 + off) % self.cols
            }
        };
        self.prev = Some(col);
        col
    }
}
