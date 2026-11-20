//! Exact seven-bag inference from the public stream, starting at a fresh bag.
//! No simulator, RNG, seed, board or hidden-queue access is available here.
use crate::Color;
use std::convert::TryFrom;
const TYPES: [char; 7] = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];

pub struct BagObserver {
    remaining: u8,
    valid: bool,
    revealed: usize,
    checks: usize,
}
impl BagObserver {
    /// `visible` must start at the first draw of a fresh standard seven-bag game.
    pub fn from_visible(visible: &[Color]) -> Self {
        let mut observer = Self {
            remaining: 127,
            valid: true,
            revealed: 0,
            checks: 0,
        };
        for &color in visible {
            observer.observe(color);
        }
        observer
    }
    /// Consume only a piece that has now appeared in the public preview stream.
    pub fn observe(&mut self, color: Color) {
        self.revealed += 1;
        if !self.valid {
            return;
        }
        if self.remaining == 0 {
            self.remaining = 127;
        }
        let index = match TYPES.iter().position(|&c| c == color.as_char()) {
            Some(i) => i,
            None => {
                self.valid = false;
                return;
            }
        };
        let bit = 1u8 << index;
        if self.remaining & bit == 0 {
            self.valid = false;
            return;
        }
        if self.remaining.count_ones() == 1 {
            self.checks += 1;
        }
        self.remaining &= !bit;
    }
    pub fn inferred(&self) -> Option<Color> {
        if self.valid && self.remaining.count_ones() == 1 {
            Some(Color::try_from(TYPES[self.remaining.trailing_zeros() as usize]).unwrap())
        } else {
            None
        }
    }
    pub fn valid(&self) -> bool {
        self.valid
    }
    pub fn remaining_mask(&self) -> u8 {
        self.remaining
    }
    pub fn revealed(&self) -> usize {
        self.revealed
    }
    pub fn checks(&self) -> usize {
        self.checks
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_5040_bags_predict_only_the_unique_last_type() {
        fn visit(prefix: &mut Vec<Color>, remaining: Vec<Color>, count: &mut usize) {
            if remaining.is_empty() {
                let mut observer = BagObserver::from_visible(&prefix[..6]);
                assert_eq!(observer.inferred(), Some(prefix[6]));
                observer.observe(prefix[6]);
                assert!(observer.valid());
                assert_eq!(observer.inferred(), None);
                assert_eq!(observer.checks(), 1);
                *count += 1;
                return;
            }
            for i in 0..remaining.len() {
                let mut next = remaining.clone();
                prefix.push(next.remove(i));
                visit(prefix, next, count);
                prefix.pop();
            }
        }
        let mut count = 0;
        visit(
            &mut vec![],
            TYPES.iter().map(|&c| Color::try_from(c).unwrap()).collect(),
            &mut count,
        );
        assert_eq!(count, 5040);
    }
    #[test]
    fn two_public_draws_can_cross_a_bag_boundary() {
        let colors: Vec<_> = "IJLOST".chars().map(Color::n).collect();
        let mut observer = BagObserver::from_visible(&colors);
        observer.observe(Color::n('Z'));
        observer.observe(Color::n('T'));
        assert_eq!(observer.inferred(), None);
        for c in "IJLOS".chars() {
            observer.observe(Color::n(c));
        }
        assert_eq!(observer.inferred(), Some(Color::n('Z')));
        assert_eq!(observer.revealed(), 13);
    }
    #[test]
    fn invalid_stream_fails_closed() {
        let observer = BagObserver::from_visible(&[Color::n('I'), Color::n('I')]);
        assert!(!observer.valid());
        assert_eq!(observer.inferred(), None);
    }
}
