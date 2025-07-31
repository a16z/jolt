use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Returns the lower WORD_SIZE/2 bits. Used to extract the lower half of a word.
pub enum LowerHalfWordSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for LowerHalfWordSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let half_word_size = WORD_SIZE / 2;
        if half_word_size == 64 {
            u128::from(b) as u64
        } else {
            (u128::from(b) % (1 << half_word_size)) as u64
        }
    }
}
