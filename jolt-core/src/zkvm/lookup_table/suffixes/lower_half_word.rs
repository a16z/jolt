use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Returns the lower XLEN/2 bits. Used to extract the lower half of a word.
pub enum LowerHalfWordSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for LowerHalfWordSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let half_word_size = XLEN / 2;
        if half_word_size == 64 {
            u128::from(b) as u64
        } else {
            (u128::from(b) % (1 << half_word_size)) as u64
        }
    }
}
