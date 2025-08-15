use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Returns the lower WORD_SIZE bits. Used to range-check values to be in
/// the range [0, 2^WORD_SIZE).
pub enum LowerWordSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for LowerWordSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        (u64::from(b) % (1 << WORD_SIZE)) as u32
    }
}
