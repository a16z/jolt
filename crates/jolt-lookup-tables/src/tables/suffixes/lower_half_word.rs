use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum LowerHalfWordSuffix {}

impl SparseDenseSuffix for LowerHalfWordSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let half_word_size = XLEN / 2;
        if half_word_size == 64 {
            u128::from(b) as u64
        } else {
            (u128::from(b) % (1 << half_word_size)) as u64
        }
    }
}
