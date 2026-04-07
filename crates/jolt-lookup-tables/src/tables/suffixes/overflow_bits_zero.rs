use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// 1 if the upper 128-XLEN bits are all zero (no overflow), 0 otherwise.
pub enum OverflowBitsZeroSuffix {}

impl SparseDenseSuffix for OverflowBitsZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let upper_bits = u128::from(b) >> XLEN;
        (upper_bits == 0).into()
    }
}
