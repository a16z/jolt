use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// 1 if the upper 128-XLEN bits are all zero (no overflow), 0 otherwise.
pub enum OverflowBitsZeroSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for OverflowBitsZeroSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let upper_bits = u128::from(b) >> XLEN;
        (upper_bits == 0).into()
    }
}
