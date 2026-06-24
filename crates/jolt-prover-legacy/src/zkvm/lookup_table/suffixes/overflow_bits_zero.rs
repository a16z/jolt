use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

pub enum OverflowBitsZeroSuffix<const XLEN: usize> {}

/// Returns 1 if the upper 128-XLEN bits are all zero (no overflow), 0 otherwise.
impl<const XLEN: usize> SparseDenseSuffix for OverflowBitsZeroSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let upper_bits = u128::from(b) >> XLEN;
        (upper_bits == 0).into()
    }
}
