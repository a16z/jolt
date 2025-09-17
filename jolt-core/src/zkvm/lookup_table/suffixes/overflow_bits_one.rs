use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the upper XLEN bits are all one, 0 otherwise.
pub enum OverflowBitsOneSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for OverflowBitsOneSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() <= XLEN {
            1
        } else {
            let upper_bits = u128::from(b) >> XLEN;
            let mask = (1u128 << (b.len() - XLEN)) - 1;
            (upper_bits == mask).into()
        }
    }
}
