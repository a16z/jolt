use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the upper XLEN bits are all zero, 0 otherwise.
pub enum UpperWordIsZeroSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for UpperWordIsZeroSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() <= XLEN {
            1
        } else {
            let upper_bits = u128::from(b) >> XLEN;
            (upper_bits == 0).into()
        }
    }
}
