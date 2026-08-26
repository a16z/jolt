use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Suffix-window value of the low `XLEN` index bits with bits 2..0 cleared
/// (suffix windows start at index bit 0).
pub enum AlignAddrSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for AlignAddrSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        ((u128::from(b) % (1 << XLEN)) as u64) & !7
    }
}
