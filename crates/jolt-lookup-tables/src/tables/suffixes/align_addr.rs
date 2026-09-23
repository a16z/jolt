use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// Suffix-window value of the low `XLEN` index bits with bits 2..0 cleared
/// (suffix windows start at index bit 0).
pub enum AlignAddrSuffix {}

impl SparseDenseSuffix for AlignAddrSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        ((u128::from(b) % (1 << XLEN)) as u64) & !7
    }
}
