use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Bitwise XOR suffix
pub enum XorSuffix {}

impl SparseDenseSuffix for XorSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        u64::from(x) ^ u64::from(y)
    }
}
