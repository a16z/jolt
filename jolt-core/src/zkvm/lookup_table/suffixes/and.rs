use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Bitwise AND suffix
pub enum AndSuffix {}

impl SparseDenseSuffix for AndSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        u64::from(x) & u64::from(y)
    }
}
