use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Bitwise OR suffix
pub enum OrSuffix {}

impl SparseDenseSuffix for OrSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        u64::from(x) | u64::from(y)
    }
}
