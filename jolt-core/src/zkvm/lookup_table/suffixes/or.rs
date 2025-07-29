use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Bitwise OR suffix
pub enum OrSuffix {}

impl SparseDenseSuffix for OrSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        u32::from(x) | u32::from(y)
    }
}
