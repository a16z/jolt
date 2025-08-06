use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Bitwise NOT-AND suffix (for ANDN operation)
/// Computes sum(2^(n-i) * (1-y_i)) = x & !y where y are the odd-indexed bits
pub enum NotAndSuffix {}

impl SparseDenseSuffix for NotAndSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        u64::from(x) & !u64::from(y)
    }
}