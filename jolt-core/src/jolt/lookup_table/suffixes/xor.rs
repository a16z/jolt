use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

/// Bitwise XOR suffix
pub enum XorSuffix {}

impl SparseDenseSuffix for XorSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        u32::from(x) ^ u32::from(y)
    }
}
