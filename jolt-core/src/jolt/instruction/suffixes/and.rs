use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

/// Bitwise AND suffix
pub enum AndSuffix {}

impl SparseDenseSuffix for AndSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        u32::from(x) & u32::from(y)
    }
}
