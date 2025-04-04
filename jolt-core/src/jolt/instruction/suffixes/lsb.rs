use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

/// Returns the least significant bit of the suffix.
pub enum LsbSuffix {}

impl SparseDenseSuffix for LsbSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        if b.len() == 0 {
            1
        } else {
            (u64::from(b) & 1) as u32
        }
    }
}
