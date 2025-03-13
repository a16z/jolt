use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum LsbSuffix {}

impl SparseDenseSuffix for LsbSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        (u64::from(b) & 1) as u32
    }
}
