use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum LeftOperandIsZeroSuffix {}

impl SparseDenseSuffix for LeftOperandIsZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, _) = b.uninterleave();
        (u64::from(x) == 0).into()
    }
}
