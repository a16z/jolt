use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum RightOperandIsZeroSuffix {}

impl SparseDenseSuffix for RightOperandIsZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (_, y) = b.uninterleave();
        (u64::from(y) == 0).into()
    }
}
