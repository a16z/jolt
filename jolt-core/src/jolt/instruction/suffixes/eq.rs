use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

#[derive(Default)]
pub struct EqSuffix;

impl SparseDenseSuffix for EqSuffix {
    fn suffix_mle(&self, b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        (x == y).into()
    }
}
