use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

#[derive(Default)]
pub struct OneSuffix;

impl SparseDenseSuffix for OneSuffix {
    fn suffix_mle(&self, _: LookupBits) -> u32 {
        1
    }
}
