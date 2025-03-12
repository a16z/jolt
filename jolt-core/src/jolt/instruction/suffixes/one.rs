use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum OneSuffix {}

impl SparseDenseSuffix for OneSuffix {
    fn suffix_mle(_: LookupBits) -> u32 {
        1
    }
}
