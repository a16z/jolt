use crate::subprotocols::sparse_dense_shout::LookupBits;

use super::SparseDenseSuffix;

pub enum GreaterThanSuffix {}

impl SparseDenseSuffix for GreaterThanSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        (u32::from(x) > u32::from(y)).into()
    }
}
