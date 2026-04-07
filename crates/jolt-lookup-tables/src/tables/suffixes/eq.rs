use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum EqSuffix {}

impl SparseDenseSuffix for EqSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (x == y).into()
    }
}
