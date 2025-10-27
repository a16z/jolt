use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

pub enum ChangeDivisorSuffix {}

impl SparseDenseSuffix for ChangeDivisorSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (((1u64 << y.len()) - 1 == u64::from(y)) && u64::from(x) == 0).into()
    }
}
