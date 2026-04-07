use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum GreaterThanSuffix {}

impl SparseDenseSuffix for GreaterThanSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (u64::from(x) > u64::from(y)).into()
    }
}
