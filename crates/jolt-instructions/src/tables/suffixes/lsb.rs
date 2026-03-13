use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum LsbSuffix {}

impl SparseDenseSuffix for LsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            1
        } else {
            (u128::from(b) & 1) as u64
        }
    }
}
