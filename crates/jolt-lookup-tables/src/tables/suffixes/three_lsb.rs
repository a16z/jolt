use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum ThreeLsbSuffix {}

impl SparseDenseSuffix for ThreeLsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            0
        } else {
            u64::from(b) & 7
        }
    }
}
