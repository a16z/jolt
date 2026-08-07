use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum ThreeLsbSuffix {}

impl SparseDenseSuffix for ThreeLsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            0
        } else {
            (b & 7) as u64
        }
    }
}
