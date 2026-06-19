use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Returns the least significant bit of the suffix.
pub enum LsbSuffix {}

impl SparseDenseSuffix for LsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            1
        } else {
            (u128::from(b) & 1) as u64
        }
    }
}
