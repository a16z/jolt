use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `2^popcount(y)` where `y` is the right operand: the scale factor a
/// prefix's pext accumulator picks up across the suffix.
pub enum PextHelperSuffix {}

impl SparseDenseSuffix for PextHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        1 << u64::from(y).count_ones()
    }
}
