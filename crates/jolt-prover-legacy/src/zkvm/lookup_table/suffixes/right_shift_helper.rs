use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Computes `2^popcount(y)`, where `y` is the right operand.
pub enum RightShiftHelperSuffix {}

impl SparseDenseSuffix for RightShiftHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        1 << u64::from(y).count_ones()
    }
}
