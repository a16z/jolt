use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// 2^(y.leading_ones()) where y is the right operand.
pub enum RightShiftHelperSuffix {}

impl SparseDenseSuffix for RightShiftHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        1 << y.leading_ones()
    }
}
