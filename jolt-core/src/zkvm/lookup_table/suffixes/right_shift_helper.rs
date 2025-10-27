use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Computes 2^(y.leading_ones()), where y is the right operand
/// e.g. if the right operand is 0b11100000000000000000000000000000
/// then this suffix would return 2^3
pub enum RightShiftHelperSuffix {}

impl SparseDenseSuffix for RightShiftHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        1 << y.leading_ones()
    }
}
