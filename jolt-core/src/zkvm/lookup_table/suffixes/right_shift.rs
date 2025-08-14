use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Right-aligns the masked bits of the left operand.
/// e.g. if the right operand (the bitmask) is 0b11100000
/// then this suffix would shift the left operand by 5.
pub enum RightShiftSuffix {}

impl SparseDenseSuffix for RightShiftSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        u32::from(x).unbounded_shr(y.trailing_zeros())
    }
}
