use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Right-aligns the masked bits of the left operand.
pub enum RightShiftSuffix {}

impl SparseDenseSuffix for RightShiftSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        u64::from(x).unbounded_shr(y.trailing_zeros())
    }
}
