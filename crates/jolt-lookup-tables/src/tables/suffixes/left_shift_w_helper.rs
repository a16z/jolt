use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// 2^(y.leading_ones()) truncated to 32 bits.
pub enum LeftShiftWHelperSuffix {}

impl SparseDenseSuffix for LeftShiftWHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        (1u32 << y.leading_ones()) as u64
    }
}
