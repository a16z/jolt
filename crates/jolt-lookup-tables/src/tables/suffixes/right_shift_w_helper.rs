use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// 2^(y.leading_ones()) for W variant, truncated to XLEN/2 bits.
pub enum RightShiftWHelperSuffix {}

impl SparseDenseSuffix for RightShiftWHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        let y = LookupBits::new(u128::from(y), y.len().min(XLEN / 2));
        1 << y.leading_ones()
    }
}
