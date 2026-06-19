use common::constants::XLEN;

use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Computes 2^(y.leading_ones()), where y is the right operand
/// e.g. if the right operand is 0b11100000000000000000000000000000
/// then this suffix would return 2^3
pub enum RightShiftWHelperSuffix {}

impl SparseDenseSuffix for RightShiftWHelperSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        let y = LookupBits::new(u128::from(y), y.len().min(XLEN / 2));
        1 << y.leading_ones()
    }
}
