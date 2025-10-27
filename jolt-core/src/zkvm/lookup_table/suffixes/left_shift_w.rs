use common::constants::XLEN;

use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Left-shifts the left operand for W variant.
/// Processes lower 32 bits.
pub enum LeftShiftWSuffix {}

impl SparseDenseSuffix for LeftShiftWSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        let y = LookupBits::new(u128::from(y), y.len().min(XLEN / 2));
        let (x, y_u) = (u64::from(x) as u32, u32::from(y));
        let x = x & !y_u;
        (x.unbounded_shl(y.leading_ones())) as u64
    }
}
