use common::constants::XLEN;

use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Right-aligns the masked bits of the left operand for W variant.
/// Processes lower 32 bits.
pub enum RightShiftWSuffix {}

impl SparseDenseSuffix for RightShiftWSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        ((u64::from(x) as u32).unbounded_shr(y.trailing_zeros().min(XLEN as u32 / 2))) as u64
    }
}
