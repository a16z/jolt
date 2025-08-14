use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the second operand is 0, 0 otherwise.
pub enum RightOperandIsZeroSuffix {}

impl SparseDenseSuffix for RightOperandIsZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (_, y) = b.uninterleave();
        (u64::from(y) == 0).into()
    }
}
