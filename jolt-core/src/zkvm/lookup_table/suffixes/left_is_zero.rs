use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the first operand is 0, 0 otherwise.
pub enum LeftOperandIsZeroSuffix {}

impl SparseDenseSuffix for LeftOperandIsZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, _) = b.uninterleave();
        (u64::from(x) == 0).into()
    }
}
