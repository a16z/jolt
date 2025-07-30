use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// 1 if the operands are equal, 0 otherwise.
pub enum EqSuffix {}

impl SparseDenseSuffix for EqSuffix {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (x, y) = b.uninterleave();
        (x == y).into()
    }
}
