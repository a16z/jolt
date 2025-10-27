use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// 1 if the operands are equal, 0 otherwise.
pub enum EqSuffix {}

impl SparseDenseSuffix for EqSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (x == y).into()
    }
}
