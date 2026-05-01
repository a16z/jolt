use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum RightOperandSuffix {}

impl SparseDenseSuffix for RightOperandSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        u64::from(y)
    }
}
