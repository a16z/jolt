use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum RightOperandWSuffix {}

impl SparseDenseSuffix for RightOperandWSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        u64::from(y) as u32 as u64
    }
}
