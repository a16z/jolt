use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum LeftOperandIsZeroSuffix {}

impl SparseDenseSuffix for LeftOperandIsZeroSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, _) = b.uninterleave();
        (u64::from(x) == 0).into()
    }
}
