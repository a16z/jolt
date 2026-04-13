use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum UpperWordSuffix {}

impl SparseDenseSuffix for UpperWordSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        (u128::from(b) >> XLEN) as u64
    }
}
