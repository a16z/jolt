use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum UpperWordSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for UpperWordSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        (u128::from(b) >> XLEN) as u64
    }
}
