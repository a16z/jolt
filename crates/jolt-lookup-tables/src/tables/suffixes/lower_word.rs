use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum LowerWordSuffix {}

impl SparseDenseSuffix for LowerWordSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        (u128::from(b) % (1 << XLEN)) as u64
    }
}
