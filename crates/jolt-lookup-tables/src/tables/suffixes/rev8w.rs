use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::tables::virtual_rev8w::rev8w;

pub enum Rev8WSuffix {}

impl SparseDenseSuffix for Rev8WSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        rev8w(u128::from(b) as u64)
    }
}
