use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum Rev8WSuffix {}

impl SparseDenseSuffix for Rev8WSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let val = u128::from(b) as u64;
        crate::tables::virtual_rev8w::rev8w(val)
    }
}
