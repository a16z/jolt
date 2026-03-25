use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum OneSuffix {}

impl SparseDenseSuffix for OneSuffix {
    fn suffix_mle(_: LookupBits) -> u64 {
        1
    }
}
