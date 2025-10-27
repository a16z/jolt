use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// The constant 1.
pub enum OneSuffix {}

impl SparseDenseSuffix for OneSuffix {
    fn suffix_mle(_: LookupBits) -> u64 {
        1
    }
}
