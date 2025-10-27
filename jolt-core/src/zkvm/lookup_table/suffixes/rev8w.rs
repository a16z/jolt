use tracer::instruction::virtual_rev8w::rev8w;

use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

pub enum Rev8W {}

impl SparseDenseSuffix for Rev8W {
    fn suffix_mle(b: LookupBits) -> u64 {
        rev8w(u128::from(b) as u64)
    }
}
