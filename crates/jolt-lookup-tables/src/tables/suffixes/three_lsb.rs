use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub struct ThreeLsbSuffix;

impl SparseDenseSuffix for ThreeLsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        (b.is_empty() || u128::from(b).trailing_zeros() >= 3).into()
    }
}
