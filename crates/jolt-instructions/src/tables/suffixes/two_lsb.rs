use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub struct TwoLsbSuffix;

impl SparseDenseSuffix for TwoLsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        // 1 if the two least significant bits are 0, else 0
        (b.is_empty() || u128::from(b).trailing_zeros() >= 2).into()
    }
}
