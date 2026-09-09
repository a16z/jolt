use crate::utils::lookup_bits::LookupBits;
use crate::zkvm::lookup_table::suffixes::SparseDenseSuffix;

pub struct ThreeLsbSuffix;

impl SparseDenseSuffix for ThreeLsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        (b.len() == 0 || u128::from(b).trailing_zeros() >= 3).into()
    }
}
