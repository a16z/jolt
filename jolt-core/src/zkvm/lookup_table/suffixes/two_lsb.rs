use crate::{utils::lookup_bits::LookupBits, zkvm::lookup_table::suffixes::SparseDenseSuffix};

pub struct TwoLsbSuffix;

impl SparseDenseSuffix for TwoLsbSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        // Returns 1 if the two least significant bits are 0
        // and 0 otherwise
        // 1 by default
        (b.len() == 0 || u128::from(b) & 0b11 == 0).into()
    }
}
