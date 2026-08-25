use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// `2^(32·ea_2)` where `ea_2` is bit 2 of the (non-interleaved) lookup index.
pub enum Pow2OffsetWSuffix {}

impl SparseDenseSuffix for Pow2OffsetWSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() < 3 {
            1
        } else {
            let bits: u128 = b.into();
            1 << (32 * ((bits >> 2) & 1) as u32)
        }
    }
}
