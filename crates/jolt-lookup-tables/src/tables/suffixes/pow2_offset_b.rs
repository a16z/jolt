use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// `2^(8·(ea mod 8))` where `ea` is the (non-interleaved) lookup index.
pub enum Pow2OffsetBSuffix {}

impl SparseDenseSuffix for Pow2OffsetBSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() < 3 {
            1
        } else {
            let bits: u128 = b.into();
            1 << (8 * (bits & 7) as u32)
        }
    }
}
