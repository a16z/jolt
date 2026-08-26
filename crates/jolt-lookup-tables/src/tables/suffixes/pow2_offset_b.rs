use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// `2^(8·(ea mod 8))` where `ea` is the (non-interleaved) lookup index.
pub enum Pow2OffsetBSuffix {}

impl SparseDenseSuffix for Pow2OffsetBSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        // The suffix owns index bits [0, b.len()); with fewer than 3 bits
        // this is the partial factor for the offset bits it owns, and the
        // Pow2OffsetB prefix supplies the rest.
        let bits: u128 = b.into();
        1 << (8 * (bits & 7) as u32)
    }
}
