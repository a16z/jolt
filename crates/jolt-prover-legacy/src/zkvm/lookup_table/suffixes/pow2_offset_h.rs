use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `2^(8·(ea mod 8 & !1))` where `ea` is the (non-interleaved) lookup index
/// (bit 0 ignored).
pub enum Pow2OffsetHSuffix {}

impl SparseDenseSuffix for Pow2OffsetHSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        // The suffix owns index bits [0, b.len()); with fewer than 3 bits
        // this is the partial factor for the offset bits it owns (bit 0
        // never contributes),
        // and the Pow2Offset prefix supplies the rest.
        let bits: u128 = b.into();
        1 << (8 * (bits & 6) as u32)
    }
}
