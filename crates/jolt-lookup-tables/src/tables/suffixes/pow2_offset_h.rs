use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// `2^(8·(ea mod 8 & !1))` where `ea` is the (non-interleaved) lookup index
/// (bit 0 ignored).
pub enum Pow2OffsetHSuffix {}

impl SparseDenseSuffix for Pow2OffsetHSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() < 3 {
            1
        } else {
            let bits: u128 = b.into();
            1 << (8 * (bits & 6) as u32)
        }
    }
}
