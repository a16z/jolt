use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum SignExtensionSuffix {}

impl SparseDenseSuffix for SignExtensionSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (_, y) = b.uninterleave();
        let padding_len = std::cmp::min(u64::from(y).trailing_zeros() as usize, y.len());
        // 0b11...100...0
        ((1u128 << XLEN) - (1u128 << (XLEN - padding_len))) as u64
    }
}
