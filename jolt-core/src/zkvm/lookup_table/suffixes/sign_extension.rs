use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum SignExtensionSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for SignExtensionSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (_, y) = b.uninterleave();
        let padding_len = std::cmp::min(u32::from(y).trailing_zeros() as usize, y.len());
        // 0b11...100...0
        ((1u64 << WORD_SIZE) - (1u64 << (WORD_SIZE - padding_len))) as u32
    }
}
