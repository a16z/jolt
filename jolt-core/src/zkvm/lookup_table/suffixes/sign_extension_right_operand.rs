use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Sign extension suffix that handles the upper half of the word
/// If suffix length >= WORD_SIZE/2, it computes sign extension based on bit 31
/// Otherwise returns 1
pub enum SignExtensionRightOperandSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for SignExtensionRightOperandSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() >= WORD_SIZE {
            // Extract bit at position (half_word_size - 1), which is the sign bit
            let bits = u128::from(b);
            let sign_bit_position = WORD_SIZE - 2;
            let sign_bit = (bits >> sign_bit_position) & 1;

            if sign_bit == 1 {
                // Return all 1s in the upper half
                ((1u128 << WORD_SIZE) - (1u128 << (WORD_SIZE / 2))) as u64
            } else {
                // Return all 0s in the upper half
                0
            }
        } else {
            // Suffix is too small, return 1 (prefix will handle)
            1
        }
    }
}
