use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Sign extension suffix that handles the upper half of the word
/// If suffix length >= XLEN/2, it computes sign extension based on bit 31
/// Otherwise returns 1
pub enum SignExtensionUpperHalfSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignExtensionUpperHalfSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let half_word_size = XLEN / 2;

        if b.len() >= half_word_size {
            // Extract bit at position (half_word_size - 1), which is the sign bit
            let bits = u128::from(b);
            let sign_bit_position = half_word_size - 1;
            let sign_bit = (bits >> sign_bit_position) & 1;

            if sign_bit == 1 {
                // Return all 1s in the upper half
                ((1u64 << half_word_size) - 1) << half_word_size
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
