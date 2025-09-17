use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// For the first term: (1-a_127)(1-a_126)...(1-a_64)(1-a_63)
/// This suffix checks that the remaining overflow bits and sign bit are all 0.
pub enum SignedOverflowBitsZeroSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignedOverflowBitsZeroSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        // The suffix contains bits from position j to 2*XLEN-1
        // For a 2*XLEN bit value:
        // - Bits 0 to XLEN-1 are overflow bits
        // - Bit XLEN is the sign bit
        // - Bits XLEN+1 to 2*XLEN-1 are the remaining lower bits
        
        // We need to determine what the suffix contains based on its length
        // and what the prefix has already processed
        
        let suffix_len = b.len();
        let value = u128::from(b);
        
        // The cutoff point j = 2*XLEN - suffix_len
        // This tells us where the prefix stopped and suffix begins
        let j = 2 * XLEN - suffix_len;
        
        
        if j > XLEN {
            // Prefix processed all overflow bits and sign bit
            // Suffix only contains lower bits (not including sign bit)
            // For this term, we don't care about these bits, return 1
            1
        } else if j == XLEN {
            // Prefix processed all overflow bits
            // Suffix contains the lower XLEN bits (bits 64-127 of the original 128-bit value)
            // The sign bit is the MSB of the lower XLEN bits (bit 63 of these 64 bits)
            let sign_bit = (value >> (XLEN - 1)) & 1;  // MSB of the lower XLEN bits
            let result = (sign_bit == 0) as u64;
            // eprintln!("  -> j={} == XLEN, sign_bit={}, result={}", j, sign_bit, result);
            result
        } else {
            // Prefix processed j overflow bits (where j < XLEN)
            // Suffix contains (XLEN - j) remaining overflow bits + sign bit + lower bits
            
            // The remaining overflow bits are at positions 0 to XLEN-j-1 in the suffix
            // The sign bit is at position XLEN-j in the suffix
            let remaining_overflow_bits = XLEN - j;
            
            // Extract the overflow bits and sign bit from the suffix
            let overflow_and_sign_bits = value >> (suffix_len - remaining_overflow_bits - 1);
            
            // All these bits should be 0 for the first term
            let result = (overflow_and_sign_bits == 0) as u64;
            // eprintln!("  -> j={}, overflow_and_sign_bits={}, result={}", j, overflow_and_sign_bits, result);
            result
        }
    }
}
