use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

pub enum SignedOverflowBitsOneSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignedOverflowBitsOneSuffix<XLEN> {
    fn suffix_mle(mut b: LookupBits) -> u64 {
        let suffix_len = b.len();

        if XLEN > suffix_len {
            // Prefix processed all overflow bits and sign bit
            1
        } else if XLEN == suffix_len {
            // Prefix processed all overflow bits
            // The sign bit is b's MSB
            (b.pop_msb() == 1) as u64
        } else {
            // The remaining overflow bits are at positions 0 to XLEN-j-1 in the suffix
            // The sign bit is at position XLEN-j in the suffix
            let overflow_sign_len = suffix_len - (XLEN - 1);
            let overflow_sign_mask = (1u128 << (overflow_sign_len)) - 1;
            let overflow_sign_bits = u128::from(b) >> (suffix_len - overflow_sign_len);
            (overflow_sign_bits == overflow_sign_mask) as u64
        }
    }
}
