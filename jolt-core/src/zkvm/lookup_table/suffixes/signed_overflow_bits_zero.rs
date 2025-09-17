use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

pub enum SignedOverflowBitsZeroSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignedOverflowBitsZeroSuffix<XLEN> {
    fn suffix_mle(mut b: LookupBits) -> u64 {
        let suffix_len = b.len();

        if XLEN > suffix_len {
            // Prefix processed all overflow bits and sign bit
            1
        } else if XLEN == suffix_len {
            // Prefix processed all overflow bits
            // The sign bit is b's MSB
            (b.pop_msb() == 0) as u64
        } else {
            // Extract the overflow bits and sign bit
            let overflow_and_sign_bits = u128::from(b) >> (XLEN - 1);
            (overflow_and_sign_bits == 0) as u64
        }
    }
}
