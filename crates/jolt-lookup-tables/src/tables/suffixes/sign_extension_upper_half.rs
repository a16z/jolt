use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

pub enum SignExtensionUpperHalfSuffix {}

impl SparseDenseSuffix for SignExtensionUpperHalfSuffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        let half_word_size = XLEN / 2;

        if b.len() >= half_word_size {
            let bits = u128::from(b);
            let sign_bit = (bits >> (half_word_size - 1)) & 1;
            if sign_bit == 1 {
                ((1u64 << half_word_size) - 1) << half_word_size
            } else {
                0
            }
        } else {
            1
        }
    }
}
