use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

pub enum SignExtensionRightOperandSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignExtensionRightOperandSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() >= XLEN {
            let bits = u128::from(b);
            let sign_bit_position = XLEN - 2;
            let sign_bit = (bits >> sign_bit_position) & 1;
            if sign_bit == 1 {
                ((1u128 << XLEN) - (1u128 << (XLEN / 2))) as u64
            } else {
                0
            }
        } else {
            1
        }
    }
}
