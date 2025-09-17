use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Returns the sign bit (MSB of the lower XLEN bits).
pub enum SignBitSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for SignBitSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() < XLEN {
            0  // Not enough bits to have a sign bit
        } else {
            let value = u128::from(b);
            ((value >> (XLEN - 1)) & 1) as u64
        }
    }
}
