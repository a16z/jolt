use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// Computes 2^shift, where shift is the lower 5 bits of the operand (for modulo 32).
pub enum Pow2WSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for Pow2WSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            1
        } else {
            // Always extract 5 bits for modulo 32
            let (_, shift) = b.split(5);
            1 << u64::from(shift)
        }
    }
}
