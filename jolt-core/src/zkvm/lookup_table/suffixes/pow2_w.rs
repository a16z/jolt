use super::SparseDenseSuffix;
use crate::utils::lookup_bits::LookupBits;

/// Computes 2^shift, where shift is the lower 5 bits of the operand (for modulo 32).
pub enum Pow2WSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for Pow2WSuffix<XLEN> {
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
