use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// 2^shift where shift is the lower 5 bits (modulo 32).
pub enum Pow2WSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for Pow2WSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            1
        } else {
            let (_, shift) = b.split(5);
            1 << u64::from(shift)
        }
    }
}
