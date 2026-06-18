use crate::{utils::lookup_bits::LookupBits, utils::math::Math};

use super::SparseDenseSuffix;

/// Computes 2^shift, where shift is the lower log_2(XLEN) bits of
/// the second operand.
pub enum Pow2Suffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for Pow2Suffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            1
        } else {
            let (_, shift) = b.split(XLEN.log_2());
            1 << u64::from(shift)
        }
    }
}
