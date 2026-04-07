use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;
use crate::XLEN;

/// 2^shift where shift is the lower log2(XLEN) bits of the second operand.
pub enum Pow2Suffix {}

impl SparseDenseSuffix for Pow2Suffix {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            1
        } else {
            let log_xlen = XLEN.trailing_zeros() as usize;
            let (_, shift) = b.split(log_xlen);
            1 << u64::from(shift)
        }
    }
}
