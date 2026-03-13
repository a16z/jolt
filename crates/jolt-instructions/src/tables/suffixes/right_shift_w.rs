use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Right-shift W variant: processes lower 32 bits.
pub enum RightShiftWSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for RightShiftWSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        let (x, y) = b.uninterleave();
        (u64::from(x) as u32).unbounded_shr(y.trailing_zeros().min(XLEN as u32 / 2)) as u64
    }
}
