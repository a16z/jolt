use crate::utils::lookup_bits::LookupBits;

use super::SparseDenseSuffix;

/// `P_s = 2^((XLEN/8)·(y mod 8 & (8−EIGHTHS)))` over the suffix bits of the
/// second operand (1 once the offset bits have been bound).
pub enum OffsetScaleSuffix<const XLEN: usize, const EIGHTHS: usize> {}

impl<const XLEN: usize, const EIGHTHS: usize> SparseDenseSuffix
    for OffsetScaleSuffix<XLEN, EIGHTHS>
{
    fn suffix_mle(b: LookupBits) -> u64 {
        // WARNING: assumes suffix windows are empty or at least 6 bits (the
        // offset bits of y sit at interleaved index bits 0/2/4); the
        // ShiftData and OffsetScale prefixes assert the matching phase-cut
        // invariant.
        if b.len() < 6 {
            return 1;
        }
        let (_, yb) = b.uninterleave();
        let offset = u64::from(yb) & (8 - EIGHTHS as u64);
        1 << ((XLEN / 8) as u64 * offset)
    }
}
