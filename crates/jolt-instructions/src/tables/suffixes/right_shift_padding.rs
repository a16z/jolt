use super::SparseDenseSuffix;
use crate::lookup_bits::LookupBits;

/// Bitmask helper for arithmetic right shift padding.
///
/// Together with `RightShiftPaddingPrefix`, computes:
/// - 2^XLEN if shift == 0
/// - 2^shift otherwise
///
/// This gets subtracted from 2^XLEN to obtain the desired padding bitmask.
pub enum RightShiftPaddingSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for RightShiftPaddingSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.is_empty() {
            return 1;
        }
        let log_xlen = XLEN.trailing_zeros() as usize;
        let (_, shift) = b.split(log_xlen);
        let shift = u64::from(shift);
        // Subtract 1 from exponent to avoid overflow; prefix compensates with factor of 2
        1 << (XLEN - 1 - shift as usize)
    }
}
