use super::SparseDenseSuffix;
use crate::utils::{lookup_bits::LookupBits, math::Math};

/// RightShiftPaddingPrefix and RightShiftPaddingSuffix are used to compute
///  a bitmask for the padding bits obtained from an arithmetic right shift.
/// `shift` := the lower log_2(XLEN) bits of the second operand.
/// The bitmask has 1s in the upper `shift` bits and 0s in the lower bits.
///
/// Together, `RightShiftPaddingPrefix and RightShiftPaddingSuffix` compute:
/// - 2^XLEN if shift == 0
/// - 2^shift otherwise
///
/// This gets subtracted from 2^XLEN to obtain the desired bitmask.
pub enum RightShiftPaddingSuffix<const XLEN: usize> {}

impl<const XLEN: usize> SparseDenseSuffix for RightShiftPaddingSuffix<XLEN> {
    fn suffix_mle(b: LookupBits) -> u64 {
        if b.len() == 0 {
            // Handled by prefix
            return 1;
        }
        let (_, shift) = b.split(XLEN.log_2());
        let shift = u64::from(shift);
        // Subtract 1 to avoid shift overflow; `RightShiftPaddingPrefix::prefix_mle`
        // will return 2 to compensate
        1 << (XLEN - 1 - shift as usize)
    }
}
