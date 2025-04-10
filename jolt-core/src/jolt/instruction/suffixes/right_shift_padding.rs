use crate::{subprotocols::sparse_dense_shout::LookupBits, utils::math::Math};

use super::SparseDenseSuffix;

/// RightShiftPaddingPrefix and RightShiftPaddingSuffix are used to compute
///  a bitmask for the padding bits obtained from an arithmetic right shift.
/// `shift` := the lower log_2(WORD_SIZE) bits of the second operand.
/// The bitmask has 1s in the upper `shift` bits and 0s in the lower bits.
///
/// Together, `RightShiftPaddingPrefix and RightShiftPaddingSuffix` compute:
/// - 2^WORD_SIZE if shift == 0
/// - 2^shift otherwise
///
/// This gets subtracted from 2^WORD_SIZE to obtain the desired bitmask.
pub enum RightShiftPaddingSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for RightShiftPaddingSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        if b.len() == 0 {
            // Handled by prefix
            return 1;
        }
        let (_, shift) = b.split(WORD_SIZE.log_2());
        let shift = u32::from(shift);
        // Subtract 1 to avoid shift overflow; `RightShiftPaddingPrefix::prefix_mle`
        // will return 2 to compensate
        1 << (WORD_SIZE - 1 - shift as usize)
    }
}
