use crate::{subprotocols::sparse_dense_shout::LookupBits, utils::math::Math};

use super::SparseDenseSuffix;

/// Computes a bitmask for the padding bits obtained from an arithmetic right shift.
/// `shift` := the lower log_2(WORD_SIZE) bits of the second operand.
/// The bitmask has 1s in the upper `shift` bits and 0s in the lower bits.
pub enum RightShiftPaddingSuffix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize> SparseDenseSuffix for RightShiftPaddingSuffix<WORD_SIZE> {
    fn suffix_mle(b: LookupBits) -> u32 {
        let (_, shift) = b.split(WORD_SIZE.log_2());
        let shift = u32::from(shift);
        if shift == 0 {
            return 0;
        }
        let ones = (1 << shift) - 1;
        ones << (WORD_SIZE as u32 - shift)
    }
}
