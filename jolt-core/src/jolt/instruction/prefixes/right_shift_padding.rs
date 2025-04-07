use crate::{
    field::JoltField,
    subprotocols::sparse_dense_shout::{current_suffix_len, LookupBits},
    utils::math::Math,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

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
pub enum RightShiftPaddingPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F>
    for RightShiftPaddingPrefix<WORD_SIZE>
{
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        if current_suffix_len(2 * WORD_SIZE, j) != 0 {
            // Suffix is off by a factor of 2 to avoid shift overflow
            return F::from_u8(2);
        }

        // Shift amount is the last WORD_SIZE bits of b
        if b.len() >= WORD_SIZE.log_2() {
            let shift = b % WORD_SIZE;
            return F::from_u64(1 << (WORD_SIZE - shift));
        }

        let mut result = F::from_u64(1 << (WORD_SIZE - usize::from(b)));
        let mut num_bits = b.len();
        let pow2 = 1 << (1 << num_bits);
        result *= F::one() - (F::one() - F::from_u64(pow2).inverse().unwrap()) * F::from_u32(c);

        // Shift amount is [c, b]
        if b.len() == WORD_SIZE.log_2() - 1 {
            return result;
        }

        // Shift amount is [(r, r_x), c, b]
        num_bits += 1;
        let pow2 = 1 << (1 << num_bits);
        if let Some(r_x) = r_x {
            result *= F::one() - (F::one() - F::from_u64(pow2).inverse().unwrap()) * r_x;
        }

        result *= checkpoints[Prefixes::RightShiftPadding].unwrap();
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if current_suffix_len(2 * WORD_SIZE, j) != 0 {
            // Handled by suffix
            return Some(F::from_u8(2)).into();
        }

        // r_y is the highest bit of the shift amount
        if j == 2 * WORD_SIZE - WORD_SIZE.log_2() {
            let pow2 = 1 << (WORD_SIZE / 2);
            return Some(F::one() - (F::one() - F::from_u64(pow2).inverse().unwrap()) * r_y).into();
        }

        // r_x and r_y are bits in the shift amount
        if 2 * WORD_SIZE - j < WORD_SIZE.log_2() {
            let mut checkpoint = checkpoints[Prefixes::RightShiftPadding].unwrap_or(F::one());
            let mut bit_index = 2 * WORD_SIZE - j;
            let pow2 = 1 << (1 << bit_index);
            checkpoint *= F::one() - (F::one() - F::from_u64(pow2).inverse().unwrap()) * r_x;
            bit_index -= 1;
            let pow2 = 1 << (1 << bit_index);
            checkpoint *= F::one() - (F::one() - F::from_u64(pow2).inverse().unwrap()) * r_y;
            if j == 2 * WORD_SIZE - 1 {
                checkpoint *= F::from_u64(1 << WORD_SIZE);
            }
            return Some(checkpoint).into();
        }

        None.into()
    }
}
