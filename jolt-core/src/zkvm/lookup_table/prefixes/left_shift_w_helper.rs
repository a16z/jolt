use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

/// Computes 2^(y.leading_ones()) for j >= WORD_SIZE
pub enum LeftShiftWHelperPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F>
    for LeftShiftWHelperPrefix<WORD_SIZE>
{
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // Only process when j >= WORD_SIZE
        if j < WORD_SIZE {
            return F::one();
        }

        let mut result = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());

        if r_x.is_some() {
            result *= F::from_u32(1 + c);
        } else {
            let y_msb = b.pop_msb();
            result *= F::from_u8(1 + y_msb);
        }

        let (_, y) = b.uninterleave();
        result *= F::from_u32(1 << y.leading_ones());

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j >= WORD_SIZE {
            let mut updated = checkpoints[Prefixes::LeftShiftWHelper].unwrap_or(F::one());
            updated *= F::one() + r_y;
            Some(updated).into()
        } else {
            Some(F::one()).into()
        }
    }
}
