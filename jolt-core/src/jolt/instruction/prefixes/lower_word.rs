use crate::{
    field::JoltField,
    subprotocols::sparse_dense_shout::{current_suffix_len, LookupBits},
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LowerWordPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for LowerWordPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        // Ignore high-order variables
        if j < WORD_SIZE {
            return F::zero();
        }
        let mut result = checkpoints[Prefixes::LowerWord].unwrap_or(F::zero());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let x_shift = 2 * WORD_SIZE - j;
            let y_shift = 2 * WORD_SIZE - j - 1;
            result += F::from_u64(1 << x_shift) * r_x;
            result += F::from_u64(1 << y_shift) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y_msb = b.pop_msb();
            let x_shift = 2 * WORD_SIZE - j - 1;
            let y_shift = 2 * WORD_SIZE - j - 2;
            result += F::from_u64(1 << x_shift) * x;
            result += F::from_u64(1 << y_shift) * F::from_u8(y_msb);
        }

        // Add in low-order bits from `b`
        let suffix_len = current_suffix_len(2 * WORD_SIZE, j);
        result += F::from_u64(u64::from(b) << suffix_len);

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j < WORD_SIZE {
            return None.into();
        }
        let x_shift = 2 * WORD_SIZE - j;
        let y_shift = 2 * WORD_SIZE - j - 1;
        let updated = checkpoints[Prefixes::LowerWord].unwrap_or(F::zero())
            + F::from_u64(1 << x_shift) * r_x
            + F::from_u64(1 << y_shift) * r_y;
        Some(updated).into()
    }
}
