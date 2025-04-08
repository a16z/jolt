use crate::{
    field::JoltField,
    subprotocols::sparse_dense_shout::{current_suffix_len, LookupBits},
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum OrPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for OrPrefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let mut result = checkpoints[Prefixes::Or].unwrap_or(F::zero());

        // OR high-order variables of x and y
        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(1 << shift) * (r_x + y - (r_x * y));
        } else {
            let y_msb = b.pop_msb() as u32;
            let shift = WORD_SIZE - 1 - j / 2;
            result += F::from_u32(c + y_msb - c * y_msb) * F::from_u32(1 << shift);
        }
        // OR remaining x and y bits
        let (x, y) = b.uninterleave();
        let suffix_len = current_suffix_len(2 * WORD_SIZE, j);
        result += F::from_u32((u32::from(x) | u32::from(y)) << (suffix_len / 2));

        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let shift = WORD_SIZE - 1 - j / 2;
        // checkpoint += 2^shift * (r_x + r_y - r_x * r_y)
        let updated = checkpoints[Prefixes::Or].unwrap_or(F::zero())
            + F::from_u32(1 << shift) * (r_x + r_y - r_x * r_y);
        Some(updated).into()
    }
}
