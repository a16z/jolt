use crate::{
    field::JoltField,
    subprotocols::sparse_dense_shout::{current_suffix_len, LookupBits},
};

use super::{PrefixCheckpoint, SparseDensePrefix};

pub enum LsbPrefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for LsbPrefix<WORD_SIZE> {
    fn prefix_mle(_: &[PrefixCheckpoint<F>], _: Option<F>, c: u32, b: LookupBits, j: usize) -> F {
        if j == 2 * WORD_SIZE - 1 {
            // in the log(K)th round, `c` corresponds to the LSB
            debug_assert_eq!(b.len(), 0);
            F::from_u32(c)
        } else if current_suffix_len(2 * WORD_SIZE, j) == 0 {
            // in the (log(K)-1)th round, the LSB of `b` is the LSB
            F::from_u32(u32::from(b) & 1)
        } else {
            F::one()
        }
    }

    fn update_prefix_checkpoint(
        _: &[PrefixCheckpoint<F>],
        _: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 2 * WORD_SIZE - 1 {
            Some(r_y).into()
        } else {
            Some(F::one()).into()
        }
    }
}
