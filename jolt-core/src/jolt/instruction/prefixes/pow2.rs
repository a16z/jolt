use crate::{
    field::JoltField,
    subprotocols::sparse_dense_shout::{current_suffix_len, LookupBits},
    utils::math::Math,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum Pow2Prefix<const WORD_SIZE: usize> {}

impl<const WORD_SIZE: usize, F: JoltField> SparseDensePrefix<F> for Pow2Prefix<WORD_SIZE> {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F {
        if current_suffix_len(2 * WORD_SIZE, j) != 0 {
            // Handled by suffix
            return F::one();
        }

        // Shift amount is the last WORD_SIZE bits of b
        if b.len() >= WORD_SIZE.log_2() {
            return F::from_u64(1 << (b % WORD_SIZE));
        }

        let mut result = F::from_u64(1 << (b % WORD_SIZE));
        let mut num_bits = b.len();
        let mut shift = 1 << (1 << num_bits);
        result *= F::from_u32(1 + (shift - 1) * c);

        // Shift amount is [c, b]
        if b.len() == WORD_SIZE.log_2() - 1 {
            return result;
        }

        // Shift amount is [(r, r_x), c, b]
        num_bits += 1;
        shift = 1 << (1 << num_bits);
        if let Some(r_x) = r_x {
            result *= F::one() + F::from_u32(shift - 1) * r_x;
        }

        result *= checkpoints[Prefixes::Pow2].unwrap_or(F::one());
        result
    }

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if current_suffix_len(2 * WORD_SIZE, j) != 0 {
            return Some(F::one()).into();
        }

        // r_y is the highest bit of the shift amount
        if j == 2 * WORD_SIZE - WORD_SIZE.log_2() {
            let shift = 1 << (WORD_SIZE / 2);
            return Some(F::one() + F::from_u64(shift - 1) * r_y).into();
        }

        // r_x and r_y are bits in the shift amount
        if 2 * WORD_SIZE - j < WORD_SIZE.log_2() {
            let mut checkpoint = checkpoints[Prefixes::Pow2].unwrap();
            let shift = 1 << (1 << (2 * WORD_SIZE - j));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_x;
            let shift = 1 << (1 << (2 * WORD_SIZE - j - 1));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_y;
            return Some(checkpoint).into();
        }

        Some(F::one()).into()
    }
}
