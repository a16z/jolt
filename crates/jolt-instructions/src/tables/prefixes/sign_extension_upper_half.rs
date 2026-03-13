use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum SignExtensionUpperHalfPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: Field> SparseDensePrefix<F> for SignExtensionUpperHalfPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let suffix_len = 2 * XLEN - j - b.len() - 1;
        let half_word_size = XLEN / 2;

        if suffix_len >= half_word_size {
            return F::one();
        }

        if j == XLEN + half_word_size {
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)).mul_u64(c as u64)
        } else if j == XLEN + half_word_size + 1 {
            F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x.unwrap()
        } else if j > XLEN + half_word_size + 1 {
            checkpoints[Prefixes::SignExtensionUpperHalf].unwrap_or(F::zero())
        } else {
            unreachable!("This case should never happen if our prefixes start at half_word_size");
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        _r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let half_word_size = XLEN / 2;

        if j == XLEN + half_word_size + 1 {
            let value = F::from_u128(((1u128 << (half_word_size)) - 1) << (half_word_size)) * r_x;
            Some(value).into()
        } else {
            checkpoints[Prefixes::SignExtensionUpperHalf].into()
        }
    }
}
