use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum Pow2Prefix<const XLEN: usize> {}

impl<const XLEN: usize, F: Field> SparseDensePrefix<F> for Pow2Prefix<XLEN> {
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
        if suffix_len != 0 {
            return F::one();
        }

        if b.len() >= XLEN.trailing_zeros() as usize {
            return F::from_u64(1 << (b & (XLEN - 1)));
        }

        let mut result = F::from_u64(1 << (b & (XLEN - 1)));
        let mut num_bits = b.len();
        let mut shift = 1u64 << (1u64 << num_bits);
        result *= F::from_u64(1 + (shift - 1) * c as u64);

        if b.len() == XLEN.trailing_zeros() as usize - 1 {
            return result;
        }

        num_bits += 1;
        shift = 1 << (1 << num_bits);
        if let Some(r_x) = r_x {
            result *= F::one() + F::from_u64(shift - 1) * r_x;
        }

        result *= checkpoints[Prefixes::Pow2].unwrap_or(F::one());
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if suffix_len != 0 {
            return Some(F::one()).into();
        }

        if j == 2 * XLEN - XLEN.trailing_zeros() as usize {
            let shift = 1 << (XLEN / 2);
            return Some(F::one() + F::from_u64(shift - 1) * r_y).into();
        }

        if 2 * XLEN - j < XLEN.trailing_zeros() as usize {
            let mut checkpoint = checkpoints[Prefixes::Pow2].unwrap();
            let shift = 1 << (1 << (2 * XLEN - j));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_x;
            let shift = 1 << (1 << (2 * XLEN - j - 1));
            checkpoint *= F::one() + F::from_u64(shift - 1) * r_y;
            return Some(checkpoint).into();
        }

        Some(F::one()).into()
    }
}
