use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum OverflowBitsZeroPrefix<const XLEN: usize> {}

impl<const XLEN: usize, F: Field> SparseDensePrefix<F> for OverflowBitsZeroPrefix<XLEN> {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let suffix_len = 2 * XLEN - j - b.len() - 1;
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());
        }

        let mut result = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u8(c as u8);
            result *= (F::one() - r_x) * (F::one() - y);
        } else {
            let x = F::from_u32(c);
            let y = F::from_u8(b.pop_msb());
            result *= (F::one() - x) * (F::one() - y);
        }

        let rest = u128::from(b);
        let temp = F::from_u64((((rest << suffix_len) >> XLEN) == 0) as u64);
        result *= temp;

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero].into();
        }
        let updated = checkpoints[Prefixes::OverflowBitsZero].unwrap_or(F::one())
            * (F::one() - r_x)
            * (F::one() - r_y);

        Some(updated).into()
    }
}
