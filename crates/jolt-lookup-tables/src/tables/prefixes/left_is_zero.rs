use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum LeftOperandIsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftOperandIsZeroPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        b: LookupBits,
        _: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let (x, _) = b.uninterleave();
        // Short-circuit if low-order bits of `x` are not 0s
        if u64::from(x) != 0 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::LeftOperandIsZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            result *= F::one() - r_x;
        } else {
            let x = F::from_u8(c as u8);
            result *= F::one() - x;
        }
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        _: C,
        _: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        // checkpoint *= (1 - r_x)
        let updated =
            checkpoints[Prefixes::LeftOperandIsZero].unwrap_or(F::one()) * (F::one() - r_x);
        Some(updated).into()
    }
}
