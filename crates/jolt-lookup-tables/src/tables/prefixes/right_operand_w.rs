use jolt_field::Field;

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::lookup_bits::LookupBits;

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::XLEN;

pub enum RightOperandWPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandWPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: Option<C>,
        c: u32,
        b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        let suffix_len = 2 * XLEN - j - b.len() - 1;
        let mut result = checkpoints[Prefixes::RightOperandW].unwrap_or(F::zero());

        if j % 2 == 1 && j > XLEN {
            // c is of the right operand
            let shift = XLEN - 1 - j / 2;
            result += F::from_u128((c as u128) << shift);
        }

        if suffix_len < XLEN {
            let (_x, y) = b.uninterleave();
            result += F::from_u128(u128::from(y) << (suffix_len / 2));
        }

        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeOps<F>,
        F: FieldOps<C>,
    {
        if j > XLEN {
            let shift = XLEN - 1 - j / 2;
            let updated = checkpoints[Prefixes::RightOperandW].unwrap_or(F::zero())
                + (F::from_u64(1 << shift) * r_y);
            Some(updated).into()
        } else {
            checkpoints[Prefixes::RightOperandW].into()
        }
    }
}
