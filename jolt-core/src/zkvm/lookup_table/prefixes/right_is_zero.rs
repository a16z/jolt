use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum RightOperandIsZeroPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for RightOperandIsZeroPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        _: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        let (_, y) = b.uninterleave();
        // Short-circuit if low-order bits of `y` are not 0s
        if u64::from(y) != 0 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one());

        if r_x.is_some() {
            let y = F::from_u8(c as u8);
            result *= F::one() - y;
        } else {
            let y = F::from_u8(b.pop_msb());
            result *= F::one() - y;
        }
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        _: C,
        r_y: C,
        _: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // checkpoint *= (1 - r_y)
        let updated =
            checkpoints[Prefixes::RightOperandIsZero].unwrap_or(F::one()) * (F::one() - r_y);
        Some(updated).into()
    }
}
