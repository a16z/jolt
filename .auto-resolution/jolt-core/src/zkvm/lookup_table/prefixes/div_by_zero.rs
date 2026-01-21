use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum DivByZeroPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for DivByZeroPrefix {
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
        let (divisor, quotient) = b.uninterleave();
        // If low-order bits of divisor are not 0s or low-order bits of quotient are not
        // 1s, short-circuit and return 0.
        if u64::from(divisor) != 0 || u64::from(quotient) != (1 << quotient.len()) - 1 {
            return F::zero();
        }

        let mut result = checkpoints[Prefixes::DivByZero].unwrap_or(F::one());

        if let Some(r_x) = r_x {
            let y = F::from_u32(c);
            result *= (F::one() - r_x) * y;
        } else {
            let x = F::from_u8(c as u8);
            let y = F::from_u8(b.pop_msb());
            result *= (F::one() - x) * y;
        }
        result
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        _: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        // checkpoint *= (1 - r_x) * r_y
        let updated = checkpoints[Prefixes::DivByZero].unwrap_or(F::one()) * (F::one() - r_x) * r_y;
        Some(updated).into()
    }
}
