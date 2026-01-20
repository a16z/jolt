use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum NegativeDivisorZeroRemainderPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for NegativeDivisorZeroRemainderPrefix {
    fn prefix_mle<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<C>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j == 0 {
            let divisor_sign = F::from_u8(b.pop_msb());
            let (remainder, _) = b.uninterleave();
            if u64::from(remainder) != 0 {
                return F::zero();
            } else {
                // `c` is the sign "bit" of the remainder.
                // This prefix handles the case where the remainder is zero
                // and the divisor is negative.
                return (F::one() - F::from_u32(c)) * divisor_sign;
            }
        }
        if j == 1 {
            let (remainder, _) = b.uninterleave();
            if u64::from(remainder) != 0 {
                return F::zero();
            } else {
                // `r_x` is the sign "bit" of the remainder.
                // `c` is the sign "bit" of the divisor.
                // This prefix handles the case where the remainder is zero
                // and the divisor is negative.
                return (F::one() - r_x.unwrap()) * F::from_u32(c);
            }
        }

        let negative_divisor_zero_remainder =
            checkpoints[Prefixes::NegativeDivisorZeroRemainder].unwrap();

        if let Some(r_x) = r_x {
            let (remainder, _) = b.uninterleave();
            // Short-circuit if low-order bits of remainder are not 0s
            if u64::from(remainder) != 0 {
                return F::zero();
            }

            negative_divisor_zero_remainder * (F::one() - r_x)
        } else {
            let _ = b.pop_msb();
            let (remainder, _) = b.uninterleave();
            // Short-circuit if low-order bits of remainder are not 0s
            if u64::from(remainder) != 0 {
                return F::zero();
            }

            negative_divisor_zero_remainder * (F::one() - F::from_u32(c))
        }
    }

    fn update_prefix_checkpoint<C>(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: C,
        r_y: C,
        j: usize,
        _suffix_len: usize,
    ) -> PrefixCheckpoint<F>
    where
        C: ChallengeFieldOps<F>,
        F: FieldChallengeOps<C>,
    {
        if j == 1 {
            // `r_x` is the sign bit of the remainder
            // `r_y` is the sign bit of the divisor
            // This prefix handles the case where the remainder is zero
            // and the divisor is negative.
            return Some((F::one() - r_x) * r_y).into();
        }

        let mut negative_divisor_zero_remainder =
            checkpoints[Prefixes::NegativeDivisorZeroRemainder].unwrap();
        negative_divisor_zero_remainder *= F::one() - r_x;
        Some(negative_divisor_zero_remainder).into()
    }
}
