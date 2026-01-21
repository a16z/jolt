use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum NegativeDivisorEqualsRemainderPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for NegativeDivisorEqualsRemainderPrefix {
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
            let (remainder, divisor) = b.uninterleave();
            if u64::from(remainder) != u64::from(divisor) {
                return F::zero();
            } else {
                // `c` is the sign "bit" of the remainder.
                // This prefix handles the case where both remainder and
                // divisor are negative, i.e. their sign bits are one.
                return F::from_u32(c) * divisor_sign;
            }
        }
        if j == 1 {
            let (remainder, divisor) = b.uninterleave();
            if u64::from(remainder) != u64::from(divisor) {
                return F::zero();
            } else {
                // `r_x` is the sign "bit" of the remainder.
                // `c` is the sign "bit" of the divisor.
                // This prefix handles the case where both remainder and
                // divisor are negative, i.e. their sign bits are one.
                return r_x.unwrap() * F::from_u32(c);
            }
        }

        let negative_divisor_equals_remainder =
            checkpoints[Prefixes::NegativeDivisorEqualsRemainder].unwrap();

        if let Some(r_x) = r_x {
            let (remainder, divisor) = b.uninterleave();
            // Short-circuit if low-order bits of remainder and divisor are not equal
            if remainder != divisor {
                return F::zero();
            }
            let y = F::from_u32(c);
            negative_divisor_equals_remainder * (r_x * y + (F::one() - r_x) * (F::one() - y))
        } else {
            let y_msb = F::from_u8(b.pop_msb());
            let (remainder, divisor) = b.uninterleave();
            // Short-circuit if low-order bits of remainder and divisor are not equal
            if remainder != divisor {
                return F::zero();
            }
            let x = F::from_u32(c);
            negative_divisor_equals_remainder * (x * y_msb + (F::one() - x) * (F::one() - y_msb))
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
            // This prefix handles the case where both remainder and
            // divisor are negative.
            return Some(r_x * r_y).into();
        }

        let mut negative_divisor_equals_remainder =
            checkpoints[Prefixes::NegativeDivisorEqualsRemainder].unwrap();
        // checkpoint *= EQ(r_x, r_y)
        negative_divisor_equals_remainder *= r_x * r_y + (F::one() - r_x) * (F::one() - r_y);
        Some(negative_divisor_equals_remainder).into()
    }
}
