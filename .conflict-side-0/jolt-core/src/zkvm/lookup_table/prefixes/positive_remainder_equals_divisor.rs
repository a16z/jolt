use crate::{
    field::{ChallengeFieldOps, FieldChallengeOps, JoltField},
    utils::lookup_bits::LookupBits,
};

use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum PositiveRemainderEqualsDivisorPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for PositiveRemainderEqualsDivisorPrefix {
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
                // This prefix handles the case where both remainder and divisor
                // are positive, i.e. their sign bits are zero.
                return (F::one() - F::from_u32(c)) * (F::one() - divisor_sign);
            }
        }
        if j == 1 {
            let (remainder, divisor) = b.uninterleave();
            if u64::from(remainder) != u64::from(divisor) {
                return F::zero();
            } else {
                // `r_x` is the sign "bit" of the remainder.
                // `c` is the sign "bit" of the divisor.
                // This prefix handles the case where both remainder and divisor
                // are positive, i.e. their sign bits are zero.
                return (F::one() - r_x.unwrap()) * (F::one() - F::from_u32(c));
            }
        }

        let positive_remainder_equals_divisor =
            checkpoints[Prefixes::PositiveRemainderEqualsDivisor].unwrap();

        if let Some(r_x) = r_x {
            let (remainder, divisor) = b.uninterleave();
            // Short-circuit if low-order bits of remainder and divisor are not equal
            if u64::from(remainder) != u64::from(divisor) {
                return F::zero();
            }
            let y = F::from_u32(c);
            positive_remainder_equals_divisor * (r_x * y + (F::one() - r_x) * (F::one() - y))
        } else {
            let y = F::from_u8(b.pop_msb());
            let (remainder, divisor) = b.uninterleave();
            // Short-circuit if low-order bits of remainder and divisor are not equal
            if u64::from(remainder) != u64::from(divisor) {
                return F::zero();
            }
            let x = F::from_u32(c);
            positive_remainder_equals_divisor * (x * y + (F::one() - x) * (F::one() - y))
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
            // This prefix handles the case where both remainder and divisor
            // are positive, i.e. their sign bits are zero.
            return Some((F::one() - r_x) * (F::one() - r_y)).into();
        }

        let mut positive_remainder_equals_divisor =
            checkpoints[Prefixes::PositiveRemainderEqualsDivisor].unwrap();
        positive_remainder_equals_divisor *= r_x * r_y + (F::one() - r_x) * (F::one() - r_y);
        Some(positive_remainder_equals_divisor).into()
    }
}
