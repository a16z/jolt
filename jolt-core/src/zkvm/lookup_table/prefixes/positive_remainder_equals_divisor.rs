use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};
use crate::field::MontU128;
use crate::{field::JoltField, utils::lookup_bits::LookupBits};

pub enum PositiveRemainderEqualsDivisorPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for PositiveRemainderEqualsDivisorPrefix {
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<MontU128>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
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
                return (F::one() - F::from_u128_mont(r_x.unwrap())) * (F::one() - F::from_u32(c));
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
            positive_remainder_equals_divisor
                * (y.mul_u128_mont_form(r_x) + (F::one() - F::from_u128_mont(r_x)) * (F::one() - y))
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

    fn prefix_mle_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
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

    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: MontU128,
        r_y: MontU128,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        let r_x_f = F::from_u128_mont(r_x);
        let r_y_f = F::from_u128_mont(r_y);
        if j == 1 {
            // `r_x` is the sign bit of the remainder
            // `r_y` is the sign bit of the divisor
            // This prefix handles the case where both remainder and divisor
            // are positive, i.e. their sign bits are zero.
            return Some((F::one() - r_x_f) * (F::one() - r_y_f)).into();
        }

        let mut positive_remainder_equals_divisor =
            checkpoints[Prefixes::PositiveRemainderEqualsDivisor].unwrap();
        positive_remainder_equals_divisor *=
            r_y_f.mul_u128_mont_form(r_x) + (F::one() - r_x_f) * (F::one() - r_y_f);
        Some(positive_remainder_equals_divisor).into()
    }

    fn update_prefix_checkpoint_field(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
    ) -> PrefixCheckpoint<F> {
        if j == 1 {
            // `r_x` is the sign bit of the remainder
            // `r_y` is the sign bit of the divisor
            // This prefix handles the case where both remainder and divisor
            // are positive, i.e. their sign bits are zero.
            return Some((F::one() - r_x) * (F::one() - r_y)).into();
        }

        let mut positive_remainder_equals_divisor =
            checkpoints[Prefixes::PositiveRemainderEqualsDivisor].unwrap();
        positive_remainder_equals_divisor *= r_y * r_x + (F::one() - r_x) * (F::one() - r_y);
        Some(positive_remainder_equals_divisor).into()
    }
}
