use crate::{field::JoltField, utils::lookup_bits::LookupBits};
use crate::field::MontU128;
use super::{PrefixCheckpoint, Prefixes, SparseDensePrefix};

pub enum NegativeDivisorEqualsRemainderPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for NegativeDivisorEqualsRemainderPrefix {
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
                return  F::from_u32(c).mul_u128_mont_form(r_x.unwrap());
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
            let r_x_f = F::from_u128_mont(r_x);
            negative_divisor_equals_remainder * (y.mul_u128_mont_form(r_x) + (F::one() - r_x_f) * (F::one() - y))
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
            // This prefix handles the case where both remainder and
            // divisor are negative.
            return Some(r_x_f * r_y_f).into();
        }

        let mut negative_divisor_equals_remainder =
            checkpoints[Prefixes::NegativeDivisorEqualsRemainder].unwrap();
        // checkpoint *= EQ(r_x, r_y)

        negative_divisor_equals_remainder *= r_x_f * r_y_f + (F::one() - r_x_f) * (F::one() - r_y_f);
        Some(negative_divisor_equals_remainder).into()
    }
}
