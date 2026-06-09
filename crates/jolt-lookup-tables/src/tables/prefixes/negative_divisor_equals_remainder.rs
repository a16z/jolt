use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum NegativeDivisorEqualsRemainderPrefix {}

impl<F: Field> SparseDensePrefix<F> for NegativeDivisorEqualsRemainderPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (remainder, divisor) = b.uninterleave();
        if remainder != divisor {
            return F::zero();
        }

        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start == 0 && !b.is_empty() {
            // Phase contains sign bits. Both must be 1 (negative).
            let rem_sign = u64::from(remainder) >> (remainder.len() - 1);
            let div_sign = u64::from(divisor) >> (divisor.len() - 1);
            if rem_sign != 1 || div_sign != 1 {
                return F::zero();
            }
        }

        checkpoints[Prefixes::NegativeDivisorEqualsRemainder]
    }

    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
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
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
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
