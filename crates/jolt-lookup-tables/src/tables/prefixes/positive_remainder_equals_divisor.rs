use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum PositiveRemainderEqualsDivisorPrefix {}

impl<F: Field> SparseDensePrefix<F> for PositiveRemainderEqualsDivisorPrefix {
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
            // Phase contains the sign bits (MSBs of remainder and divisor).
            // Both must be 0 (positive) for this prefix to be non-zero.
            let rem_sign = u64::from(remainder) >> (remainder.len() - 1);
            let div_sign = u64::from(divisor) >> (divisor.len() - 1);
            if rem_sign != 0 || div_sign != 0 {
                return F::zero();
            }
        }

        checkpoints[Prefixes::PositiveRemainderEqualsDivisor]
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
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
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
