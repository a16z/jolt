use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixCheckpoint, PrefixEval, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorWPrefix {}

impl<F: Field> SparseDensePrefix<F> for ChangeDivisorWPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();

        if j_start < XLEN {
            return F::zero();
        }

        // Same logic as ChangeDivisor but only in the lower word (j >= XLEN).
        // The initial value (2 - 2^XLEN) is set at j=XLEN, not from default_checkpoint.
        // At j=XLEN: x_msb (of lower word) must be 1, rest x=0, all y=1.
        // At j=XLEN+1: checkpoint * x * y (special first pair after MSB).
        // At j>XLEN+1: checkpoint * (1-x) * y.

        if j_start == XLEN {
            let (x, y) = b.uninterleave();
            let x_val = u64::from(x);
            let y_val = u64::from(y);

            let x_msb = (x_val >> (x.len() - 1)) & 1;
            if x_msb == 0 {
                return F::zero();
            }

            let x_rest = x_val & ((1u64 << (x.len() - 1)) - 1);
            if x_rest != 0 || y_val != (1u64 << y.len()) - 1 {
                return F::zero();
            }

            F::from_u64(2) - F::from_u128(1u128 << XLEN)
        } else {
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            checkpoints[Prefixes::ChangeDivisorW]
        }
    }

    #[expect(clippy::unwrap_used)]
    fn prefix_mle(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: Option<F>,
        c: u32,
        mut b: LookupBits,
        j: usize,
    ) -> F {
        let _ = (checkpoints, r_x, c, b, j);
        if j < XLEN {
            return F::zero();
        }

        let mut result = if j == XLEN || j == XLEN + 1 {
            F::from_u64(2) - F::from_u128(1u128 << XLEN)
        } else {
            checkpoints[Prefixes::ChangeDivisorW].unwrap()
        };

        if j == XLEN {
            let x_msb = b.pop_msb() as u32;
            if x_msb == 0 {
                return F::zero();
            }
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result = result.mul_u64(c as u64);
        } else if let Some(r_x) = r_x {
            if j > XLEN {
                let (x, y) = b.uninterleave();
                if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 || c == 0 {
                    return F::zero();
                }

                if j == XLEN + 1 {
                    result *= (r_x) * F::from_u64(c as u64);
                } else {
                    result *= (F::one() - r_x) * F::from_u64(c as u64);
                }
            }
        } else if j > XLEN {
            let (x, y) = b.uninterleave();
            if !b.is_empty() && u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            result *= F::one() - F::from_u64(c as u64);
        }
        result
    }

    #[expect(clippy::unwrap_used)]
    fn update_prefix_checkpoint(
        checkpoints: &[PrefixCheckpoint<F>],
        r_x: F,
        r_y: F,
        j: usize,
        suffix_len: usize,
    ) -> PrefixCheckpoint<F> {
        let _ = (checkpoints, r_x, r_y, j, suffix_len);
        if j < XLEN {
            return Some(F::zero()).into();
        }

        let updated = if j == XLEN + 1 {
            (F::from_u64(2) - F::from_u128(1u128 << XLEN)) * r_x * r_y
        } else {
            checkpoints[Prefixes::ChangeDivisorW].unwrap() * ((F::one() - r_x) * r_y)
        };
        Some(updated).into()
    }
}
