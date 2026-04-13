use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum ChangeDivisorPrefix {}

impl<F: Field> SparseDensePrefix<F> for ChangeDivisorPrefix {
    fn default_checkpoint() -> F {
        F::from_u64(2) - F::from_u128(1u128 << XLEN)
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();

        // change_divisor computes: checkpoint * x_msb * prod((1-x_i) * y_i) for remaining pairs
        // where x_msb is the first x bit (at j=0).
        //
        // At j=0: x_msb must be 1, all remaining x bits must be 0, all y bits must be 1.
        // At j=1: checkpoint * r_x (from j=1) * c (y bit) — special case for first y bit.
        // At j>1: checkpoint * (1-x_i) * y_i for each pair.
        //
        // At binary points, non-zero only when x_msb=1, all subsequent x bits=0, all y bits=1.
        // Exception: j=1 uses x*y instead of (1-x)*y.

        if j_start == 0 {
            // Phase includes the MSB x bit. Extract it.
            let (x, y) = b.uninterleave();
            let x_val = u64::from(x);
            let y_val = u64::from(y);

            // x_msb must be 1
            let x_msb = (x_val >> (x.len() - 1)) & 1;
            if x_msb == 0 {
                return F::zero();
            }

            // For j=0 round: remaining x bits (below MSB) must be 0, all y bits must be 1
            let x_rest = x_val & ((1u64 << (x.len() - 1)) - 1);
            if x_rest != 0 || y_val != (1u64 << y.len()) - 1 {
                return F::zero();
            }

            // j=1 contributes x*y = x_msb * y_0, j>1 contributes (1-x_i)*y_i.
            // Since x_rest=0 and all y=1, each (1-0)*1 = 1, and x_msb*y_0 = 1.
            checkpoints[Prefixes::ChangeDivisor]
        } else {
            // All x bits must be 0, all y bits must be 1 for non-zero result
            let (x, y) = b.uninterleave();
            if u64::from(x) != 0 || u64::from(y) != (1u64 << y.len()) - 1 {
                return F::zero();
            }
            checkpoints[Prefixes::ChangeDivisor]
        }
    }
}
