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

        // change_divisor restricted to binary points is checkpoint * x_0 * y_0
        // * prod_{i>0}((1-x_i) * y_i): non-zero only when the operand MSB x_0 is 1,
        // every later x bit is 0, and every y bit is 1. When the phase does not
        // contain the MSB pair (j_start > 0), the x_0 * y_0 factor is already
        // folded into the checkpoint.

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

            // With x_msb=1, x_rest=0, and all y bits 1, every product factor is 1.
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
