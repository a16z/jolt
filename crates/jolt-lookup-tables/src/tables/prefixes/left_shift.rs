use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftShiftPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();

        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let n = y.len();

        // Per-round recurrence at binary points:
        //   result += x_i * (1-y_i) * prod * 2^(XLEN-1-j_start/2-i)
        //   prod *= (1 + y_i)       [= 1 when y_i=0, 2 when y_i=1]
        let mut result = checkpoints[Prefixes::LeftShift];
        let mut prod = checkpoints[Prefixes::LeftShiftHelper];
        for i in 0..n {
            let x_i = (x_val >> (n - 1 - i)) & 1;
            let y_i = (y_val >> (n - 1 - i)) & 1;
            if y_i == 0 && x_i == 1 {
                result += prod * F::from_u64(1u64 << (XLEN - 1 - j_start / 2 - i));
            }
            if y_i == 1 {
                prod = prod + prod;
            }
        }

        result
    }
}
