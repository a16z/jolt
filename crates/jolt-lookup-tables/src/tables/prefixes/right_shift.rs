use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightShiftPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightShiftPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        // Per-round recurrence at binary points:
        //   result = result * (1 + y_i) + x_i * y_i
        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let n = y.len();

        let mut result = checkpoints[Prefixes::RightShift];
        for i in 0..n {
            let x_i = (x_val >> (n - 1 - i)) & 1;
            let y_i = (y_val >> (n - 1 - i)) & 1;
            // result *= (1 + y_i); result += x_i * y_i
            if y_i == 1 {
                result = result + result + F::from_u64(x_i);
            }
        }

        result
    }
}
