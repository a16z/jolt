use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightShiftWPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightShiftWPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::zero();
        }

        let (x, y) = b.uninterleave();
        let (x_val, y_val) = (u64::from(x), u64::from(y));
        let n = y.len();

        let mut result = checkpoints[Prefixes::RightShiftW];
        for i in 0..n {
            let x_i = (x_val >> (n - 1 - i)) & 1;
            let y_i = (y_val >> (n - 1 - i)) & 1;
            if y_i == 1 {
                result = result + result + F::from_u64(x_i);
            }
        }

        result
    }
}
