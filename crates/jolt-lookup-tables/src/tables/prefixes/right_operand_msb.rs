use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightOperandMsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandMsbPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start > 0 {
            return checkpoints[Prefixes::RightOperandMsb];
        }
        let (_, y) = b.uninterleave();
        F::from_u64(u64::from(y) >> (y.len() - 1))
    }
}
