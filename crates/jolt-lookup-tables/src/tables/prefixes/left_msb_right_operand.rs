use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftMsbRightOperandPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftMsbRightOperandPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        let (x, y) = b.uninterleave();
        let left_msb = if j_start == 0 {
            F::from_u64(u64::from(x) >> (x.len() - 1))
        } else {
            checkpoints[Prefixes::LeftOperandMsb]
        };

        checkpoints[Prefixes::LeftMsbRightOperand]
            + left_msb * F::from_u128(u128::from(y) << (suffix_len / 2))
    }
}
