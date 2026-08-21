use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftMsbRightOperandIsZeroPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftMsbRightOperandIsZeroPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (_, y) = b.uninterleave();
        if u64::from(y) != 0 {
            return F::zero();
        }

        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start == 0 {
            let (x, _) = b.uninterleave();
            F::from_u64(u64::from(x) >> (x.len() - 1))
        } else {
            checkpoints[Prefixes::LeftMsbRightOperandIsZero]
        }
    }
}
