use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightOperandIsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandIsZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (_, y) = b.uninterleave();
        if u64::from(y) != 0 {
            F::zero()
        } else {
            checkpoints[Prefixes::RightOperandIsZero]
        }
    }
}
