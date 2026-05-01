use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum RightOperandPrefix {}

impl<F: Field> SparseDensePrefix<F> for RightOperandPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (_, y) = b.uninterleave();
        checkpoints[Prefixes::RightOperand] + F::from_u128(u128::from(y) << (suffix_len / 2))
    }
}
