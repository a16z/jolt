use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftOperandIsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftOperandIsZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, _) = b.uninterleave();
        if u64::from(x) != 0 {
            F::zero()
        } else {
            checkpoints[Prefixes::LeftOperandIsZero]
        }
    }
}
