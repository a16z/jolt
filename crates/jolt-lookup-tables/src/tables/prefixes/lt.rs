use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LessThanPrefix {}

impl<F: Field> SparseDensePrefix<F> for LessThanPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        if u64::from(x) < u64::from(y) {
            checkpoints[Prefixes::LessThan] + checkpoints[Prefixes::Eq]
        } else {
            checkpoints[Prefixes::LessThan]
        }
    }
}
