use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum ThreeLsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for ThreeLsbPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len == 0 {
            checkpoints[Prefixes::ThreeLsb] + F::from_u64(u64::from(b) & 7)
        } else {
            F::zero()
        }
    }
}
