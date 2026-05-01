use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, SparseDensePrefix};

pub enum LsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for LsbPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(_checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len == 0 {
            F::from_u64(u64::from(b) & 1)
        } else {
            F::one()
        }
    }
}
