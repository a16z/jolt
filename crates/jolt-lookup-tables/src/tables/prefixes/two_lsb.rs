use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum TwoLsbPrefix {}

impl<F: Field> SparseDensePrefix<F> for TwoLsbPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len == 0 {
            if u32::from(b).trailing_zeros() >= 2 {
                F::one()
            } else {
                F::zero()
            }
        } else {
            checkpoints[Prefixes::TwoLsb]
        }
    }
}
