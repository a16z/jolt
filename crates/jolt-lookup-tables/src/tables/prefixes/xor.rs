use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum XorPrefix {}

impl<F: Field> SparseDensePrefix<F> for XorPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        checkpoints[Prefixes::Xor] + F::from_u64((u64::from(x) ^ u64::from(y)) << (suffix_len / 2))
    }
}
