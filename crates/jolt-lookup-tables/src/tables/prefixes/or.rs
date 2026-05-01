use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum OrPrefix {}

impl<F: Field> SparseDensePrefix<F> for OrPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        checkpoints[Prefixes::Or] + F::from_u64((u64::from(x) | u64::from(y)) << (suffix_len / 2))
    }
}
