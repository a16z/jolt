use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum EqPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for EqPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (x, y) = b.uninterleave();
        if x == y {
            checkpoints[Prefixes::Eq]
        } else {
            F::zero()
        }
    }
}
