use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum ThreeLsbPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for ThreeLsbPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let bound_bits = 3usize.saturating_sub(suffix_len).min(b.len());
        if b.trailing_zeros() >= bound_bits as u32 {
            checkpoints[Prefixes::ThreeLsb]
        } else {
            F::zero()
        }
    }
}
