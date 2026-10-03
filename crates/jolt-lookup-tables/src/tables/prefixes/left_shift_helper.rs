use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftHelperPrefix {}

impl<F: JoltField> SparseDensePrefix<F> for LeftShiftHelperPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        let (_x, y) = b.uninterleave();
        checkpoints[Prefixes::LeftShiftHelper] * F::from_u64(1u64 << u64::from(y).count_ones())
    }
}
