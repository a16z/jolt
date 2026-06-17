use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftHelperPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftShiftHelperPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, _suffix_len: usize) -> F {
        // Tracks the product of (1 + y_i) across rounds. At binary points each
        // factor is 1 when y_i=0 and 2 when y_i=1, so the phase contribution is
        // 2^(popcount of the phase's y bits).
        let (_x, y) = b.uninterleave();
        checkpoints[Prefixes::LeftShiftHelper] * F::from_u64(1u64 << u64::from(y).count_ones())
    }
}
