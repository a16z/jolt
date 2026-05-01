use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::tables::virtual_rev8w::rev8w;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum Rev8WPrefix {}

impl<F: Field> SparseDensePrefix<F> for Rev8WPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len >= 64 {
            return F::zero();
        }

        // Each bit at position `i` (from MSB=0) in the interleaved stream maps to
        // bit position `rev8w(1 << i)` in the output. At binary points, both x and y
        // bits land at concrete positions. The interleaved b bits represent positions
        // starting at `suffix_len` upward.
        let b_contribution = rev8w(u64::from(b) << suffix_len);
        checkpoints[Prefixes::Rev8W] + F::from_u64(b_contribution)
    }
}
