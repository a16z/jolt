use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum LeftShiftWHelperPrefix {}

impl<F: Field> SparseDensePrefix<F> for LeftShiftWHelperPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start < XLEN {
            return F::one();
        }

        let (_x, y) = b.uninterleave();
        checkpoints[Prefixes::LeftShiftWHelper] * F::from_u64(1u64 << u64::from(y).count_ones())
    }
}
