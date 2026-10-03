use jolt_field::JoltField;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum Pow2Prefix {}

impl<F: JoltField> SparseDensePrefix<F> for Pow2Prefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len != 0 {
            return F::one();
        }

        checkpoints[Prefixes::Pow2] * F::from_u64(1u64 << (b & (crate::XLEN - 1)))
    }
}
