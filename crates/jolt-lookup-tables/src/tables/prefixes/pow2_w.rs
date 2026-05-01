use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum Pow2WPrefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2WPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        if suffix_len != 0 {
            return F::one();
        }

        // pow2w computes 2^(shift mod 32). The shift is the last 5 interleaved bits.
        checkpoints[Prefixes::Pow2W] * F::from_u64(1u64 << (b & 0b11111))
    }
}
