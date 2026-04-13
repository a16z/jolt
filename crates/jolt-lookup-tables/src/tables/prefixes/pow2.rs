use jolt_field::Field;

use crate::lookup_bits::LookupBits;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum Pow2Prefix {}

impl<F: Field> SparseDensePrefix<F> for Pow2Prefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        // pow2 computes 2^(shift) where shift is the last log2(XLEN) interleaved
        // index bits. When the suffix still contains all shift bits, the prefix is 1.
        if suffix_len != 0 {
            return F::one();
        }

        // At suffix_len == 0, the shift bits are in the low bits of `b` (raw interleaved).
        // `b & (XLEN-1)` extracts the last 6 bits.
        checkpoints[Prefixes::Pow2] * F::from_u64(1u64 << (b & (crate::XLEN - 1)))
    }
}
