use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum OverflowBitsZeroPrefix {}

impl<F: Field> SparseDensePrefix<F> for OverflowBitsZeroPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start >= 128 - XLEN {
            return checkpoints[Prefixes::OverflowBitsZero];
        }

        // Overflow region = interleaved positions 0..XLEN.
        // Phase bits in overflow = top portion of `b`.
        let overflow_bits = if suffix_len >= XLEN {
            u128::from(b)
        } else {
            u128::from(b) >> (XLEN - suffix_len)
        };

        if overflow_bits != 0 {
            F::zero()
        } else {
            checkpoints[Prefixes::OverflowBitsZero]
        }
    }
}
