use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum PositiveRemainderEqualsDivisorPrefix {}

impl<F: Field> SparseDensePrefix<F> for PositiveRemainderEqualsDivisorPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (remainder, divisor) = b.uninterleave();
        if remainder != divisor {
            return F::zero();
        }

        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start == 0 && !b.is_empty() {
            // Phase contains the sign bits (MSBs of remainder and divisor).
            // Both must be 0 (positive) for this prefix to be non-zero.
            let rem_sign = u64::from(remainder) >> (remainder.len() - 1);
            let div_sign = u64::from(divisor) >> (divisor.len() - 1);
            if rem_sign != 0 || div_sign != 0 {
                return F::zero();
            }
        }

        checkpoints[Prefixes::PositiveRemainderEqualsDivisor]
    }
}
