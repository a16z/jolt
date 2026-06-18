use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum PositiveRemainderLessThanDivisorPrefix {}

impl<F: Field> SparseDensePrefix<F> for PositiveRemainderLessThanDivisorPrefix {
    fn default_checkpoint() -> F {
        F::zero()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (remainder, divisor) = b.uninterleave();

        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start == 0 && !b.is_empty() {
            // Phase contains sign bits. Both must be 0 (positive).
            let rem_sign = u64::from(remainder) >> (remainder.len() - 1);
            let div_sign = u64::from(divisor) >> (divisor.len() - 1);
            if rem_sign != 0 || div_sign != 0 {
                return F::zero();
            }
        }

        if u64::from(remainder) < u64::from(divisor) {
            checkpoints[Prefixes::PositiveRemainderLessThanDivisor]
                + checkpoints[Prefixes::PositiveRemainderEqualsDivisor]
        } else {
            checkpoints[Prefixes::PositiveRemainderLessThanDivisor]
        }
    }
}
