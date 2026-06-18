use jolt_field::Field;

use crate::lookup_bits::LookupBits;
use crate::XLEN;

use super::{PrefixEval, Prefixes, SparseDensePrefix};

pub enum NegativeDivisorZeroRemainderPrefix {}

impl<F: Field> SparseDensePrefix<F> for NegativeDivisorZeroRemainderPrefix {
    fn default_checkpoint() -> F {
        F::one()
    }

    fn evaluate(checkpoints: &[PrefixEval<F>], b: LookupBits, suffix_len: usize) -> F {
        let (remainder, divisor) = b.uninterleave();
        if u64::from(remainder) != 0 {
            return F::zero();
        }

        let j_start = 2 * XLEN - suffix_len - b.len();
        if j_start == 0 && !b.is_empty() {
            // Phase contains sign bits. Remainder sign must be 0 (positive/zero),
            // divisor sign must be 1 (negative).
            let div_sign = u64::from(divisor) >> (divisor.len() - 1);
            // rem_sign is already checked via remainder == 0 (which includes sign bit = 0)
            if div_sign != 1 {
                return F::zero();
            }
        }

        checkpoints[Prefixes::NegativeDivisorZeroRemainder]
    }
}
