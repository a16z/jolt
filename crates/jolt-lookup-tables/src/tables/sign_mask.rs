use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::XLEN;

/// Returns all-ones if the MSB of the first operand is set, else zero.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignMaskTable;

impl LookupTable for SignMaskTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let sign_bit_pos = 2 * XLEN - 1;
        let sign_bit = 1u128 << sign_bit_pos;
        if index & sign_bit != 0 {
            ((1u128 << XLEN) - 1) as u64
        } else {
            0
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let sign_bit = r[0];
        let ones: u64 = ((1u128 << XLEN) - 1) as u64;
        sign_bit * F::from_u64(ones)
    }
}

impl PrefixSuffixDecomposition for SignMaskTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one] = suffixes.try_into().unwrap();
        let ones: u64 = ((1u128 << XLEN) - 1) as u64;
        F::from_u64(ones) * prefixes[Prefixes::LeftOperandMsb] * one
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, SignMaskTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, SignMaskTable>();
    }
}
