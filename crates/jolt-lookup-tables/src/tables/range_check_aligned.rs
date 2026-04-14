use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::XLEN;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RangeCheckAlignedTable;

impl LookupTable for RangeCheckAlignedTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        if XLEN == 64 {
            (index as u64) & !1
        } else {
            ((index % (1u128 << XLEN)) as u64) & !1
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        // Skip the LSB (last bit position)
        for i in 0..XLEN - 1 {
            let shift = XLEN - 1 - i;
            result += F::from_u128(1u128 << shift) * r[XLEN + i];
        }
        result
    }
}

impl PrefixSuffixDecomposition for RangeCheckAlignedTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::LowerWord, Suffixes::Lsb]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, lower_word, lsb] = suffixes.try_into().unwrap();
        let lower_word_contribution = prefixes[Prefixes::LowerWord] * one + lower_word;
        let lsb_contribution = prefixes[Prefixes::Lsb] * lsb;
        lower_word_contribution - lsb_contribution
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, RangeCheckAlignedTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, RangeCheckAlignedTable>();
    }
}
