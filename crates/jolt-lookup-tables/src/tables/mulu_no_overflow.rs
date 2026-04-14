use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;
use crate::XLEN;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MulUNoOverflowTable;

impl LookupTable for MulUNoOverflowTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        let upper_bits = index >> XLEN;
        (upper_bits == 0) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::one();
        for r_i in &r[..XLEN] {
            result *= F::one() - *r_i;
        }
        result
    }
}

impl PrefixSuffixDecomposition for MulUNoOverflowTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::OverflowBitsZero]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [overflow_bits_zero] = suffixes.try_into().unwrap();
        prefixes[Prefixes::OverflowBitsZero] * overflow_bits_zero
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, MulUNoOverflowTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, MulUNoOverflowTable>();
    }
}
