use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AlignAddrTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for AlignAddrTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        ((index & (1u128 << XLEN).wrapping_sub(1)) as u64) & !7
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for i in 0..XLEN - 3 {
            result += F::from_u128(1u128 << (XLEN - 1 - i)) * r[XLEN + i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for AlignAddrTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::LowerWord, Prefixes::ThreeLsb]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::LowerWord, Suffixes::ThreeLsb]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, lower_word, three_lsb] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerWord] * one + lower_word
            - prefixes[Prefixes::ThreeLsb] * one
            - three_lsb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, AlignAddrTable<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, AlignAddrTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AlignAddrTable<XLEN>>();
    }
}
