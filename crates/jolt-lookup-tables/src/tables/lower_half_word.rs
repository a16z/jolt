use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

/// Extracts the lower half of a word.
/// For XLEN=64 this extracts the lower 32 bits; for XLEN=32, the lower 16 bits.
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LowerHalfWordTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for LowerHalfWordTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let half_word_size = XLEN / 2;
        (index % (1u128 << half_word_size)) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let half_word_size = XLEN / 2;
        let mut result = F::zero();
        for i in 0..half_word_size {
            result += F::from_u64(1 << (half_word_size - 1 - i)) * r[XLEN + half_word_size + i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for LowerHalfWordTable<XLEN> {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::LowerHalfWord]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, lower_half_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerHalfWord] * one + lower_half_word
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
        mle_full_hypercube_test::<8, Fr, LowerHalfWordTable<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, LowerHalfWordTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, LowerHalfWordTable<XLEN>>();
    }
}
