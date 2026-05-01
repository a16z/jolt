use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct UpperWordTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for UpperWordTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index >> XLEN) as u64
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        for (i, r_i) in r[..XLEN].iter().enumerate() {
            result += F::from_u64(1 << (XLEN - 1 - i)) * *r_i;
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for UpperWordTable<XLEN> {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::UpperWord]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, upper_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::UpperWord] * one + upper_word
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
        mle_full_hypercube_test::<8, Fr, UpperWordTable<8>>();
    }

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, UpperWordTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, UpperWordTable<XLEN>>();
    }
}
