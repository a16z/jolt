use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct WordAlignmentTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for WordAlignmentTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index.is_multiple_of(4)).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let lsb0 = r[r.len() - 1];
        let lsb1 = r[r.len() - 2];
        (F::one() - lsb0) * (F::one() - lsb1)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for WordAlignmentTable<XLEN> {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::TwoLsb]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [two_lsb] = suffixes.try_into().unwrap();
        prefixes[Prefixes::TwoLsb] * two_lsb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_full_hypercube_test, mle_random_test, prefix_suffix_test};
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, WordAlignmentTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, WordAlignmentTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, WordAlignmentTable<8>>();
    }
}
