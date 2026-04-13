use jolt_field::Field;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct HalfwordAlignmentTable;

impl LookupTable for HalfwordAlignmentTable {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index.is_multiple_of(2)).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: Field + FieldOps<C>,
    {
        let lsb = r[r.len() - 1];
        F::one() - lsb
    }
}

impl PrefixSuffixDecomposition for HalfwordAlignmentTable {
    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::One, Suffixes::Lsb]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: Field>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lsb] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Lsb] * lsb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{mle_random_test, prefix_suffix_test};
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<Fr, HalfwordAlignmentTable>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, HalfwordAlignmentTable>();
    }
}
