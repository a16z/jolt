use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

use crate::challenge_ops::{ChallengeOps, FieldOps};
use crate::tables::prefixes::{PrefixEval, Prefixes};
use crate::tables::suffixes::{SuffixEval, Suffixes};
use crate::tables::PrefixSuffixDecomposition;
use crate::traits::LookupTable;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DoublewordAlignmentTable<const XLEN: usize>;

impl<const XLEN: usize> LookupTable for DoublewordAlignmentTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        index.is_multiple_of(8).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeOps<F>,
        F: JoltField + FieldOps<C>,
    {
        let lsb0 = r[r.len() - 1];
        let lsb1 = r[r.len() - 2];
        let lsb2 = r[r.len() - 3];
        (F::one() - lsb0) * (F::one() - lsb1) * (F::one() - lsb2)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for DoublewordAlignmentTable<XLEN> {
    fn prefixes(&self) -> &'static [Prefixes] {
        &[Prefixes::ThreeLsb]
    }

    fn suffixes(&self) -> &'static [Suffixes] {
        &[Suffixes::ThreeLsb]
    }

    #[expect(clippy::unwrap_used)]
    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [three_lsb] = suffixes.try_into().unwrap();
        prefixes[Prefixes::ThreeLsb] * three_lsb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::test_utils::{
        mle_full_hypercube_test, mle_random_test, prefix_suffix_materialization_test,
        prefix_suffix_test,
    };
    use crate::XLEN;
    use jolt_field::Fr;

    #[test]
    fn mle_random() {
        mle_random_test::<XLEN, Fr, DoublewordAlignmentTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, DoublewordAlignmentTable<XLEN>>();
        prefix_suffix_materialization_test::<XLEN, Fr, DoublewordAlignmentTable<XLEN>>(2, 20);
    }

    #[test]
    fn mle_full_hypercube() {
        mle_full_hypercube_test::<8, Fr, DoublewordAlignmentTable<8>>();
    }
}
