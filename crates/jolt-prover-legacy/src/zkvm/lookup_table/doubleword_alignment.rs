use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct DoublewordAlignmentTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for DoublewordAlignmentTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        index.is_multiple_of(8).into()
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        let lsb0 = r[r.len() - 1];
        let lsb1 = r[r.len() - 2];
        let lsb2 = r[r.len() - 3];
        (F::one() - lsb0) * (F::one() - lsb1) * (F::one() - lsb2)
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for DoublewordAlignmentTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::ThreeLsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [three_lsb] = suffixes.try_into().unwrap();
        prefixes[Prefixes::ThreeLsb] * three_lsb
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
        prefix_suffix_test_with_phase_size,
    };
    use common::constants::XLEN;

    use super::DoublewordAlignmentTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, DoublewordAlignmentTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, DoublewordAlignmentTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, DoublewordAlignmentTable<XLEN>>();
        prefix_suffix_test_with_phase_size::<XLEN, Fr, DoublewordAlignmentTable<XLEN>>(2, 20);
    }
}
