use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RangeCheckAlignedTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for RangeCheckAlignedTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        if XLEN == 64 {
            (index as u64) & !1
        } else {
            ((index % (1u128 << XLEN)) as u64) & !1
        }
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
    {
        debug_assert_eq!(r.len(), 2 * XLEN);
        let mut result = F::zero();
        // Skip the LSB
        for i in 0..XLEN - 1 {
            let shift = XLEN - 1 - i;
            result += F::from_u128(1u128 << shift) * r[XLEN + i];
        }
        result
    }
}

impl<const XLEN: usize> PrefixSuffixDecomposition<XLEN> for RangeCheckAlignedTable<XLEN> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWord, Suffixes::Lsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_word, lsb] = suffixes.try_into().unwrap();
        let lower_word_contribution = prefixes[Prefixes::LowerWord] * one + lower_word;
        let lsb_contribution = prefixes[Prefixes::Lsb] * lsb;
        lower_word_contribution - lsb_contribution
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::RangeCheckAlignedTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, RangeCheckAlignedTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, RangeCheckAlignedTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, RangeCheckAlignedTable<XLEN>>();
    }
}
