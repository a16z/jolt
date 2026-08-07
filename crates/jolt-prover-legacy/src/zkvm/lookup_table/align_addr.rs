use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltLookupTable, PrefixSuffixDecomposition};
use crate::field::{ChallengeFieldOps, FieldChallengeOps, JoltField};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AlignAddrTable<const XLEN: usize>;

impl<const XLEN: usize> JoltLookupTable for AlignAddrTable<XLEN> {
    fn materialize_entry(&self, index: u128) -> u64 {
        ((index & (1u128 << XLEN).wrapping_sub(1)) as u64) & !7
    }

    fn evaluate_mle<F, C>(&self, r: &[C]) -> F
    where
        C: ChallengeFieldOps<F>,
        F: JoltField + FieldChallengeOps<C>,
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
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWord, Suffixes::ThreeLsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        let [one, lower_word, three_lsb] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerWord] * one + lower_word
            - prefixes[Prefixes::ThreeLsb] * one
            - three_lsb
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use ark_bn254::Fr;
    use common::constants::XLEN;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, AlignAddrTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, AlignAddrTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AlignAddrTable<XLEN>>();
    }
}
