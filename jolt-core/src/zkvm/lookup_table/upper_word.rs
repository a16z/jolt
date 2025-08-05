use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct UpperWordTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for UpperWordTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index >> WORD_SIZE) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            result += F::from_u64(1 << (WORD_SIZE - 1 - i)) * r[i];
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for UpperWordTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::UpperWord]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, upper_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::UpperWord] * one + upper_word
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::UpperWordTable;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, UpperWordTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, UpperWordTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, UpperWordTable<XLEN>>();
    }
}
