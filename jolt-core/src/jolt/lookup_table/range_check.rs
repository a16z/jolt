use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct RangeCheckTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for RangeCheckTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        (index % (1u128 << WORD_SIZE)) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            result += F::from_u64(1 << (WORD_SIZE - 1 - i)) * r[WORD_SIZE + i];
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for RangeCheckTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LowerWord]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lower_word] = suffixes.try_into().unwrap();
        prefixes[Prefixes::LowerWord] * one + lower_word
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::RangeCheckTable;
    use crate::jolt::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, RangeCheckTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, RangeCheckTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, RangeCheckTable<32>>();
    }
}
