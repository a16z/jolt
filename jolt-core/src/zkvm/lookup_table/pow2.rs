use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::math::Math;
use crate::zkvm::lookup_table::prefixes::Prefixes;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct Pow2Table<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for Pow2Table<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        1 << (index % WORD_SIZE as u128) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::one();
        for i in 0..WORD_SIZE.log_2() {
            result *= F::one() + (F::from_u64((1 << (1 << i)) - 1)) * r[r.len() - i - 1];
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for Pow2Table<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::Pow2]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [pow2] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Pow2] * pow2
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::Pow2Table;
    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, Pow2Table<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, Pow2Table<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, Pow2Table<32>>();
    }
}
