use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{JoltField, MontU128};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MovsignTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for MovsignTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit = 1 << (2 * WORD_SIZE - 1);
        if index & sign_bit != 0 {
            (1 << WORD_SIZE) - 1
        } else {
            0
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        // 2 ^ {WORD_SIZE - 1} * x_0
        debug_assert!(r.len() == 2 * WORD_SIZE);

        let sign_bit = r[0];
        let ones: u64 = (1 << WORD_SIZE) - 1;
        F::from_u64(ones).mul_u128_mont_form(sign_bit)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for MovsignTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one] = suffixes.try_into().unwrap();
        let ones: u64 = (1 << WORD_SIZE) - 1;
        F::from_u64(ones) * prefixes[Prefixes::LeftOperandMsb] * one
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::MovsignTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, MovsignTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, MovsignTable<32>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, MovsignTable<32>>();
    }
}
