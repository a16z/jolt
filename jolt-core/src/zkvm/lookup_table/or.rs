use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::{JoltField, MontU128};
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct OrTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for OrTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x | y) as u64
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[MontU128]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (WORD_SIZE - 1 - i)) * (F::from_u128_mont(x_i) + F::from_u128_mont(y_i) - F::from_u128_mont(x_i) * F::from_u128_mont(y_i));
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for OrTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Or]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, or] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Or] * one + or
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };

    use super::OrTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<Fr, OrTable<32>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, OrTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, OrTable<32>>();
    }
}
