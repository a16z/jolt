use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;
use crate::utils::uninterleave_bits;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AndnTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for AndnTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let (x, y) = uninterleave_bits(index);
        x & !y
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);

        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (WORD_SIZE - 1 - i)) * x_i * (F::one() - y_i);
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for AndnTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::NotAnd]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, andn] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Andn] * one + andn
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::AndnTable;

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, AndnTable<XLEN>>();
    }

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, AndnTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, AndnTable<XLEN>>();
    }
}
