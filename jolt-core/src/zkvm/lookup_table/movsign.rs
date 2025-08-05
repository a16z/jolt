use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MovsignTable<const WORD_SIZE: usize>;

impl<const WORD_SIZE: usize> JoltLookupTable for MovsignTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u128) -> u64 {
        let sign_bit_pos = 2 * WORD_SIZE - 1;
        let sign_bit = 1u128 << sign_bit_pos;
        if index & sign_bit != 0 {
            if WORD_SIZE == 64 {
                u64::MAX
            } else {
                (1u64 << WORD_SIZE) - 1
            }
        } else {
            0
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        // 2 ^ {WORD_SIZE - 1} * x_0
        debug_assert!(r.len() == 2 * WORD_SIZE);

        let sign_bit = r[0];
        let ones: u64 = ((1u128 << WORD_SIZE) - 1) as u64;
        sign_bit * F::from_u64(ones)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for MovsignTable<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one] = suffixes.try_into().unwrap();
        let ones: u64 = ((1u128 << WORD_SIZE) - 1) as u64;
        F::from_u64(ones) * prefixes[Prefixes::LeftOperandMsb] * one
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::zkvm::lookup_table::test::{
        lookup_table_mle_full_hypercube_test, lookup_table_mle_random_test, prefix_suffix_test,
    };
    use common::constants::XLEN;

    use super::MovsignTable;

    #[test]
    fn mle_full_hypercube() {
        lookup_table_mle_full_hypercube_test::<Fr, MovsignTable<8>>();
    }

    #[test]
    fn mle_random() {
        lookup_table_mle_random_test::<Fr, MovsignTable<XLEN>>();
    }

    #[test]
    fn prefix_suffix() {
        prefix_suffix_test::<XLEN, Fr, MovsignTable<XLEN>>();
    }
}
