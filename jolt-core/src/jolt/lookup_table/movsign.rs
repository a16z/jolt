use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltLookupTable;
use super::PrefixSuffixDecomposition;
use crate::field::JoltField;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MovsignTable<const WORD_SIZE: usize>;

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

impl<const WORD_SIZE: usize> JoltLookupTable for MovsignTable<WORD_SIZE> {
    fn materialize_entry(&self, index: u64) -> u64 {
        let sign_bit = 1 << (2 * WORD_SIZE - 1);
        if index & sign_bit != 0 {
            (1 << WORD_SIZE) - 1
        } else {
            0
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        // 2 ^ {WORD_SIZE - 1} * x_0
        debug_assert!(r.len() == 2 * WORD_SIZE);

        let sign_bit = r[0];
        let ones: u64 = (1 << WORD_SIZE) - 1;
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
        let ones: u64 = (1 << WORD_SIZE) - 1;
        F::from_u64(ones) * prefixes[Prefixes::LeftOperandMsb] * one
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::lookup_table::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::MovsignTable;

    #[test]
    fn virtual_movsign_materialize_entry() {
        materialize_entry_test::<Fr, MovsignTable<32>>();
    }

    #[test]
    fn virtual_movsign_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, MovsignTable<8>>();
    }

    #[test]
    fn virtual_movsign_mle_random() {
        instruction_mle_random_test::<Fr, MovsignTable<32>>();
    }

    #[test]
    fn virtual_movsign_prefix_suffix() {
        prefix_suffix_test::<Fr, MovsignTable<32>>();
    }
}
