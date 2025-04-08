use crate::field::JoltField;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SUBInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SUBInstruction<WORD_SIZE> {
    fn to_lookup_index(&self) -> u64 {
        let x = self.0 as u128;
        let y = (1u128 << WORD_SIZE) - self.1 as u128;
        (x + y) as u64
    }

    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        index % (1 << WORD_SIZE)
    }

    fn lookup_entry(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8).overflowing_sub(self.1 as u8).0.into(),
            32 => (self.0 as u32).overflowing_sub(self.1 as u32).0.into(),
            64 => self.0.overflowing_sub(self.1).0,
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64, rng.next_u32() as u64),
            64 => Self(rng.next_u64(), rng.next_u64()),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
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

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for SUBInstruction<WORD_SIZE> {
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

    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::SUBInstruction;

    #[test]
    fn sub_prefix_suffix() {
        prefix_suffix_test::<Fr, SUBInstruction<32>>();
    }

    #[test]
    fn sub_materialize_entry() {
        materialize_entry_test::<Fr, SUBInstruction<32>>();
    }

    #[test]
    fn sub_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, SUBInstruction<8>>();
    }

    #[test]
    fn sub_mle_random() {
        instruction_mle_random_test::<Fr, SUBInstruction<32>>();
    }
}
