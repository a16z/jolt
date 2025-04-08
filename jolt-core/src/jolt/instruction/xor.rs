use crate::field::JoltField;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::uninterleave_bits;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct XORInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for XORInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x ^ y) as u64
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for both 32-bit and 64-bit word sizes
        self.0 ^ self.1
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
            let x_i = r[2 * i];
            let y_i = r[2 * i + 1];
            result += F::from_u64(1u64 << (WORD_SIZE - 1 - i))
                * ((F::one() - x_i) * y_i + x_i * (F::one() - y_i));
        }
        result
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for XORInstruction<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Xor]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, xor] = suffixes.try_into().unwrap();
        prefixes[Prefixes::Xor] * one + xor
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, instruction_mle_random_test, materialize_entry_test,
        prefix_suffix_test,
    };

    use super::XORInstruction;

    #[test]
    fn xor_prefix_suffix() {
        prefix_suffix_test::<Fr, XORInstruction<32>>();
    }

    #[test]
    fn xor_materialize_entry() {
        materialize_entry_test::<Fr, XORInstruction<32>>();
    }

    #[test]
    fn xor_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, XORInstruction<8>>();
    }

    #[test]
    fn xor_mle_random() {
        instruction_mle_random_test::<Fr, XORInstruction<32>>();
    }
}
