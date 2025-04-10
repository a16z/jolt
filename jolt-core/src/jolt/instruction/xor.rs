use crate::field::JoltField;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::uninterleave_bits;
use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::JoltInstruction;
use crate::jolt::instruction::SubtableIndices;
use crate::jolt::subtable::{xor::XorSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct XORInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for XORInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(XorSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
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
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{
        jolt::instruction::{
            test::{
                instruction_mle_full_hypercube_test, instruction_mle_random_test,
                materialize_entry_test, prefix_suffix_test,
            },
            JoltInstruction,
        },
        jolt_instruction_test,
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

    #[test]
    fn xor_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = XORInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            XORInstruction::<WORD_SIZE>(100, 0),
            XORInstruction::<WORD_SIZE>(0, 100),
            XORInstruction::<WORD_SIZE>(1, 0),
            XORInstruction::<WORD_SIZE>(0, u32_max),
            XORInstruction::<WORD_SIZE>(u32_max, 0),
            XORInstruction::<WORD_SIZE>(u32_max, u32_max),
            XORInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            XORInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn xor_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = XORInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            XORInstruction::<WORD_SIZE>(100, 0),
            XORInstruction::<WORD_SIZE>(0, 100),
            XORInstruction::<WORD_SIZE>(1, 0),
            XORInstruction::<WORD_SIZE>(0, u64_max),
            XORInstruction::<WORD_SIZE>(u64_max, 0),
            XORInstruction::<WORD_SIZE>(u64_max, u64_max),
            XORInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            XORInstruction::<WORD_SIZE>(1 << 8, u64_max),
            XORInstruction::<WORD_SIZE>(u64_max, 1 << (32 - 1)),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
