use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{or::OrSubtable, LassoSubtable};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};
use crate::utils::{interleave_bits, uninterleave_bits};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ORInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ORInstruction<WORD_SIZE> {
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
        vec![(Box::new(OrSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        (x | y) as u64
    }

    fn to_lookup_index(&self) -> u64 {
        interleave_bits(self.0 as u32, self.1 as u32)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for both 32-bit and 64-bit word sizes
        self.0 | self.1
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
            result += F::from_u64(1u64 << (WORD_SIZE - 1 - i)) * (x_i + y_i - x_i * y_i);
        }
        result
    }
}

impl<const WORD_SIZE: usize, F: JoltField> PrefixSuffixDecomposition<WORD_SIZE, F>
    for ORInstruction<WORD_SIZE>
{
    fn prefixes() -> Vec<Prefixes> {
        vec![Prefixes::Or]
    }

    fn suffixes() -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Or]
    }

    fn combine(prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        prefixes[Prefixes::Or] * suffixes[Suffixes::One] + suffixes[Suffixes::Or]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{
        instruction_mle_test_large, instruction_mle_test_small,
        jolt::instruction::{test::prefix_suffix_test, JoltInstruction},
        jolt_instruction_test,
    };

    use super::ORInstruction;

    #[test]
    fn or_prefix_suffix() {
        prefix_suffix_test::<Fr, ORInstruction<32>>();
    }

    instruction_mle_test_small!(or_mle_small, ORInstruction<8>);
    instruction_mle_test_large!(or_mle_large, ORInstruction<32>);

    #[test]
    fn or_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let y = rng.next_u32() as u64;
            let instruction = ORInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ORInstruction::<WORD_SIZE>(100, 0),
            ORInstruction::<WORD_SIZE>(0, 100),
            ORInstruction::<WORD_SIZE>(1, 0),
            ORInstruction::<WORD_SIZE>(0, u32_max),
            ORInstruction::<WORD_SIZE>(u32_max, 0),
            ORInstruction::<WORD_SIZE>(u32_max, u32_max),
            ORInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            ORInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn or_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let y = rng.next_u64();
            let instruction = ORInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ORInstruction::<WORD_SIZE>(100, 0),
            ORInstruction::<WORD_SIZE>(0, 100),
            ORInstruction::<WORD_SIZE>(1, 0),
            ORInstruction::<WORD_SIZE>(0, u64_max),
            ORInstruction::<WORD_SIZE>(u64_max, 0),
            ORInstruction::<WORD_SIZE>(u64_max, u64_max),
            ORInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            ORInstruction::<WORD_SIZE>(1 << 8, u64_max),
            ORInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
