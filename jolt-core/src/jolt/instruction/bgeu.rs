use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    sltu::SLTUInstruction,
    suffixes::SuffixEval,
    JoltInstruction, SubtableIndices,
};
use crate::{
    field::JoltField,
    jolt::{
        instruction::suffixes::Suffixes,
        subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    },
    subprotocols::sparse_dense_shout::PrefixSuffixDecomposition,
    utils::{instruction_utils::chunk_and_concatenate_operands, uninterleave_bits},
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct BGEUInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for BGEUInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // 1 - SLTU(x, y) =
        F::one() - SLTUInstruction::<WORD_SIZE>(self.0, self.1).combine_lookups(vals, C, M)
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(LtuSubtable::new()), SubtableIndices::from(0..C)),
            (Box::new(EqSubtable::new()), SubtableIndices::from(0..C - 1)),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for 32-bit and 64-bit
        (self.0 >= self.1).into()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x >= y).into(),
            32 => (x >= y).into(),
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
        F::one() - SLTUInstruction::<WORD_SIZE>::default().evaluate_mle::<F>(r)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for BGEUInstruction<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::LessThan]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, less_than] = suffixes.try_into().unwrap();
        // 1 - LTU(x, y)
        one - prefixes[Prefixes::LessThan] * one - prefixes[Prefixes::Eq] * less_than
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

    use super::BGEUInstruction;

    #[test]
    fn bgeu_materialize_entry() {
        materialize_entry_test::<Fr, BGEUInstruction<32>>();
    }

    #[test]
    fn bgeu_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, BGEUInstruction<8>>();
    }

    #[test]
    fn bgeu_mle_random() {
        instruction_mle_random_test::<Fr, BGEUInstruction<32>>();
    }

    #[test]
    fn bgeu_prefix_suffix() {
        prefix_suffix_test::<Fr, BGEUInstruction<32>>();
    }

    #[test]
    fn bgeu_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = BGEUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = BGEUInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            BGEUInstruction::<WORD_SIZE>(100, 0),
            BGEUInstruction::<WORD_SIZE>(0, 100),
            BGEUInstruction::<WORD_SIZE>(1, 0),
            BGEUInstruction::<WORD_SIZE>(0, u32_max),
            BGEUInstruction::<WORD_SIZE>(u32_max, 0),
            BGEUInstruction::<WORD_SIZE>(u32_max, u32_max),
            BGEUInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            BGEUInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn bgeu_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = BGEUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Ones
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = BGEUInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge-cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            BGEUInstruction::<WORD_SIZE>(100, 0),
            BGEUInstruction::<WORD_SIZE>(0, 100),
            BGEUInstruction::<WORD_SIZE>(1, 0),
            BGEUInstruction::<WORD_SIZE>(0, u64_max),
            BGEUInstruction::<WORD_SIZE>(u64_max, 0),
            BGEUInstruction::<WORD_SIZE>(u64_max, u64_max),
            BGEUInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            BGEUInstruction::<WORD_SIZE>(1 << 32, u64_max),
            BGEUInstruction::<WORD_SIZE>(1 << 63, 1),
            BGEUInstruction::<WORD_SIZE>(1, 1 << 63),
            BGEUInstruction::<WORD_SIZE>(u64_max - 1, u64_max),
            BGEUInstruction::<WORD_SIZE>(u64_max, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
