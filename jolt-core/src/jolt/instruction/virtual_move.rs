use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{
    prefixes::{PrefixEval, Prefixes},
    suffixes::{SuffixEval, Suffixes},
    JoltInstruction,
};
use crate::{
    field::JoltField,
    jolt::{
        instruction::SubtableIndices,
        subtable::{identity::IdentitySubtable, LassoSubtable},
    },
    subprotocols::sparse_dense_shout::PrefixSuffixDecomposition,
    utils::instruction_utils::{chunk_operand_usize, concatenate_lookups},
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MOVEInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for MOVEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert!(M == 1 << 16);
        vec![(
            // Implicitly range-checks all query chunks
            Box::new(IdentitySubtable::<F>::new()),
            SubtableIndices::from(0..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn lookup_entry(&self) -> u64 {
        // Same for both 32-bit and 64-bit word sizes
        self.0
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        index % (1 << WORD_SIZE)
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64),
            64 => Self(rng.next_u64()),
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

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for MOVEInstruction<WORD_SIZE> {
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

    use super::MOVEInstruction;

    #[test]
    fn virtual_move_materialize_entry() {
        materialize_entry_test::<Fr, MOVEInstruction<32>>();
    }

    #[test]
    fn virtual_move_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, MOVEInstruction<8>>();
    }

    #[test]
    fn virtual_move_mle_random() {
        instruction_mle_random_test::<Fr, MOVEInstruction<32>>();
    }

    #[test]
    fn virtual_move_prefix_suffix() {
        prefix_suffix_test::<Fr, MOVEInstruction<32>>();
    }

    #[test]
    fn virtual_move_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = MOVEInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            MOVEInstruction::<WORD_SIZE>(0),
            MOVEInstruction::<WORD_SIZE>(1),
            MOVEInstruction::<WORD_SIZE>(100),
            MOVEInstruction::<WORD_SIZE>(1 << 16),
            MOVEInstruction::<WORD_SIZE>(u32_max),
            MOVEInstruction::<WORD_SIZE>(u32_max - 10),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn virtual_move_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = MOVEInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            MOVEInstruction::<WORD_SIZE>(0),
            MOVEInstruction::<WORD_SIZE>(1),
            MOVEInstruction::<WORD_SIZE>(100),
            MOVEInstruction::<WORD_SIZE>(1 << 16),
            MOVEInstruction::<WORD_SIZE>(1 << 32),
            MOVEInstruction::<WORD_SIZE>(1 << 48),
            MOVEInstruction::<WORD_SIZE>(u64_max),
            MOVEInstruction::<WORD_SIZE>(u64_max - 2),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
