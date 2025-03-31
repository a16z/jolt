use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::utils::instruction_utils::{
    add_and_chunk_operands, assert_valid_parameters, concatenate_lookups,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ADDInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for ADDInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(vals.len() == C / 2);
        // The output is the identity of lower chunks
        concatenate_lookups(vals, C / 2, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(msb_chunk_index + 1..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(self.0 as u128, self.1 as u128, C, log_M)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        index % (1 << WORD_SIZE)
    }

    fn to_lookup_index(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => self.0 + self.1,
            32 => self.0 + self.1,
            // 64 => (self.0 as u128) + (self.1 as u128),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_entry(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => (self.0 as u8).overflowing_add(self.1 as u8).0.into(),
            32 => (self.0 as u32).overflowing_add(self.1 as u32).0.into(),
            64 => self.0.overflowing_add(self.1).0,
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

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for ADDInstruction<WORD_SIZE> {
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

    use super::ADDInstruction;
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

    #[test]
    fn add_prefix_suffix() {
        prefix_suffix_test::<Fr, ADDInstruction<32>>();
    }

    #[test]
    fn add_materialize_entry() {
        materialize_entry_test::<Fr, ADDInstruction<32>>();
    }

    #[test]
    fn add_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, ADDInstruction<8>>();
    }

    #[test]
    fn add_mle_random() {
        instruction_mle_random_test::<Fr, ADDInstruction<32>>();
    }

    #[test]
    fn add_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = ADDInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ADDInstruction::<WORD_SIZE>(100, 0),
            ADDInstruction::<WORD_SIZE>(0, 100),
            ADDInstruction::<WORD_SIZE>(1, 0),
            ADDInstruction::<WORD_SIZE>(0, u32_max),
            ADDInstruction::<WORD_SIZE>(u32_max, 0),
            ADDInstruction::<WORD_SIZE>(u32_max, u32_max),
            ADDInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            ADDInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn add_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = ADDInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            ADDInstruction::<WORD_SIZE>(100, 0),
            ADDInstruction::<WORD_SIZE>(0, 100),
            ADDInstruction::<WORD_SIZE>(1, 0),
            ADDInstruction::<WORD_SIZE>(0, u64_max),
            ADDInstruction::<WORD_SIZE>(u64_max, 0),
            ADDInstruction::<WORD_SIZE>(u64_max, u64_max),
            ADDInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            ADDInstruction::<WORD_SIZE>(1 << 32, u64_max),
            ADDInstruction::<WORD_SIZE>(1 << 63, 1),
            ADDInstruction::<WORD_SIZE>(1, 1 << 63),
            ADDInstruction::<WORD_SIZE>(u64_max - 1, 1),
            ADDInstruction::<WORD_SIZE>(1, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
