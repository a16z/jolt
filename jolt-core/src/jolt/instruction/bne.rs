use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::{PrefixEval, Prefixes};
use super::suffixes::{SuffixEval, Suffixes};
use super::{beq::BEQInstruction, JoltInstruction};
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{
    field::JoltField,
    jolt::{
        instruction::SubtableIndices,
        subtable::{eq::EqSubtable, LassoSubtable},
    },
    utils::{instruction_utils::chunk_and_concatenate_operands, uninterleave_bits},
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct BNEInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for BNEInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, _: usize) -> F {
        F::one() - vals.iter().product::<F>()
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(EqSubtable::new()), SubtableIndices::from(0..C))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // This is the same for 32-bit and 64-bit word sizes
        (self.0 != self.1).into()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            #[cfg(test)]
            8 => (x != y).into(),
            32 => (x != y).into(),
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
        F::one() - BEQInstruction::<WORD_SIZE>::default().evaluate_mle::<F>(r)
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE> for BNEInstruction<WORD_SIZE> {
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Eq]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, eq] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Eq] * eq
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

    use super::BNEInstruction;

    #[test]
    fn bne_materialize_entry() {
        materialize_entry_test::<Fr, BNEInstruction<32>>();
    }

    #[test]
    fn bne_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, BNEInstruction<8>>();
    }

    #[test]
    fn bne_mle_random() {
        instruction_mle_random_test::<Fr, BNEInstruction<32>>();
    }

    #[test]
    fn bne_prefix_suffix() {
        prefix_suffix_test::<Fr, BNEInstruction<32>>();
    }

    #[test]
    fn bne_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = BNEInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = BNEInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            BNEInstruction::<WORD_SIZE>(100, 0),
            BNEInstruction::<WORD_SIZE>(0, 100),
            BNEInstruction::<WORD_SIZE>(1, 0),
            BNEInstruction::<WORD_SIZE>(0, u32_max),
            BNEInstruction::<WORD_SIZE>(u32_max, 0),
            BNEInstruction::<WORD_SIZE>(u32_max, u32_max),
            BNEInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            BNEInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn bne_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = BNEInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = BNEInstruction::<WORD_SIZE>(x, x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            BNEInstruction::<WORD_SIZE>(100, 0),
            BNEInstruction::<WORD_SIZE>(0, 100),
            BNEInstruction::<WORD_SIZE>(1, 0),
            BNEInstruction::<WORD_SIZE>(0, u64_max),
            BNEInstruction::<WORD_SIZE>(u64_max, 0),
            BNEInstruction::<WORD_SIZE>(u64_max, u64_max),
            BNEInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            BNEInstruction::<WORD_SIZE>(1 << 8, u64_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
