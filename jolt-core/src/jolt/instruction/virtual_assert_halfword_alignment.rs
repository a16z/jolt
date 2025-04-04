use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::prefixes::PrefixEval;
use super::suffixes::{SuffixEval, Suffixes};
use super::{JoltInstruction, SubtableIndices};
use crate::jolt::instruction::prefixes::Prefixes;
use crate::subprotocols::sparse_dense_shout::PrefixSuffixDecomposition;
use crate::{
    field::JoltField,
    jolt::subtable::{low_bit::LowBitSubtable, LassoSubtable},
    utils::instruction_utils::{add_and_chunk_operands, assert_valid_parameters},
};

/// (address, offset)
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct AssertHalfwordAlignmentInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for AssertHalfwordAlignmentInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, _: usize) -> F {
        assert_eq!(vals.len(), 1);
        let lowest_bit = vals[0];
        F::one() - lowest_bit
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(
            Box::new(LowBitSubtable::<F>::new()),
            SubtableIndices::from(C - 1),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(self.0 as u128, self.1 as u128, C, log_M)
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
        if WORD_SIZE == 32 {
            ((self.0 as u32 as i32 + self.1 as u32 as i32) % 2 == 0) as u64
        } else if WORD_SIZE == 64 {
            ((self.0 as i64 + self.1 as i64) % 2 == 0) as u64
        } else {
            panic!("Only 32-bit and 64-bit word sizes are supported");
        }
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        (index % 2 == 0).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        match WORD_SIZE {
            #[cfg(test)]
            8 => Self(rng.next_u64() % (1 << 8), rng.next_u64() % (1 << 8)),
            32 => Self(rng.next_u32() as u64, (rng.next_u32() % (1 << 12)) as u64),
            64 => Self(rng.next_u64(), rng.next_u64() % (1 << 12)),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        let lsb = r[r.len() - 1];
        F::one() - lsb
    }
}

impl<const WORD_SIZE: usize> PrefixSuffixDecomposition<WORD_SIZE>
    for AssertHalfwordAlignmentInstruction<WORD_SIZE>
{
    fn suffixes(&self) -> Vec<Suffixes> {
        vec![Suffixes::One, Suffixes::Lsb]
    }

    fn combine<F: JoltField>(&self, prefixes: &[PrefixEval<F>], suffixes: &[SuffixEval<F>]) -> F {
        debug_assert_eq!(self.suffixes().len(), suffixes.len());
        let [one, lsb] = suffixes.try_into().unwrap();
        one - prefixes[Prefixes::Lsb] * lsb
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

    use super::AssertHalfwordAlignmentInstruction;

    #[test]
    fn assert_halfword_alignment_materialize_entry() {
        materialize_entry_test::<Fr, AssertHalfwordAlignmentInstruction<32>>();
    }

    #[test]
    fn assert_halford_alignment_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, AssertHalfwordAlignmentInstruction<8>>();
    }

    #[test]
    fn assert_halford_alignment_mle_random() {
        instruction_mle_random_test::<Fr, AssertHalfwordAlignmentInstruction<32>>();
    }

    #[test]
    fn assert_halfword_alignment_prefix_suffix() {
        prefix_suffix_test::<Fr, AssertHalfwordAlignmentInstruction<32>>();
    }

    #[test]
    fn assert_halfword_alignment_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let x = rng.next_u64();
            let imm = rng.next_u64() as i64 % (1 << 12);
            let instruction = AssertHalfwordAlignmentInstruction::<WORD_SIZE>(x, imm as u64);
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn assert_halfword_alignment_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        for _ in 0..256 {
            let x = rng.next_u64();
            let imm = rng.next_u64() as i64 % (1 << 12);
            let instruction = AssertHalfwordAlignmentInstruction::<WORD_SIZE>(x, imm as u64);
            jolt_instruction_test!(instruction);
        }
    }
}
