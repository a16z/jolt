use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
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
            Box::new(LowBitSubtable::<F, 0>::new()),
            SubtableIndices::from(C - 1),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(self.0 as u128, self.1 as u128, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        if WORD_SIZE == 32 {
            ((self.0 as u32 as i32 + self.1 as u32 as i32) % 2 as i32 == 0) as u64
        } else if WORD_SIZE == 64 {
            ((self.0 as i64 + self.1 as i64) % 2 as i64 == 0) as u64
        } else {
            panic!("Only 32-bit and 64-bit word sizes are supported");
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        if WORD_SIZE == 32 {
            Self(rng.next_u32() as u64, (rng.next_u32() % (1 << 12)) as u64)
        } else if WORD_SIZE == 64 {
            Self(rng.next_u64(), rng.next_u64() % (1 << 12))
        } else {
            panic!("Only 32-bit and 64-bit word sizes are supported");
        }
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::AssertHalfwordAlignmentInstruction;

    #[test]
    fn assert_halfword_alignment_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // ALIGN = 2
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

        // ALIGN = 2
        for _ in 0..256 {
            let x = rng.next_u64();
            let imm = rng.next_u64() as i64 % (1 << 12);
            let instruction = AssertHalfwordAlignmentInstruction::<WORD_SIZE>(x, imm as u64);
            jolt_instruction_test!(instruction);
        }
    }
}
