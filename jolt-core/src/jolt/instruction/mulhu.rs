use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::utils::instruction_utils::{
    assert_valid_parameters, concatenate_lookups, multiply_and_chunk_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct MULHUInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for MULHUInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, M: usize) -> F {
        concatenate_lookups(vals, vals.len(), log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        assert_eq!(C * log2(M) as usize, 2 * WORD_SIZE);
        vec![(
            Box::new(IdentitySubtable::new()),
            SubtableIndices::from(0..C / 2),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        multiply_and_chunk_operands(self.0 as u128, self.1 as u128, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        if WORD_SIZE == 32 {
            (self.0).wrapping_mul(self.1) >> 32
        } else if WORD_SIZE == 64 {
            ((self.0 as u128).wrapping_mul(self.1 as u128) >> 64) as u64
        } else {
            panic!("MULHU is only implemented for 32-bit or 64-bit word sizes")
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        if WORD_SIZE == 32 {
            Self(rng.next_u32() as u64, rng.next_u32() as u64)
        } else if WORD_SIZE == 64 {
            Self(rng.next_u64(), rng.next_u64())
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

    use super::MULHUInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn mulhu_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = MULHUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            MULHUInstruction::<WORD_SIZE>(100, 0),
            MULHUInstruction::<WORD_SIZE>(0, 100),
            MULHUInstruction::<WORD_SIZE>(1, 0),
            MULHUInstruction::<WORD_SIZE>(0, u32_max),
            MULHUInstruction::<WORD_SIZE>(u32_max, 0),
            MULHUInstruction::<WORD_SIZE>(u32_max, u32_max),
            MULHUInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            MULHUInstruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn mulhu_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = MULHUInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            MULHUInstruction::<WORD_SIZE>(100, 0),
            MULHUInstruction::<WORD_SIZE>(0, 100),
            MULHUInstruction::<WORD_SIZE>(1, 0),
            MULHUInstruction::<WORD_SIZE>(0, u64_max),
            MULHUInstruction::<WORD_SIZE>(u64_max, 0),
            MULHUInstruction::<WORD_SIZE>(u64_max, u64_max),
            MULHUInstruction::<WORD_SIZE>(u64_max, 1 << 32),
            MULHUInstruction::<WORD_SIZE>(1 << 32, u64_max),
            MULHUInstruction::<WORD_SIZE>(1 << 63, 1),
            MULHUInstruction::<WORD_SIZE>(1, 1 << 63),
            MULHUInstruction::<WORD_SIZE>(u64_max - 1, 1),
            MULHUInstruction::<WORD_SIZE>(1, u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
