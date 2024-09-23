use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::JoltInstruction;
use crate::{
    field::JoltField,
    jolt::{
        instruction::SubtableIndices,
        subtable::{identity::IdentitySubtable, LassoSubtable},
    },
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

    fn lookup_entry(&self) -> u64 {
        // Same for both 32-bit and 64-bit word sizes
        self.0
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        if WORD_SIZE == 32 {
            Self(rng.next_u32() as u64)
        } else if WORD_SIZE == 64 {
            Self(rng.next_u64())
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

    use super::MOVEInstruction;

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
