use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SHInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SHInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (0, self.0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vals[0]
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![
            (
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(C - 1),
            ),
            (
                // Not used for lookup, but this implicitly range-checks
                // the remaining query chunks
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(0..C - 1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Lower 16 bits of the rs2 value, no sign extension
        // Same for both 32-bit and 64-bit word sizes
        self.0 & 0xffff
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

    use super::SHInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn sh_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 2;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = SHInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SHInstruction::<WORD_SIZE>(0),
            SHInstruction::<WORD_SIZE>(1),
            SHInstruction::<WORD_SIZE>(100),
            SHInstruction::<WORD_SIZE>(1 << 8),
            SHInstruction::<WORD_SIZE>(u32_max),
            SHInstruction::<WORD_SIZE>(u32_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn sh_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = SHInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            SHInstruction::<WORD_SIZE>(0),
            SHInstruction::<WORD_SIZE>(1),
            SHInstruction::<WORD_SIZE>(100),
            SHInstruction::<WORD_SIZE>(1 << 16),
            SHInstruction::<WORD_SIZE>(1 << 48),
            SHInstruction::<WORD_SIZE>(u64_max),
            SHInstruction::<WORD_SIZE>(u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
