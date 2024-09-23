use crate::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::utils::instruction_utils::chunk_operand_usize;

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SWInstruction<const WORD_SIZE: usize>(pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SWInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (0, self.0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _C: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        // Only concatenate the first two lookup results
        vals[0] * F::from_u64(M as u64).unwrap() + vals[1]
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need two IdentitySubtables
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![
            (
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(C - 2..C),
            ),
            // quang : disabling this right now since it may add overhead for 32-bit word sizes
            // (
            //     // Not used for lookup, but this implicitly range-checks
            //     // the remaining query chunks (only relevant for 64-bit word sizes)
            //     Box::new(IdentitySubtable::<F>::new()),
            //     SubtableIndices::from(0..C - 2),
            // ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Lower 32 bits of the rs2 value, no sign extension
        // Same for both 32-bit and 64-bit word sizes
        self.0 & 0xffffffff
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

    use super::SWInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn sw_instruction_32_e2e() {
        let mut rng = test_rng();
        // This works for any `C >= 2`
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = SWInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SWInstruction::<WORD_SIZE>(0),
            SWInstruction::<WORD_SIZE>(1),
            SWInstruction::<WORD_SIZE>(100),
            SWInstruction::<WORD_SIZE>(1 << 8),
            SWInstruction::<WORD_SIZE>(u32_max),
            SWInstruction::<WORD_SIZE>(u32_max - 2),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn sw_instruction_64_e2e() {
        let mut rng = test_rng();
        // This works for any `C >= 2`
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = SWInstruction::<WORD_SIZE>(x);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            SWInstruction::<WORD_SIZE>(0),
            SWInstruction::<WORD_SIZE>(1),
            SWInstruction::<WORD_SIZE>(100),
            SWInstruction::<WORD_SIZE>(1 << 8),
            SWInstruction::<WORD_SIZE>(1 << 40),
            SWInstruction::<WORD_SIZE>(u64_max),
            SWInstruction::<WORD_SIZE>(u64_max - 1),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
