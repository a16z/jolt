use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::JoltInstruction;
use crate::{
    field::JoltField,
    jolt::{
        instruction::SubtableIndices,
        subtable::{identity::IdentitySubtable, sign_extend::SignExtendSubtable, LassoSubtable},
    },
    utils::instruction_utils::{chunk_operand_usize, concatenate_lookups},
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct MOVSIGNInstruction<const WORD_SIZE: usize>(pub u64);

// Constants for 32-bit and 64-bit word sizes
const ALL_ONES_32: u64 = 0xFFFF_FFFF;
const ALL_ONES_64: u64 = 0xFFFF_FFFF_FFFF_FFFF;
const SIGN_BIT_32: u64 = 0x8000_0000;
const SIGN_BIT_64: u64 = 0x8000_0000_0000_0000;

impl<const WORD_SIZE: usize> JoltInstruction for MOVSIGNInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        let val = vals[0];
        let repeat = WORD_SIZE / 16;
        concatenate_lookups(&vec![val; repeat], repeat, log2(M) as usize)
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
        let msb_chunk_index = C - (WORD_SIZE / 16);
        vec![
            (
                Box::new(SignExtendSubtable::<F, 16>::new()),
                SubtableIndices::from(msb_chunk_index),
            ),
            (
                // Not used for lookup, but this implicitly range-checks
                // the remaining query chunks
                Box::new(IdentitySubtable::<F>::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        match WORD_SIZE {
            32 => {
                if self.0 & SIGN_BIT_32 != 0 {
                    ALL_ONES_32
                } else {
                    0
                }
            }
            64 => {
                if self.0 & SIGN_BIT_64 != 0 {
                    ALL_ONES_64
                } else {
                    0
                }
            }
            _ => panic!("only implemented for u32 / u64"),
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::MOVSIGNInstruction;

    #[test]
    fn virtual_movsign_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        // Random
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let instruction = MOVSIGNInstruction::<32>(x);
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn virtual_movsign_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = MOVSIGNInstruction::<64>(x);
            jolt_instruction_test!(instruction);
        }
    }
}
