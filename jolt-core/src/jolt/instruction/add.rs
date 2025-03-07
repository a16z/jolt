use ark_std::log2;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::field::JoltField;
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
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

    fn eta(&self) -> usize {
        1
    }

    fn subtable_entry(&self, _: usize, index: u64) -> u64 {
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

    // m_\ell(r_j, j, b_j)
    fn multiplicative_update<F: JoltField>(
        &self,
        l: usize,
        j: usize,
        r_j: F,
        b_j: u8,
        r_prev: Option<F>,
        b_next: Option<u8>,
    ) -> F {
        F::one()
    }

    // a_\ell(r_j, j, b_j)
    fn additive_update<F: JoltField>(
        &self,
        l: usize,
        j: usize,
        r_j: F,
        b_j: u8,
        r_prev: Option<F>,
        b_next: Option<u8>,
    ) -> F {
        if j < WORD_SIZE {
            return F::zero();
        }
        let d_j = F::from_u32(1 << (2 * WORD_SIZE - 1 - j));
        // (r_j - b_j) * d_j
        if b_j == 1 {
            r_j * d_j - d_j
        } else {
            debug_assert_eq!(b_j, 0);
            r_j * d_j
        }
    }

    fn subtable_mle<F: JoltField>(&self, _: usize, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            result += F::from_u64(1 << (WORD_SIZE - 1 - i)) * r[WORD_SIZE + i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::ADDInstruction;
    use crate::{
        instruction_mle_test_large, instruction_mle_test_small, instruction_update_function_test,
        jolt::instruction::JoltInstruction, jolt_instruction_test,
    };

    instruction_mle_test_small!(add_mle_small, ADDInstruction<8>);
    instruction_mle_test_large!(add_mle_large, ADDInstruction<32>);
    instruction_update_function_test!(add_update_fn, ADDInstruction<32>);

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
