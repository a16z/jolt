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

    #[cfg(test)]
    fn materialize(&self) -> Vec<u64> {
        assert_eq!(WORD_SIZE, 8);
        (0..1 << 16).map(|i| self.materialize_entry(i)).collect()
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        index >> WORD_SIZE
    }

    fn to_lookup_index(&self) -> u64 {
        match WORD_SIZE {
            #[cfg(test)]
            8 => self.0 * self.1,
            32 => self.0 * self.1,
            // 64 => (self.0 as u128) + (self.1 as u128),
            _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        }
    }

    fn lookup_entry(&self) -> u64 {
        self.materialize_entry(self.to_lookup_index())
        // match WORD_SIZE {
        //     #[cfg(test)]
        //     8 => (self.0).wrapping_mul(self.1) >> 8,
        //     32 => (self.0).wrapping_mul(self.1) >> 32,
        //     64 => ((self.0 as u128).wrapping_mul(self.1 as u128) >> 64) as u64,
        //     _ => panic!("{WORD_SIZE}-bit word size is unsupported"),
        // }
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
    fn multiplicative_update<F: JoltField>(&self, _: F, _: usize, _: bool) -> F {
        F::one()
    }

    // a_\ell(r_j, j, b_j)
    fn additive_update<F: JoltField>(&self, r_j: F, j: usize, b_j: bool) -> F {
        if j >= WORD_SIZE {
            return F::zero();
        }
        let d_j = F::from_u32(1 << (WORD_SIZE - 1 - j));
        // (r_j - b_j) * d_j
        if b_j {
            r_j * d_j - d_j
        } else {
            r_j * d_j
        }
    }

    fn evaluate_mle<F: JoltField>(&self, r: &[F]) -> F {
        debug_assert_eq!(r.len(), 2 * WORD_SIZE);
        let mut result = F::zero();
        for i in 0..WORD_SIZE {
            result += F::from_u64(1 << (WORD_SIZE - 1 - i)) * r[i];
        }
        result
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
    fn mulhu_materialize() {
        for (i, entry) in MULHUInstruction::<8>::default()
            .materialize()
            .iter()
            .enumerate()
        {
            assert_eq!(
                *entry,
                MULHUInstruction::<8>::default().materialize_entry(i as u64)
            );
        }
    }

    #[test]
    fn mulhu_index_entry() {
        for (i, entry) in MULHUInstruction::<8>::default()
            .materialize()
            .iter()
            .enumerate()
        {
            assert_eq!(
                *entry,
                MULHUInstruction::<8>::default().materialize_entry(i as u64)
            );
        }
    }

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
