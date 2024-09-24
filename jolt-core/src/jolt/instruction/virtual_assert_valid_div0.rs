use crate::{
    field::JoltField,
    jolt::subtable::{div_by_zero::DivByZeroSubtable, left_is_zero::LeftIsZeroSubtable},
};
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::JoltInstruction;
use crate::{
    jolt::{instruction::SubtableIndices, subtable::LassoSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
/// (divisor, quotient)
pub struct AssertValidDiv0Instruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for AssertValidDiv0Instruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        let vals_by_subtable = self.slice_values(vals, C, M);
        let divisor_is_zero: F = vals_by_subtable[0].iter().product();
        let is_valid_div_by_zero: F = vals_by_subtable[1].iter().product();

        F::one() - divisor_is_zero + is_valid_div_by_zero
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (
                Box::new(LeftIsZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
            (
                Box::new(DivByZeroSubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        let divisor = self.0;
        let quotient = self.1;
        if divisor == 0 {
            match WORD_SIZE {
                32 => (quotient == u32::MAX as u64).into(),
                64 => (quotient == u64::MAX).into(),
                _ => panic!("Unsupported WORD_SIZE: {}", WORD_SIZE),
            }
        } else {
            1
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

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::AssertValidDiv0Instruction;

    #[test]
    fn assert_valid_div0_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = AssertValidDiv0Instruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            jolt_instruction_test!(AssertValidDiv0Instruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            AssertValidDiv0Instruction::<WORD_SIZE>(100, 0),
            AssertValidDiv0Instruction::<WORD_SIZE>(0, 100),
            AssertValidDiv0Instruction::<WORD_SIZE>(1, 0),
            AssertValidDiv0Instruction::<WORD_SIZE>(0, u32_max),
            AssertValidDiv0Instruction::<WORD_SIZE>(u32_max, 0),
            AssertValidDiv0Instruction::<WORD_SIZE>(u32_max, u32_max),
            AssertValidDiv0Instruction::<WORD_SIZE>(u32_max, 1 << 8),
            AssertValidDiv0Instruction::<WORD_SIZE>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn assert_valid_div0_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = AssertValidDiv0Instruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // x == y
        for _ in 0..256 {
            let x = rng.next_u64();
            jolt_instruction_test!(AssertValidDiv0Instruction::<WORD_SIZE>(x, x));
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            AssertValidDiv0Instruction::<WORD_SIZE>(100, 0),
            AssertValidDiv0Instruction::<WORD_SIZE>(0, 100),
            AssertValidDiv0Instruction::<WORD_SIZE>(1, 0),
            AssertValidDiv0Instruction::<WORD_SIZE>(0, u64_max),
            AssertValidDiv0Instruction::<WORD_SIZE>(u64_max, 0),
            AssertValidDiv0Instruction::<WORD_SIZE>(u64_max, u64_max),
            AssertValidDiv0Instruction::<WORD_SIZE>(u64_max, 1 << 8),
            AssertValidDiv0Instruction::<WORD_SIZE>(1 << 8, u64_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
