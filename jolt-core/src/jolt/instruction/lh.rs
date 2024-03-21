use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{
    sign_extend::SignExtendByteSubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::utils::instruction_utils::chunk_and_concatenate_operands;

#[derive(Copy, Clone, Default, Debug)]
pub struct LHInstruction(pub u64, pub u64);

impl JoltInstruction for LHInstruction {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 3);

        let half = vals[0] * F::from_u64(1 << 8).unwrap() + vals[1];
        let sign_extension = vals[2];

        let mut result = half;
        for i in 2..C {
            result += F::from_u64(1 << (8 * i)).unwrap() * sign_extension;
        }
        result
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need one TruncateOverflowSubtable
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![
            (
                // Truncate all but the lowest eight bits of the last two chunks,
                // which contains the lower 16 bits of the loaded value.
                Box::new(TruncateOverflowSubtable::<F, 8>::new()),
                SubtableIndices::from(C - 2..C),
            ),
            (
                // Sign extend the lowest 16 bits of the loaded value,
                // Which will be in the second-to-last chunk.
                Box::new(SignExtendByteSubtable::<F>::new()),
                SubtableIndices::from(C - 2),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Sign-extend lower 16 bits of the loaded value
        (self.1 & 0xffff) as i16 as i32 as u32 as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand_core::RngCore;
        Self(rng.next_u32() as u64, rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use super::LHInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn lh_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = LHInstruction(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            LHInstruction(100, 0),
            LHInstruction(0, 100),
            LHInstruction(1, 0),
            LHInstruction(0, u32_max),
            LHInstruction(u32_max, 0),
            LHInstruction(u32_max, u32_max),
            LHInstruction(u32_max, 1 << 8),
            LHInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
