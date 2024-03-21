use ark_ff::PrimeField;
use ark_std::log2;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{truncate_overflow::TruncateOverflowSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};

#[derive(Copy, Clone, Default, Debug)]
pub struct SHInstruction(pub u64, pub u64);

impl JoltInstruction for SHInstruction {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], _: usize, M: usize) -> F {
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        assert!(vals.len() == 2);
        concatenate_lookups(vals, 2, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // This assertion ensures that we only need two TruncateOverflowSubtables
        // TODO(moodlezoup): make this work with different M
        assert!(M == 1 << 16);
        vec![(
            Box::new(TruncateOverflowSubtable::<F, 8>::new()),
            SubtableIndices::from(C - 2..C),
        )]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        // Lower 16 bits of the rs2 value
        self.1 & 0xffff
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

    use super::SHInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn sh_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = SHInstruction(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SHInstruction(100, 0),
            SHInstruction(0, 100),
            SHInstruction(1, 0),
            SHInstruction(0, u32_max),
            SHInstruction(u32_max, 0),
            SHInstruction(u32_max, u32_max),
            SHInstruction(u32_max, 1 << 8),
            SHInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
