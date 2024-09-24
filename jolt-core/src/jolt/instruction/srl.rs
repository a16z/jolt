use crate::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{srl::SrlSubtable, LassoSubtable};
use crate::utils::instruction_utils::{assert_valid_parameters, chunk_and_concatenate_for_shift};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SRLInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SRLInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, _: usize) -> F {
        assert!(C <= 10);
        assert!(vals.len() == C);
        vals.iter().sum()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // We have to pre-define subtables in this way because `CHUNK_INDEX` needs to be a constant,
        // i.e. known at compile time (so we cannot do a `map` over the range of `C`,
        // which only happens at runtime).
        let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
            Box::new(SrlSubtable::<F, 0, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 1, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 2, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 3, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 4, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 5, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 6, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 7, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 8, WORD_SIZE>::new()),
            Box::new(SrlSubtable::<F, 9, WORD_SIZE>::new()),
        ];
        subtables.truncate(C);
        subtables.reverse();

        let indices = (0..C).map(SubtableIndices::from);
        subtables.into_iter().zip(indices).collect()
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        if WORD_SIZE == 32 {
            let x = self.0 as u32;
            let y = (self.1 % 32) as u32;
            (x.wrapping_shr(y)).into()
        } else if WORD_SIZE == 64 {
            let x = self.0;
            let y = (self.1 % 64) as u32;
            x.wrapping_shr(y)
        } else {
            panic!("SRL is only implemented for 32-bit or 64-bit word sizes")
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

    use super::SRLInstruction;

    #[test]
    fn srl_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32(), rng.next_u32());
            let instruction = SRLInstruction::<WORD_SIZE>(x as u64, y as u64);
            jolt_instruction_test!(instruction);
        }
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SRLInstruction::<32>(100, 0),
            SRLInstruction::<32>(0, 100),
            SRLInstruction::<32>(1, 0),
            SRLInstruction::<32>(0, u32_max),
            SRLInstruction::<32>(u32_max, 0),
            SRLInstruction::<32>(u32_max, u32_max),
            SRLInstruction::<32>(u32_max, 1 << 8),
            SRLInstruction::<32>(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    fn srl_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = SRLInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            SRLInstruction::<64>(100, 0),
            SRLInstruction::<64>(0, 2),
            SRLInstruction::<64>(1, 2),
            SRLInstruction::<64>(0, 64),
            SRLInstruction::<64>(u64_max, 0),
            SRLInstruction::<64>(u64_max, 63),
            SRLInstruction::<64>(u64_max, 1 << 8),
            SRLInstruction::<64>(1 << 32, 1 << 16),
            SRLInstruction::<64>(1 << 63, 1),
            SRLInstruction::<64>((1 << 63) - 1, 1),
        ];

        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
