use crate::field::JoltField;
use crate::utils::uninterleave_bits;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{sra_sign::SraSignSubtable, srl::SrlSubtable, LassoSubtable};
use crate::utils::instruction_utils::{assert_valid_parameters, chunk_and_concatenate_for_shift};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SRAInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SRAInstruction<WORD_SIZE> {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, _: usize) -> F {
        assert!(C <= 10);
        assert_eq!(vals.len(), C + 1);
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
        let mut subtables_and_indices: Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> =
            subtables.into_iter().zip(indices).collect();

        subtables_and_indices.push((
            Box::new(SraSignSubtable::<F, WORD_SIZE>::new()),
            SubtableIndices::from(0),
        ));

        subtables_and_indices
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        if WORD_SIZE == 32 {
            let x = self.0 as i32;
            let y = self.1 as u32 % 32;
            (x.wrapping_shr(y) as u32).into()
        } else if WORD_SIZE == 64 {
            let x = self.0 as i64;
            let y = (self.1 % 64) as u32;
            x.wrapping_shr(y) as u64
        } else {
            panic!("SRA is only implemented for 32-bit or 64-bit word sizes")
        }
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let (x, y) = uninterleave_bits(index);
        match WORD_SIZE {
            #[cfg(test)]
            8 => ((x as i8).wrapping_shr(y % 8)) as u8 as u64,
            32 => ((x as i32).wrapping_shr(y % 32)) as u32 as u64,
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

    fn evaluate_mle<F: JoltField>(&self, _: &[F]) -> F {
        todo!("Placeholder; will use virtual sequence when we switch to Shout")
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::SRAInstruction;

    #[test]
    fn sra_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u32(), rng.next_u32());
            let instruction = SRAInstruction::<WORD_SIZE>(x as u64, y as u64);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SRAInstruction::<WORD_SIZE>(100, 0),
            SRAInstruction::<WORD_SIZE>(0, 2),
            SRAInstruction::<WORD_SIZE>(1, 2),
            SRAInstruction::<WORD_SIZE>(0, 32),
            SRAInstruction::<WORD_SIZE>(u32_max, 0),
            SRAInstruction::<WORD_SIZE>(u32_max, 31),
            SRAInstruction::<WORD_SIZE>(u32_max, 1 << 8),
            SRAInstruction::<WORD_SIZE>(1 << 8, 1 << 16),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }

    #[test]
    #[ignore]
    fn sra_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 64;

        // Random
        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            let instruction = SRAInstruction::<WORD_SIZE>(x, y);
            jolt_instruction_test!(instruction);
        }

        // Edge cases
        let u64_max: u64 = u64::MAX;
        let instructions = vec![
            SRAInstruction::<WORD_SIZE>(100, 0),
            SRAInstruction::<WORD_SIZE>(0, 2),
            SRAInstruction::<WORD_SIZE>(1, 2),
            SRAInstruction::<WORD_SIZE>(0, 64),
            SRAInstruction::<WORD_SIZE>(u64_max, 0),
            SRAInstruction::<WORD_SIZE>(u64_max, 63),
            SRAInstruction::<WORD_SIZE>(u64_max, 1 << 8),
            SRAInstruction::<WORD_SIZE>(1 << 32, 1 << 16),
            SRAInstruction::<WORD_SIZE>(1 << 63, 1),
            SRAInstruction::<WORD_SIZE>((1 << 63) - 1, 1),
        ];

        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
