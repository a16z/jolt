use ark_ff::PrimeField;
use ark_std::log2;
use fixedbitset::FixedBitSet;
use rand::prelude::StdRng;

use super::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{
    identity::IdentitySubtable, truncate_overflow::TruncateOverflowSubtable, LassoSubtable,
};
use crate::utils::instruction_utils::{
    add_and_chunk_operands, assert_valid_parameters, concatenate_lookups,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct ADDInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

pub type ADD32Instruction = ADDInstruction<32>;
pub type ADD64Instruction = ADDInstruction<64>;

impl<const WORD_SIZE: usize> JoltInstruction for ADDInstruction<WORD_SIZE> {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // The first C are from IDEN and the last C are from LOWER9
        assert!(vals.len() == 2 * C);

        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;

        let mut vals_by_subtable = vals.chunks_exact(C);
        let identity = vals_by_subtable.next().unwrap();
        let truncate_overflow = vals_by_subtable.next().unwrap();

        // The output is the LOWER9(most significant chunk) || IDEN of other chunks
        concatenate_lookups(
            [
                &truncate_overflow[0..=msb_chunk_index],
                &identity[msb_chunk_index + 1..C],
            ]
            .concat()
            .as_slice(),
            C,
            log2(M) as usize,
        )
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(
        &self,
        C: usize,
        M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        let msb_chunk_index = C - (WORD_SIZE / log2(M) as usize) - 1;
        vec![
            (
                Box::new(IdentitySubtable::new()),
                SubtableIndices::from(0..msb_chunk_index + 1),
            ),
            (
                Box::new(TruncateOverflowSubtable::<F, WORD_SIZE>::new()),
                SubtableIndices::from(msb_chunk_index + 1..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        add_and_chunk_operands(self.0 as u128, self.1 as u128, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        if WORD_SIZE == 32 {
            (self.0 as u32).overflowing_add(self.1 as u32).0.into()
        } else if WORD_SIZE == 64 {
            self.0.overflowing_add(self.1).0
        } else {
            panic!("only implemented for u32 / u64")
        }
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

    use super::ADDInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    #[test]
    fn add_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = ADDInstruction::<32>(x, y);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ADDInstruction::<32>(100, 0),
            ADDInstruction::<32>(0, 100),
            ADDInstruction::<32>(1, 0),
            ADDInstruction::<32>(0, u32_max),
            ADDInstruction::<32>(u32_max, 0),
            ADDInstruction::<32>(u32_max, u32_max),
            ADDInstruction::<32>(u32_max, 1 << 8),
            ADDInstruction::<32>(1 << 8, u32_max),
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

        for _ in 0..256 {
            let (x, y) = (rng.next_u32() as u64, rng.next_u32() as u64);
            let instruction = ADDInstruction::<64>(x, y);
            jolt_instruction_test!(instruction);
        }
    }
}
