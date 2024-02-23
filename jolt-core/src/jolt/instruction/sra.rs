use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::JoltInstruction;
use crate::jolt::subtable::{sra_sign::SraSignSubtable, srl::SrlSubtable, LassoSubtable};
use crate::utils::instruction_utils::{assert_valid_parameters, chunk_and_concatenate_for_shift};

#[derive(Copy, Clone, Default, Debug)]
pub struct SRAInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SRAInstruction<WORD_SIZE> {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(C <= 10);
        assert!(vals.len() == (C + 1) * C);

        let mut subtable_vals = vals.chunks_exact(C);
        let mut vals_filtered: Vec<F> = Vec::with_capacity(C);
        for i in 0..C {
            let subtable_val = subtable_vals.next().unwrap();
            vals_filtered.extend_from_slice(&subtable_val[i..i + 1]);
        }

        // SRASign subtable applied to the most significant index
        vals_filtered.extend_from_slice(&subtable_vals.next().unwrap()[0..1]);

        vals_filtered.iter().sum()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(&self, C: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
        let mut subtables: Vec<Box<dyn LassoSubtable<F>>> = vec![
            Box::new(SraSignSubtable::<F, WORD_SIZE>::new()),
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
        subtables.truncate(C + 1);
        subtables.reverse();
        subtables
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
    }

    fn lookup_entry_u64(&self) -> u64 {
        // TODO(sragss): Not sure this is correct.
        (self.0 as i32).checked_shr(self.1 as u32).unwrap_or(0) as u64
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

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::SRAInstruction;

    #[test]
    fn sra_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;
        const WORD_SIZE: usize = 32;

        for _ in 0..8 {
            let (x, y) = (rng.next_u32(), rng.next_u32());

            let entry: i32 = (x as i32).checked_shr(y % WORD_SIZE as u32).unwrap_or(0);

            jolt_instruction_test!(
                SRAInstruction::<WORD_SIZE>(x as u64, y as u64),
                (entry as u32).into()
            );
            assert_eq!(
                SRAInstruction::<WORD_SIZE>(x as u64, y as u64).lookup_entry::<Fr>(C, M),
                (entry as u32).into()
            );
        }
    }

    use crate::jolt::instruction::test::{lookup_entry_u64_parity_random, lookup_entry_u64_parity};

    #[test]
    fn u64_parity() {
        let concrete_instruction = SRAInstruction(0, 0);
        lookup_entry_u64_parity_random::<Fr, SRAInstruction::<32>>(100, concrete_instruction);

        // Test edge-cases
        let u32_max: u64 = ((1u64 << 32u64 - 1) as u32) as u64;
        let instructions = vec![
            SRAInstruction::<32>(100, 0),
            SRAInstruction::<32>(0, 100),
            SRAInstruction::<32>(1 , 0),
            SRAInstruction::<32>(0, u32_max),
            SRAInstruction::<32>(u32_max, 0),
            SRAInstruction::<32>(u32_max, u32_max),
            SRAInstruction::<32>(u32_max, 1 << 8),
            SRAInstruction::<32>(1 << 8, u32_max),
        ];
        lookup_entry_u64_parity::<Fr, _>(instructions);
    }
}
