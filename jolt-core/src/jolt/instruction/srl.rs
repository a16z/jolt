use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::JoltInstruction;
use crate::jolt::subtable::{srl::SrlSubtable, LassoSubtable};
use crate::utils::instruction_utils::{assert_valid_parameters, chunk_and_concatenate_for_shift};

#[derive(Copy, Clone, Default, Debug)]
pub struct SRLInstruction<const WORD_SIZE: usize>(pub u64, pub u64);

impl<const WORD_SIZE: usize> JoltInstruction for SRLInstruction<WORD_SIZE> {
    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        assert!(C <= 10);
        assert!(vals.len() == C * C);

        let mut subtable_vals = vals.chunks_exact(C);
        let mut vals_filtered: Vec<F> = Vec::with_capacity(C);
        for i in 0..C {
            let subtable_val = subtable_vals.next().unwrap();
            vals_filtered.extend_from_slice(&subtable_val[i..i + 1]);
        }

        vals_filtered.iter().sum()
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(&self, C: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
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
        subtables
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        assert_valid_parameters(WORD_SIZE, C, log_M);
        chunk_and_concatenate_for_shift(self.0, self.1, C, log_M)
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

    use super::SRLInstruction;

    #[test]
    fn srl_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 3;
        const M: usize = 1 << 22;
        const WORD_SIZE: usize = 32;

        for _ in 0..8 {
            let (x, y) = (rng.next_u32(), rng.next_u32());

            let entry: u64 = x.checked_shr(y % WORD_SIZE as u32).unwrap_or(0) as u64;

            jolt_instruction_test!(
                SRLInstruction::<WORD_SIZE>(x as u64, y as u64),
                entry.into()
            );
            assert_eq!(
                SRLInstruction::<WORD_SIZE>(x as u64, y as u64).lookup_entry::<Fr>(C, M),
                entry.into()
            );
        }
    }
}
