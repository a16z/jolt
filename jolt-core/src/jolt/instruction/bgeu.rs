use ark_ff::PrimeField;
use rand::prelude::StdRng;

use super::{sltu::SLTUInstruction, JoltInstruction};
use crate::{
    jolt::subtable::{eq::EqSubtable, ltu::LtuSubtable, LassoSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug)]
pub struct BGEUInstruction(pub u64, pub u64);

impl JoltInstruction for BGEUInstruction {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // 1 - LTU(x, y) =
        F::one() - SLTUInstruction(self.0, self.1).combine_lookups(vals, C, M)
    }

    fn g_poly_degree(&self, C: usize) -> usize {
        C
    }

    fn subtables<F: PrimeField>(&self, _: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
        vec![Box::new(LtuSubtable::new()), Box::new(EqSubtable::new())]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry_u64(&self) -> u64 {
        (self.0 >= self.1).into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand_core::RngCore;
        Self(rng.next_u32() as u64, rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;
    use ark_std::{test_rng, One};
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::BGEUInstruction;

    #[test]
    fn bgeu_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            jolt_instruction_test!(BGEUInstruction(x, y), (x >= y).into());
            assert_eq!(
                BGEUInstruction(x, y).lookup_entry::<Fr>(C, M),
                (x >= y).into()
            );
        }
        for _ in 0..256 {
            let x = rng.next_u64();
            jolt_instruction_test!(BGEUInstruction(x, x), Fr::one());
            assert_eq!(BGEUInstruction(x, x).lookup_entry::<Fr>(C, M), Fr::one());
        }
    }

    use crate::jolt::instruction::test::{lookup_entry_u64_parity_random, lookup_entry_u64_parity};

    #[test]
    fn u64_parity() {
        let concrete_instruction = BGEUInstruction(0, 0);
        lookup_entry_u64_parity_random::<Fr, BGEUInstruction>(100, concrete_instruction);

        // Test edge-cases
        let u32_max: u64 = ((1u64 << 32u64 - 1) as u32) as u64;
        let instructions = vec![
            BGEUInstruction(100, 0),
            BGEUInstruction(0, 100),
            BGEUInstruction(1 , 0),
            BGEUInstruction(0, u32_max),
            BGEUInstruction(u32_max, 0),
            BGEUInstruction(u32_max, u32_max),
            BGEUInstruction(u32_max, 1 << 8),
            BGEUInstruction(1 << 8, u32_max),
        ];
        lookup_entry_u64_parity::<Fr, _>(instructions);
    }
}
