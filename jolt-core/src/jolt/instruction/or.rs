use ark_ff::PrimeField;
use ark_std::log2;
use rand::prelude::StdRng;

use super::JoltInstruction;
use crate::jolt::subtable::{or::OrSubtable, LassoSubtable};
use crate::utils::instruction_utils::{chunk_and_concatenate_operands, concatenate_lookups};

#[derive(Copy, Clone, Default, Debug)]
pub struct ORInstruction(pub u64, pub u64);

impl JoltInstruction for ORInstruction {
    fn operands(&self) -> [u64; 2] {
        [self.0, self.1]
    }

    fn combine_lookups<F: PrimeField>(&self, vals: &[F], C: usize, M: usize) -> F {
        concatenate_lookups(vals, C, log2(M) as usize / 2)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: PrimeField>(&self, _: usize) -> Vec<Box<dyn LassoSubtable<F>>> {
        vec![Box::new(OrSubtable::new())]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry_u64(&self) -> u64 {
        (self.0 | self.1).into()
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

    use super::ORInstruction;

    #[test]
    fn or_instruction_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let (x, y) = (rng.next_u64(), rng.next_u64());
            jolt_instruction_test!(ORInstruction(x, y), (x | y).into());
            assert_eq!(
                ORInstruction(x as u64, y as u64).lookup_entry::<Fr>(C, M),
                (x | y).into()
            );
        }
    }

    use crate::jolt::instruction::test::{lookup_entry_u64_parity_random, lookup_entry_u64_parity};

    #[test]
    fn u64_parity() {
        let concrete_instruction = ORInstruction(0, 0);
        lookup_entry_u64_parity_random::<Fr, ORInstruction>(100, concrete_instruction);

        // Test edge-cases
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ORInstruction(100, 0),
            ORInstruction(0, 100),
            ORInstruction(1 , 0),
            ORInstruction(0, u32_max),
            ORInstruction(u32_max, 0),
            ORInstruction(u32_max, u32_max),
            ORInstruction(u32_max, 1 << 8),
            ORInstruction(1 << 8, u32_max),
        ];
        lookup_entry_u64_parity::<Fr, _>(instructions);
    }
}
