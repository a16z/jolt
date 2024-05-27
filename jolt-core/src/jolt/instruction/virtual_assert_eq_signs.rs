use crate::poly::field::JoltField;
use rand::prelude::StdRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::{JoltInstruction, SubtableIndices};
use crate::{
    jolt::subtable::{eq_msb::EqMSBSubtable, LassoSubtable},
    utils::instruction_utils::chunk_and_concatenate_operands,
};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize)]
pub struct EQSIGNSInstruction(pub u64, pub u64);

impl JoltInstruction for EQSIGNSInstruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, self.1)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], _: usize, _: usize) -> F {
        vals[0]
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        1
    }

    fn subtables<F: JoltField>(
        &self,
        _: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![(Box::new(EqMSBSubtable::new()), SubtableIndices::from(0))]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_and_concatenate_operands(self.0, self.1, C, log_M)
    }

    fn lookup_entry(&self) -> u64 {
        ((self.0 as i32).signum() == (self.1 as i32).signum()
            || (self.0 as i32).signum() >= 0 && (self.1 as i32).signum() >= 0)
            .into()
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        Self(rng.next_u32() as u64, rng.next_u32() as u64)
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_chacha::rand_core::RngCore;

    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};

    use super::EQSIGNSInstruction;

    #[test]
    fn slt_instruction_32_e2e() {
        let mut rng = test_rng();
        const C: usize = 4;
        const M: usize = 1 << 16;

        for _ in 0..256 {
            let x = rng.next_u32() as u64;
            let y = rng.next_u32() as u64;
            let instruction = EQSIGNSInstruction(x, y);
            jolt_instruction_test!(instruction);
        }
        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            EQSIGNSInstruction(100, 0),
            EQSIGNSInstruction(0, 100),
            EQSIGNSInstruction(1, 0),
            EQSIGNSInstruction(0, u32_max),
            EQSIGNSInstruction(u32_max, 0),
            EQSIGNSInstruction(u32_max, u32_max),
            EQSIGNSInstruction(u32_max, 1 << 8),
            EQSIGNSInstruction(1 << 8, u32_max),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
