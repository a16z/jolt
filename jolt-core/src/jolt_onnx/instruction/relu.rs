//! ReLU instruction for Jolt ONNX

use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::jolt_onnx::subtable::is_pos::IsPosSubtable;
use crate::utils::instruction_utils::{chunk_operand_usize, concatenate_lookups};
use ark_std::log2;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

/// Represents a ReLU instruction in Jolt ONNX
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReLUInstruction(pub u64);

impl JoltInstruction for ReLUInstruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // The output is the ReLU(most significant chunk) || identity of other chunks
        vals[0] * concatenate_lookups(&vals[1..vals.len()], C, log2(M) as usize)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        2
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (
                Box::new(IsPosSubtable::<F>::new()),
                SubtableIndices::from(0),
            ),
            (
                Box::new(IdentitySubtable::new()),
                SubtableIndices::from(0..C),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    // TODO: make sure it works with u32
    fn lookup_entry(&self) -> u64 {
        // check if msb is 1
        if self.0 & (1 << 63) != 0 {
            // if so, return 0
            0
        } else {
            // otherwise, return x
            self.0
        }
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand::RngCore;
        Self(rng.next_u32() as u64)
    }

    fn materialize_entry(&self, _: u64) -> u64 {
        todo!()
    }

    fn evaluate_mle<F>(&self, point: &[F]) -> F
    where
        F: JoltField,
    {
        let mut result = F::zero();
        for i in 0..point.len() - 1 {
            result += F::from_u64(1u64 << i) * point[point.len() - 1 - i];
        }
        result *= F::one() - point[0];
        result
    }
}

#[cfg(test)]
mod test {
    use super::ReLUInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};
    use ark_bn254::Fr;
    use ark_std::rand::RngCore;
    use ark_std::test_rng;

    #[test]
    fn relu_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 8;

        for i in 0..256 {
            let instruction = ReLUInstruction(i as u64);
            jolt_instruction_test!(instruction);
        }

        for _ in 0..256 {
            let x = rng.next_u64();
            let instruction = ReLUInstruction(x);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            ReLUInstruction(0),
            ReLUInstruction(1),
            ReLUInstruction(8374),
            ReLUInstruction((-100_i64) as u64),
            ReLUInstruction((-1_i64) as u64),
            ReLUInstruction(u32_max),
            ReLUInstruction(u32_max + 100),
            ReLUInstruction(u32_max + (1 << 8)),
            ReLUInstruction(1 << 8),
            ReLUInstruction(1 << 30),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
