use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{LassoSubtable};
use crate::jolt_onnx::subtable::is_max::IsMaxSubtable;
use crate::jolt_onnx::subtable::is_pos::IsPosSubtable;
use crate::jolt_onnx::subtable::is_zero::IsZeroSubtable;
use crate::jolt_onnx::subtable::sigmoid_neg::SigmoidNegSubtable;
use crate::jolt_onnx::subtable::sigmoid_pos::SigmoidPosSubtable;
use crate::utils::instruction_utils::{chunk_operand_usize};
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]


/// Sigmoid instruction
pub struct SigmoidInstruction(pub u64);

impl JoltInstruction for SigmoidInstruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, _M: usize) -> F {
        let is_pos = vals[0] == F::one();
        let is_small_pos = vals[1..C].iter().all(|x| *x == F::one());
        let is_small_neg = vals[C..2*C - 1].iter().all(|x| *x == F::one());
        if is_pos {
            if is_small_pos {
                vals[2*C - 1]
            } else {
                F::from_u64(u32::MAX as u64)
            }
        } else {
            if is_small_neg {
                vals[2*C]
            } else {
                F::zero()
            }
        }
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        5
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        // TODO: Match M
        vec![
            (
                Box::new(IsPosSubtable::new()),
                SubtableIndices::from(0),
            ),
            (
                Box::new(IsZeroSubtable::new()),
                SubtableIndices::from(0..C-1),
            ),
            (
                Box::new(IsMaxSubtable::new()),
                SubtableIndices::from(0..C-1),
            ),
            (
                Box::new(SigmoidPosSubtable::<F>::new()),
                SubtableIndices::from(C-1),
            ),
            (
                Box::new(SigmoidNegSubtable::<F>::new()),
                SubtableIndices::from(C-1),
            ),
        ]
    }

    fn to_indices(&self, C: usize, log_M: usize) -> Vec<usize> {
        chunk_operand_usize(self.0, C, log_M)
    }

    fn to_lookup_index(&self) -> u64 {
        self.0
    }

    fn lookup_entry(&self) -> u64 {
        let x = self.0 as i64;
        let output = 1.0 / (1.0 + (-(x) as f32).exp());
        let quantized_output = (output * (u32::MAX as f32)) as u32;
        quantized_output as u64
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand::RngCore;
        Self(rng.next_u32() as u64)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let x = index as i64;
        let output = 1.0 / (1.0 + (-(x) as f32).exp());
        let quantized_output = (output * (u32::MAX as f32)) as u32;
        quantized_output as u64
    }

    fn evaluate_mle<F>(&self, point: &[F]) -> F
    where
        F: JoltField,
    {
        let mut f_eval = vec![F::from_u32(4294967295); 1 << point.len()];
        f_eval[0] = F::from_u32(2147483648);
        f_eval[1] = F::from_u32(3139872768);
        f_eval[2] = F::from_u32(3782994432);
        f_eval[3] = F::from_u32(4091274752);
        f_eval[4] = F::from_u32(4217716992);
        f_eval[5] = F::from_u32(4266221824);
        f_eval[6] = F::from_u32(4284347648);
        f_eval[7] = F::from_u32(4291054592);
        f_eval[8] = F::from_u32(4293527040);
        f_eval[9] = F::from_u32(4294437376);
        f_eval[10] = F::from_u32(4294772224);
        f_eval[11] = F::from_u32(4294895616);
        f_eval[12] = F::from_u32(4294940672);
        f_eval[13] = F::from_u32(4294957568);
        f_eval[14] = F::from_u32(4294963712);
        f_eval[15] = F::from_u32(4294965760);
        f_eval[16] = F::from_u32(4294966784);
        let mut idx = F::zero();
        for i in 0..point.len() {
            idx += F::from_u64(1u64 << i) * point[point.len() - 1 - i];
        }
        // TODO: Add boolean sum
        let result = f_eval[idx.to_u64().unwrap() as usize];
        result
    }
}

#[cfg(test)]
mod test {
    use super::SigmoidInstruction;
    use crate::jolt::instruction::test::{instruction_mle_full_hypercube_test, materialize_entry_test};
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};
    use ark_bn254::Fr;
    use ark_std::rand::RngCore;
    use ark_std::test_rng;


    #[test]
    fn sigmoid_mle_full_hypercube() {
        instruction_mle_full_hypercube_test::<Fr, SigmoidInstruction>();
    }

    #[test]
    fn sigmoid_materialize_entry() {
        materialize_entry_test::<Fr, SigmoidInstruction>();
    }

    #[test]
    fn sigmoid_instruction_64_e2e() {
        let mut rng = test_rng();
        const C: usize = 8;
        const M: usize = 1 << 8;

        for i in 0..256 {
            let instruction = SigmoidInstruction(i as u64);
            jolt_instruction_test!(instruction);
        }

        for _ in 0..256 {
            let x = rng.next_u32();
            let instruction = SigmoidInstruction(x as u64);
            jolt_instruction_test!(instruction);
        }

        let u32_max: u64 = u32::MAX as u64;
        let instructions = vec![
            SigmoidInstruction(0),
            SigmoidInstruction(1),
            SigmoidInstruction(8374),
            SigmoidInstruction((-100_i32) as u64),
            SigmoidInstruction((-1_i32) as u64),
            SigmoidInstruction(u32_max),
            SigmoidInstruction(u32_max + 100),
            SigmoidInstruction(u32_max + (1 << 8)),
            SigmoidInstruction(1 << 8),
            SigmoidInstruction(1 << 30),
        ];
        for instruction in instructions {
            jolt_instruction_test!(instruction);
        }
    }
}
