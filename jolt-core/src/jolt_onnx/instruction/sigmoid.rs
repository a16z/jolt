use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::{identity::IdentitySubtable, LassoSubtable};
use crate::jolt_onnx::subtable::is_max::IsMaxSubtable;
use crate::jolt_onnx::subtable::is_pos::IsPosSubtable;
use crate::jolt_onnx::subtable::is_zero::IsZeroSubtable;
use crate::jolt_onnx::subtable::sigmoid_neg::SigmoidNegSubtable;
use crate::jolt_onnx::subtable::sigmoid_pos::SigmoidPosSubtable;
use crate::utils::instruction_utils::{chunk_operand_usize, concatenate_lookups};
use ark_std::log2;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]


pub struct SigmoidInstruction(pub u64);

impl JoltInstruction for SigmoidInstruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
        // (self.0 as i32 as u64, 0)
    }

    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, M: usize) -> F {
        // TODO: Match M
        println!("vals: {:?}", vals);
        println!("vals length: {:?}", vals.len());
        println!("C: {:?}", C);
        println!("M: {:?}", M);
        // Is positive?
        let is_pos = vals[0] == F::one();
        println!("is_pos: {:?}", is_pos);
        // Is small (i.e. vals[i] = 0)
        println!("vals[1..C]: {:?}", vals[1..C].to_vec());
        let is_small = vals[1..C].iter().all(|x| *x == F::one());
        println!("is_small: {:?}", is_small);
        // Is large (i.e. vals[i] = M - 1)
        println!("vals[C..2*C - 1]: {:?}", vals[C..2*C - 1].to_vec());
        let is_large = vals[C..2*C - 1].iter().all(|x| *x == F::one());
        println!("is_large: {:?}", is_large);
        // Compute Sigmoid
        if is_pos && is_small {
            println!("is_pos && is_small");
            vals[2*C-1]
        } else if !is_pos && is_large {
            println!("!is_pos && is_large");
            vals[2*C]
        } else if is_pos {
            println!("is_pos");
            F::from_u64(u32::MAX as u64)
        } else {
            println!("!is_pos");
            F::zero()
        }
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        2
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        M: usize,
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

    fn materialize_entry(&self, _: u64) -> u64 {
        todo!()
    }

    fn evaluate_mle<F>(&self, point: &[F]) -> F
    where
        F: JoltField,
    {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use super::SigmoidInstruction;
    use crate::{jolt::instruction::JoltInstruction, jolt_instruction_test};
    use ark_bn254::Fr;
    use ark_std::rand::RngCore;
    use ark_std::test_rng;

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
