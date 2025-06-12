//! Sigmoid instruction for Jolt ONNX

use crate::field::JoltField;
use crate::jolt::instruction::{JoltInstruction, SubtableIndices};
use crate::jolt::subtable::LassoSubtable;
use crate::jolt_onnx::subtable::is_pos::IsPosSubtable;
use crate::jolt_onnx::subtable::is_zero::IsZeroSubtable;
use crate::jolt_onnx::subtable::sigmoid::{
    SigmoidSubtable, INPUT_SCALE, INPUT_ZERO_POINT, OUTPUT_SCALE, OUTPUT_ZERO_POINT,
    QUANTIZED_SIGMOID_TABLE,
};
use crate::poly::eq_poly::EqPolynomial;
use crate::utils::instruction_utils::chunk_operand_usize;
use itertools::Itertools;
use rand::prelude::StdRng;
use serde::{Deserialize, Serialize};

/// Sigmoid instruction
#[derive(Copy, Clone, Default, Debug, Serialize, Deserialize, PartialEq)]
pub struct SigmoidInstruction(pub u64);

impl JoltInstruction for SigmoidInstruction {
    fn operands(&self) -> (u64, u64) {
        (self.0, 0)
    }

    /// We only care about small values since sigmoid goes to 0 or 1 exponentially fast.
    /// We quantize the input 0 to 128. Any negative quantized value will point to an
    /// unquantized value much smaller 0 whose sigmoid output will be certainly 0.
    /// If you look at QUANTIZED_SIGMOID_TABLE, you will see that values are between 0 or 255.
    /// Any value outside this table can only be 0 or 255. That is,
    /// [0, ..., 0, QUANTIZED_SIGMOID_TABLE, 255, ..., 255].
    ///
    /// is_pos * is_small_pos checks that the inputs belong to this table QUANTIZED_SIGMOID_TABLE.
    /// If it's positive but not small, then it will be 255. Otherwise it will be 0.
    fn combine_lookups<F: JoltField>(&self, vals: &[F], C: usize, _M: usize) -> F {
        let is_pos = vals[0];
        let is_small_pos = vals[1..C].iter().fold(F::one(), |acc, x| acc * x);

        is_pos * is_small_pos * vals[C]
            + is_pos * (F::one() - is_small_pos) * F::from_u64(u8::MAX as u64)
    }

    fn g_poly_degree(&self, _: usize) -> usize {
        3
    }

    fn subtables<F: JoltField>(
        &self,
        C: usize,
        _M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F>>, SubtableIndices)> {
        vec![
            (Box::new(IsPosSubtable::new()), SubtableIndices::from(0)),
            (
                Box::new(IsZeroSubtable::new()),
                SubtableIndices::from(0..C - 1),
            ),
            (
                Box::new(SigmoidSubtable::<F>::new()),
                SubtableIndices::from(C - 1),
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
        let x_dequant = (self.0 as i64 - INPUT_ZERO_POINT) as f32 * INPUT_SCALE;
        let sigmoid = 1.0 / (1.0 + (-x_dequant).exp());
        let x_requant = (sigmoid / OUTPUT_SCALE + OUTPUT_ZERO_POINT as f32).round();
        let res = x_requant.clamp(0.0, 255.0) as u64;
        res
    }

    fn random(&self, rng: &mut StdRng) -> Self {
        use rand::RngCore;
        Self(rng.next_u32() as u64)
    }

    fn materialize_entry(&self, index: u64) -> u64 {
        let x_dequant = (index as i64 - INPUT_ZERO_POINT) as f32 * INPUT_SCALE;
        let sigmoid = 1.0 / (1.0 + (-x_dequant).exp());

        let x_requant = (sigmoid / OUTPUT_SCALE + OUTPUT_ZERO_POINT as f32).round();
        x_requant.clamp(0.0, 255.0) as u64
    }

    fn evaluate_mle<F>(&self, point: &[F]) -> F
    where
        F: JoltField,
    {
        let mut f_eval: Vec<F> = vec![F::from_u8(255); 1 << point.len()];
        for i in 0..256 {
            f_eval[i] = F::from_u8(QUANTIZED_SIGMOID_TABLE[i]);
        }

        let eq_evals = EqPolynomial::evals(point);
        f_eval
            .iter()
            .zip_eq(eq_evals.iter())
            .map(|(x, e)| *x * e)
            .sum()
    }
}

#[cfg(test)]
mod test {
    use super::SigmoidInstruction;
    use crate::jolt::instruction::test::{
        instruction_mle_full_hypercube_test, materialize_entry_test,
    };
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
