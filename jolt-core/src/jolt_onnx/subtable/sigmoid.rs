//! Lookup table for sigmoid of positive values.

use itertools::Itertools;

use crate::{field::JoltField, jolt::subtable::LassoSubtable, poly::eq_poly::EqPolynomial};
use std::marker::PhantomData;

/// Input scale for sigmoid. Input values are between -8 and 8. Quantized input values are between 0 and 255.
pub const INPUT_SCALE: f32 = 16.0 / 256.0;
/// Quantized input zero
pub const INPUT_ZERO_POINT: i64 = 128;
/// Output scale for sigmoid. Output values are between 0 and 1. Quantized output values are between 0 and 255.
pub const OUTPUT_SCALE: f32 = 1.0 / 256.0;
/// Quantized output zero
pub const OUTPUT_ZERO_POINT: i64 = 0;

/// Materialised sigmoid table for quantized values.
pub const QUANTIZED_SIGMOID_TABLE: [u8; 256] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4,
    5, 5, 5, 6, 6, 6, 7, 7, 8, 8, 8, 9, 10, 10, 11, 11, 12, 13, 14, 15, 15, 16, 17, 18, 19, 21, 22,
    23, 24, 26, 27, 29, 31, 32, 34, 36, 38, 40, 42, 44, 47, 49, 52, 54, 57, 60, 63, 66, 69, 72, 75,
    79, 82, 86, 89, 93, 97, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148, 152,
    156, 159, 163, 167, 170, 174, 177, 181, 184, 187, 190, 193, 196, 199, 202, 204, 207, 209, 212,
    214, 216, 218, 220, 222, 224, 225, 227, 229, 230, 232, 233, 234, 235, 237, 238, 239, 240, 241,
    241, 242, 243, 244, 245, 245, 246, 246, 247, 248, 248, 248, 249, 249, 250, 250, 250, 251, 251,
    251, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 254, 254, 254, 254,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
];

/// A lookup table that returns the sigmoid of a value.
#[derive(Default)]
pub struct SigmoidSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> SigmoidSubtable<F> {
    /// Creates a new instance of [`SigmoidSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for SigmoidSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        for i in 0..M {
            let x_dequant = (i as i64 - INPUT_ZERO_POINT) as f32 * INPUT_SCALE;
            let sigmoid = 1.0 / (1.0 + (-x_dequant).exp());
            let x_requant = (sigmoid / OUTPUT_SCALE + OUTPUT_ZERO_POINT as f32).round();
            entries[i] = x_requant.clamp(0.0, 255.0) as u32;
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
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
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::sigmoid::SigmoidSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        sigmoid_materialize_mle_parity,
        SigmoidSubtable<Fr>,
        Fr,
        256
    );
}
