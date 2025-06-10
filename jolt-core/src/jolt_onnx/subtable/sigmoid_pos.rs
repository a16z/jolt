//! Lookup table for sigmoid of positive values.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that returns the sigmoid of a value.
#[derive(Default)]
pub struct SigmoidPosSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> SigmoidPosSubtable<F> {
    /// Creates a new instance of [`SigmoidPosSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for SigmoidPosSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        for i in 0..M {
            // println!("i: {:?}", i);
            let output = 1.0 / (1.0 + (-(i as i64) as f32).exp());
            // println!("output: {:?}", output);
            let quantized_output = (output * (u32::MAX as f32));
            // println!("quantized_output: {:?}", quantized_output);
            entries[i] = quantized_output as u32;
        }
        // for i in M/2..M {
        //     let output = 1.0 / (1.0 + ((i - M/2 + 1) as f32).exp());
        //     let quantized_output = (output * (u32::MAX as f32)) as u32;
        //     entries[M + M/2 - i - 1] = quantized_output;
        // }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::sigmoid_pos::SigmoidPosSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        sigmoid_pos_materialize_mle_parity,
        SigmoidPosSubtable<Fr>,
        Fr,
        256
    );
}
