//! Lookup table for sigmoid of negative values.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that returns the sigmoid of a value.
#[derive(Default)]
pub struct SigmoidNegSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> SigmoidNegSubtable<F> {
    /// Creates a new instance of [`SigmoidNegSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for SigmoidNegSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        for i in 1..M {
            let output = 1.0 / (1.0 + (i as f32).exp());
            let quantized_output = (output * (u32::MAX as f32)) as u32;
            entries[M-i] = quantized_output;
        }
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
        jolt_onnx::subtable::sigmoid_neg::SigmoidNegSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        sigmoid_neg_materialize_mle_parity,
        SigmoidNegSubtable<Fr>,
        Fr,
        256
    );
}
