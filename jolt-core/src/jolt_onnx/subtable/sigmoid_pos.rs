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
            let output = 1.0 / (1.0 + (-(i as i64) as f32).exp());
            let quantized_output = output * (u32::MAX as f32);
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

        // TODO: Add boolean sum
        let mut idx = F::zero();
        for i in 0..point.len() {
            idx += F::from_u64(1u64 << i) * point[point.len() - 1 - i];
        }
        let result = f_eval[idx.to_u64().unwrap() as usize];
        result
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
