use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use crate::{jolt::subtable::LassoSubtable, utils::split_bits};

#[derive(Default)]
pub struct IsPosSubtable<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> IsPosSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: PrimeField> LassoSubtable<F> for IsPosSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries = vec![F::zero(); M];
        for i in 0..M / 2 {
            entries[i] = F::one();
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        F::one() - point[0]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        jolt::subtable::LassoSubtable, jolt_onnx::subtable::is_pos::IsPosSubtable,
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        is_neg_materialize_mle_parity,
        IsPosSubtable<Fr>,
        Fr,
        256
    );
}
