use super::LassoSubtable;
use crate::field::JoltField;
use std::marker::PhantomData;

#[derive(Default)]
pub struct ReLUSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> ReLUSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for ReLUSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0_u32; M];
        for i in 0..M / 2 {
            entries[i] = i as u32;
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_i 2^i * x_{b - i - 1}
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
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{relu::ReLUSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(and_materialize_mle_parity, ReLUSubtable<Fr>, Fr, 256);
}
