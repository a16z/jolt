use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default, Debug)]
pub struct AndSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> AndSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for AndSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = x & y
        let mut entries = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;

        // Materialize table entries in order where (x | y) ranges 0..M
        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            let row = (x & y) as u32;
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_i 2^i * x_{b - i - 1} * y_{b - i - 1}
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();
        for i in 0..b {
            let x = x[b - i - 1];
            let y = y[b - i - 1];
            result += F::from_u64(1u64 << i) * x * y;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{and::AndSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(and_materialize_mle_parity, AndSubtable<Fr>, Fr, 256);
}
