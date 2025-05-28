use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default, Debug)]
pub struct LtuSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> LtuSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for LtuSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = (x < y)
        let mut entries = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;

        // Materialize table entries in order where (x | y) ranges 0..M
        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            let row = if x < y { 1 } else { 0 };
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_i (1 - x_i) * y_i * \prod_{j < i} ((1 - x_j) * (1 - y_j) + x_j * y_j)
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();
        let mut eq_term = F::one();
        for i in 0..b {
            result += (F::one() - x[i]) * y[i] * eq_term;
            eq_term *= F::one() - x[i] - y[i] + x[i] * y[i] + x[i] * y[i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{ltu::LtuSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(ltu_ark_materialize_mle_parity, LtuSubtable<Fr>, Fr, 256);
}
