use ark_std::log2;

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

#[derive(Default)]
pub struct HigherHalfSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> HigherHalfSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for HigherHalfSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x] = x & !((1 << operand_chunk_width) - 1)
        let operand_chunk_width: usize = (log2(M) / 2) as usize;
        let mut entries = Vec::with_capacity(M);
        // Materialize table entries in order from 0..M
        for idx in 0..M {
            entries.push((idx as u32) >> operand_chunk_width);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_i 2^i * x_{b - i - 1}
        let half_len = point.len() / 2;
        let mut result = F::zero();
        for i in 0..half_len {
            result += F::from_u64(1u64 << i) * point[half_len - 1 - i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use super::HigherHalfSubtable;
    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        lower_half_materialize_mle_parity,
        HigherHalfSubtable<Fr>,
        Fr,
        256
    );
}
