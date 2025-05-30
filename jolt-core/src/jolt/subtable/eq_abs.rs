use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default, Debug)]
pub struct EqAbsSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> EqAbsSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for EqAbsSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // Materialize table entries in order where (x | y) ranges 0..M
        // Below is the optimized loop for the condition:
        // lower_bits_mask = 0b01111...11
        // table[x | y] == (x & lower_bits_mask) == (y & lower_bits_mask)
        let mut entries = vec![0; M];
        let bits_per_operand = (log2(M) / 2) as usize;

        for idx in 0..(1 << (bits_per_operand)) {
            // we set the bit in the table where x == y
            // e.g. 01010011 | 01010011 = 1
            let concat_index_1 = idx | (idx << bits_per_operand);
            // we also set the bit where x == y except for their leading bit
            // e.g. 11010011 | 01010011 = 0
            let concat_index_2 = idx | ((idx ^ (1 << (bits_per_operand - 1))) << bits_per_operand);
            entries[concat_index_1] = 1;
            entries[concat_index_2] = 1;
        }

        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i x_i * y_i + (1 - x_i) * (1 - y_i) for i > 0
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::one();
        // Skip i=0
        for i in 1..b {
            result *= x[i] * y[i] + (F::one() - x[i]) * (F::one() - y[i]);
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{eq_abs::EqAbsSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        eq_abs_materialize_mle_parity,
        EqAbsSubtable<Fr>,
        Fr,
        256
    );
}
