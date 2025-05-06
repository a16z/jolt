use crate::{field::JoltField, utils::split_bits};
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default, Debug)]
pub struct DivByZeroSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> DivByZeroSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for DivByZeroSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = (x == 0) && (y == 2^b - 1)
        let mut entries = vec![0; M];
        let bits_per_operand = (log2(M) / 2) as usize;

        for idx in 0..M {
            let (x, y) = split_bits(idx, bits_per_operand);
            if x == 0 && (y == (1 << bits_per_operand) - 1) {
                entries[idx] = 1;
            }
        }

        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i (1 - x_i) * y_i
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::one();
        for i in 0..b {
            result *= (F::one() - x[i]) * y[i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{div_by_zero::DivByZeroSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        div_by_zero_materialize_mle_parity,
        DivByZeroSubtable<Fr>,
        Fr,
        256
    );
}
