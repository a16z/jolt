use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default, Debug)]
pub struct RightIsZeroSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> RightIsZeroSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for RightIsZeroSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = (y == 0)
        let mut entries = vec![0; M];
        let right_operand_bits = (1 << (log2(M) / 2)) - 1;

        for idx in 0..M {
            if (idx & right_operand_bits) == 0 {
                entries[idx] = 1;
            }
        }

        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i (1 - y_i)
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (_, y) = point.split_at(b);

        let mut result = F::one();
        for i in 0..b {
            result *= F::one() - y[i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{right_is_zero::RightIsZeroSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        right_is_zero_materialize_mle_parity,
        RightIsZeroSubtable<Fr>,
        Fr,
        256
    );
}
