use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
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
    fn materialize(&self, M: usize) -> Vec<F> {
        // table[x | y] = (y == 0)
        let mut entries: Vec<F> = vec![F::zero(); M];
        let right_operand_bits = (1 << (log2(M) / 2)) - 1;

        for idx in 0..M {
            if (idx & right_operand_bits) == 0 {
                entries[idx] = F::one();
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
    use binius_field::BinaryField128b;

    use crate::{
        field::binius::BiniusField,
        jolt::subtable::{right_is_zero::RightIsZeroSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        right_is_zero_materialize_mle_parity,
        RightIsZeroSubtable<Fr>,
        Fr,
        256
    );
    subtable_materialize_mle_parity_test!(
        right_is_zero_binius_materialize_mle_parity,
        RightIsZeroSubtable<BiniusField<BinaryField128b>>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
