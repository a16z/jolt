use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
pub struct RightIsOnesSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> RightIsOnesSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for RightIsOnesSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        let mut entries: Vec<F> = vec![F::zero(); M];
        let right_operand_bits = (1 << (log2(M) / 2)) - 1;

        for idx in 0..M {
            if (idx & right_operand_bits) == right_operand_bits {
                entries[idx] = F::one();
            }
        }

        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i y_i
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (_, y) = point.split_at(b);

        let mut result = F::one();
        for i in 0..b {
            result *= y[i];
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
        jolt::subtable::{right_is_ones::RightIsOnesSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        right_is_ones_materialize_mle_parity,
        RightIsOnesSubtable<Fr>,
        Fr,
        256
    );
    subtable_materialize_mle_parity_test!(
        right_is_ones_binius_materialize_mle_parity,
        RightIsOnesSubtable<BiniusField<BinaryField128b>>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
