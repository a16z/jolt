use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
pub struct LeftIsZeroSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> LeftIsZeroSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for LeftIsZeroSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        // table[x | y] = (x == 0)
        let mut entries: Vec<F> = vec![F::zero(); M];

        for idx in 0..(1 << (log2(M) / 2)) {
            entries[idx] = F::one();
        }

        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \prod_i (1 - x_i)
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, _) = point.split_at(b);

        let mut result = F::one();
        for i in 0..b {
            result *= F::one() - x[i];
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
        jolt::subtable::{left_is_zero::LeftIsZeroSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        left_is_zero_materialize_mle_parity,
        LeftIsZeroSubtable<Fr>,
        Fr,
        256
    );
    subtable_materialize_mle_parity_test!(
        left_is_zero_binius_materialize_mle_parity,
        LeftIsZeroSubtable<BiniusField<BinaryField128b>>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
