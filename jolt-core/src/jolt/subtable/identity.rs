use crate::field::JoltField;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default)]
pub struct IdentitySubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> IdentitySubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for IdentitySubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x] = x
        (0..M).map(|i| i as u32).collect()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_i 2^i * x_{b - i - 1}
        let mut result = F::zero();
        for i in 0..point.len() {
            result += F::from_u64(1u64 << i) * point[point.len() - 1 - i];
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
        field::JoltField,
        jolt::subtable::{identity::IdentitySubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        iden_materialize_mle_parity,
        IdentitySubtable<Fr>,
        Fr,
        1 << 16
    );

    subtable_materialize_mle_parity_test!(
        iden_binius_materialize_mle_parity,
        IdentitySubtable<BiniusField<BinaryField128b>>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
