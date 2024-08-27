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
    fn materialize(&self, M: usize) -> Vec<F> {
        // table[x] = x
        (0..M).map(|i| F::from_u64(i as u64).unwrap()).collect()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_i 2^i * x_{b - i - 1}
        let mut result = F::zero();
        for i in 0..point.len() {
            result += F::from_u64(1u64 << i).unwrap() * point[point.len() - 1 - i];
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
        jolt::subtable::{identity::IdentitySubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        iden_materialize_mle_parity,
        IdentitySubtable<Fr>,
        Fr,
        256
    );

    subtable_materialize_mle_parity_test!(
        iden_binius_materialize_mle_parity,
        IdentitySubtable<BiniusField<BinaryField128b>>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
