use allocative::Allocative;

use crate::poly::field::JoltField;
use std::marker::PhantomData;

use super::LassoSubtable;

#[derive(Default, Allocative)]
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
        (0..M).map(|i| F::from_u64(i as u64).unwrap()).collect()
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
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

    use crate::{
        jolt::subtable::{identity::IdentitySubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        iden_materialize_mle_parity,
        IdentitySubtable<Fr>,
        Fr,
        256
    );
}
