//! Lookup table that always returns 0.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that always returns 1.
#[derive(Default)]
pub struct OneSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> OneSubtable<F> {
    /// Creates a new instance of [`OneSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for OneSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        vec![1; M]
    }

    fn evaluate_mle(&self, _: &[F]) -> F {
        F::one()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::one::OneSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        one_materialize_mle_parity,
        OneSubtable<Fr>,
        Fr,
        256
    );
}
