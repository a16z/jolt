//! Lookup table that always returns 0.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that always returns 0.
#[derive(Default)]
pub struct ZeroSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> ZeroSubtable<F> {
    /// Creates a new instance of [`ZeroSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for ZeroSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        vec![0; M]
    }

    fn evaluate_mle(&self, _: &[F]) -> F {
        F::zero()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::zero::ZeroSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        zero_materialize_mle_parity,
        ZeroSubtable<Fr>,
        Fr,
        256
    );
}
