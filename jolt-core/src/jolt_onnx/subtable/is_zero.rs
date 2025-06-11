//! Lookup table for checking if a value is zero.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that checks if a value is zero.
#[derive(Default)]
pub struct IsZeroSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> IsZeroSubtable<F> {
    /// Creates a new instance of [`IsZeroSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for IsZeroSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        entries[0] = 1;
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        point.iter().map(|r| (F::one() - r)).product()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::is_zero::IsZeroSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        is_zero_materialize_mle_parity,
        IsZeroSubtable<Fr>,
        Fr,
        256
    );
}
