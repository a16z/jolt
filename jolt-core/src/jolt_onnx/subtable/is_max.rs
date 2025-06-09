//! Lookup table for checking if a value is max.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that checks if a value is max.
#[derive(Default)]
pub struct IsMaxSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> IsMaxSubtable<F> {
    /// Creates a new instance of [`IsMaxSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for IsMaxSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        entries[M-1] = 1;
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::is_max::IsMaxSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        is_max_materialize_mle_parity,
        IsMaxSubtable<Fr>,
        Fr,
        256
    );
}
