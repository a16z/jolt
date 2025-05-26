//! Lookup table for checking if a value is positive.

use crate::{field::JoltField, jolt::subtable::LassoSubtable};
use std::marker::PhantomData;

/// A lookup table that checks if a value is positive.
#[derive(Default)]
pub struct IsPosSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> IsPosSubtable<F> {
    /// Creates a new instance of [`IsPosSubtable`].
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for IsPosSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        let mut entries = vec![0; M];
        for i in 0..M / 2 {
            entries[i] = 1;
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        F::one() - point[0]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField, jolt::subtable::LassoSubtable,
        jolt_onnx::subtable::is_pos::IsPosSubtable, subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        is_pos_materialize_mle_parity,
        IsPosSubtable<Fr>,
        Fr,
        256
    );
}
