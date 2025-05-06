use super::LassoSubtable;
use crate::field::JoltField;
use std::marker::PhantomData;

#[derive(Default, Debug)]
pub struct LowBitSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> LowBitSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for LowBitSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x] = x & 1
        let mut entries = Vec::with_capacity(M);

        // Materialize table entries in order from 0..M
        for idx in 0..M {
            entries.push(idx as u32 & 1);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        point[point.len() - 1]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{low_bit::LowBitSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(lsb_materialize_mle_parity, LowBitSubtable<Fr>, Fr, 256);
}
