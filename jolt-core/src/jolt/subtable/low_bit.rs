use super::LassoSubtable;
use crate::field::JoltField;
use std::marker::PhantomData;

#[derive(Default)]
pub struct LowBitSubtable<F: JoltField, const OFFSET: usize> {
    _field: PhantomData<F>,
}

impl<F: JoltField, const OFFSET: usize> LowBitSubtable<F, OFFSET> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, const OFFSET: usize> LassoSubtable<F> for LowBitSubtable<F, OFFSET> {
    fn materialize(&self, M: usize) -> Vec<F> {
        // table[x] = x & (1 << OFFSET)
        let mut entries: Vec<F> = Vec::with_capacity(M);
        let low_bit = 1usize << OFFSET;

        // Materialize table entries in order from 0..M
        for idx in 0..M {
            entries.push(if idx & low_bit != 0 {
                F::one()
            } else {
                F::zero()
            });
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        point[point.len() - 1 - OFFSET]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use binius_field::BinaryField128b;

    use crate::{
        field::binius::BiniusField,
        jolt::subtable::{low_bit::LowBitSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        lsb_materialize_mle_parity,
        LowBitSubtable<Fr, 0>,
        Fr,
        256
    );
    subtable_materialize_mle_parity_test!(
        lsb_binius_materialize_mle_parity,
        LowBitSubtable<BiniusField<BinaryField128b>, 0>,
        BiniusField<BinaryField128b>,
        1 << 16
    );

    subtable_materialize_mle_parity_test!(
        second_lsb_materialize_mle_parity,
        LowBitSubtable<Fr, 1>,
        Fr,
        256
    );
    subtable_materialize_mle_parity_test!(
        second_lsb_binius_materialize_mle_parity,
        LowBitSubtable<BiniusField<BinaryField128b>, 1>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
