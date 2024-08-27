use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default)]
pub struct RightMSBSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> RightMSBSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for RightMSBSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        // table[x | y] = (y & 0b100..0) = msb(y)
        let mut entries: Vec<F> = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;
        let high_bit = 1usize << (bits_per_operand - 1);

        // Materialize table entries in order from 0..M
        for idx in 0..M {
            let (_, y) = split_bits(idx, bits_per_operand);
            entries.push(if y & high_bit != 0 {
                F::one()
            } else {
                F::zero()
            });
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // y_0
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (_, y) = point.split_at(b);
        y[0]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use binius_field::BinaryField128b;

    use crate::{
        field::binius::BiniusField,
        jolt::subtable::{right_msb::RightMSBSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        right_msb_materialize_mle_parity,
        RightMSBSubtable<Fr>,
        Fr,
        256
    );
    subtable_materialize_mle_parity_test!(
        right_msb_binius_materialize_mle_parity,
        RightMSBSubtable<BiniusField<BinaryField128b>>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
