use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::split_bits;

#[derive(Default, Debug)]
pub struct LeftMSBSubtable<F: JoltField> {
    _field: PhantomData<F>,
}

impl<F: JoltField> LeftMSBSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField> LassoSubtable<F> for LeftMSBSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = (x & 0b100..0) = msb(x)
        let mut entries = Vec::with_capacity(M);
        let bits_per_operand = (log2(M) / 2) as usize;
        let high_bit = 1usize << (bits_per_operand - 1);

        // Materialize table entries in order from 0..M
        for idx in 0..M {
            let (x, _) = split_bits(idx, bits_per_operand);
            entries.push(if x & high_bit != 0 { 1 } else { 0 });
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // x_0
        debug_assert!(point.len() % 2 == 0);
        let b = point.len() / 2;
        let (x, _) = point.split_at(b);
        x[0]
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{left_msb::LeftMSBSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        left_msb_materialize_mle_parity,
        LeftMSBSubtable<Fr>,
        Fr,
        256
    );
}
