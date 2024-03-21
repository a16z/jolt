use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default)]
pub struct SignExtendByteSubtable<F: PrimeField> {
    _field: PhantomData<F>,
}

impl<F: PrimeField> SignExtendByteSubtable<F> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

const SIGN_BIT_INDEX: usize = 7;

impl<F: PrimeField> LassoSubtable<F> for SignExtendByteSubtable<F> {
    fn materialize(&self, M: usize) -> Vec<F> {
        // TODO(moodlezoup): This subtable currently only works for M = 2^16
        assert_eq!(M, 1 << 16);
        let mut entries: Vec<F> = Vec::with_capacity(M);

        let operand_chunk_width: usize = (log2(M) / 2) as usize;
        let ones: u64 = (1 << operand_chunk_width) - 1;

        for idx in 0..M {
            let (_, y) = split_bits(idx, operand_chunk_width);
            let y_sign = ((y >> SIGN_BIT_INDEX) & 1) as u64;
            let sign_extension = y_sign * ones;
            let row = F::from_u64(sign_extension).unwrap();
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        assert_eq!(point.len(), 16);
        let operand_chunk_width: usize = point.len() / 2;

        let (_, y) = point.split_at(operand_chunk_width);
        let y_sign = y[0];

        let ones: u64 = (1 << operand_chunk_width) - 1;
        y_sign * F::from_u64(ones).unwrap()
    }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;

    use crate::{
        jolt::subtable::{sign_extend::SignExtendByteSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
        sign_extend_materialize_mle_parity,
        SignExtendByteSubtable<Fr>,
        Fr,
        1 << 16
    );
}
