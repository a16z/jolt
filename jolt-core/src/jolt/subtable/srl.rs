use crate::field::JoltField;
use ark_std::log2;
use std::cmp::min;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default, Debug)]
pub struct SrlSubtable<F: JoltField, const CHUNK_INDEX: usize, const WORD_SIZE: usize> {
    _field: PhantomData<F>,
}

impl<F: JoltField, const CHUNK_INDEX: usize, const WORD_SIZE: usize>
    SrlSubtable<F, CHUNK_INDEX, WORD_SIZE>
{
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, const CHUNK_INDEX: usize, const WORD_SIZE: usize> LassoSubtable<F>
    for SrlSubtable<F, CHUNK_INDEX, WORD_SIZE>
{
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = (x << suffix_length) >> (y % WORD_SIZE)
        // where `suffix_length = operand_chunk_width * CHUNK_INDEX`
        let mut entries = Vec::with_capacity(M);

        let operand_chunk_width: usize = (log2(M) / 2) as usize;
        let suffix_length = operand_chunk_width * CHUNK_INDEX;

        for idx in 0..M {
            let (x, y) = split_bits(idx, operand_chunk_width);

            let row = (x as u64)
                .checked_shl(suffix_length as u32)
                .unwrap_or(0)
                .checked_shr((y % WORD_SIZE) as u32)
                .unwrap_or(0);

            entries.push(row as u32);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_{k = 0}^{2^b - 1} eq(y, bin(k)) * (\sum_{j = m}^{chunk_length - 1} 2^{b * CHUNK_INDEX + j - k} * x_{b - j - 1}),
        // where m = max(0, min(b, k - b * CHUNK_INDEX))
        // and chunk_length = min(b, WORD_SIZE - b * CHUNK_INDEX)

        // We assume the first half is chunk(X_i) and the second half is always chunk(Y_0)
        debug_assert!(point.len() % 2 == 0);

        let log_WORD_SIZE = log2(WORD_SIZE) as usize;

        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();

        // min with 1 << b is included for test cases with subtables of bit-length smaller than 6
        for k in 0..min(WORD_SIZE, 1 << b) {
            // bit-decompose k
            let k_bits = k
                .get_bits(log_WORD_SIZE)
                .iter()
                .map(|bit| if *bit { F::one() } else { F::zero() })
                .collect::<Vec<F>>(); // big-endian

            // Compute eq(y, bin(k))
            let mut eq_term = F::one();
            // again, min with b is included when subtables of bit-length less than 6 are used
            for i in 0..min(log_WORD_SIZE, b) {
                eq_term *= k_bits[log_WORD_SIZE - 1 - i] * y[b - 1 - i]
                    + (F::one() - k_bits[log_WORD_SIZE - 1 - i]) * (F::one() - y[b - 1 - i]);
            }

            let m = if k > b * CHUNK_INDEX {
                min(b, k - b * CHUNK_INDEX)
            } else {
                0
            };

            // the most significant chunk might be shorter
            let chunk_length = min(b, WORD_SIZE - b * CHUNK_INDEX);

            let shift_x_by_k = (m..chunk_length)
                .map(|j| F::from_u64(1_u64 << (b * CHUNK_INDEX + j - k)) * x[b - 1 - j])
                .fold(F::zero(), |acc, val: F| acc + val);

            result += eq_term * shift_x_by_k;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{srl::SrlSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(srl_materialize_mle_parity0, SrlSubtable<Fr, 0, 32>, Fr, 1 << 10);
    subtable_materialize_mle_parity_test!(srl_materialize_mle_parity1, SrlSubtable<Fr, 1, 32>, Fr, 1 << 10);
    subtable_materialize_mle_parity_test!(srl_materialize_mle_parity2, SrlSubtable<Fr, 2, 32>, Fr, 1 << 10);
    subtable_materialize_mle_parity_test!(srl_materialize_mle_parity3, SrlSubtable<Fr, 3, 32>, Fr, 1 << 10);
}
