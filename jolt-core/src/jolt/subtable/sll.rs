use crate::field::JoltField;
use ark_std::log2;
use std::cmp::min;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default)]
pub struct SllSubtable<F: JoltField, const CHUNK_INDEX: usize, const WORD_SIZE: usize> {
    _field: PhantomData<F>,
}

impl<F: JoltField, const CHUNK_INDEX: usize, const WORD_SIZE: usize>
    SllSubtable<F, CHUNK_INDEX, WORD_SIZE>
{
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, const CHUNK_INDEX: usize, const WORD_SIZE: usize> LassoSubtable<F>
    for SllSubtable<F, CHUNK_INDEX, WORD_SIZE>
{
    fn materialize(&self, M: usize) -> Vec<F> {
        // table[x | y] = (x << (y % WORD_SIZE)) % (1 << (WORD_SIZE - suffix_length))
        // where `suffix_length = operand_chunk_width * CHUNK_INDEX`
        let mut entries: Vec<F> = Vec::with_capacity(M);

        let operand_chunk_width: usize = (log2(M) / 2) as usize;
        let suffix_length = operand_chunk_width * CHUNK_INDEX;

        for idx in 0..M {
            let (x, y) = split_bits(idx, operand_chunk_width);

            let row = (x as u64)
                .checked_shl((y % WORD_SIZE) as u32)
                .unwrap_or(0)
                .rem_euclid(1 << (WORD_SIZE - suffix_length));

            entries.push(F::from_u64(row).unwrap());
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_{k = 0}^{2^b - 1} eq(y, bin(k)) * (\sum_{j = 0}^{m'-1} 2^{k + j} * x_{b - j - 1}),
        // where m = min(b, max( 0, (k + b * (CHUNK_INDEX + 1)) - WORD_SIZE))
        // and m' = b - m

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
                .map(|bit| F::from_u64(*bit as u64).unwrap())
                .collect::<Vec<F>>(); // big-endian

            // Compute eq(y, bin(k))
            let mut eq_term = F::one();
            // again, min with b is included when subtables of bit-length less than 6 are used
            for i in 0..min(log_WORD_SIZE, b) {
                eq_term *= k_bits[log_WORD_SIZE - 1 - i] * y[b - 1 - i]
                    + (F::one() - k_bits[log_WORD_SIZE - 1 - i]) * (F::one() - y[b - 1 - i]);
            }

            let m = if (k + b * (CHUNK_INDEX + 1)) > WORD_SIZE {
                min(b, (k + b * (CHUNK_INDEX + 1)) - WORD_SIZE)
            } else {
                0
            };
            let m_prime = b - m;

            // Compute \sum_{j = 0}^{m'-1} 2^{k + j} * x_{b - j - 1}
            let shift_x_by_k = (0..m_prime)
                .enumerate()
                .map(|(j, _)| F::from_u64(1_u64 << (j + k)).unwrap() * x[b - 1 - j])
                .fold(F::zero(), |acc, val| acc + val);

            result += eq_term * shift_x_by_k;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;
    use binius_field::BinaryField128b;

    use crate::{
        field::binius::BiniusField,
        jolt::subtable::{sll::SllSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(sll_materialize_mle_parity0, SllSubtable<Fr, 0, 32>, Fr, 1 << 10);
    subtable_materialize_mle_parity_test!(sll_materialize_mle_parity1, SllSubtable<Fr, 1, 32>, Fr, 1 << 10);
    subtable_materialize_mle_parity_test!(sll_materialize_mle_parity2, SllSubtable<Fr, 2, 32>, Fr, 1 << 10);
    subtable_materialize_mle_parity_test!(sll_materialize_mle_parity3, SllSubtable<Fr, 3, 32>, Fr, 1 << 10);

    // This test is noticeably slow
    subtable_materialize_mle_parity_test!(
        sll_binius_materialize_mle_parity3,
        SllSubtable<BiniusField<BinaryField128b>, 3, 32>,
        BiniusField<BinaryField128b>,
        1 << 16
    );
}
