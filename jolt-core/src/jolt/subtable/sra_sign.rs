use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default, Debug)]
pub struct SraSignSubtable<F: JoltField, const WORD_SIZE: usize> {
    _field: PhantomData<F>,
}

impl<F: JoltField, const WORD_SIZE: usize> SraSignSubtable<F, WORD_SIZE> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, const WORD_SIZE: usize> LassoSubtable<F> for SraSignSubtable<F, WORD_SIZE> {
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x | y] = (x_sign == 0) ? 0 : 0b11..100..0,
        // where x_sign = (x >> ((WORD_SIZE - 1) & (log2(M) / 2))) & 1,
        // `0b11..100..0` has `WORD_SIZE` bits and `y % WORD_SIZE` ones
        let mut entries = Vec::with_capacity(M);

        let operand_chunk_width: usize = (log2(M) / 2) as usize;

        // find position of sign bit in the chunk
        let sign_bit_index = (WORD_SIZE - 1) % operand_chunk_width;

        for idx in 0..M {
            let (x, y) = split_bits(idx, operand_chunk_width);

            let x_sign = (x >> sign_bit_index) & 1;

            if x_sign == 0 {
                entries.push(0);
            } else {
                let row = (0..(y % WORD_SIZE)).fold(0, |acc, i| acc + (1 << (WORD_SIZE - 1 - i)));
                entries.push(row as u32);
            }
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_{k = 0}^{WORD_SIZE - 1} eq(y, bin(k)) * x_sign * \prod_{j = 0}^{k-1} 2^{WORD_SIZE - j - 1},
        // where x_sign = x_{b - 1 - (WORD_SIZE - 1) % b}

        // first half is chunk X_last
        // and second half is always chunk Y_0
        debug_assert!(point.len() % 2 == 0);

        let log_WORD_SIZE = log2(WORD_SIZE) as usize;

        let b = point.len() / 2;
        let (x, y) = point.split_at(b);

        let mut result = F::zero();

        let sign_index = (WORD_SIZE - 1) % b;
        let x_sign = x[b - 1 - sign_index];

        // min with 1 << b is included for test cases with subtables of bit-length smaller than 6
        for k in 0..std::cmp::min(WORD_SIZE, 1 << b) {
            // bit-decompose k
            let k_bits = k
                .get_bits(log_WORD_SIZE)
                .iter()
                .map(|bit| if *bit { F::one() } else { F::zero() })
                .collect::<Vec<F>>(); // big-endian

            // Compute eq(y, bin(k))
            let mut eq_term = F::one();
            // again, min with b is included when subtables of bit-length less than 6 are used
            for i in 0..std::cmp::min(log_WORD_SIZE, b) {
                eq_term *= k_bits[log_WORD_SIZE - 1 - i] * y[b - 1 - i]
                    + (F::one() - k_bits[log_WORD_SIZE - 1 - i]) * (F::one() - y[b - 1 - i]);
            }

            let x_sign_upper = (0..k).fold(F::zero(), |acc, i| {
                acc + F::from_u64(1_u64 << (WORD_SIZE - 1 - i)) * x_sign
            });

            result += eq_term * x_sign_upper;
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{sra_sign::SraSignSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
      sra_sign_materialize_mle_parity,
      SraSignSubtable<Fr, 32>,
      Fr,
      256
    );
}
