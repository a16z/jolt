use crate::field::JoltField;
use ark_std::log2;
use std::marker::PhantomData;

use crate::utils::split_bits;

use super::LassoSubtable;

/// Example usage in ADD:
/// Input z is of 65 bits, which is split into 20-bit chunks.
/// This subtable is used to remove the overflow bits (bits 60 to 64) from the 4th chunk.
#[derive(Default, Debug)]
pub struct TruncateOverflowSubtable<F: JoltField, const WORD_SIZE: usize> {
    _field: PhantomData<F>,
}

impl<F: JoltField, const WORD_SIZE: usize> TruncateOverflowSubtable<F, WORD_SIZE> {
    pub fn new() -> Self {
        Self {
            _field: PhantomData,
        }
    }
}

impl<F: JoltField, const WORD_SIZE: usize> LassoSubtable<F>
    for TruncateOverflowSubtable<F, WORD_SIZE>
{
    fn materialize(&self, M: usize) -> Vec<u32> {
        // table[x] = x & (0b00..011..1), where the number of 0s is `cutoff`.
        // Truncates overflow bits beyond nearest multiple of `log2(M)`
        let cutoff = WORD_SIZE % log2(M) as usize;

        let mut entries = Vec::with_capacity(M);
        for idx in 0..M {
            let (_, lower_bits) = split_bits(idx, cutoff);
            let row = lower_bits as u32;
            entries.push(row);
        }
        entries
    }

    fn evaluate_mle(&self, point: &[F]) -> F {
        // \sum_{i = 0}^{cutoff - 1} 2^i * x_{b - i - 1}
        let log_M = point.len();
        let cutoff = WORD_SIZE % log_M;

        let mut result = F::zero();
        for i in 0..cutoff {
            result += F::from_u64(1u64 << i) * point[point.len() - 1 - i];
        }
        result
    }
}

#[cfg(test)]
mod test {
    use ark_bn254::Fr;

    use crate::{
        field::JoltField,
        jolt::subtable::{truncate_overflow::TruncateOverflowSubtable, LassoSubtable},
        subtable_materialize_mle_parity_test,
    };

    subtable_materialize_mle_parity_test!(
      truncate_overflow_materialize_mle_parity,
      TruncateOverflowSubtable<Fr, 32>,
      Fr,
      256
    );
}
