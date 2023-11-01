use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;
use std::sync::Arc;

use typenum::Unsigned;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Copy, Clone, Default)]
pub struct SllSubtable<F: PrimeField, ChunkIndex: Unsigned> {
  _field: PhantomData<F>,
  _chunk_index: PhantomData<ChunkIndex>,
}

impl<F: PrimeField, ChunkIndex: Unsigned> SllSubtable<F, ChunkIndex> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
      _chunk_index: PhantomData,
    }
  }
}

impl<F: PrimeField, ChunkIndex: Unsigned> LassoSubtable<F> for SllSubtable<F, ChunkIndex> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);

    let operand_chunk_width: usize = (log2(M) / 2) as usize;
    let suffix_length = operand_chunk_width * ChunkIndex::to_usize();

    for idx in 0..M {
      let (x, y) = split_bits(idx, operand_chunk_width);

      let row = x
        .checked_shl((y as u32) % 64 + suffix_length as u32)
        .unwrap_or(0)
        .checked_shr(suffix_length as u32)
        .unwrap_or(0);

      entries.push(F::from(row as u64));
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    // first half is chunk X_i
    // and second half is always chunk Y_0
    debug_assert!(point.len() % 2 == 0);

    let MAX_SHIFT = 64;
    let log_MAX_SHIFT = log2(MAX_SHIFT) as usize;

    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::zero();

    // min with 1 << b is included for test cases with subtables of bit-length smaller than 6
    for k in 0..std::cmp::min(MAX_SHIFT, 1 << b) {
      let k_bits = (k as usize)
        .get_bits(log_MAX_SHIFT)
        .iter()
        .map(|bit| F::from(*bit as u64))
        .collect::<Vec<F>>(); // big-endian

      let mut eq_term = F::one();
      // again, min with b is included when subtables of bit-length less than 6 are used
      for i in 0..std::cmp::min(log_MAX_SHIFT, b) {
        eq_term *= k_bits[log_MAX_SHIFT - 1 - i] * y[b - 1 - i]
          + (F::one() - k_bits[log_MAX_SHIFT - 1 - i]) * (F::one() - y[b - 1 - i]);
      }

      let m = if (k + b * (ChunkIndex::to_usize() + 1)) > 64 {
        std::cmp::min(b, (k + b * (ChunkIndex::to_usize() + 1)) - 64)
      } else {
        0
      };

      let m_prime = b - (m as usize);

      let shift_x_by_k = (0..m_prime)
        .enumerate()
        .map(|(j, _)| F::from(1_u64 << (j + k)) * x[b - 1 - j])
        .fold(F::zero(), |acc, val| acc + val);

      result += eq_term * shift_x_by_k;
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;
  use typenum::U0;

  use crate::{
    jolt::subtable::{sll::SllSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(sll_materialize_mle_parity, SllSubtable<Fr, U0>, Fr, 256);
}
