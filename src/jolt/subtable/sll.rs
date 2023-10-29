use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;
use std::sync::Arc;

use super::LassoSubtable;
use crate::utils::split_bits;
use crate::utils::math::Math;

#[derive(Default)]
pub struct SllSubtable<F: PrimeField> {
  _field: PhantomData<F>,
  chunk_idx: usize,
}

impl<F: PrimeField> SllSubtable<F> {
  pub fn new(chunk_idx: usize) -> Self {
    Self {
      _field: PhantomData,
      chunk_idx: chunk_idx,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for SllSubtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);
    let bits_per_operand = (log2(M) / 2) as usize;

    // Materialize table entries in order where (x | y) ranges 0..M
    for idx in 0..M {
      let (x, y) = split_bits(idx, bits_per_operand);
      let x_in_position = x << (bits_per_operand * self.chunk_idx);

      // shift x by the length represented by the lower 6 bits of y
      let row = match x_in_position.checked_shl((y as u32) %64 as u32) {
        None => F::zero(),
        Some(x_shifted) => F::from((x_shifted >> (bits_per_operand * self.chunk_idx)) as u64),
      };

      entries.push(row );
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    // first half is chunk X_i 
    // and second half is always chunk Y_0
    debug_assert!(point.len() % 2 == 0);

    let MAX_SHIFT= 64;
    let log_MAX_SHIFT = log2(MAX_SHIFT) as usize;

    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::zero();

    // min with 1 << b is included for test cases with subtables of bit-length smaller than 6 
    for k in 0..std::cmp::min(MAX_SHIFT, 1<<b) {
      let k_bits = (k as usize)
        .get_bits(log_MAX_SHIFT).iter()
        .map(|bit| F::from(*bit as u64))
        .collect::<Vec<F>>(); // big-endian

      let mut eq_term = F::one();

      // again, min with b is included when subtables of bit-length less than 6 are used 
      for i in 0..std::cmp::min(log_MAX_SHIFT, b) {
        eq_term *= k_bits[log_MAX_SHIFT-1-i] * y[b-1-i] + (F::one() - k_bits[log_MAX_SHIFT-1-i]) * (F::one() - y[b-1-i]);
      }

      let mut shift_x_by_k = F::zero();

      let m = if (k + b * (self.chunk_idx+1)) > 64 {
        std::cmp::min(b, (k + b * (self.chunk_idx+1)) - 64)
      } else {
        0
      };
      let m_prime = b - (m as usize);
      for j in 0..m_prime {
        shift_x_by_k += F::from((1_u64 << (j+k)) as u64) * x[b-1-j];
      }

      result += eq_term * shift_x_by_k;
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{
    jolt::subtable::{sll::SllSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(sll_materialize_mle_parity, SllSubtable<Fr>, Fr, 256, 16);
}
