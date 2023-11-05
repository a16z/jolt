use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;
use std::sync::Arc;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default)]
pub struct SrlSubtable<F: PrimeField, const CHUNK_INDEX: usize> {
  _field: PhantomData<F>,
}

impl<F: PrimeField, const CHUNK_INDEX: usize> SrlSubtable<F, CHUNK_INDEX> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

impl<F: PrimeField, const CHUNK_INDEX: usize> LassoSubtable<F> for SrlSubtable<F, CHUNK_INDEX> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);

    let operand_chunk_width: usize = (log2(M) / 2) as usize;
    let suffix_length = operand_chunk_width * CHUNK_INDEX;

    for idx in 0..M {
      let (x, y) = split_bits(idx, operand_chunk_width);

      let row = x
        .checked_shl(suffix_length as u32)
        .unwrap_or(0)
        .checked_shr((y as u32) % 64)
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

      let m = if (k - b * CHUNK_INDEX) > 0 {
        std::cmp::min(b, (k - b * CHUNK_INDEX))
      } else {
        0
      };

      let shift_x_by_k = (m..b)
        .enumerate()
        .map(|(_, j)| F::from(1_u64 << (b * CHUNK_INDEX + j - k)) * x[b - 1 - j])
        .fold(F::zero(), |acc, val| acc + val);

      result += eq_term * shift_x_by_k;
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{
    jolt::subtable::{srl::SrlSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(srl_materialize_mle_parity, SrlSubtable<Fr, 0>, Fr, 256);
}
