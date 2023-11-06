use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;
use std::sync::Arc;

use super::LassoSubtable;
use crate::utils::math::Math;
use crate::utils::split_bits;

#[derive(Default)]
pub struct SraSignSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> SraSignSubtable<F> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for SraSignSubtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);

    let operand_chunk_width: usize = (log2(M) / 2) as usize;

    // find position of sign bit in the chunk 
    let sign_bit_index = 63 % operand_chunk_width; 

    for idx in 0..M {
      let (x, y) = split_bits(idx, operand_chunk_width);

      let x_sign = F::from(((x >> sign_bit_index) & 1) as u64);

      let row = (0..(y as u32) % 64).into_iter()
        .fold(F::zero(), |acc, i: u32| acc + F::from(1_u64 << (64-1-i)) * x_sign);

      entries.push(F::from(row));
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    // first half is chunk X_i
    // and second half is always chunk Y_0
    debug_assert!(point.len() % 2 == 0);

    const MAX_SHIFT: usize = 64;
    let log_MAX_SHIFT = log2(MAX_SHIFT) as usize;

    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::zero();

    let sign_index = 63 % b;
    let x_sign = x[b-1-sign_index];

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

      let x_sign_upper = (0..k).into_iter()
      .fold(F::zero(), |acc, i| acc + F::from(1_u64 << (64-1-i)) * x_sign);

      result += eq_term * x_sign_upper;
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{
    jolt::subtable::{sra_sign::SraSignSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(sra_sign_materialize_mle_parity, SraSignSubtable<Fr>, Fr, 256);
}
