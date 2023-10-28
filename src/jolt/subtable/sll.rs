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
      // shift x by the length represented by the lower 6 bits of y
      let row = F::from((x << (y % 64)) as u64);
      entries.push(row);
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;

    let MAX_SHIFT= 64;
    let log_MAX_SHIFT: usize= log2(MAX_SHIFT) as usize;

    let (x, y) = point.split_at(b);

    println!("x: {:?}, y: {:?}", x, y);

    let mut result = F::zero();
    for k in 0..std::cmp::min(MAX_SHIFT, 1<<b) {
      let k_bits = (k as usize)
        .get_bits(log_MAX_SHIFT).iter()
        .map(|bit| F::from(*bit as u64))
        .collect::<Vec<F>>(); // big-endian
      let mut eq_term = F::one();
      for i in 0..std::cmp::min(log_MAX_SHIFT, b) {
        eq_term *= k_bits[log_MAX_SHIFT-1-i] * y[b-1-i] + (F::one() - k_bits[log_MAX_SHIFT-1-i]) * (F::one() - y[b-1-i]);
      }

      let mut shift_x_by_k = F::zero();
      // let m = std::cmp::min(
      //   b as i32, 
      //   std::cmp::max(
      //     0_i32, 
      //     (k + b * (self.chunk_idx+1) - 64) as i32 
      //   )
      // );

      let m = if (k + b * (self.chunk_idx+1)) > 64 {
        std::cmp::min(b, (k + b * (self.chunk_idx+1)) - 64)
      } else {
        0
      };

      let m_prime = b - (m as usize) - 1;
      println!("k: {}, m_prime: {}", k, m_prime);
      for j in 0..m_prime+1 {
        shift_x_by_k += F::from((1_u64 << (j+k)) as u64) * x[b-1-j];
        println!("k: {}, eq_term: {}, j: {}, x[j]: {}", k, eq_term, j, x[j]);
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

  subtable_materialize_mle_parity_test!(sll_materialize_mle_parity, SllSubtable<Fr>, Fr, 256, 0);
}
