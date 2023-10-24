use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use crate::utils::split_bits;
use std::cmp;

use super::LassoSubtable;

#[derive(Default)]
pub struct LOWER9Subtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> LOWER9Subtable<F> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

/// Used in ADD
/// Input z is of 65 bits, which is split into 5 x 11-bit chunks + 1 x 10-bit chunk.
/// This subtable removes the MSB from the 10-bit, returning only the lower 9 bits. 
impl<F: PrimeField> LassoSubtable<F> for LOWER9Subtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);

    for idx in 0..M {
      let (_, lower9) = split_bits(idx, 9);
      let row = F::from(lower9 as u64);
      entries.push(row);
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    let mut result = F::zero();
    let b = std::cmp::min(point.len(), 9);

    for i in 0..b {
      result += F::from(1u64 << i) * point[point.len() - 1 - i];
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{
    jolt::subtable::{lower9::LOWER9Subtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(lower9_materialize_mle_parity, LOWER9Subtable<Fr>, Fr, 256);
}
