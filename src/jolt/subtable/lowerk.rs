use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use crate::utils::split_bits;
use std::cmp;

use super::LassoSubtable;

/// Example usage in ADD:
/// Input z is of 65 bits, which is split into 5 x 11-bit chunks + 1 x 10-bit chunk.
/// This subtable removes the MSB from the 10-bit, returning only the lower k bits. 
#[derive(Default)]
pub struct LOWERKSubtable<F: PrimeField> {
  _field: PhantomData<F>,
  k: usize,
}

impl<F: PrimeField> LOWERKSubtable<F> {
  pub fn new(k: usize) -> Self {
    Self {
      _field: PhantomData,
      k: k,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for LOWERKSubtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let mut entries: Vec<F> = Vec::with_capacity(M);

    for idx in 0..M {
      let (_, lowerk) = split_bits(idx, self.k);
      let row = F::from(lowerk as u64);
      entries.push(row);
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    let mut result = F::zero();
    let b = std::cmp::min(point.len(), self.k);

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
    jolt::subtable::{lowerk::LOWERKSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(lowerk_materialize_mle_parity, LOWERKSubtable<Fr>, Fr, 256, 9);
}
