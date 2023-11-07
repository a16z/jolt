use ark_ff::PrimeField;
use ark_std::log2;
use std::marker::PhantomData;

use crate::utils::split_bits;

use super::LassoSubtable;

/// Example usage in ADD:
/// Input z is of 65 bits, which is split into 20-bit chunks.
/// This subtable is used to remove the overflow bit from the 4th chunk.
#[derive(Default)]
pub struct TruncateOverflowSubtable<F: PrimeField> {
  _field: PhantomData<F>,
}

impl<F: PrimeField> TruncateOverflowSubtable<F> {
  pub fn new() -> Self {
    Self {
      _field: PhantomData,
    }
  }
}

impl<F: PrimeField> LassoSubtable<F> for TruncateOverflowSubtable<F> {
  fn materialize(&self, M: usize) -> Vec<F> {
    let cutoff = 64 % log2(M) as usize;

    let mut entries: Vec<F> = Vec::with_capacity(M);
    for idx in 0..M {
      let (_, lower_bits) = split_bits(idx, cutoff);
      let row = F::from(lower_bits as u64);
      entries.push(row);
    }
    entries
  }

  fn evaluate_mle(&self, point: &[F]) -> F {
    let log_M = point.len();
    let cutoff = 64 % log_M as usize;

    let mut result = F::zero();
    for i in 0..cutoff {
      result += F::from(1u64 << i) * point[point.len() - 1 - i];
    }
    result
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{
    jolt::subtable::{truncate_overflow::TruncateOverflowSubtable, LassoSubtable},
    subtable_materialize_mle_parity_test,
  };

  subtable_materialize_mle_parity_test!(
    truncate_overflow_materialize_mle_parity,
    TruncateOverflowSubtable<Fr>,
    Fr,
    256
  );
}
