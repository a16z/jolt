use ark_ff::PrimeField;
use ark_std::log2;

use super::SubtableStrategy;

/// Used for lookups in the range [0, 2^LOG_R)
pub enum RangeCheckSubtableStrategy<const LOG_R: usize> {}

impl<F: PrimeField, const C: usize, const M: usize, const LOG_R: usize> SubtableStrategy<F, C, M>
  for RangeCheckSubtableStrategy<LOG_R>
{
  const NUM_SUBTABLES: usize = 3;
  const NUM_MEMORIES: usize = C;

  fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
    assert!(M.is_power_of_two());

    let full: Vec<F> = (0..M).map(|i| F::from(i as u64)).collect();

    let cutoff = 1 << (LOG_R % log2(M) as usize);
    let remainder: Vec<F> = (0..M)
      .map(|i| {
        if i < cutoff {
          F::from(i as u64)
        } else {
          F::zero()
        }
      })
      .collect();

    let zeros: Vec<F> = vec![F::zero(); M];

    [full, remainder, zeros]
  }

  fn evaluate_subtable_mle(subtable_index: usize, point: &[F]) -> F {
    if subtable_index == 0 {
      let b = point.len();
      let mut result = F::zero();
      for i in 0..b {
        result += F::from(1u64 << (i)) * point[b - i - 1];
      }
      result
    } else if subtable_index == 1 {
      let b = point.len();
      let cutoff = LOG_R % (log2(M) as usize);
      let mut result = F::zero();
      for i in 0..b {
        if i < cutoff {
          result += F::from(1u64 << (i)) * point[b - i - 1];
        } else {
          result *= F::one() - point[b - i - 1];
        }
      }
      result
    } else {
      assert_eq!(subtable_index, 2);
      F::zero()
    }
  }

  fn memory_to_subtable_index(memory_index: usize) -> usize {
    let log_m = log2(M) as usize;
    if memory_index * log_m > LOG_R {
      2
    } else {
      usize::from((memory_index + 1) * log_m > LOG_R)
    }
  }

  fn memory_to_dimension_index(memory_index: usize) -> usize {
    memory_index
  }

  /// Combine AND table subtable evaluations
  /// T = T'[0] + 2^16*T'[1] + 2^32*T'[2] + 2^48*T'[3]
  /// T'[3] | T'[2] | T'[1] | T'[0]
  fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
    let log_m = log2(M) as usize;
    let mut sum = F::zero();
    for (i, val) in vals.iter().enumerate() {
      let weight: u64 = 1u64 << (i * log_m);
      sum += F::from(weight) * val;
    }
    sum
  }

  fn g_poly_degree() -> usize {
    1
  }
}

#[cfg(test)]
mod test {
  use crate::{materialization_mle_parity_test, utils::index_to_field_bitvector};

  use super::*;
  use ark_curve25519::Fr;
  use ark_ff::Zero;

  #[test]
  fn table_materialization() {
    const M: usize = 1 << 16;
    let subtables: [Vec<Fr>; 3] =
      <RangeCheckSubtableStrategy<40> as SubtableStrategy<Fr, 4, M>>::materialize_subtables();
    assert_eq!(subtables.len(), 3);

    subtables
      .iter()
      .for_each(|subtable| assert_eq!(subtable.len(), M));

    subtables[0]
      .iter()
      .enumerate()
      .for_each(|(i, &entry)| assert_eq!(entry, Fr::from(i as u64)));

    subtables[1].iter().enumerate().for_each(|(i, &entry)| {
      if i < (1 << 8) {
        assert_eq!(entry, Fr::from(i as u64));
      } else {
        assert_eq!(entry, Fr::zero());
      }
    });

    subtables[2]
      .iter()
      .for_each(|&entry| assert_eq!(entry, Fr::zero()));
  }

  materialization_mle_parity_test!(
    materialization_parity,
    RangeCheckSubtableStrategy::<40>,
    Fr,
    1 << 16,
    3
  );
}
