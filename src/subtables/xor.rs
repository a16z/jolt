use ark_ff::PrimeField;
use ark_std::log2;

use crate::utils::split_bits;

use super::SubtableStrategy;

pub enum XorSubtableStrategy {}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
  for XorSubtableStrategy
{
  const NUM_SUBTABLES: usize = 1;
  const NUM_MEMORIES: usize = C;

  fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
    let mut materialized: Vec<F> = Vec::with_capacity(M);
    let bits_per_operand = (log2(M) / 2) as usize;

    // Materialize table in counting order where lhs | rhs counts 0->m
    for idx in 0..M {
      let (lhs, rhs) = split_bits(idx, bits_per_operand);
      materialized.push(F::from((lhs ^ rhs) as u64));
    }

    [materialized]
  }

  fn evaluate_subtable_mle(_: usize, point: &[F]) -> F {
    // (1-x)*y + x*(1-y)
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::zero();
    for i in 0..b {
      let x = x[b - i - 1];
      let y = y[b - i - 1];
      result += F::from(1u64 << (i)) * ((F::one() - x) * y + x * (F::one() - y));
    }
    result
  }

  /// Combine AND table subtable evaluations
  /// T = T'[0] + 2^16*T'[1] + 2^32*T'[2] + 2^48*T'[3]
  /// T'[3] | T'[2] | T'[1] | T'[0]
  fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
    let increment = log2(M) as usize / 2;
    let mut sum = F::zero();
    for (i, val) in vals.iter().enumerate() {
      let weight: u64 = 1u64 << (i * increment);
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
  use crate::{
    materialization_mle_parity_test, subtables::Subtables, utils::index_to_field_bitvector,
  };

  use super::*;
  use ark_curve25519::Fr;

  #[test]
  fn table_materialization_hardcoded() {
    const C: usize = 4;
    const M: usize = 1 << 4;

    let materialized: [Vec<Fr>; 1] =
      <XorSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].len(), M);

    let table: Vec<Fr> = materialized[0].clone();
    assert_eq!(table[0], Fr::from(0b00)); // 00 ^ 00
    assert_eq!(table[1], Fr::from(0b01)); // 00 ^ 01
    assert_eq!(table[2], Fr::from(0b10)); // 00 ^ 10
    assert_eq!(table[3], Fr::from(0b11)); // 00 ^ 11
    assert_eq!(table[4], Fr::from(0b01)); // 01 ^ 00
    assert_eq!(table[5], Fr::from(0b00)); // 01 ^ 01
    assert_eq!(table[6], Fr::from(0b11)); // 01 ^ 10
    assert_eq!(table[7], Fr::from(0b10)); // 01 ^ 11
    assert_eq!(table[8], Fr::from(0b10)); // 10 ^ 00
    assert_eq!(table[9], Fr::from(0b11)); // 10 ^ 01
    assert_eq!(table[10], Fr::from(0b00)); // 10 ^ 10
                                           // ...
  }

  #[test]
  fn combine() {
    const M: usize = 1 << 16;
    let combined: Fr = <XorSubtableStrategy as SubtableStrategy<Fr, 4, M>>::combine_lookups(&[
      Fr::from(100),
      Fr::from(200),
      Fr::from(300),
      Fr::from(400),
    ]);

    // 2^0 * 100 + 2^8 * 200 + 2^16 * 300 + 2^24 * 400
    let expected = (100u64)
      + ((1u64 << 8u64) * 200u64)
      + ((1u64 << 16u64) * 300u64)
      + ((1u64 << 24u64) * 400u64);
    assert_eq!(combined, Fr::from(expected));
  }

  #[test]
  fn valid_merged_poly() {
    const C: usize = 2;
    const M: usize = 1 << 4;

    let x_indices: Vec<usize> = vec![0, 2];
    let y_indices: Vec<usize> = vec![5, 9];

    let subtable_evals: Subtables<Fr, C, M, XorSubtableStrategy> =
      Subtables::new(&[x_indices, y_indices], 2);

    // Real equation here is log2(sparsity) + log2(C)
    let combined_table_index_bits = 2;

    for (x, expected) in [
      (0, 0b00), // xor(0) -> 00 | 00 = 00
      (1, 0b10), // xor(2) -> 00 | 10 = 10
      (2, 0b00), // xor(5) -> 01 | 01 = 00
      (3, 0b11), // xor(9) -> 10 | 01 = 11
    ] {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, combined_table_index_bits));
      assert_eq!(calculated, Fr::from(expected));
    }
  }

  materialization_mle_parity_test!(materialization_parity, XorSubtableStrategy, Fr, 16, 1);
  materialization_mle_parity_test!(
    materialization_parity_nonzero_c,
    XorSubtableStrategy,
    Fr,
    16,
    2
  );
}
