use ark_ff::PrimeField;
use ark_std::log2;

use crate::utils::{pack_field_xyz, split_bits};

use super::SubtableStrategy;

pub enum LTSubtableStrategy {}

impl<F: PrimeField, const C: usize, const M: usize> SubtableStrategy<F, C, M>
  for LTSubtableStrategy
{
  const NUM_SUBTABLES: usize = 2;
  const NUM_MEMORIES: usize = 2 * C;

  fn materialize_subtables() -> [Vec<F>; <Self as SubtableStrategy<F, C, M>>::NUM_SUBTABLES] {
    let bits_per_operand = (log2(M) / 2) as usize;

    let mut materialized_lt: Vec<F> = Vec::with_capacity(M);
    let mut materialized_eq: Vec<F> = Vec::with_capacity(M);

    // Materialize table in counting order where lhs | rhs counts 0->m
    for idx in 0..M {
      let (lhs, rhs) = split_bits(idx, bits_per_operand);

      // Note packs memory T[row] = lhs | rhs | 0 / 1 -- x controls highest order bits
      let row_lt = if lhs < rhs {
        pack_field_xyz(lhs, rhs, 1, bits_per_operand)
      } else {
        pack_field_xyz(lhs, rhs, 0, bits_per_operand)
      };
      materialized_lt.push(row_lt);

      let row_eq = if lhs == rhs {
        pack_field_xyz(lhs, rhs, 1, bits_per_operand)
      } else {
        pack_field_xyz(lhs, rhs, 0, bits_per_operand)
      };
      materialized_eq.push(row_eq);
    }

    [materialized_lt, materialized_eq]
  }

  /// LT = (1-x_i)* y_i * eq(x_{>i}, y_{>i})
  fn evaluate_subtable_mle(subtable_index: usize, point: &Vec<F>) -> F {
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    // TODO: Improve
    if subtable_index % 2 == 0 {
      // LT subtable
      let mut bitpacked = F::zero();
      let mut result = F::zero();
      let mut eq_term = F::one();
      for i in 0..b {
        bitpacked += F::from(1u64 << (2 * b + i)) * x[b - i - 1];
        bitpacked += F::from(1u64 << (b + i)) * y[b - i - 1];

        result += (F::one() - x[i]) * y[i] * eq_term;
        eq_term *= F::one() - x[i] - y[i] + F::from(2u64) * x[i] * y[i];
      }
      bitpacked + result
    } else {
      // EQ subtable
      let mut bitpacked = F::zero();
      let mut eq_term = F::one();
      for i in 0..b {
        bitpacked += F::from(1u64 << (2 * b + i)) * x[b - i - 1];
        bitpacked += F::from(1u64 << (b + i)) * y[b - i - 1];
        eq_term *= F::one() - x[i] - y[i] + F::from(2u64) * x[i] * y[i];
      }
      bitpacked + eq_term
    }
  }

  /// Combines lookups into the LT subtables.
  /// Assumes ALPHA lookups are ordered: LT[0], EQ[0], ... LT[C], EQ[C]
  fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C, M>>::NUM_MEMORIES]) -> F {
    let mut sum = F::zero();
    let mut eq_prod = F::one();

    for i in 0..C {
      sum += vals[2 * i] * eq_prod;
      eq_prod *= vals[2 * i + 1];
    }
    sum
  }

  fn g_poly_degree() -> usize {
    C
  }
}

#[cfg(test)]
mod test {
  use ark_curve25519::Fr;

  use crate::{materialization_mle_parity_test, utils::index_to_field_bitvector};

  use super::*;

  #[test]
  fn mle() {
    const C: usize = 1;
    const M: usize = 64;
    let point: Vec<Fr> = index_to_field_bitvector(0b011_101, 6);
    let eval = <LTSubtableStrategy as SubtableStrategy<Fr, C, M>>::evaluate_subtable_mle(0, &point);
    assert_eq!(eval, pack_field_xyz(0b011, 0b101, 1, 3));

    let point: Vec<Fr> = index_to_field_bitvector(0b111_011, 6);
    let eval = <LTSubtableStrategy as SubtableStrategy<Fr, C, M>>::evaluate_subtable_mle(0, &point);
    assert_eq!(eval, pack_field_xyz(0b111, 0b011, 0, 3));

    // Eq
    let point: Vec<Fr> = index_to_field_bitvector(0b011_011, 6);
    let eval = <LTSubtableStrategy as SubtableStrategy<Fr, C, M>>::evaluate_subtable_mle(0, &point);
    assert_eq!(eval, pack_field_xyz(0b011, 0b011, 0, 3));
  }

  #[test]
  fn combine() {
    const C: usize = 4;
    const M: usize = 16;
    let vals: [Fr; C * 2] = [
      Fr::from(10u64), // LT[0]
      Fr::one(),       // EQ[0]
      Fr::from(20u64), // LT[1]
      Fr::zero(),      // EQ[1]
      Fr::from(30u64), // LT[2]
      Fr::one(),       // EQ[2]
      Fr::from(40u64), // LT[3]
      Fr::one(),       // EQ[3]
    ];

    // LT = LT[0]
    //  + LT[1] * EQ[0]
    //  + LT[2] * EQ[0] * EQ[1]
    //  + LT[3] * EQ[0] * EQ[1] * EQ[2]
    let expected = Fr::from(10u64)
      + Fr::from(20u64) * Fr::one()
      + Fr::from(30u64) * Fr::one() * Fr::zero()
      + Fr::from(40u64) * Fr::one() * Fr::zero() * Fr::one();

    let combined = <LTSubtableStrategy as SubtableStrategy<_, C, M>>::combine_lookups(&vals);
    assert_eq!(combined, expected);
  }

  #[test]
  fn table_materialization_hardcoded() {
    const C: usize = 2;
    const M: usize = 16;
    let materialized: [Vec<Fr>; 2] =
      <LTSubtableStrategy as SubtableStrategy<Fr, C, M>>::materialize_subtables();
    let lt = materialized[0].clone();
    let eq = materialized[1].clone();

    assert_eq!(lt[0], Fr::from(0b00_00_00));
    assert_eq!(lt[1], Fr::from(0b00_01_01));
    assert_eq!(lt[2], Fr::from(0b00_10_01));
    assert_eq!(lt[3], Fr::from(0b00_11_01));
    assert_eq!(lt[4], Fr::from(0b01_00_00));
    assert_eq!(lt[5], Fr::from(0b01_01_00));
    assert_eq!(lt[6], Fr::from(0b01_10_01));
    // ...

    assert_eq!(eq[0], Fr::from(0b00_00_01));
    assert_eq!(eq[1], Fr::from(0b00_01_00));
    assert_eq!(eq[2], Fr::from(0b00_10_00));
    assert_eq!(eq[3], Fr::from(0b00_11_00));
    assert_eq!(eq[4], Fr::from(0b01_00_00));
    assert_eq!(eq[5], Fr::from(0b01_01_01));
    assert_eq!(eq[6], Fr::from(0b01_10_00));
    // ...
  }

  materialization_mle_parity_test!(
    lt_materialization_parity_test,
    LTSubtableStrategy,
    Fr,
    /* m = */ 16,
    /* NUM_SUBTABLES = */ 2
  );
}
