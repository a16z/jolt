use ark_ff::PrimeField;
use ark_std::log2;

use crate::utils::{pack_field_xyz, split_bits};

use super::SubtableStrategy;

pub enum AndSubtableStrategy {}

impl<F: PrimeField, const C: usize> SubtableStrategy<F, C> for AndSubtableStrategy {
  const NUM_SUBTABLES: usize = 1;
  const NUM_MEMORIES: usize = C;

  fn materialize_subtables(
    m: usize,
    _r: &[Vec<F>; C],
  ) -> [Vec<F>; <Self as SubtableStrategy<F, C>>::NUM_SUBTABLES] {
    let mut materialized: Vec<F> = Vec::with_capacity(m);
    let bits_per_operand = (log2(m) / 2) as usize;

    // Materialize table in counting order where lhs | rhs counts 0->m
    for idx in 0..m {
      let (lhs, rhs) = split_bits(idx, bits_per_operand);
      let out = lhs & rhs;

      // Note packs memory T[row] = lhs | rhs | out -- x controls highest order bits
      let row = pack_field_xyz(lhs, rhs, out, bits_per_operand);
      materialized.push(row);
    }

    [materialized]
  }

  fn evaluate_subtable_mle(_: usize, _: &[Vec<F>; C], point: &Vec<F>) -> F {
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    let mut result = F::zero();
    for i in 0..b {
      result += F::from(1u64 << (i)) * x[b - i - 1] * y[b - i - 1];
      result += F::from(1u64 << (b + i)) * y[b - i - 1];
      result += F::from(1u64 << (2 * b + i)) * x[b - i - 1];
    }
    result
  }

  /// Combine AND table subtable evaluations
  /// T = T'[0] + 2^16*T'[1] + 2^32*T'[2] + 2^48*T'[3]
  /// T'[3] | T'[2] | T'[1] | T'[0]
  /// x3 | y3 | z3 | x2 | y2 | z2 | x1 | y1 | z1 | x0 | y0 | z0 |
  fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C>>::NUM_MEMORIES]) -> F {
    let increment = 64 / C; // TODO: Generalize 64 to M
    let mut sum = F::zero();
    for i in 0..C {
      let weight: u64 = 1u64 << (i * increment);
      sum += F::from(weight) * vals[i];
    }
    sum
  }

  fn sumcheck_poly_degree() -> usize {
    1
  }
}

#[cfg(test)]
mod test {
  use crate::{sparse_mlpoly::subtables::Subtables, utils::index_to_field_bitvector, materialization_mle_parity_test};

  use super::*;
  use ark_curve25519::Fr;
  use ark_ff::Zero;

  #[test]
  fn table_materialization_hardcoded() {
    const C: usize = 4;
    const M: usize = 1 << 4;

    let materialized: [Vec<Fr>; 1] =
      AndSubtableStrategy::materialize_subtables(M, &[vec![], vec![], vec![], vec![]]);
    assert_eq!(materialized.len(), 1);
    assert_eq!(materialized[0].len(), M);

    let table: Vec<Fr> = materialized[0].clone();
    assert_eq!(table[0], Fr::from(0b00_00_00));
    assert_eq!(table[1], Fr::from(0b00_01_00));
    assert_eq!(table[2], Fr::from(0b00_10_00));
    assert_eq!(table[3], Fr::from(0b00_11_00));
    assert_eq!(table[4], Fr::from(0b01_00_00));
    assert_eq!(table[5], Fr::from(0b01_01_01));
    assert_eq!(table[6], Fr::from(0b01_10_00));
    assert_eq!(table[7], Fr::from(0b01_11_01));
    assert_eq!(table[8], Fr::from(0b10_00_00));
    assert_eq!(table[9], Fr::from(0b10_01_00));
    assert_eq!(table[10], Fr::from(0b10_10_10));
    // ...
  }

  #[test]
  fn evaluate_mle() {
    let x = vec![Fr::from(0), Fr::from(1), Fr::from(1), Fr::from(0)];
    let y = vec![Fr::from(1), Fr::from(0), Fr::from(1), Fr::from(1)];
    let x_and_y = AndSubtableStrategy::evaluate_subtable_mle(0, &[], &[x, y].concat());
    assert_eq!(x_and_y, Fr::from(0b0110_1011_0010));
  }

  #[test]
  fn combine() {
    let combined: Fr = <AndSubtableStrategy as SubtableStrategy<Fr, 4>>::combine_lookups(&[
      Fr::from(100),
      Fr::from(200),
      Fr::from(300),
      Fr::from(400),
    ]);

    // 2^0 * 100 + 2^16 * 200 + 2^32 * 300 + 2^48 * 400
    let expected = (1u64 * 100u64)
      + ((1u64 << 16u64) * 200u64)
      + ((1u64 << 32u64) * 300u64)
      + ((1u64 << 48u64) * 400u64);
    assert_eq!(combined, Fr::from(expected));
  }

  #[test]
  fn valid_merged_poly() {
    const C: usize = 2;
    const M: usize = 1 << 4;
    let log_m = log2(M) as usize;

    let x_indices: Vec<usize> = vec![0, 2];
    let y_indices: Vec<usize> = vec![5, 9];

    let r_x: Vec<Fr> = vec![Fr::zero(), Fr::zero()]; // unused
    let r_y: Vec<Fr> = vec![Fr::zero(), Fr::zero()]; // unused

    let subtable_evals: Subtables<Fr, C, AndSubtableStrategy> =
      Subtables::new(&[x_indices, y_indices], &[r_x, r_y], 1 << log_m, 2);

    // Real equation here is log2(sparsity) + log2(C)
    let combined_table_index_bits = 2;

    for (x, expected) in vec![
      (0, 0b00_00_00), // and(0) -> 00 & 00 = 00 -> 00_00_00
      (1, 0b00_10_00), // and(2) -> 00 & 10 = 00 -> 00_10_00
      (2, 0b01_01_01), // and(5) -> 01 & 01 = 01 -> 01_01_01
      (3, 0b10_01_00), // and(9)  -> 10 & 01 = 00 -> 10_01_00
    ] {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, combined_table_index_bits));
      assert_eq!(calculated, Fr::from(expected));
    }
  }

  materialization_mle_parity_test!(materialization_parity, AndSubtableStrategy, Fr, 16, 1);
}
