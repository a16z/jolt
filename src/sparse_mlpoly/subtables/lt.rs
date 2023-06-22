use ark_ff::{PrimeField, One, Zero};
use ark_std::log2;

use crate::{
  dense_mlpoly::DensePolynomial,
  sparse_mlpoly::{densified::DensifiedRepresentation, memory_checking::GrandProducts},
  utils::{pack_field_xyz, split_bits},
};

use super::SubtableStrategy;


pub enum LTSubtableStrategy {}

impl<F: PrimeField, const C: usize> SubtableStrategy<F, C, { C * 2 }> for LTSubtableStrategy {
  fn materialize_subtables(m: usize, _r: &[Vec<F>; C]) -> [Vec<F>; C * 2] {
    let bits_per_operand = (log2(m) / 2) as usize;

    let mut materialized_lt: Vec<F> = Vec::with_capacity(m);
    let mut materialized_eq: Vec<F> = Vec::with_capacity(m);
    
    // Materialize table in counting order where lhs | rhs counts 0->m
    for idx in 0..m {
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

    // TODO: Hack until alpha is removed
    std::array::from_fn(|i| 
        if i % 2 == 0 {
          materialized_lt.clone()
        } else {
          materialized_eq.clone()
        }
    )
  }

  /// LT = (1-x_i)* y_i * eq(x_{>i}, y_{>i})
  fn evalute_subtable_mle(subtable_index: usize, _: &[Vec<F>; C], point: &Vec<F>) -> F {
    debug_assert!(point.len() % 2 == 0);
    let b = point.len() / 2;
    let (x, y) = point.split_at(b);

    // TODO: Improve
    if subtable_index % 2 == 0 { // LT subtable
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
    } else { // EQ subtable
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

  fn to_lookup_polys(
    subtable_entries: &[Vec<F>; C * 2],
    nz: &[Vec<usize>; C],
    s: usize,
  ) -> [DensePolynomial<F>; C * 2] {
    std::array::from_fn(|i| {
      let mut subtable_lookups: Vec<F> = Vec::with_capacity(s);
      for j in 0..s {
        subtable_lookups.push(subtable_entries[i][nz[i / 2][j]]);
      }
      DensePolynomial::new(subtable_lookups)
    })
  }

  fn to_grand_products(
    subtable_entries: &[Vec<F>; C * 2],
    dense: &DensifiedRepresentation<F, C>,
    r_mem_check: &(F, F),
  ) -> [GrandProducts<F>; C * 2] {
    std::array::from_fn(|i| {
      GrandProducts::new(
        &subtable_entries[i],
        &dense.dim[i/2],
        &dense.dim_usize[i/2],
        &dense.read[i/2],
        &dense.r#final[i/2],
        r_mem_check,
      )
    })
  }

  /// Combines lookups into the LT subtables.
  /// Assumes ALPHA lookups are ordered: LT[0], EQ[0], ... LT[C], EQ[C]
  fn combine_lookups(vals: &[F; C * 2]) -> F {
    let mut sum = F::zero();
    let mut eq_prod = F::one();

    for i in 0..C {
        sum += vals[2*i] * eq_prod;
        eq_prod *= vals[2*i + 1];
    }
    sum
  }

  fn sumcheck_poly_degree() -> usize {
    C
  }
}

#[cfg(test)]
mod test {
    use ark_curve25519::Fr;

    use crate::utils::index_to_field_bitvector;

    use super::*; 

    #[test] 
    fn mle() {
      let point: Vec<Fr> = index_to_field_bitvector(0b011_101, 6);
      let eval = LTSubtableStrategy::evalute_subtable_mle(0, &[vec![]], &point);
      assert_eq!(eval, pack_field_xyz(0b011, 0b101, 1, 3));

      let point: Vec<Fr> = index_to_field_bitvector(0b111_011, 6);
      let eval = LTSubtableStrategy::evalute_subtable_mle(0, &[vec![]], &point);
      assert_eq!(eval, pack_field_xyz(0b111, 0b011, 0, 3));

      // Eq
      let point: Vec<Fr> = index_to_field_bitvector(0b011_011, 6);
      let eval = LTSubtableStrategy::evalute_subtable_mle(0, &[vec![]], &point);
      assert_eq!(eval, pack_field_xyz(0b011, 0b011, 0, 3));
    }

    #[test]
    fn combine() {
      const C: usize = 4;
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

      let combined = <LTSubtableStrategy as SubtableStrategy<_, C, { C * 2 }>>::combine_lookups(&vals);
      assert_eq!(combined, expected);
    }

    #[test]
    fn table_materialization() {
      const C: usize = 2;
      let materialized: [Vec<Fr>; C * 2] = LTSubtableStrategy::materialize_subtables(16, &[vec![], vec![]]);
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

      assert_eq!(lt[0], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b00_00, 4)));
      assert_eq!(lt[1], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b00_01, 4)));
      assert_eq!(lt[2], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b00_10, 4)));
      assert_eq!(lt[3], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b00_11, 4)));
      assert_eq!(lt[4], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b01_00, 4)));
      assert_eq!(lt[5], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b01_01, 4)));
      assert_eq!(lt[6], LTSubtableStrategy::evalute_subtable_mle(0, &[], &index_to_field_bitvector(0b01_10, 4)));
      // ...

      assert_eq!(eq[0], Fr::from(0b00_00_01));
      assert_eq!(eq[1], Fr::from(0b00_01_00));
      assert_eq!(eq[2], Fr::from(0b00_10_00));
      assert_eq!(eq[3], Fr::from(0b00_11_00));
      assert_eq!(eq[4], Fr::from(0b01_00_00));
      assert_eq!(eq[5], Fr::from(0b01_01_01));
      assert_eq!(eq[6], Fr::from(0b01_10_00));
      // ...

      assert_eq!(eq[0], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b00_00, 4)));
      assert_eq!(eq[1], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b00_01, 4)));
      assert_eq!(eq[2], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b00_10, 4)));
      assert_eq!(eq[3], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b00_11, 4)));
      assert_eq!(eq[4], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b01_00, 4)));
      assert_eq!(eq[5], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b01_01, 4)));
      assert_eq!(eq[6], LTSubtableStrategy::evalute_subtable_mle(1, &[], &index_to_field_bitvector(0b01_10, 4)));
      // ...
    }
}