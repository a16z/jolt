use ark_ff::PrimeField;

use crate::{dense_mlpoly::{EqPolynomial, DensePolynomial}, sparse_mlpoly::{densified::DensifiedRepresentation, memory_checking::GrandProducts}};

use super::SubtableStrategy;

pub enum SparkSubtableStrategy {}

impl<F: PrimeField, const C: usize> SubtableStrategy<F, C, C> for SparkSubtableStrategy {
  fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> [Vec<F>; C] {
    std::array::from_fn(|i| {
      let eq_evals = EqPolynomial::new(r[i].clone()).evals();
      assert_eq!(eq_evals.len(), m);
      eq_evals
    })
  }

  fn to_lookup_polys(
    subtable_entries: &[Vec<F>; C],
    nz: &[Vec<usize>; C],
    s: usize,
  ) -> [DensePolynomial<F>; C] {
    std::array::from_fn(|i| {
      let mut subtable_lookups: Vec<F> = Vec::with_capacity(s);
      for j in 0..s {
        subtable_lookups.push(subtable_entries[i][nz[i][j]]);
      }
      DensePolynomial::new(subtable_lookups)
    })
  }

  fn to_grand_products(
    subtable_entries: &[Vec<F>; C],
    dense: &DensifiedRepresentation<F, C>,
    r_mem_check: &(F, F),
  ) -> [GrandProducts<F>; C] {
    std::array::from_fn(|i| {
      GrandProducts::new(
        &subtable_entries[i],
        &dense.dim[i],
        &dense.dim_usize[i],
        &dense.read[i],
        &dense.r#final[i],
        r_mem_check,
      )
    })
  }

  fn combine_lookups(vals: &[F; C]) -> F {
    vals.iter().product()
  }

  fn sumcheck_poly_degree() -> usize {
    C
  }
}


#[cfg(test)]
mod test {
  use super::*;

  use crate::sparse_mlpoly::subtables::Subtables;
  use crate::utils::index_to_field_bitvector;
  use ark_bls12_381::Fr;

  #[test]
  fn forms_valid_merged_dense_poly() {
    // Pass in the eq evaluations over log_m boolean variables and log_m fixed variables r
    let log_m = 2;
    const C: usize = 2;

    let r_x: Vec<Fr> = vec![Fr::from(3), Fr::from(4)];
    let r_y: Vec<Fr> = vec![Fr::from(5), Fr::from(6)];

    let eq_index_bits = 2;
    // eq(x,y) = prod{x_i * y_i + (1-x_i) * (1-y_i)}
    // eq(0) = eq(0, 0, 3, 4) = (0 * 3 + (1-0) * (1-3)) * (0 * 4 + (1-0) * (1-4)) = (-2)(-3) = 6
    // eq(2) = eq(0, 1, 3, 4) = (0 * 3 + (1-0) * (1-3)) * (1 * 4 + (1-1) * (1-4)) = (-2)(4) = -8
    // Second poly...
    // eq(2) = eq(1, 0, 5, 6) = (1 * 5 + (1-1) * (1-5)) * (0 * 6 + (1-0) * (1-6)) = (5)(-5) = -25
    // eq(2) = eq(1, 0, 5, 6) = (1 * 5 + (1-1) * (1-5)) * (0 * 6 + (1-0) * (1-6)) = (5)(-5) = -25

    let subtable_evals: Subtables<Fr, C, C, SparkSubtableStrategy> =
      Subtables::new(&[vec![0,2], vec![2,2]], &[r_x, r_y], 1 << log_m, 2);

    for (x, expected) in vec![
      (0, 6),
      (1, -9),
      (2, -25),
      (3, -25),
    ] {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, eq_index_bits));
      assert_eq!(
        calculated,
        Fr::from(expected)
      );
    }
  }
}
