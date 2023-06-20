use ark_ff::PrimeField;

use crate::{dense_mlpoly::{EqPolynomial, DensePolynomial}, sparse_mlpoly::{densified::DensifiedRepresentation, memory_checking::GrandProducts}};

use super::SubtableStrategy;

pub enum AndSubtableStrategy {}

impl<F: PrimeField, const C: usize> SubtableStrategy<F, C, C> for AndSubtableStrategy {
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