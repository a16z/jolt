use ark_ff::PrimeField;

use crate::dense_mlpoly::EqPolynomial;

use super::SubtableStrategy;

pub enum SparkSubtableStrategy {}

impl<F: PrimeField, const C: usize> SubtableStrategy<F, C> for SparkSubtableStrategy {
  const NUM_SUBTABLES: usize = C;
  const NUM_MEMORIES: usize = C;

  fn materialize_subtables(
    m: usize,
    r: &[Vec<F>; C],
  ) -> [Vec<F>; <Self as SubtableStrategy<F, C>>::NUM_SUBTABLES] {
    std::array::from_fn(|i| {
      let eq_evals = EqPolynomial::new(r[i].clone()).evals();
      assert_eq!(eq_evals.len(), m);
      eq_evals
    })
  }

  fn memory_to_subtable_index(memory_index: usize) -> usize {
    memory_index
  }

  fn memory_to_dimension_index(memory_index: usize) -> usize {
    memory_index
  }

  fn evalute_subtable_mle(subtable_index: usize, r: &[Vec<F>; C], point: &Vec<F>) -> F {
    EqPolynomial::new(r[subtable_index].clone()).evaluate(point)
  }

  fn combine_lookups(vals: &[F; <Self as SubtableStrategy<F, C>>::NUM_MEMORIES]) -> F {
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
  use ark_curve25519::Fr;

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

    let subtable_evals: Subtables<Fr, C, SparkSubtableStrategy> =
      Subtables::new(&[vec![0, 2], vec![2, 2]], &[r_x, r_y], 1 << log_m, 2);

    for (x, expected) in vec![(0, 6), (1, -9), (2, -25), (3, -25)] {
      let calculated = subtable_evals
        .combined_poly
        .evaluate(&index_to_field_bitvector(x, eq_index_bits));
      assert_eq!(calculated, Fr::from(expected));
    }
  }
}
