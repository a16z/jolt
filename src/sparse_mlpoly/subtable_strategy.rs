use ark_ff::PrimeField;

use crate::dense_mlpoly::EqPolynomial;

pub trait SubtableStrategy<F: PrimeField, const C: usize> {
  /// Materialize 'c' copies of 'k' different subtables indexed [0, ..., \alpha] where \alpha = c * k
  /// Note: Some materializations will not use the parameter r.
  /// Note: Some materializations will have \alpha copies of the same subtable, which is a wasteful artifact
  /// of supporting Sparkplug and Surge within the same repo.
  /// 
  /// Params
  /// - `m`: size of subtable / number of evaluations to materialize
  /// - `r`: point at which to materialize the table (potentially unused)
  fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> Vec<Vec<F>>;

  fn k() -> usize;

  fn alpha() -> usize {
    C * Self::k()
  }
}

pub enum EqSubtableStrategy {}

impl<F: PrimeField, const C: usize> SubtableStrategy<F, C> for EqSubtableStrategy {
  fn materialize_subtables(m: usize, r: &[Vec<F>; C]) -> Vec<Vec<F>> {
    r
      .iter()
      .map(|r_dim| {
        let eq_evals = EqPolynomial::new(r_dim.clone()).evals();
        assert_eq!(eq_evals.len(), m);
        eq_evals
      })
      .collect()
  }

  fn k() -> usize {
    1usize
  }
}
