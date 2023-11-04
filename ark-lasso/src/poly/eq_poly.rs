use ark_ff::PrimeField;

use crate::utils::math::Math;

pub struct EqPolynomial<F> {
  r: Vec<F>,
}

impl<F: PrimeField> EqPolynomial<F> {
  pub fn new(r: Vec<F>) -> Self {
    EqPolynomial { r }
  }

  pub fn evaluate(&self, rx: &[F]) -> F {
    assert_eq!(self.r.len(), rx.len());
    (0..rx.len())
      .map(|i| self.r[i] * rx[i] + (F::one() - self.r[i]) * (F::one() - rx[i]))
      .product()
  }

  #[tracing::instrument(skip_all, name = "EqPolynomial.evals")]
  pub fn evals(&self) -> Vec<F> {
    let ell = self.r.len();

    let mut evals: Vec<F> = vec![F::one(); ell.pow2()];
    let mut size = 1;
    for j in 0..ell {
      // in each iteration, we double the size of chis
      size *= 2;
      for i in (0..size).rev().step_by(2) {
        // copy each element from the prior iteration twice
        let scalar = evals[i / 2];
        evals[i] = scalar * self.r[j];
        evals[i - 1] = scalar - evals[i];
      }
    }
    evals
  }

  pub fn compute_factored_lens(ell: usize) -> (usize, usize) {
    (ell / 2, ell - ell / 2)
  }

  pub fn compute_factored_evals(&self) -> (Vec<F>, Vec<F>) {
    let ell = self.r.len();
    let (left_num_vars, _right_num_vars) = Self::compute_factored_lens(ell);

    let L = EqPolynomial::new(self.r[..left_num_vars].to_vec()).evals();
    let R = EqPolynomial::new(self.r[left_num_vars..ell].to_vec()).evals();

    (L, R)
  }
}
