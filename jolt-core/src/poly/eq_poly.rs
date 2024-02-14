use ark_ff::PrimeField;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::utils::math::Math;

pub struct EqPolynomial<F> {
    r: Vec<F>,
}

const PARALLEL_THRESHOLD: usize = 16;

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

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals")]
    pub fn evals(&self) -> Vec<F> {
        let ell = self.r.len();

        match ell {
            0..=PARALLEL_THRESHOLD => self.evals_serial(ell),
            _ => self.evals_parallel(ell),
        }
    }

    /// Computes evals serially. Uses less memory (and fewer allocations) than `evals_parallel`.
    fn evals_serial(&self, ell: usize) -> Vec<F> {
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

    /// Computes evals in parallel. Uses more memory and allocations than `evals_serial`, but
    /// evaluates biggest layers of the dynamic programming tree in parallel.
    fn evals_parallel(&self, ell: usize) -> Vec<F> {
        let mut previous_evals: Vec<F>;
        let mut evals: Vec<F> = vec![F::one()];
        for j in 1..=ell {
            previous_evals = evals;
            evals = match j {
                0..=PARALLEL_THRESHOLD => previous_evals
                    .iter()
                    .flat_map(|&eval| {
                        let x1 = eval * self.r[j - 1];
                        let x0 = eval - x1;
                        vec![x0, x1]
                    })
                    .collect(),
                _ => previous_evals
                    .par_iter()
                    .flat_map_iter(|&eval| {
                        let x1 = eval * self.r[j - 1];
                        let x0 = eval - x1;
                        vec![x0, x1]
                    })
                    .collect(),
            };
        }
        evals
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial.compute_factored_evals")]
    pub fn compute_factored_evals(&self) -> (Vec<F>, Vec<F>) {
        let ell = self.r.len();
        let (left_num_vars, _right_num_vars) = super::hyrax::matrix_dimensions(ell);

        let L = EqPolynomial::new(self.r[..left_num_vars].to_vec()).evals();
        let R = EqPolynomial::new(self.r[left_num_vars..ell].to_vec()).evals();

        (L, R)
    }
}
