use crate::poly::field::JoltField;
use allocative::Allocative;
use rayon::prelude::*;

use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};

#[derive(Allocative)]
pub struct EqPolynomial<F> {
    r: Vec<F>,
}

const PARALLEL_THRESHOLD: usize = 16;

impl<F: JoltField> EqPolynomial<F> {
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
    #[tracing::instrument(skip_all, "EqPolynomial::evals_parallel")]
    pub fn evals_parallel(&self, ell: usize) -> Vec<F> {
        let final_size = (2usize).pow(ell as u32);
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
        let mut size = 1;
        evals[0] = F::one();

        for r in self.r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * r;
                    *x -= &*y;
                });

            size *= 2;
        }

        evals
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::compute_factored_evals")]
    pub fn compute_factored_evals(&self, L_size: usize) -> (Vec<F>, Vec<F>) {
        let ell = self.r.len();
        let left_num_vars = L_size.log_2();

        let L = EqPolynomial::new(self.r[..left_num_vars].to_vec()).evals();
        let R = EqPolynomial::new(self.r[left_num_vars..ell].to_vec()).evals();

        (L, R)
    }
}
