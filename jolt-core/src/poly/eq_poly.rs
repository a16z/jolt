use crate::field::JoltField;
use rayon::prelude::*;

use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};

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
    pub fn evals(r: &[F]) -> Vec<F> {
        let ell = r.len();

        match ell {
            0..=PARALLEL_THRESHOLD => Self::evals_serial(r, ell, None),
            _ => Self::evals_parallel(r, ell, None),
        }
    }

    /// When evaluating a multilinear polynomial on a point `r`, we first compute the EQ evaluation table
    /// for `r`, then dot-product those values with the coefficients of the polynomial.
    ///
    /// However, if the polynomial in question is a `CompactPolynomial`, its coefficients are represented
    /// by primitive integers while the dot product needs to be computed using Montgomery multiplication.
    ///
    /// To avoid converting every polynomial coefficient to Montgomery form, we can instead introduce an
    /// additional R^2 factor to every element in the EQ evaluation table and performing the dot product
    /// using `JoltField::mul_u64_unchecked`.
    ///
    /// We can efficiently compute the EQ table with this additional R^2 factor by initializing the root of
    /// the dynamic programming tree to R^2 instead of 1.
    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_with_r2")]
    pub fn evals_with_r2(r: &[F]) -> Vec<F> {
        let ell = r.len();

        match ell {
            0..=PARALLEL_THRESHOLD => Self::evals_serial(r, ell, F::montgomery_r2()),
            _ => Self::evals_parallel(r, ell, F::montgomery_r2()),
        }
    }

    /// Computes evals serially. Uses less memory (and fewer allocations) than `evals_parallel`.
    fn evals_serial(r: &[F], ell: usize, r2: Option<F>) -> Vec<F> {
        let mut evals: Vec<F> = vec![r2.unwrap_or(F::one()); ell.pow2()];
        let mut size = 1;
        for j in 0..ell {
            // in each iteration, we double the size of chis
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                // copy each element from the prior iteration twice
                let scalar = evals[i / 2];
                evals[i] = scalar * r[j];
                evals[i - 1] = scalar - evals[i];
            }
        }
        evals
    }

    /// Computes evals in parallel. Uses more memory and allocations than `evals_serial`, but
    /// evaluates biggest layers of the dynamic programming tree in parallel.
    #[tracing::instrument(skip_all, "EqPolynomial::evals_parallel")]
    pub fn evals_parallel(r: &[F], ell: usize, r2: Option<F>) -> Vec<F> {
        let final_size = (2usize).pow(ell as u32);
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
        let mut size = 1;
        evals[0] = r2.unwrap_or(F::one());

        for r in r.iter().rev() {
            let (evals_left, evals_right) = evals.split_at_mut(size);
            let (evals_right, _) = evals_right.split_at_mut(size);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter_mut())
                .for_each(|(x, y)| {
                    *y = *x * *r;
                    *x -= *y;
                });

            size *= 2;
        }

        evals
    }
}

/* This MLE is 1 if y = x + 1 for x in the range [0... 2^l-2].
That is, it ignores the case where x is all 1s, outputting 0.
Assumes x and y are provided big-endian. */
pub fn eq_plus_one<F: JoltField>(x: &[F], y: &[F], l: usize) -> F {
    let one = F::one();

    /* If y+1 = x, then the two bit vectors are of the following form.
        Let k be the longest suffix of 1s in x.
        In y, those k bits are 0.
        Then, the next bit in x is 0 and the next bit in y is 1.
        The remaining higher bits are the same in x and y.
    */
    (0..l)
        .into_par_iter()
        .map(|k| {
            let lower_bits_product = (0..k)
                .map(|i| x[l - 1 - i] * (F::one() - y[l - 1 - i]))
                .product::<F>();
            let kth_bit_product = (F::one() - x[l - 1 - k]) * y[l - 1 - k];
            let higher_bits_product = ((k + 1)..l)
                .map(|i| x[l - 1 - i] * y[l - 1 - i] + (one - x[l - 1 - i]) * (one - y[l - 1 - i]))
                .product::<F>();
            lower_bits_product * kth_bit_product * higher_bits_product
        })
        .sum()
}
