use crate::field::JoltField;
use crate::poly::opening_proof::{Endianness, OpeningPoint};
use crate::utils::{math::Math, thread::unsafe_allocate_zero_vec};
use rayon::prelude::*;
use std::{
    marker::PhantomData,
    ops::{Mul, Sub},
};

const PARALLEL_THRESHOLD: usize = 16;

pub struct EqPolynomial<F: JoltField>(PhantomData<F>);
impl<F: JoltField> EqPolynomial<F> {
    pub fn mle<X, Y>(x: &[X], y: &[Y]) -> F
    where
        X: Copy + Send + Sync,
        Y: Copy + Send + Sync,
        F: JoltField + Sub<X, Output = F> + Sub<Y, Output = F>,
        X: Mul<Y, Output = F>,
    {
        assert_eq!(x.len(), y.len());
        x.par_iter()
            .zip(y.par_iter())
            .map(|(x_i, y_i)| *x_i * *y_i + (F::one() - *x_i) * (F::one() - *y_i))
            .product()
    }

    /// Computes the MLE evaluation EQ(x, y)
    pub fn mle_endian<const E1: Endianness, const E2: Endianness>(
        x: &OpeningPoint<E1, F>,
        y: &OpeningPoint<E2, F>,
    ) -> F {
        assert_eq!(x.len(), y.len());
        if E1 == E2 {
            x.r.par_iter()
                .zip(y.r.par_iter())
                .map(|(x_i, y_i)| *x_i * y_i + (F::one() - x_i) * (F::one() - y_i))
                .product()
        } else {
            x.r.par_iter()
                .zip(y.r.par_iter().rev())
                .map(|(x_i, y_i)| *x_i * y_i + (F::one() - x_i) * (F::one() - y_i))
                .product()
        }
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals")]
    /// Computes the table of coefficients: `{eq(r, x) for all x in {0, 1}^n}`
    /// If `scaling_factor` is provided, computes `scaling_factor * eq(r, x)` instead.
    pub fn evals<C>(r: &[C]) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        Self::evals_with_scaling(r, None)
    }

    /// Computes the table of coefficients: `scaling_factor * eq(r, x) for all x in {0, 1}^n`
    /// If `scaling_factor` is None, defaults to 1 (no scaling).
    #[inline]
    pub fn evals_with_scaling<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        match r.len() {
            0..=PARALLEL_THRESHOLD => Self::evals_serial(r, scaling_factor),
            _ => Self::evals_parallel(r, scaling_factor),
        }
    }

    #[tracing::instrument(skip_all, name = "EqPolynomial::evals_cached")]
    /// Computes the table of coefficients like `evals`, but also caches the intermediate results
    ///
    /// In other words, computes `{eq(r[i..], x) for all x in {0, 1}^{n - i}}` and for all `i in
    /// 0..r.len()`.
    pub fn evals_cached<C>(r: &[C]) -> Vec<Vec<F>>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        // TODO: implement parallel version & determine switchover point
        Self::evals_serial_cached(r, None)
    }

    /// Same as evals_cached but for high-to-low (reverse) binding order
    pub fn evals_cached_rev(r: &[F::Challenge]) -> Vec<Vec<F>> {
        Self::evals_serial_cached_rev(r, None)
    }

    /// Computes the table of coefficients:
    ///     scaling_factor * eq(r, x) for all x in {0, 1}^n
    /// serially. More efficient for short `r`.
    #[inline]
    pub fn evals_serial<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let mut evals: Vec<F> = vec![scaling_factor.unwrap_or(F::one()); r.len().pow2()];
        let mut size = 1;
        for j in 0..r.len() {
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

    /// Computes the table of coefficients like `evals_serial`, but also caches the intermediate results.
    ///
    /// Returns a vector of vectors, where the `j`th vector contains the coefficients for the polynomial
    /// `eq(r[..j], x)` for all `x in {0, 1}^{j}`.
    ///
    /// Performance seems at most 10% worse than `evals_serial`
    #[inline]
    pub fn evals_serial_cached<C>(r: &[C], scaling_factor: Option<F>) -> Vec<Vec<F>>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            size *= 2;
            for i in (0..size).rev().step_by(2) {
                let scalar = evals[j][i / 2];
                evals[j + 1][i] = scalar * r[j];
                evals[j + 1][i - 1] = scalar - evals[j + 1][i];
            }
        }
        evals
    }
    /// evals_serial_cached but for "high to low" ordering, used specifically in the Gruen x Dao Thaler optimization.
    pub fn evals_serial_cached_rev(r: &[F::Challenge], scaling_factor: Option<F>) -> Vec<Vec<F>> {
        let rev_r = r.iter().rev().collect::<Vec<_>>();
        let mut evals: Vec<Vec<F>> = (0..r.len() + 1)
            .map(|i| vec![scaling_factor.unwrap_or(F::one()); 1 << i])
            .collect();
        let mut size = 1;
        for j in 0..r.len() {
            for i in 0..size {
                let scalar = evals[j][i];
                let multiple = 1 << j;
                evals[j + 1][i + multiple] = scalar * *rev_r[j];
                evals[j + 1][i] = scalar - evals[j + 1][i + multiple];
            }
            size *= 2;
        }
        evals
    }

    /// Computes the table of coefficients:
    ///
    /// scaling_factor * eq(r, x) for all x in {0, 1}^n
    ///
    /// computing biggest layers of the dynamic programming tree in parallel.
    #[tracing::instrument(skip_all, "EqPolynomial::evals_parallel")]
    #[inline]
    pub fn evals_parallel<C>(r: &[C], scaling_factor: Option<F>) -> Vec<F>
    where
        C: Copy + Send + Sync + Into<F>,
        F: std::ops::Mul<C, Output = F> + std::ops::SubAssign<F>,
    {
        let final_size = r.len().pow2();
        let mut evals: Vec<F> = unsafe_allocate_zero_vec(final_size);
        let mut size = 1;
        evals[0] = scaling_factor.unwrap_or(F::one());

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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;
    use std::time::Instant;

    #[test]
    /// Test that the results of running `evals_serial`, `evals_parallel`, and `evals_serial_cached`
    /// (taking the last vector) are the same (and also benchmark them)
    fn test_evals() {
        let mut rng = test_rng();
        for len in 5..22 {
            let r = (0..len)
                .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                .collect::<Vec<_>>();
            let start = Instant::now();
            let evals_serial: Vec<Fr> = EqPolynomial::evals_serial(&r, None);
            let end_first = Instant::now();
            let evals_parallel = EqPolynomial::evals_parallel(&r, None);
            let end_second = Instant::now();
            let evals_serial_cached = EqPolynomial::evals_serial_cached(&r, None);
            let end_third = Instant::now();
            println!(
                "len: {}, Time taken to compute evals_serial: {:?}",
                len,
                end_first - start
            );
            println!(
                "len: {}, Time taken to compute evals_parallel: {:?}",
                len,
                end_second - end_first
            );
            println!(
                "len: {}, Time taken to compute evals_serial_cached: {:?}",
                len,
                end_third - end_second
            );
            assert_eq!(evals_serial, evals_parallel);
            assert_eq!(evals_serial, *evals_serial_cached.last().unwrap());
        }
    }

    #[test]
    /// Test that the `i`th vector of `evals_serial_cached` is equivalent to
    /// `evals(&r[..i])`, for all `i`.
    fn test_evals_cached() {
        let mut rng = test_rng();
        for len in 2..22 {
            let r = (0..len)
                .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
                .collect::<Vec<_>>();
            let evals_serial_cached = EqPolynomial::<Fr>::evals_serial_cached(&r, None);
            for i in 0..len {
                let evals = EqPolynomial::<Fr>::evals(&r[..i]);
                assert_eq!(evals_serial_cached[i], evals);
            }
        }
    }
}
