#![allow(clippy::too_many_arguments)]
#![allow(clippy::uninlined_format_args)]
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::PolynomialEvaluation;
use crate::utils::thread::unsafe_allocate_zero_vec;
use crate::utils::{compute_dotproduct, compute_dotproduct_low_optimized};

use crate::field::{JoltField, OptimizedMul};
use crate::utils::math::Math;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use core::ops::Index;
use rand_core::{CryptoRng, RngCore};
use rayon::prelude::*;

use super::multilinear_polynomial::BindingOrder;

#[derive(Default, Debug, PartialEq, CanonicalSerialize, CanonicalDeserialize)]
pub struct DensePolynomial<F: JoltField> {
    pub num_vars: usize, // the number of variables in the multilinear polynomial
    pub len: usize,
    pub Z: Vec<F>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
    binding_scratch_space: Option<Vec<F>>,
}

impl<F: JoltField> DensePolynomial<F> {
    pub fn new(Z: Vec<F>) -> Self {
        assert!(
            Z.len().is_power_of_two(),
            "Dense multi-linear polynomials must be made from a power of 2 (not {})",
            Z.len()
        );

        DensePolynomial {
            num_vars: Z.len().log_2(),
            len: Z.len(),
            Z,
            binding_scratch_space: None,
        }
    }

    pub fn new_padded(evals: Vec<F>) -> Self {
        // Pad non-power-2 evaluations to fill out the dense multilinear polynomial
        let mut poly_evals = evals;
        while !(poly_evals.len().is_power_of_two()) {
            poly_evals.push(F::zero());
        }

        DensePolynomial {
            num_vars: poly_evals.len().log_2(),
            len: poly_evals.len(),
            Z: poly_evals,
            binding_scratch_space: None,
        }
    }

    pub fn get_num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn is_bound(&self) -> bool {
        self.len != self.Z.len()
    }

    pub fn bind(&mut self, r: F, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bound_poly_var_bot(&r),
            BindingOrder::HighToLow => self.bound_poly_var_top(&r),
        }
    }

    pub fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        match order {
            BindingOrder::LowToHigh => self.bound_poly_var_bot_01_optimized(&r),
            BindingOrder::HighToLow => self.bound_poly_var_top_zero_optimized(&r),
        }
    }

    pub fn bound_poly_var_top(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
            *a += *r * (*b - *a);
        });

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_top_many_ones(&mut self, r: &F) {
        let n = self.len() / 2;
        let (left, right) = self.Z.split_at_mut(n);

        left.iter_mut()
            .zip(right.iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                let m = *b - *a;
                if m.is_one() {
                    *a += *r;
                } else {
                    *a += *r * m;
                }
            });

        self.num_vars -= 1;
        self.len = n;
    }

    /// Bounds the polynomial's most significant index bit to 'r' optimized for a
    /// high P(eval = 0).
    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_top_zero_optimized(&mut self, r: &F) {
        let n = self.len() / 2;

        let (left, right) = self.Z.split_at_mut(n);

        left.par_iter_mut()
            .zip(right.par_iter())
            .filter(|(&mut a, &b)| a != b)
            .for_each(|(a, b)| {
                *a += *r * (*b - *a);
            });

        self.num_vars -= 1;
        self.len = n;
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            let low = self.Z[i];
            let high = self.Z[i + n];
            if !(low.is_zero() && high.is_zero()) {
                let m = high - low;
                new_evals[i] = low + *r * m;
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
            binding_scratch_space: None,
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn new_poly_from_bound_poly_var_top_flags(&self, r: &F) -> Self {
        let n = self.len() / 2;
        let mut new_evals: Vec<F> = unsafe_allocate_zero_vec(n);

        for i in 0..n {
            // let low' = low + r * (high - low)
            // Special truth table here
            //         high 0   high 1
            // low 0     0        r
            // low 1   (1-r)      1
            let low = self.Z[i];
            let high = self.Z[i + n];

            if low.is_zero() {
                if high.is_one() {
                    new_evals[i] = *r;
                } else if !high.is_zero() {
                    panic!("Shouldn't happen for a flag poly");
                }
            } else if low.is_one() {
                if high.is_one() {
                    new_evals[i] = F::one();
                } else if high.is_zero() {
                    new_evals[i] = F::one() - *r;
                } else {
                    panic!("Shouldn't happen for a flag poly");
                }
            }
        }
        let num_vars = self.num_vars - 1;
        let len = n;

        Self {
            num_vars,
            len,
            Z: new_evals,
            binding_scratch_space: None,
        }
    }

    /// Note: does not truncate
    #[tracing::instrument(skip_all)]
    pub fn bound_poly_var_bot(&mut self, r: &F) {
        let n = self.len() / 2;
        for i in 0..n {
            self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
        }

        self.num_vars -= 1;
        self.len = n;
    }

    pub fn bound_poly_var_bot_01_optimized(&mut self, r: &F) {
        let n = self.len() / 2;

        if self.binding_scratch_space.is_none() {
            self.binding_scratch_space = Some(unsafe_allocate_zero_vec(n));
        }

        let scratch_space = self.binding_scratch_space.as_mut().unwrap();

        scratch_space
            .par_iter_mut()
            .take(n)
            .enumerate()
            .for_each(|(i, z)| {
                let m = self.Z[2 * i + 1] - self.Z[2 * i];
                *z = if m.is_zero() {
                    self.Z[2 * i]
                } else if m.is_one() {
                    self.Z[2 * i] + r
                } else {
                    self.Z[2 * i] + *r * m
                }
            });

        std::mem::swap(&mut self.Z, scratch_space);

        self.num_vars -= 1;
        self.len = n;
    }

    // returns Z(r) in O(n) time
    pub fn evaluate(&self, r: &[F]) -> F {
        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let chis = EqPolynomial::evals(r);
        assert_eq!(chis.len(), self.Z.len());
        compute_dotproduct(&self.Z, &chis)
    }
    pub fn split_eq_evaluate(&self, r: &[F], eq_one: &[F], eq_two: &[F]) -> F {
        const PARALLEL_THRESHOLD: usize = 16;
        if r.len() < PARALLEL_THRESHOLD {
            self.evaluate_split_eq_serial(eq_one, eq_two)
        } else {
            self.evaluate_split_eq_parallel(eq_one, eq_two)
        }
    }
    fn evaluate_split_eq_parallel(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .into_par_iter()
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .into_par_iter()
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        OptimizedMul::mul_01_optimized(eq_two[x2], self.Z[idx])
                    })
                    .reduce(|| F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(eq_one[x1], partial_sum)
            })
            .reduce(|| F::zero(), |acc, val| acc + val);
        eval
    }

    fn evaluate_split_eq_serial(&self, eq_one: &[F], eq_two: &[F]) -> F {
        let eval: F = (0..eq_one.len())
            .map(|x1| {
                let partial_sum = (0..eq_two.len())
                    .map(|x2| {
                        let idx = x1 * eq_two.len() + x2;
                        OptimizedMul::mul_01_optimized(eq_two[x2], self.Z[idx])
                    })
                    .fold(F::zero(), |acc, val| acc + val);
                OptimizedMul::mul_01_optimized(eq_one[x1], partial_sum)
            })
            .fold(F::zero(), |acc, val| acc + val);
        eval
    }

    // Faster evaluation based on
    // https://randomwalks.xyz/publish/fast_polynomial_evaluation.html
    // Shaves a factor of 2 from run time.
    pub fn inside_out_evaluate(&self, r: &[F]) -> F {
        // Copied over from eq_poly
        // If the number of variables are greater
        // than 2^16 -- use parallel evaluate
        // Below that it's better to just do things linearly.
        const PARALLEL_THRESHOLD: usize = 16;

        // r must have a value for each variable
        assert_eq!(r.len(), self.get_num_vars());
        let m = r.len();
        if m < PARALLEL_THRESHOLD {
            self.inside_out_serial(r)
        } else {
            self.inside_out_parallel(r)
        }
    }

    fn inside_out_serial(&self, r: &[F]) -> F {
        // r is expected to be big endinan
        // r[0] is the most significant digit
        let mut current = self.Z.clone();
        let m = r.len();
        for i in (0..m).rev() {
            let stride = 1 << i;

            // Note that as r is big endian
            // and i is reversed
            // r[m-1-i] actually starts at the big endian digit
            // and moves towards the little endian digit.
            for j in 0..stride {
                let f0 = current[j];
                let f1 = current[j + stride];
                let slope = f1 - f0;
                if slope.is_zero() {
                    current[j] = f0;
                } else if slope.is_one() {
                    current[j] = f0 + r[m - 1 - i];
                } else {
                    current[j] = f0 + r[m - 1 - i] * slope;
                }
            }
            // No benefit to truncating really.
            //current.truncate(stride);
        }
        current[0]
    }

    fn inside_out_parallel(&self, r: &[F]) -> F {
        let mut current: Vec<_> = self.Z.par_iter().cloned().collect();
        let m = r.len();
        // Invoking the same parallelisation structure
        // currently in evaluating in Lagrange bases.
        // See eq_poly::evals()
        for i in (0..m).rev() {
            let stride = 1 << i;
            let r_val = r[m - 1 - i];
            let (evals_left, evals_right) = current.split_at_mut(stride);
            let (evals_right, _) = evals_right.split_at_mut(stride);

            evals_left
                .par_iter_mut()
                .zip(evals_right.par_iter())
                .for_each(|(x, y)| {
                    let slope = *y - *x;
                    if slope.is_zero() {
                        return;
                    }
                    if slope.is_one() {
                        *x += r_val;
                    } else {
                        *x += r_val * slope;
                    }
                });
        }
        current[0]
    }
    pub fn evaluate_at_chi(&self, chis: &[F]) -> F {
        compute_dotproduct(&self.Z, chis)
    }

    pub fn evaluate_at_chi_low_optimized(&self, chis: &[F]) -> F {
        assert_eq!(self.Z.len(), chis.len());
        compute_dotproduct_low_optimized(&self.Z, chis)
    }

    pub fn evals(&self) -> Vec<F> {
        self.Z.clone()
    }

    pub fn evals_ref(&self) -> &[F] {
        self.Z.as_ref()
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_usize(Z: &[usize]) -> Self {
        DensePolynomial::new(
            (0..Z.len())
                .map(|i| F::from_u64(Z[i] as u64))
                .collect::<Vec<F>>(),
        )
    }

    #[tracing::instrument(skip_all, name = "DensePolynomial::from")]
    pub fn from_u64(Z: &[u64]) -> Self {
        DensePolynomial::new((0..Z.len()).map(|i| F::from_u64(Z[i])).collect::<Vec<F>>())
    }

    pub fn random<R: RngCore + CryptoRng>(num_vars: usize, mut rng: &mut R) -> Self {
        Self::new(
            std::iter::from_fn(|| Some(F::random(&mut rng)))
                .take(1 << num_vars)
                .collect(),
        )
    }
}

impl<F: JoltField> Clone for DensePolynomial<F> {
    fn clone(&self) -> Self {
        Self::new(self.Z[0..self.len].to_vec())
    }
}

impl<F: JoltField> Index<usize> for DensePolynomial<F> {
    type Output = F;

    #[inline(always)]
    fn index(&self, _index: usize) -> &F {
        &(self.Z[_index])
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for DensePolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        self.evaluate(r)
    }

    fn batch_evaluate(polys: &[&Self], r: &[F]) -> Vec<F> {
        let eq = EqPolynomial::evals(r);
        let evals: Vec<F> = polys
            .into_par_iter()
            .map(|&poly| poly.evaluate_at_chi_low_optimized(&eq))
            .collect();
        evals
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.len() / 2);

        let mut evals = vec![F::zero(); degree];
        match order {
            BindingOrder::HighToLow => {
                evals[0] = self[index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self[index + self.len() / 2];
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
            BindingOrder::LowToHigh => {
                evals[0] = self[2 * index];
                if degree == 1 {
                    return evals;
                }
                let mut eval = self[2 * index + 1];
                let m = eval - evals[0];
                for i in 1..degree {
                    eval += m;
                    evals[i] = eval;
                }
            }
        };
        evals
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    pub fn compute_chis_at_r<F: JoltField>(r: &[F]) -> Vec<F> {
        let ell = r.len();
        let n = ell.pow2();
        let mut chis: Vec<F> = Vec::new();
        for i in 0..n {
            let mut chi_i = F::one();
            for j in 0..r.len() {
                let bit_j = (i & (1 << (r.len() - j - 1))) > 0;
                if bit_j {
                    chi_i *= r[j];
                } else {
                    chi_i *= F::one() - r[j];
                }
            }
            chis.push(chi_i);
        }
        chis
    }

    #[test]
    fn check_memoized_chis() {
        check_memoized_chis_helper::<Fr>()
    }

    fn check_memoized_chis_helper<F: JoltField>() {
        let mut prng = test_rng();

        let s = 10;
        let mut r: Vec<F> = Vec::new();
        for _i in 0..s {
            r.push(F::random(&mut prng));
        }
        let chis = compute_chis_at_r::<F>(&r);
        let chis_m = EqPolynomial::<F>::evals(&r);
        assert_eq!(chis, chis_m);
    }

    #[test]
    fn evaluation() {
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }
    #[test]
    fn compare_random_evaluations() {
        // Compares optimised polynomial evaluation
        // with the old polynomial evaluation
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        let mut rng = ChaCha20Rng::seed_from_u64(42);

        for &exp in &[2, 4, 6, 8] {
            let num_evals = 1 << exp; // must be a power of 2
            let num_vars = exp;

            // Generate random coefficients for the multilinear polynomial
            let evals: Vec<Fr> = (0..num_evals).map(|_| Fr::random(&mut rng)).collect();
            let poly = DensePolynomial::<Fr>::new(evals);

            // Try 10 random evaluation points
            for _ in 0..10 {
                let eval_point: Vec<Fr> = (0..num_vars).map(|_| Fr::random(&mut rng)).collect();

                let eval1 = poly.evaluate(&eval_point);
                let eval2 = poly.inside_out_evaluate(&eval_point);

                assert_eq!(
                    eval1, eval2,
                    "Mismatch at point {:?} for num_vars = {}: eval = {:?}, opt = {:?}",
                    eval_point, num_vars, eval1, eval2
                );
            }
        }
    }
    #[test]
    fn fast_evaluation() {
        // Uses the above test with the new polynomial evaluation
        // algorithm.
        let num_evals = 4;
        let mut evals: Vec<Fr> = Vec::with_capacity(num_evals);
        for _ in 0..num_evals {
            evals.push(Fr::from(8));
        }
        let dense_poly: DensePolynomial<Fr> = DensePolynomial::new(evals.clone());

        // Evaluate at 3:
        // (0, 0) = 1
        // (0, 1) = 1
        // (1, 0) = 1
        // (1, 1) = 1
        // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
        // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
        // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
        assert_eq!(
            dense_poly.inside_out_evaluate(vec![Fr::from(3), Fr::from(4)].as_slice()),
            Fr::from(8)
        );
    }
}
