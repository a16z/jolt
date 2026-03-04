//! Dense multilinear polynomial stored as evaluations over the Boolean hypercube.

use std::borrow::Cow;

use jolt_field::Field;
use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use crate::eq::EqPolynomial;
use crate::serde_canonical::vec_canonical;
use crate::traits::MultilinearPolynomial;

/// Parallelism threshold for bind and evaluation operations.
const PAR_THRESHOLD: usize = 1024;

/// Dense multilinear polynomial: stores all $2^n$ evaluations as `Vec<F>`.
///
/// A multilinear polynomial in $n$ variables is uniquely determined by its
/// $2^n$ evaluations on the Boolean hypercube $\{0,1\}^n$. This type stores
/// those evaluations explicitly and supports efficient in-place variable binding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DensePolynomial<F: Field> {
    #[serde(with = "vec_canonical")]
    evaluations: Vec<F>,
    num_vars: usize,
}

impl<F: Field> DensePolynomial<F> {
    /// Creates a polynomial from its evaluations over the Boolean hypercube.
    ///
    /// # Panics
    /// Panics if `evaluations.len()` is not a power of two (or zero).
    pub fn new(evaluations: Vec<F>) -> Self {
        let len = evaluations.len();
        if len == 0 {
            return Self {
                evaluations,
                num_vars: 0,
            };
        }
        assert!(
            len.is_power_of_two(),
            "evaluation count must be a power of two, got {len}"
        );
        let num_vars = len.trailing_zeros() as usize;
        Self {
            evaluations,
            num_vars,
        }
    }

    /// Creates the zero polynomial with $2^n$ evaluations all equal to zero.
    pub fn zeros(num_vars: usize) -> Self {
        Self {
            evaluations: vec![F::zero(); 1 << num_vars],
            num_vars,
        }
    }

    /// Creates a polynomial with random evaluations.
    pub fn random(num_vars: usize, rng: &mut impl RngCore) -> Self {
        let evaluations = (0..(1 << num_vars)).map(|_| F::random(rng)).collect();
        Self {
            evaluations,
            num_vars,
        }
    }

    /// Fixes the first variable to `scalar` in place, halving the number of evaluations.
    ///
    /// The evaluations table is laid out so that the first variable controls the
    /// upper/lower half split: indices `0..half` have $x_1 = 0$ and indices
    /// `half..2*half` have $x_1 = 1$. The result is:
    /// $$g(x_2, \ldots, x_n) = f(0, x_2, \ldots) + s \cdot (f(1, x_2, \ldots) - f(0, x_2, \ldots))$$
    ///
    /// This is the hot inner loop of sumcheck — every allocation matters.
    #[inline]
    pub fn bind_in_place(&mut self, scalar: F) {
        let half = self.evaluations.len() / 2;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                let (lo, hi) = self.evaluations.split_at_mut(half);
                lo.par_iter_mut().zip(hi.par_iter()).for_each(|(a, b)| {
                    *a = *a + scalar * (*b - *a);
                });
            } else {
                for i in 0..half {
                    let lo = self.evaluations[i];
                    let hi = self.evaluations[i + half];
                    self.evaluations[i] = lo + scalar * (hi - lo);
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..half {
                let lo = self.evaluations[i];
                let hi = self.evaluations[i + half];
                self.evaluations[i] = lo + scalar * (hi - lo);
            }
        }

        self.evaluations.truncate(half);
        self.num_vars -= 1;
    }

    /// Evaluates the polynomial by sequentially binding each variable, consuming `self`.
    ///
    /// More memory-efficient than `evaluate` when the polynomial is no longer needed,
    /// as it avoids materializing the full eq table.
    pub fn evaluate_and_consume(mut self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        for &r in point {
            self.bind_in_place(r);
        }
        debug_assert_eq!(self.evaluations.len(), 1);
        self.evaluations[0]
    }

    /// Direct read access to the evaluation buffer.
    #[inline]
    pub fn evaluations(&self) -> &[F] {
        &self.evaluations
    }

    /// Mutable access to the evaluation buffer.
    #[inline]
    pub fn evaluations_mut(&mut self) -> &mut [F] {
        &mut self.evaluations
    }
}

impl<F: Field> MultilinearPolynomial<F> for DensePolynomial<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn len(&self) -> usize {
        self.evaluations.len()
    }

    fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        let eq_evals = EqPolynomial::new(point.to_vec()).evaluations();

        #[cfg(feature = "parallel")]
        {
            if self.evaluations.len() >= PAR_THRESHOLD {
                use rayon::prelude::*;
                return self
                    .evaluations
                    .par_iter()
                    .zip(eq_evals.par_iter())
                    .map(|(&f, &e)| f * e)
                    .sum();
            }
        }

        self.evaluations
            .iter()
            .zip(eq_evals.iter())
            .map(|(&f, &e)| f * e)
            .sum()
    }

    fn bind(&self, scalar: F) -> DensePolynomial<F> {
        let half = self.evaluations.len() / 2;
        let mut result = Vec::with_capacity(half);

        for i in 0..half {
            let lo = self.evaluations[i];
            let hi = self.evaluations[i + half];
            result.push(lo + scalar * (hi - lo));
        }

        DensePolynomial {
            evaluations: result,
            num_vars: self.num_vars - 1,
        }
    }

    fn evaluations(&self) -> Cow<'_, [F]> {
        Cow::Borrowed(&self.evaluations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use jolt_field::Field;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn bind_then_evaluate_equals_direct_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let n = 5;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let direct = poly.evaluate(&point);

        let bound = poly.bind(point[0]);
        let via_bind = bound.evaluate(&point[1..]);

        assert_eq!(direct, via_bind);
    }

    #[test]
    fn zeros_evaluates_to_zero() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let n = 4;
        let poly = DensePolynomial::<Fr>::zeros(n);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        assert!(poly.evaluate(&point).is_zero());
    }

    #[test]
    fn bind_in_place_matches_bind() {
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let n = 6;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let scalar = Fr::random(&mut rng);

        let bound = poly.bind(scalar);

        let mut poly_mut = poly;
        poly_mut.bind_in_place(scalar);

        assert_eq!(bound.evaluations(), poly_mut.evaluations());
    }

    #[test]
    fn evaluate_and_consume_matches_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(4);
        let n = 4;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let expected = poly.evaluate(&point);
        let consumed = poly.clone().evaluate_and_consume(&point);
        assert_eq!(expected, consumed);
    }

    #[test]
    fn empty_polynomial() {
        let poly = DensePolynomial::<Fr>::new(vec![]);
        assert_eq!(MultilinearPolynomial::num_vars(&poly), 0);
        assert!(MultilinearPolynomial::is_empty(&poly));
    }

    #[test]
    fn single_evaluation() {
        let val = Fr::from_u64(42);
        let poly = DensePolynomial::new(vec![val]);
        assert_eq!(MultilinearPolynomial::num_vars(&poly), 0);
        assert_eq!(poly.evaluate(&[]), val);
    }

    #[test]
    fn sequential_bind_equals_full_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(5);
        let n = 4;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let mut p = poly.clone();
        for &r in &point {
            p.bind_in_place(r);
        }
        assert_eq!(p.evaluations.len(), 1);
        assert_eq!(p.evaluations[0], poly.evaluate(&point));
    }

    #[allow(unused)]
    fn uses_one_trait() {
        let _ = Fr::one();
    }

    #[test]
    fn serde_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let poly = DensePolynomial::<Fr>::random(4, &mut rng);
        let bytes = bincode::serialize(&poly).unwrap();
        let recovered: DensePolynomial<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn serde_round_trip_empty() {
        let poly = DensePolynomial::<Fr>::new(vec![]);
        let bytes = bincode::serialize(&poly).unwrap();
        let recovered: DensePolynomial<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn serde_round_trip_single() {
        let poly = DensePolynomial::new(vec![Fr::from_u64(99)]);
        let bytes = bincode::serialize(&poly).unwrap();
        let recovered: DensePolynomial<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn parallel_bind_in_place_matches_bind() {
        // n=11 -> 2048 evaluations, above PAR_THRESHOLD=1024
        let mut rng = ChaCha20Rng::seed_from_u64(201);
        let n = 11;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let scalar = Fr::random(&mut rng);

        let bound = poly.bind(scalar);

        let mut poly_mut = poly;
        poly_mut.bind_in_place(scalar);

        assert_eq!(bound.evaluations(), poly_mut.evaluations());
    }

    #[test]
    fn parallel_bind_in_place_equals_evaluate_and_consume() {
        // Verifies the parallel bind_in_place path produces the same
        // result as evaluate_and_consume (which also uses bind_in_place).
        let mut rng = ChaCha20Rng::seed_from_u64(202);
        let n = 11;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let consumed = poly.clone().evaluate_and_consume(&point);

        let mut p = poly;
        for &r in &point {
            p.bind_in_place(r);
        }
        assert_eq!(p.evaluations.len(), 1);
        assert_eq!(p.evaluations[0], consumed);
    }

    #[test]
    fn parallel_bind_then_evaluate_and_consume() {
        // bind at n=11 triggers parallel path, then evaluate_and_consume the rest
        let mut rng = ChaCha20Rng::seed_from_u64(203);
        let n = 11;
        let poly = DensePolynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let expected = poly.clone().evaluate_and_consume(&point);

        let bound = poly.bind(point[0]);
        let via_bind = bound.evaluate_and_consume(&point[1..]);
        assert_eq!(expected, via_bind);
    }
}
