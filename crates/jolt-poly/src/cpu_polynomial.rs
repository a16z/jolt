//! Polynomial stored as evaluations over the Boolean hypercube.

use std::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use jolt_field::Field;
use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use crate::eq::EqPolynomial;

/// Minimum number of evaluations before parallelizing bind/evaluate.
///
/// Below this threshold the overhead of Rayon work-stealing exceeds the
/// benefit. 1024 field elements is roughly one L1 cache line's worth of
/// useful work per core, keeping synchronization cost negligible.
const PAR_THRESHOLD: usize = 1024;

/// Multilinear polynomial stored as evaluations over the Boolean hypercube $\{0,1\}^n$.
///
/// Generic over the coefficient type `T`:
/// - When `T` is a [`Field`] type: full polynomial with in-place [`bind`](Polynomial::bind),
///   [`evaluate`](Polynomial::evaluate), and arithmetic operators.
/// - When `T` is a small type (`u8`, `bool`, `i64`, etc.): compact storage with
///   [`bind_to_field`](Polynomial::bind_to_field) for on-demand field promotion.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: for<'a> Deserialize<'a>",))]
pub struct Polynomial<T> {
    coefficients: Vec<T>,
    num_vars: usize,
}

impl<T> Polynomial<T> {
    /// Creates a polynomial from its evaluations over the Boolean hypercube.
    ///
    /// # Panics
    /// Panics if `coefficients.len()` is not a power of two (or zero).
    pub fn new(coefficients: Vec<T>) -> Self {
        let len = coefficients.len();
        if len == 0 {
            return Self {
                coefficients,
                num_vars: 0,
            };
        }
        assert!(
            len.is_power_of_two(),
            "evaluation count must be a power of two, got {len}"
        );
        let num_vars = len.trailing_zeros() as usize;
        Self {
            coefficients,
            num_vars,
        }
    }

    /// Number of variables `n`. The polynomial has `2^n` evaluations.
    #[inline]
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of evaluations (`2^n`).
    #[inline]
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns `true` if the polynomial has no evaluations.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The raw evaluation table over the Boolean hypercube.
    #[inline]
    pub fn coefficients(&self) -> &[T] {
        &self.coefficients
    }
}

impl<T: Copy> Polynomial<T> {
    /// Fixes the first variable to `scalar`, promoting all coefficients to field elements.
    ///
    /// Produces a `Polynomial<F>` with `n − 1` variables:
    /// $$g(x_2, \ldots, x_n) = (1 - s) \cdot f(0, x_2, \ldots) + s \cdot f(1, x_2, \ldots)$$
    ///
    /// When `T = F`, the `From` conversion is the identity and the compiler
    /// eliminates it, making this equivalent to an allocating bind.
    pub fn bind_to_field<F: Field + From<T>>(&self, scalar: F) -> Polynomial<F> {
        let half = self.coefficients.len() / 2;
        let mut result = Vec::with_capacity(half);
        for i in 0..half {
            let lo: F = self.coefficients[i].into();
            let hi: F = self.coefficients[i + half].into();
            result.push(lo + scalar * (hi - lo));
        }
        Polynomial {
            coefficients: result,
            num_vars: self.num_vars - 1,
        }
    }
}

impl<F: Field> Polynomial<F> {
    /// Creates the zero polynomial with $2^n$ evaluations all equal to zero.
    pub fn zeros(num_vars: usize) -> Self {
        Self {
            coefficients: vec![F::zero(); 1 << num_vars],
            num_vars,
        }
    }

    /// Creates a polynomial with random evaluations.
    pub fn random(num_vars: usize, rng: &mut impl RngCore) -> Self {
        let coefficients = (0..(1 << num_vars)).map(|_| F::random(rng)).collect();
        Self {
            coefficients,
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
    pub fn bind(&mut self, scalar: F) {
        let half = self.coefficients.len() / 2;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                let (lo, hi) = self.coefficients.split_at_mut(half);
                lo.par_iter_mut().zip(hi.par_iter()).for_each(|(a, b)| {
                    *a = *a + scalar * (*b - *a);
                });
            } else {
                for i in 0..half {
                    let lo = self.coefficients[i];
                    let hi = self.coefficients[i + half];
                    self.coefficients[i] = lo + scalar * (hi - lo);
                }
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            for i in 0..half {
                let lo = self.coefficients[i];
                let hi = self.coefficients[i + half];
                self.coefficients[i] = lo + scalar * (hi - lo);
            }
        }

        self.coefficients.truncate(half);
        self.num_vars -= 1;
    }

    /// Evaluates the polynomial at `point` using the multilinear extension formula:
    /// $$f(r) = \sum_{x \in \{0,1\}^n} f(x) \cdot \widetilde{eq}(x, r)$$
    pub fn evaluate(&self, point: &[F]) -> F {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point dimension must match num_vars"
        );
        let eq_evals = EqPolynomial::new(point.to_vec()).evaluations();

        #[cfg(feature = "parallel")]
        {
            if self.coefficients.len() >= PAR_THRESHOLD {
                use rayon::prelude::*;
                return self
                    .coefficients
                    .par_iter()
                    .zip(eq_evals.par_iter())
                    .map(|(&f, &e)| f * e)
                    .sum();
            }
        }

        self.coefficients
            .iter()
            .zip(eq_evals.iter())
            .map(|(&f, &e)| f * e)
            .sum()
    }

    /// Evaluates by sequentially binding each variable, consuming `self`.
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
            self.bind(r);
        }
        debug_assert_eq!(self.coefficients.len(), 1);
        self.coefficients[0]
    }

    /// The evaluation table as a shared slice.
    #[inline]
    pub fn evaluations(&self) -> &[F] {
        &self.coefficients
    }

    /// The evaluation table as a mutable slice.
    #[inline]
    pub fn evaluations_mut(&mut self) -> &mut [F] {
        &mut self.coefficients
    }
}

impl<F: Field> crate::MultilinearEvaluation<F> for Polynomial<F> {
    #[inline]
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    #[inline]
    fn len(&self) -> usize {
        self.coefficients.len()
    }

    fn evaluate(&self, point: &[F]) -> F {
        Polynomial::evaluate(self, point)
    }
}

impl<F: Field> crate::MultilinearBinding<F> for Polynomial<F> {
    fn bind(&mut self, scalar: F) {
        Polynomial::bind(self, scalar);
    }
}

#[inline]
fn assert_matching_dims<F: Field>(a: &Polynomial<F>, b: &Polynomial<F>) -> (usize, usize) {
    assert_eq!(
        a.num_vars, b.num_vars,
        "num_vars mismatch: {} vs {}",
        a.num_vars, b.num_vars
    );
    (a.num_vars, a.coefficients.len())
}

impl<F: Field> Add for Polynomial<F> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self {
        self += &rhs;
        self
    }
}

impl<F: Field> Add<&Self> for Polynomial<F> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self {
        self += rhs;
        self
    }
}

impl<F: Field> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self += &rhs;
    }
}

impl<F: Field> AddAssign<&Self> for Polynomial<F> {
    fn add_assign(&mut self, rhs: &Self) {
        let (_nv, len) = assert_matching_dims(self, rhs);

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.coefficients
                    .par_iter_mut()
                    .zip(rhs.coefficients.par_iter())
                    .for_each(|(a, b)| *a += *b);
                return;
            }
        }

        for i in 0..len {
            self.coefficients[i] += rhs.coefficients[i];
        }
    }
}

impl<F: Field> Sub for Polynomial<F> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self {
        self -= &rhs;
        self
    }
}

impl<F: Field> Sub<&Self> for Polynomial<F> {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self {
        self -= rhs;
        self
    }
}

impl<F: Field> SubAssign for Polynomial<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self -= &rhs;
    }
}

impl<F: Field> SubAssign<&Self> for Polynomial<F> {
    fn sub_assign(&mut self, rhs: &Self) {
        let (_nv, len) = assert_matching_dims(self, rhs);

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.coefficients
                    .par_iter_mut()
                    .zip(rhs.coefficients.par_iter())
                    .for_each(|(a, b)| *a -= *b);
                return;
            }
        }

        for i in 0..len {
            self.coefficients[i] -= rhs.coefficients[i];
        }
    }
}

impl<F: Field> Mul<F> for Polynomial<F> {
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self {
        let len = self.coefficients.len();

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.coefficients.par_iter_mut().for_each(|a| *a *= rhs);
                return self;
            }
        }

        for i in 0..len {
            self.coefficients[i] *= rhs;
        }
        self
    }
}

impl<F: Field> Mul<F> for &Polynomial<F> {
    type Output = Polynomial<F>;

    fn mul(self, rhs: F) -> Polynomial<F> {
        self.clone() * rhs
    }
}

impl<F: Field> Neg for Polynomial<F> {
    type Output = Self;

    fn neg(mut self) -> Self {
        let len = self.coefficients.len();

        #[cfg(feature = "parallel")]
        {
            if len >= PAR_THRESHOLD {
                use rayon::prelude::*;
                self.coefficients.par_iter_mut().for_each(|a| *a = -*a);
                return self;
            }
        }

        for i in 0..len {
            self.coefficients[i] = -self.coefficients[i];
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Field;
    use jolt_field::Fr;
    use num_traits::{One, Zero};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    #[test]
    fn bind_to_field_then_evaluate_equals_direct_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let n = 5;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let direct = poly.evaluate(&point);

        let bound = poly.bind_to_field(point[0]);
        let via_bind = bound.evaluate(&point[1..]);

        assert_eq!(direct, via_bind);
    }

    #[test]
    fn zeros_evaluates_to_zero() {
        let mut rng = ChaCha20Rng::seed_from_u64(2);
        let n = 4;
        let poly = Polynomial::<Fr>::zeros(n);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();
        assert!(poly.evaluate(&point).is_zero());
    }

    #[test]
    fn bind_matches_bind_to_field() {
        let mut rng = ChaCha20Rng::seed_from_u64(3);
        let n = 6;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let scalar = Fr::random(&mut rng);

        let bound = poly.bind_to_field(scalar);

        let mut poly_mut = poly;
        poly_mut.bind(scalar);

        assert_eq!(bound.evaluations(), poly_mut.evaluations());
    }

    #[test]
    fn evaluate_and_consume_matches_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(4);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let expected = poly.evaluate(&point);
        let consumed = poly.clone().evaluate_and_consume(&point);
        assert_eq!(expected, consumed);
    }

    #[test]
    fn empty_polynomial() {
        let poly = Polynomial::<Fr>::new(vec![]);
        assert_eq!(poly.num_vars(), 0);
        assert!(poly.is_empty());
    }

    #[test]
    fn single_evaluation() {
        let val = Fr::from_u64(42);
        let poly = Polynomial::new(vec![val]);
        assert_eq!(poly.num_vars(), 0);
        assert_eq!(poly.evaluate(&[]), val);
    }

    #[test]
    fn sequential_bind_equals_full_evaluate() {
        let mut rng = ChaCha20Rng::seed_from_u64(5);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let mut p = poly.clone();
        for &r in &point {
            p.bind(r);
        }
        assert_eq!(p.coefficients.len(), 1);
        assert_eq!(p.coefficients[0], poly.evaluate(&point));
    }

    #[allow(unused)]
    fn uses_one_trait() {
        let _ = Fr::one();
    }

    #[test]
    fn serde_round_trip() {
        let mut rng = ChaCha20Rng::seed_from_u64(100);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        let bytes = bincode::serialize(&poly).unwrap();
        let recovered: Polynomial<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn serde_round_trip_empty() {
        let poly = Polynomial::<Fr>::new(vec![]);
        let bytes = bincode::serialize(&poly).unwrap();
        let recovered: Polynomial<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn serde_round_trip_single() {
        let poly = Polynomial::new(vec![Fr::from_u64(99)]);
        let bytes = bincode::serialize(&poly).unwrap();
        let recovered: Polynomial<Fr> = bincode::deserialize(&bytes).unwrap();
        assert_eq!(poly, recovered);
    }

    #[test]
    fn parallel_bind_matches_bind_to_field() {
        // n=11 -> 2048 evaluations, above PAR_THRESHOLD=1024
        let mut rng = ChaCha20Rng::seed_from_u64(201);
        let n = 11;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let scalar = Fr::random(&mut rng);

        let bound = poly.bind_to_field(scalar);

        let mut poly_mut = poly;
        poly_mut.bind(scalar);

        assert_eq!(bound.evaluations(), poly_mut.evaluations());
    }

    #[test]
    fn parallel_bind_equals_evaluate_and_consume() {
        let mut rng = ChaCha20Rng::seed_from_u64(202);
        let n = 11;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let consumed = poly.clone().evaluate_and_consume(&point);

        let mut p = poly;
        for &r in &point {
            p.bind(r);
        }
        assert_eq!(p.coefficients.len(), 1);
        assert_eq!(p.coefficients[0], consumed);
    }

    #[test]
    fn parallel_bind_then_evaluate_and_consume() {
        let mut rng = ChaCha20Rng::seed_from_u64(203);
        let n = 11;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let expected = poly.clone().evaluate_and_consume(&point);

        let bound = poly.bind_to_field(point[0]);
        let via_bind = bound.evaluate_and_consume(&point[1..]);
        assert_eq!(expected, via_bind);
    }

    #[test]
    fn add_element_wise() {
        let mut rng = ChaCha20Rng::seed_from_u64(500);
        let n = 4;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);

        let sum = a.clone() + &b;
        for i in 0..sum.evaluations().len() {
            assert_eq!(
                sum.evaluations()[i],
                a.evaluations()[i] + b.evaluations()[i]
            );
        }
    }

    #[test]
    fn sub_element_wise() {
        let mut rng = ChaCha20Rng::seed_from_u64(501);
        let n = 4;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);

        let diff = a.clone() - &b;
        for i in 0..diff.evaluations().len() {
            assert_eq!(
                diff.evaluations()[i],
                a.evaluations()[i] - b.evaluations()[i]
            );
        }
    }

    #[test]
    fn scalar_mul() {
        let mut rng = ChaCha20Rng::seed_from_u64(502);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let s = Fr::random(&mut rng);

        let scaled = poly.clone() * s;
        for i in 0..scaled.evaluations().len() {
            assert_eq!(scaled.evaluations()[i], poly.evaluations()[i] * s);
        }
    }

    #[test]
    fn negation() {
        let mut rng = ChaCha20Rng::seed_from_u64(503);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);

        let neg = -poly.clone();
        for i in 0..neg.evaluations().len() {
            assert_eq!(neg.evaluations()[i], -poly.evaluations()[i]);
        }
    }

    #[test]
    fn add_preserves_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(504);
        let n = 5;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let sum = a.clone() + &b;
        assert_eq!(
            sum.evaluate(&point),
            a.evaluate(&point) + b.evaluate(&point)
        );
    }

    #[test]
    fn sub_preserves_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(505);
        let n = 5;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let diff = a.clone() - &b;
        assert_eq!(
            diff.evaluate(&point),
            a.evaluate(&point) - b.evaluate(&point)
        );
    }

    #[test]
    fn scalar_mul_preserves_evaluation() {
        let mut rng = ChaCha20Rng::seed_from_u64(506);
        let n = 5;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let s = Fr::random(&mut rng);
        let point: Vec<Fr> = (0..n).map(|_| Fr::random(&mut rng)).collect();

        let scaled = poly.clone() * s;
        assert_eq!(scaled.evaluate(&point), poly.evaluate(&point) * s);
    }

    #[test]
    #[should_panic(expected = "num_vars mismatch")]
    fn add_mismatched_num_vars_panics() {
        let mut rng = ChaCha20Rng::seed_from_u64(510);
        let a = Polynomial::<Fr>::random(3, &mut rng);
        let b = Polynomial::<Fr>::random(4, &mut rng);
        let _ = a + b;
    }

    #[test]
    fn add_assign_accumulation() {
        let mut rng = ChaCha20Rng::seed_from_u64(511);
        let n = 4;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);
        let c = Polynomial::<Fr>::random(n, &mut rng);

        let mut acc = a.clone();
        acc += &b;
        acc += &c;

        let expected = a.clone() + &b + &c;
        assert_eq!(acc, expected);
    }

    #[test]
    fn neg_double_is_identity() {
        let mut rng = ChaCha20Rng::seed_from_u64(512);
        let poly = Polynomial::<Fr>::random(4, &mut rng);
        assert_eq!(-(-poly.clone()), poly);
    }

    #[test]
    fn add_sub_inverse() {
        let mut rng = ChaCha20Rng::seed_from_u64(513);
        let n = 4;
        let a = Polynomial::<Fr>::random(n, &mut rng);
        let b = Polynomial::<Fr>::random(n, &mut rng);

        let result = (a.clone() + &b) - &b;
        assert_eq!(result, a);
    }

    #[test]
    fn ref_scalar_mul() {
        let mut rng = ChaCha20Rng::seed_from_u64(514);
        let n = 4;
        let poly = Polynomial::<Fr>::random(n, &mut rng);
        let s = Fr::random(&mut rng);

        let owned_result = poly.clone() * s;
        let ref_result = &poly * s;
        assert_eq!(owned_result, ref_result);
    }

    #[test]
    fn compact_u8_bind_to_field_matches_dense() {
        let scalars: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_u8_sequential_bind_matches_evaluate() {
        let scalars: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(10);
        let point: Vec<Fr> = (0..3).map(|_| Fr::random(&mut rng)).collect();

        let mut bound = compact.bind_to_field::<Fr>(point[0]);
        for &r in &point[1..] {
            bound.bind(r);
        }
        assert_eq!(bound.coefficients[0], dense.evaluate(&point));
    }

    #[test]
    fn compact_bool_bind_to_field_matches_dense() {
        let scalars: Vec<bool> = vec![true, false, false, true];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(20);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_u16_bind_to_field_matches_dense() {
        let scalars: Vec<u16> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(30);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_i64_bind_to_field_matches_dense() {
        let scalars: Vec<i64> = vec![-1, 0, 1, -100, i64::MIN, i64::MAX, -42, 42];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(50);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_i128_bind_to_field_matches_dense() {
        let scalars: Vec<i128> = vec![-1, 0, 1, -999, i128::MIN, i128::MAX, -7, 7];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(60);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_u128_bind_to_field_matches_dense() {
        let scalars: Vec<u128> = vec![u128::MAX, u128::MAX - 1, 0, 1];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(70);
        let scalar = Fr::random(&mut rng);

        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            dense.bind_to_field(scalar)
        );
    }

    #[test]
    fn compact_bind_chain_consistency() {
        let scalars: Vec<u32> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let compact = Polynomial::new(scalars.clone());
        let dense_evals: Vec<Fr> = scalars.iter().map(|&s| Fr::from(s)).collect();
        let dense = Polynomial::new(dense_evals);

        let mut rng = ChaCha20Rng::seed_from_u64(80);
        let r1 = Fr::random(&mut rng);
        let r2 = Fr::random(&mut rng);
        let remaining: Vec<Fr> = (0..1).map(|_| Fr::random(&mut rng)).collect();

        // bind_to_field(r1) then bind(r2) should match dense evaluate
        let mut bound = compact.bind_to_field::<Fr>(r1);
        bound.bind(r2);
        let result = bound.evaluate(&remaining);

        let mut full_point = vec![r1, r2];
        full_point.extend_from_slice(&remaining);
        assert_eq!(result, dense.evaluate(&full_point));
    }

    #[test]
    fn compact_empty() {
        let compact = Polynomial::<u8>::new(vec![]);
        assert_eq!(compact.num_vars(), 0);
        assert!(compact.is_empty());
    }

    #[test]
    fn compact_single_element() {
        let compact = Polynomial::<u64>::new(vec![42]);
        assert_eq!(compact.num_vars(), 0);
        assert_eq!(compact.coefficients(), &[42u64]);
    }

    #[test]
    fn serde_round_trip_compact_u8() {
        let scalars: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let compact = Polynomial::new(scalars);
        let bytes = bincode::serialize(&compact).unwrap();
        let recovered: Polynomial<u8> = bincode::deserialize(&bytes).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(40);
        let scalar = Fr::random(&mut rng);
        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            recovered.bind_to_field::<Fr>(scalar)
        );
    }

    #[test]
    fn serde_round_trip_compact_bool() {
        let scalars: Vec<bool> = vec![true, false, true, false];
        let compact = Polynomial::new(scalars);
        let bytes = bincode::serialize(&compact).unwrap();
        let recovered: Polynomial<bool> = bincode::deserialize(&bytes).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(41);
        let scalar = Fr::random(&mut rng);
        assert_eq!(
            compact.bind_to_field::<Fr>(scalar),
            recovered.bind_to_field::<Fr>(scalar)
        );
    }
}
